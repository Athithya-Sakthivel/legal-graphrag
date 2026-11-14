"""Orchestration — IndexingPipeline stages.

Ingest-side stages live here: manifest, chunk, dedup.
Load-side stages live in pipeline_load.py: embed, load, validate, index.

Manifest Bloat Removed (v1.6):
  - The filesystem is now the single source of truth for skip logic.
  - Removed `_load_manifest`, `_persist_chunk_state`, and `_cleanup_orphans`.
  - Skip logic checks file existence and content hashes/line counts directly on disk.
  - A lightweight `_manifest.jsonl` is generated at the end of the `chunk` stage
    purely to satisfy the `dedup` stage's file-agreement checks.
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import json
import logging
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

import pyarrow.parquet as pq

from . import db
from .embedder import Embedder
from .filters import is_noise_chunk
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_jsonl_from_dict,
)
from .pipeline_helpers import (
    FAILURE_ABSOLUTE,
    FAILURE_CHECK_EVERY,
    FAILURE_RATE,
    Config,
    PipelineError,
    PlanEntry,
    StageReport,
    chunk_worker,
    resolve_corpus_rows,
    worker_init,
)

logger = logging.getLogger("indexing_pipeline")


# ────────────────────────────────────────────────────────────────────────
# Base: infrastructure shared by ingest and load stages
# ────────────────────────────────────────────────────────────────────────


class IndexingPipelineBase:
    """Shared infrastructure for ingest and downstream pipeline stages."""

    def __init__(self, config: Config) -> None:
        self.config = config.validated()
        self._embedder: Any = None
        self._database: db.Database | None = None
        self._tokenizer_fn: Callable[[str], int] | None = None
        self._closed = False
        self._lock_path = self.config.data_dir / ".indexing.lock"

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        errors: list[Exception] = []

        if self._embedder is not None:
            try:
                self._embedder.close()
            except Exception as exc:
                errors.append(exc)
            finally:
                self._embedder = None

        if self._database is not None:
            try:
                self._database.close()
            except Exception as exc:
                errors.append(exc)
            finally:
                self._database = None

        if errors:
            raise PipelineError(
                "close",
                "; ".join(str(error) for error in errors),
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("pipeline is closed")

    # ── Backend resolution ──────────────────────────────────────────

    def _get_embedder(self) -> Any:
        self._require_open()

        if self._embedder is None:
            if self.config.device.startswith("cuda"):
                import torch

                from .embedder import TorchEmbedder

                cfg_dev = self.config.device.strip()

                if "," in cfg_dev:
                    device: str | list[str] = [
                        value.strip() for value in cfg_dev.split(",") if value.strip()
                    ]
                    if not device:
                        raise PipelineError("embed", "CUDA device list is empty")
                elif cfg_dev == "cuda":
                    device = "cuda"
                else:
                    device = cfg_dev

                gpu_count = torch.cuda.device_count()

                logger.info(
                    "CUDA devices available: %d (%s)",
                    gpu_count,
                    ", ".join(torch.cuda.get_device_name(i) for i in range(gpu_count))
                    if gpu_count
                    else "none",
                )

                self._embedder = TorchEmbedder(
                    model_dir=self.config.gpu_model_dir,
                    max_seq_length=self.config.max_seq_length,
                    device=device,
                    batch_size=self.config.embed_batch,
                )
            else:
                logger.info(
                    "loading OpenVINO Embedder device=CPU model_dir=%s",
                    self.config.model_dir,
                )
                self._embedder = Embedder(
                    model_dir=self.config.model_dir,
                    max_seq_length=self.config.max_seq_length,
                    threads=self.config.inference_threads,
                    batch_size=self.config.embed_batch,
                    device="CPU",
                )

        return self._embedder

    def _get_database(self) -> db.Database:
        self._require_open()

        if not self.config.db_dsn:
            raise PipelineError(
                "database",
                "DATABASE_URL is required for this stage "
                "(load, validate, index, drop-index). "
                "The embed stage runs without a database.",
            )

        if self._database is None:
            self._database = db.Database(
                dsn=self.config.db_dsn,
                pool_min=self.config.pool_min,
                pool_max=self.config.pool_max,
            )

        return self._database

    def _get_tokenizer(self) -> Callable[[str], int]:
        """Load the tokenizer belonging to the active embedding backend.

        Fails loudly if the tokenizer cannot be loaded. We do not fall back
        to character-based heuristics, as that causes ID drift and boundary
        shifts on pipeline resume.
        """
        if self._tokenizer_fn is None:
            from transformers import AutoTokenizer

            if self.config.device.startswith("cuda"):
                primary = self.config.gpu_model_dir
                fallback = self.config.model_dir
            else:
                primary = self.config.model_dir
                fallback = self.config.gpu_model_dir

            if (primary / "tokenizer.json").is_file():
                tokenizer_dir = primary
            elif (fallback / "tokenizer.json").is_file():
                tokenizer_dir = fallback
                logger.warning(
                    "chunker tokenizer: using fallback %s (primary %s missing)",
                    tokenizer_dir,
                    primary,
                )
            else:
                raise PipelineError(
                    "chunk",
                    f"no tokenizer.json under {primary} or {fallback}",
                )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_dir),
                    local_files_only=True,
                )
            except Exception as exc:
                raise PipelineError(
                    "chunk",
                    f"could not load tokenizer from {tokenizer_dir}: {type(exc).__name__}: {exc}",
                ) from exc

            self._tokenizer_fn = lambda text: len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )

            logger.info(
                "chunker tokenizer loaded from %s (device=%s)",
                tokenizer_dir,
                self.config.device,
            )

        return self._tokenizer_fn

    # ── Locking ─────────────────────────────────────────────────────

    @contextmanager
    def _stage_lock(self) -> Iterator[None]:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        with self._lock_path.open("a+") as handle:
            try:
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise PipelineError(
                    "lock",
                    "another pipeline process is active",
                ) from exc

            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # ── Input checks ────────────────────────────────────────────────

    def _check_inputs(self, *, include_metadata: bool = True) -> None:
        required = [self.config.corpus_file]

        if include_metadata:
            required.extend(
                [
                    self.config.clusters_file,
                    self.config.edges_file,
                ]
            )

        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)

    # ── Parquet iteration ───────────────────────────────────────────

    def _iter_opinions(
        self,
        keep_ids: set[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        parquet = pq.ParquetFile(self.config.corpus_file)

        required = {"cl_opinion_id", "text", "text_kind", "opinion_type"}
        missing = required - set(parquet.schema_arrow.names)

        if missing:
            raise PipelineError(
                "ingest",
                f"corpus missing columns: {sorted(missing)}",
            )

        for batch in parquet.iter_batches(
            batch_size=self.config.batch_size,
            use_threads=True,
        ):
            for row in batch.to_pylist():
                if keep_ids is None:
                    yield row
                    continue

                value = row.get("cl_opinion_id")
                if value is not None and int(value) in keep_ids:
                    yield row

    def _iter_parquet(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(path)

        parquet = pq.ParquetFile(path)

        for batch in parquet.iter_batches(
            batch_size=self.config.batch_size,
            use_threads=True,
        ):
            yield from batch.to_pylist()

    def _read_all_case_ids(self) -> list[int]:
        """Read and validate every opinion ID in the source corpus."""
        parquet = pq.ParquetFile(self.config.corpus_file)

        if "cl_opinion_id" not in parquet.schema_arrow.names:
            raise PipelineError(
                "manifest",
                "corpus missing required column: cl_opinion_id",
            )

        ids: list[int] = []
        seen: set[int] = set()

        for batch in parquet.iter_batches(
            batch_size=self.config.batch_size,
            columns=["cl_opinion_id"],
            use_threads=True,
        ):
            for value in batch.column(0).to_pylist():
                if value is None:
                    raise PipelineError(
                        "manifest",
                        "corpus contains NULL cl_opinion_id",
                    )

                try:
                    case_id = int(value)
                except (TypeError, ValueError) as exc:
                    raise PipelineError(
                        "manifest",
                        f"invalid cl_opinion_id: {value!r}",
                    ) from exc

                if case_id in seen:
                    raise PipelineError(
                        "manifest",
                        f"duplicate cl_opinion_id in corpus: {case_id}",
                    )

                seen.add(case_id)
                ids.append(case_id)

        return ids

    # ── Manifest helpers ────────────────────────────────────────────

    def _load_plan(self) -> dict[int, PlanEntry]:
        result: dict[int, PlanEntry] = {}
        plan_path = self.config.manifest_dir / "plan.jsonl"
        if not plan_path.is_file():
            return result

        for row in read_jsonl(plan_path):
            case_id = int(row["case_id"])

            if case_id in result:
                raise PipelineError(
                    "manifest",
                    f"duplicate case_id in plan: {case_id}",
                )

            cluster_id = row.get("cluster_id")

            result[case_id] = PlanEntry(
                case_id=case_id,
                source_text_hash=str(row["source_text_hash"]),
                text_kind=str(row.get("text_kind") or ""),
                cluster_id=(int(cluster_id) if cluster_id is not None else None),
            )

        return result

    def _load_manifest_meta(self) -> dict | None:
        path = self.config.manifest_dir / "_manifest.jsonl"
        if not path.is_file():
            return None
        rows = list(read_jsonl(path))
        return rows[0] if rows else None

    def _get_failures(self, directory: Path) -> dict[int, dict]:
        path = directory / "_failures.jsonl"
        if not path.is_file():
            return {}
        return {int(row["case_id"]): row for row in read_jsonl(path)}

    def _persist_failures(self, directory: Path, failures: dict[int, dict]) -> None:
        write_jsonl_from_dict(directory / "_failures.jsonl", failures)

    def _chunked_file(self, case_id: int) -> Path:
        return self.config.chunked_dir / f"{case_id}.jsonl"

    def _embedded_file(self, case_id: int) -> Path:
        return self.config.embedded_dir / f"{case_id}.jsonl"

    # ── Skip logic (Filesystem as Source of Truth) ──────────────────

    def _is_valid_chunk_output(self, case_id: int, plan_entry: PlanEntry) -> bool:
        """Verify chunk file exists and its hash/version match the plan."""
        path = self._chunked_file(case_id)
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as f:
                first_line = f.readline()
                if not first_line:
                    return False
                first_row = json.loads(first_line)
            return (
                first_row.get("source_text_hash") == plan_entry.source_text_hash
                and first_row.get("chunker_version") == self.config.chunker_version
            )
        except Exception:
            return False

    def _is_valid_embed_output(self, case_id: int) -> bool:
        """Verify embed file exists and vector count matches non-parent chunk count."""
        chunk_path = self._chunked_file(case_id)
        embed_path = self._embedded_file(case_id)
        if not chunk_path.is_file() or not embed_path.is_file():
            return False

        try:
            chunk_children = 0
            with chunk_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not row.get("is_parent"):
                        chunk_children += 1

            embed_count = 0
            with embed_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        embed_count += 1

            return chunk_children == embed_count
        except Exception:
            return False

    # ── Halt handling ───────────────────────────────────────────────

    def _is_halted(self, stage: str) -> bool:
        directory = self.config.chunked_dir if stage == "chunk" else self.config.embedded_dir
        return (directory / "_HALTED").exists()

    def _clear_halted(self, stage: str) -> None:
        directory = self.config.chunked_dir if stage == "chunk" else self.config.embedded_dir

        try:
            (directory / "_HALTED").unlink()
        except FileNotFoundError:
            pass

    def _write_halted(
        self,
        stage: str,
        reason: str,
        failed: int,
        processed: int,
    ) -> None:
        directory = self.config.chunked_dir if stage == "chunk" else self.config.embedded_dir

        directory.mkdir(parents=True, exist_ok=True)

        atomic_write_json(
            directory / "_HALTED",
            {
                "reason": reason,
                "failed": failed,
                "processed": processed,
                "threshold_absolute": FAILURE_ABSOLUTE,
                "threshold_rate": FAILURE_RATE,
                "ts": utc_now(),
            },
        )

    @staticmethod
    def _check_failure_threshold(failed: int, processed: int) -> bool:
        if failed <= 0:
            return False
        if failed > FAILURE_ABSOLUTE:
            return True
        if processed <= 0:
            return False
        return failed / processed > FAILURE_RATE

    # ── Fingerprint ─────────────────────────────────────────────────

    def _dataset_fingerprint(self) -> str:
        """Fingerprint source files, plan, versions, and embedding settings."""
        plan_path = self.config.manifest_dir / "plan.jsonl"

        plan_hash = sha256_file(plan_path) if plan_path.is_file() else None
        plan_rows = sum(1 for _ in read_jsonl(plan_path)) if plan_path.is_file() else 0

        embedding_backend = "torch" if self.config.device.startswith("cuda") else "openvino"
        active_model_dir = (
            self.config.gpu_model_dir if embedding_backend == "torch" else self.config.model_dir
        )

        components: dict[str, Any] = {
            "corpus_file_sha256": sha256_file(self.config.corpus_file),
            "clusters_file_sha256": sha256_file(self.config.clusters_file),
            "edges_file_sha256": sha256_file(self.config.edges_file),
            "plan_file_sha256": plan_hash,
            "corpus_rows_effective": plan_rows,
            "chunker_version": self.config.chunker_version,
            "embedder_version": self.config.embedder_version,
            "filters_version": self.config.filters_version,
            "dedup_version": self.config.dedup_version,
            "dedup_enabled": self.config.dedup_enabled,
            "noise_filter_enabled": self.config.noise_filter_enabled,
            "embedding_backend": embedding_backend,
            "embedding_device": self.config.device,
            "embedding_model_dir": str(active_model_dir),
            "max_seq_length": self.config.max_seq_length,
        }

        return sha256_text(
            json.dumps(
                components,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


# ────────────────────────────────────────────────────────────────────────
# LoadMixin import — after base definition to avoid circular import
# ────────────────────────────────────────────────────────────────────────

from .pipeline_load import LoadMixin  # noqa: E402

# ────────────────────────────────────────────────────────────────────────
# IndexingPipeline: load mixin + ingest stages + orchestration
# ────────────────────────────────────────────────────────────────────────


class IndexingPipeline(LoadMixin, IndexingPipelineBase):
    # ── Stage: manifest ─────────────────────────────────────────────

    def manifest(self) -> StageReport:
        self._require_open()

        with self._stage_lock():
            started = time.monotonic()

            self._check_inputs(include_metadata=False)

            source_hash = sha256_file(self.config.corpus_file)
            requested = resolve_corpus_rows(self.config.corpus_rows)

            all_ids = self._read_all_case_ids()
            effective = min(requested, len(all_ids))

            if effective < requested:
                logger.warning(
                    "corpus_rows requested %d, corpus has only %d",
                    requested,
                    effective,
                )

            meta = self._load_manifest_meta()
            plan_path = self.config.manifest_dir / "plan.jsonl"

            if (
                not self.config.force
                and meta is not None
                and meta.get("source_file_hash") == source_hash
                and int(meta.get("row_count", -1)) == len(all_ids)
                and int(meta.get("corpus_rows_effective", -1)) == effective
                and plan_path.is_file()
                and meta.get("plan_file_sha256") == sha256_file(plan_path)
            ):
                logger.info(
                    "manifest skipped: unchanged (effective=%d)",
                    effective,
                )
                return StageReport("manifest", 0, effective, 0, time.monotonic() - started)

            if self.config.dry_run:
                logger.info(
                    "manifest dry-run: would build %d cases (requested=%d, corpus=%d)",
                    effective,
                    requested,
                    len(all_ids),
                )
                return StageReport("manifest", 0, effective, 0, time.monotonic() - started)

            self.config.manifest_dir.mkdir(parents=True, exist_ok=True)

            keep_ids = set(sorted(all_ids)[:effective])

            plan_rows: list[dict[str, Any]] = []
            for record in self._iter_opinions(keep_ids):
                case_id = int(record["cl_opinion_id"])
                cluster_id = record.get("cluster_id")
                plan_rows.append(
                    {
                        "case_id": case_id,
                        "source_text_hash": sha256_text(record.get("text")),
                        "text_kind": str(record.get("text_kind") or ""),
                        "cluster_id": (int(cluster_id) if cluster_id is not None else None),
                    }
                )

            if len(plan_rows) != effective:
                raise PipelineError(
                    "manifest",
                    f"expected {effective} plan rows, got {len(plan_rows)}",
                )

            atomic_write_jsonl(
                plan_path,
                sorted(plan_rows, key=lambda row: row["case_id"]),
            )

            plan_hash = sha256_file(plan_path)

            atomic_write_jsonl(
                self.config.manifest_dir / "_manifest.jsonl",
                [
                    {
                        "source_file_hash": source_hash,
                        "plan_file_sha256": plan_hash,
                        "row_count": len(all_ids),
                        "corpus_rows_requested": requested,
                        "corpus_rows_effective": effective,
                        "created_at": utc_now(),
                    }
                ],
            )

            logger.info(
                "manifest built: %d cases (requested=%d, corpus=%d)",
                len(plan_rows),
                requested,
                len(all_ids),
            )

            return StageReport("manifest", len(plan_rows), 0, 0, time.monotonic() - started)

    # ── Stage: chunk — tokenizer-aware, resumable, deterministic ────

    def chunk(self) -> StageReport:
        self._require_open()

        with self._stage_lock():
            started = time.monotonic()

            if self._is_halted("chunk") and not self.config.force:
                raise PipelineError("chunk", "halted")

            if self.config.force:
                self._clear_halted("chunk")

            self._check_inputs(include_metadata=False)

            plan = self._load_plan()

            if not plan:
                raise PipelineError("chunk", "plan is empty or missing")

            if self.config.dry_run:
                logger.info("chunk dry-run: %d cases", len(plan))
                return StageReport("chunk", 0, len(plan), 0, time.monotonic() - started)

            self.config.chunked_dir.mkdir(parents=True, exist_ok=True)

            # Load existing failures to preserve them if we don't reprocess
            failures = self._get_failures(self.config.chunked_dir)

            # Load known drops so we skip re-processing them on rerun.
            drops_path = self.config.chunked_dir / "_corruption_drops.jsonl"
            known_drops: dict[int, dict] = {}
            if drops_path.is_file() and not self.config.force:
                for row in read_jsonl(drops_path):
                    cid = int(row.get("case_id", -1))
                    if cid > 0:
                        known_drops[cid] = row

            processed = 0
            skipped = 0
            failed = 0
            dropped = 0

            logger.info(
                "chunk starting: %d cases, %d workers (chunker %s, noise_filter=%s)",
                len(plan),
                self.config.workers,
                self.config.chunker_version,
                self.config.noise_filter_enabled,
            )

            tokenize = self._get_tokenizer() if self.config.workers == 1 else None
            noise = is_noise_chunk if self.config.noise_filter_enabled else None

            executor: concurrent.futures.ProcessPoolExecutor | None = None

            if self.config.workers > 1:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.config.workers,
                    initializer=worker_init,
                    initargs=(
                        str(self.config.model_dir),
                        str(self.config.gpu_model_dir),
                        self.config.noise_filter_enabled,
                    ),
                )

            try:
                records_iter = self._iter_opinions(set(plan.keys()))
                pending: list[tuple[dict[str, Any], concurrent.futures.Future]] = []

                def submit_next() -> bool:
                    nonlocal skipped, dropped

                    try:
                        record = next(records_iter)
                    except StopIteration:
                        return False

                    case_id = int(record["cl_opinion_id"])
                    plan_entry = plan[case_id]

                    # Skip if valid output already exists on disk
                    if not self.config.force and self._is_valid_chunk_output(case_id, plan_entry):
                        failures.pop(case_id, None)
                        skipped += 1
                        return True

                    # Skip known-dropped cases on rerun
                    if not self.config.force:
                        drop_entry = known_drops.get(case_id)
                        if (
                            drop_entry is not None
                            and drop_entry.get("source_text_hash") == plan_entry.source_text_hash
                            and drop_entry.get("chunker_version") == self.config.chunker_version
                        ):
                            failures.pop(case_id, None)
                            skipped += 1
                            dropped += 1
                            return True

                    if executor is None:
                        future: concurrent.futures.Future = concurrent.futures.Future()
                        try:
                            result = chunk_worker(record, tokenize, noise)
                            future.set_result(result)
                        except Exception as exc:
                            future.set_exception(exc)
                    else:
                        future = executor.submit(chunk_worker, record)

                    pending.append((record, future))
                    return True

                max_pending = max(1, self.config.workers * 2)

                while len(pending) < max_pending and submit_next():
                    pass

                while pending:
                    record, future = pending.pop(0)

                    try:
                        case_id, chunks, error, trace = future.result()
                    except Exception as exc:
                        case_id = int(record["cl_opinion_id"])
                        chunks = None
                        error = f"{type(exc).__name__}: {exc}"
                        trace = traceback.format_exc()

                    processed += 1

                    if error is not None or chunks is None:
                        failed += 1
                        failures[case_id] = {
                            "case_id": case_id,
                            "error": error or "unknown",
                            "traceback": trace or "",
                            "stage": "chunk",
                            "ts": utc_now(),
                        }
                    else:
                        from .chunker import quality_filter

                        keep, qstats = quality_filter(chunks)
                        if not keep:
                            with drops_path.open("a", encoding="utf-8") as fh:
                                fh.write(
                                    json.dumps(
                                        {
                                            "case_id": case_id,
                                            "source_text_hash": plan[case_id].source_text_hash,
                                            "chunker_version": self.config.chunker_version,
                                            **qstats,
                                            "ts": utc_now(),
                                        }
                                    )
                                    + "\n"
                                )
                            logger.warning(
                                "chunk: dropped case %d (%s)",
                                case_id,
                                qstats.get("reason"),
                            )
                            dropped += 1
                            failures.pop(case_id, None)
                        else:
                            output_path = self._chunked_file(case_id)
                            atomic_write_jsonl(output_path, chunks)
                            failures.pop(case_id, None)

                    if processed % FAILURE_CHECK_EVERY == 0 or processed + skipped == len(plan):
                        logger.info(
                            "chunk progress: %d/%d, skipped=%d, failed=%d, dropped=%d",
                            processed + skipped,
                            len(plan),
                            skipped,
                            failed,
                            dropped,
                        )

                    should_check = failed > FAILURE_ABSOLUTE or (
                        processed > 0 and processed % FAILURE_CHECK_EVERY == 0
                    )

                    if should_check:
                        self._persist_failures(self.config.chunked_dir, failures)

                        if self._check_failure_threshold(failed, processed):
                            reason = f"failure threshold exceeded: {failed}/{processed}"
                            self._write_halted("chunk", "systemic", failed, processed)
                            raise PipelineError(
                                "chunk",
                                "halted",
                                StageReport(
                                    "chunk",
                                    processed,
                                    skipped,
                                    failed,
                                    time.monotonic() - started,
                                    reason,
                                ),
                            )

                    while len(pending) < max_pending and submit_next():
                        pass

            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            self._persist_failures(self.config.chunked_dir, failures)

            # Write a lightweight manifest purely for the dedup stage to consume
            self._write_chunk_manifest_for_dedup()

            if self._check_failure_threshold(failed, processed):
                reason = f"failure threshold exceeded: {failed}/{processed}"
                self._write_halted("chunk", "systemic", failed, processed)
                raise PipelineError(
                    "chunk",
                    "halted",
                    StageReport(
                        "chunk",
                        processed,
                        skipped,
                        failed,
                        time.monotonic() - started,
                        reason,
                    ),
                )

            logger.info(
                "chunk complete: %d processed, %d skipped, %d failed, %d dropped (corruption)",
                processed,
                skipped,
                failed,
                dropped,
            )

            return StageReport("chunk", processed, skipped, failed, time.monotonic() - started)

    def _write_chunk_manifest_for_dedup(self) -> None:
        """Generate a lightweight _manifest.jsonl purely for the dedup stage.

        The dedup stage expects a manifest to verify file agreement. Since we
        removed the bloated pipeline manifests, we generate a minimal one here
        by scanning the directory, ensuring dedup doesn't crash.
        """
        manifest_rows = []
        for path in sorted(self.config.chunked_dir.glob("*.jsonl")):
            if path.name.startswith("_") or not path.stem.isdigit():
                continue
            case_id = int(path.stem)
            manifest_rows.append(
                {"case_id": case_id, "file_sha256": sha256_file(path), "completed_at": utc_now()}
            )
        atomic_write_jsonl(self.config.chunked_dir / "_manifest.jsonl", manifest_rows)

    # ── Stage: dedup — cross-case exact-hash dedup ──────────────────

    def dedup(self) -> StageReport:
        """Run cross-case deduplication on chunked/.

        After dedup, chunked/_manifest.jsonl reflects new chunk_count and
        embedding_count for rewritten cases. embedded/ files retain pre-dedup
        state; next embed detects the mismatch and re-embeds affected cases.
        """
        self._require_open()

        with self._stage_lock():
            started = time.monotonic()

            if self.config.dry_run:
                logger.info("dedup dry-run")
                return StageReport("dedup", 0, 0, 0, time.monotonic() - started)

            if not self.config.dedup_enabled:
                logger.info("dedup disabled (dedup_enabled=False)")
                return StageReport("dedup", 0, 0, 0, time.monotonic() - started)

            if not (self.config.chunked_dir / "_manifest.jsonl").is_file():
                raise PipelineError(
                    "dedup",
                    "precondition: chunk must run first (missing chunked/_manifest.jsonl)",
                )

            from .dedup import DedupError, dedup_chunks

            try:
                report = dedup_chunks(self.config.chunked_dir)
            except DedupError as exc:
                raise PipelineError("dedup", str(exc)) from exc

            logger.info(
                "dedup complete: %d/%d duplicate chunks removed, %d cases rewritten (%.1fs)",
                report.chunks_dropped,
                report.chunks_scanned,
                report.cases_rewritten,
                report.elapsed_s,
            )

            return StageReport(
                stage="dedup",
                processed=report.chunks_scanned,
                skipped=(report.chunks_scanned - report.cases_rewritten),
                failed=0,
                duration_s=time.monotonic() - started,
            )

    # ── Stage: run_all — orchestration ──────────────────────────────

    def run_all(self) -> list[StageReport]:
        """Run all stages in order.

        Stage order:
            manifest -> chunk -> dedup -> embed -> load -> validate -> index

        A database connection is established before expensive stages.
        With --force, managed indexes are dropped before processing.
        """
        self._require_open()

        stage_names = (
            "manifest",
            "chunk",
            "dedup",
            "embed",
            "load",
            "validate",
            "index",
        )

        if self.config.dry_run:
            return [StageReport(name, 0, 0, 0, 0.0) for name in stage_names]

        if not self.config.db_dsn:
            raise PipelineError(
                "run_all",
                "DATABASE_URL is required because run_all includes "
                "load, validate, and index. Run the ingest/embed "
                "stages individually for a database-free workflow.",
            )

        database = self._get_database()

        if self.config.force:
            try:
                self.drop_indexes()
            except Exception as exc:
                raise PipelineError(
                    "run_all",
                    f"could not drop indexes before forced reload: {exc}",
                ) from exc
        elif database.index_exists():
            raise PipelineError(
                "run_all",
                "the embedding index already exists. Drop indexes "
                "before loading, or use --force to drop managed "
                "indexes first.",
            )

        reports: list[StageReport] = []

        for stage in (
            self.manifest,
            self.chunk,
            self.dedup,
            self.embed,
            self.load,
            self.validate,
            self.index,
        ):
            reports.append(stage())

        return reports


__all__ = [
    "IndexingPipeline",
    "IndexingPipelineBase",
]
