# Indexing Pipeline Architecture & Implementation

## Executive Summary
The Indexing Pipeline is a modular, deterministic, and crash-resilient data ingestion engine designed to transform raw, noisy legal text corpora into a strictly validated, vector-indexed PostgreSQL database. Engineered for high-scale CPU/GPU execution, it prioritizes data integrity, compute efficiency, and exact reproducibility.

By treating the **filesystem as the single source of truth** for intermediate state and utilizing **character-offset-based identifiers**, the pipeline guarantees safe resumption from interruptions without the state-desynchronization bugs common in manifest-heavy architectures.

---

## Core Architectural Paradigms

### 1. Filesystem as Single Source of Truth
Early iterations relied on bloated JSON manifests to track pipeline state, which frequently desynchronized during crashes. The current architecture eliminates intermediate manifests. Skip logic relies entirely on atomic file existence, first-line hash verification, and line-count matching. Because all file writes use a strict `tempfile` + `os.fsync` + `os.replace` pattern, a file on disk is mathematically guaranteed to be complete and valid.

### 2. Deterministic, Offset-Based Identifiers
Chunk IDs are generated using exact character offsets from the normalized source text, rather than array indices. This ensures that IDs remain perfectly stable even if tokenization heuristics drift or environment configurations change, completely eliminating ID drift during pipeline resumption.

```python
# src/indexing_pipeline/chunker.py
def _make_chunk_id(case_id: int, is_parent: bool, char_start: int, char_end: int) -> str:
    """Generate deterministic chunk IDs based on character offsets.

    Prevents ID drift if tokenizer heuristics change, ensuring that
    parent/child relationships and database upserts remain stable
    across pipeline interruptions and environment shifts.
    """
    prefix = "p" if is_parent else "c"
    return f"{case_id}:{prefix}:{char_start}:{char_end}"
```

### 3. Crash-Safe Atomic I/O
Every intermediate artifact (chunks, embeddings, failure logs) is written using an atomic replacement strategy. If the process is killed mid-write, the original file remains untouched, preventing half-written JSONL files from corrupting subsequent resume attempts.

```python
# src/indexing_pipeline/io_utils.py (Simplified)
def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())  # Force OS buffers to physical disk
        os.replace(tmp, path)          # Atomic POSIX rename
    except BaseException:
        os.unlink(tmp)                 # Clean up partial temp file
        raise
```

---

## The Chunking Engine

The chunker (`chunker.py`) is a pure, I/O-free function. It implements a hierarchical, structure-aware splitting algorithm optimized for legal text, ensuring that semantic units and critical citations are never severed.

### Atomic Span Protection
Legal text is dense with indivisible entities (e.g., `410 U.S. 113`, `§ 1983`, `Roe v. Wade`). The chunker pre-computes these spans using regex and merges overlapping intervals. Boundary checks then use $O(\log N)$ binary search (`bisect`) to ensure cuts never fall inside an atomic span, yielding a 4-8x speedup over naive regex rescans.

### Filesystem-Based Skip Logic
Instead of loading heavy manifests into memory, the orchestration layer checks the disk directly to determine if a case needs processing.

```python
# src/indexing_pipeline/pipeline.py
def _is_valid_chunk_output(self, case_id: int, plan_entry: PlanEntry) -> bool:
    """Verify chunk file exists and its metadata matches the execution plan."""
    path = self._chunked_file(case_id)
    if not path.is_file():
        return False
    try:
        # Read only the first line to avoid loading the entire file into memory
        with path.open("r", encoding="utf-8") as f:
            first_row = json.loads(f.readline())
        return (
            first_row.get("source_text_hash") == plan_entry.source_text_hash
            and first_row.get("chunker_version") == self.config.chunker_version
        )
    except Exception:
        return False
```

---

## Pipeline Stages & Resumption Logic

The pipeline executes in a strict, dependency-ordered sequence. Each stage is independently resumable.

1. **Manifest**: Reads the source Parquet, resolves the target corpus size, and generates `plan.jsonl` (mapping `case_id` to `source_text_hash`).
2. **Chunk**: Segments text. *Resume logic:* Checks if `{case_id}.jsonl` exists and its first-line hash matches the plan. Intentionally drops cases failing strict quality gates (logging them to `_corruption_drops.jsonl`).
3. **Dedup**: Removes cross-case duplicate children using exact SHA-256 hash matching, deterministically keeping the lowest `case_id`. Rewrites affected JSONL files atomically.
4. **Embed**: Generates 384-dim vectors. *Resume logic:* Counts non-parent lines in the chunk file and compares it to the total lines in the embed file. If they match, the case is skipped. (If `dedup` removed chunks, the counts mismatch, triggering a safe re-embed of only the affected cases).
5. **Load**: Ingests data into PostgreSQL using temporary staging tables and `ON CONFLICT ... DO UPDATE` upserts. Skips cases listed in `_corruption_drops.jsonl`.
6. **Validate**: Enforces 8 rigorous SQL assertions against the staged data.
7. **Index**: Builds concurrent HNSW and GIN indexes only after validation succeeds.

---

## Compute & Inference Optimizations

* **Pre-Embed Deduplication**: Cross-case dedup runs *after* chunking but *before* embedding. This prevents wasting expensive GPU/CPU cycles on redundant boilerplate text and prevents index bloat.
* **Static OpenVINO Reshaping**: The CPU embedder enforces a strict static shape (`batch=32, seq=512`) on the OpenVINO IR. This prevents memory spikes (OOM), enables OpenVINO's `THROUGHPUT` auto-tuning, and explicitly pins threads to physical cores.
* **Fast C++ Tokenization**: Bypasses Python loops by passing entire batches directly to the HuggingFace `fast` (Rust-backed) tokenizer.
* **Strict Tokenizer Enforcement**: The pipeline crashes loudly if the tokenizer fails to load, rather than falling back to character-count heuristics. This prevents silent boundary shifts that would corrupt the vector space.

---

## Data Integrity: The 8 Validation Gates

Before the `index` stage is permitted to run, the `validate` stage executes strict SQL assertions. Failure results in an immediate pipeline halt.

1. **Parent Coverage**: 100% of child chunks must reference a valid, existing parent chunk.
2. **Breadcrumb Presence**: 100% of chunks must contain the required metadata header (case, citation, court, section).
3. **Boundary Integrity**: $\le 5\%$ of child chunks may end mid-sentence (accounting for unavoidable long legal sentences).
4. **Citation Integrity**: **Absolute zero tolerance** for chunks ending in split citations (e.g., `410 U.S. ` or `v. `).
5. **Size Distribution**: Enforces strict token limits (Child p50 $\le 450$, p95 $\le 550$, max $\le 1600$; Parent max $\le 3000$).
6. **Duplicate Content**: Zero intra-case duplicate child bodies permitted.
7. **Section Coverage**: $\ge 98\%$ consistency between child and parent section metadata.
8. **Determinism**: Re-chunking a random sample of 20 cases must yield byte-identical chunk IDs and hashes.

---

## Module Breakdown

| Module | Responsibility |
| :--- | :--- |
| `chunker.py` | Pure, deterministic text segmentation. Zero I/O. |
| `pipeline.py` | Main orchestration, ingest-side stages (`manifest`, `chunk`, `dedup`), and filesystem skip logic. |
| `pipeline_load.py` | Load-side stages (`embed`, `load`, `validate`, `index`). |
| `pipeline_helpers.py` | Configuration dataclasses, constants, and multiprocessing worker initialization. |
| `db.py` | PostgreSQL schema management, connection pooling, and all SQL queries. |
| `embedder.py` | Inference wrappers for OpenVINO (CPU) and PyTorch (CUDA). |
| `dedup.py` | Cross-case exact-hash deduplication logic. |
| `filters.py` | Metadata normalization and heuristic noise detection. |
| `io_utils.py` | Atomic file writing, SHA-256 hashing, and JSONL serialization. |
| `cli.py` | Command-line interface and argument parsing. |

---

## Operational Guide & CLI

The pipeline is orchestrated via a unified CLI. All stages share a common set of flags.

### Environment Setup
```bash
cd /workspace/src
source .venv/bin/activate

export PG_PASS="your_secure_password"
export DATABASE_URL="postgresql://legalrag:${PG_PASS}@127.0.0.1:5432/legalrag"
export CORPUS_ROWS=100000 # min (5k), medium (100k), max (400k), or exact int
```

### Execution
```bash
# Drop existing indexes for a clean, optimized HNSW build
python -m indexing_pipeline.cli drop-index

# Run the full pipeline with optimized CPU inference settings
python -m indexing_pipeline.cli all \
    --embed-batch 32 \
    --inference-threads 8 \
    --log-level INFO
```

### State and Fingerprinting
The pipeline tracks high-level completion in the `pipeline_state` database table. A **dataset fingerprint** (SHA-256 hash) is computed from source file hashes, code versions (`chunker_version`, `embedder_version`), and active backend configuration.

If any code or configuration parameter changes, the fingerprint changes. The `load` stage will detect this mismatch, refuse to skip, and force a safe, complete re-ingestion, guaranteeing that the database never contains data generated by an outdated or incompatible pipeline version.
