#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN=${HF_TOKEN}
export PG_PASS=${PG_PASS:-e6a1c2e5bf8d}

echo "[1/5] Downloading Embedder..."
hf download Athithya117/bge-small-en-v1.5-openvino-qint8 \
  --repo-type model \
  --local-dir /workspace/src/indexing_pipeline/embedders
# hf download is idempotent — re-runs hash existing files and only fetch missing ones.

echo "[2/5] Starting PostgreSQL..."
if docker inspect legal-pgvector >/dev/null 2>&1; then
  docker start legal-pgvector >/dev/null
  echo "  ↳ reused existing container"
else
  docker run -d \
    --name legal-pgvector \
    --restart unless-stopped \
    --shm-size=2g \
    -p 127.0.0.1:5432:5432 \
    -e POSTGRES_DB=legalrag \
    -e POSTGRES_USER=legalrag \
    -e POSTGRES_PASSWORD="${PG_PASS:?PG_PASS must be set in env}" \
    -v legal_pgdata:/var/lib/postgresql/data \
    pgvector/pgvector:0.8.7-pg17-bookworm@sha256:7d0a41464797aeb9862dfdc0bbf1526b82cfd324cdd8036df19e3545f4a9ccb8 \
    -c shared_buffers=2GB -c effective_cache_size=4GB -c maintenance_work_mem=1GB \
    -c work_mem=16MB -c temp_buffers=16MB -c wal_buffers=64MB -c max_wal_size=2GB \
    -c min_wal_size=512MB -c checkpoint_timeout=15min -c max_connections=20 \
    -c max_parallel_workers=2 -c max_parallel_maintenance_workers=2 \
    -c random_page_cost=1.1 -c effective_io_concurrency=200
  echo "  ↳ created new container"
fi

echo "[3/5] Setting up Python Environment..."
cd /workspace/src
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r indexing_pipeline/requirements.txt

 echo "[4/5] Downloading Raw Data..."
 python3 /workspace/data/download_raw_data.py --dir /workspace/tmp/data/raw --scale medium || true

META_PATH=/workspace/tmp/data/raw/metadata/clusters_filtered.parquet
if [ ! -f "$META_PATH" ]; then
  echo "ERROR: raw data not found at $META_PATH after download step." >&2
  echo "       Fix the download failure and re-run this script." >&2
  exit 1
fi

echo "[5/5] Preprocessing & Deduplicating Raw Metadata..."
python3 - <<'PYEOF'
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

path = Path("/workspace/tmp/data/raw/metadata/clusters_filtered.parquet")
original = pq.read_table(path)
orig_schema = original.schema

df = original.to_pandas()
df["_score"] = df.notna().sum(axis=1)
df = df.sort_values(["id", "_score"], ascending=[True, False], kind="mergesort")
df = df.drop_duplicates(subset="id", keep="first")
df = df.drop(columns=["_score"]).reset_index(drop=True)

new_table = pa.Table.from_pandas(df, schema=orig_schema, preserve_index=False)

# Atomic write — a crash mid-write leaves the previous parquet intact.
tmp = path.parent / (path.name + ".tmp")
pq.write_table(new_table, tmp)
tmp.replace(path)
print("Metadata deduplication complete.")
PYEOF

echo "Bootstrap complete. You can now run: python -m indexing_pipeline all"


bash src/evals/test_e2e_locally.sh --top-k 5 --rebuild-gold
