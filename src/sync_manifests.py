import json
import os

# These are the absolute paths based on your workspace structure
chunked_manifest = "/workspace/tmp/data/chunked/_manifest.jsonl"
embedded_manifest = "/workspace/tmp/data/embedded/_manifest.jsonl"


def get_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(int(json.loads(line)["case_id"]))
    return ids


# 1. Read all IDs from both manifests
chunked_ids = get_ids(chunked_manifest)
embedded_ids = get_ids(embedded_manifest)

# 2. Find the exact intersection (cases that are in BOTH)
common_ids = chunked_ids.intersection(embedded_ids)
print(f"Chunked: {len(chunked_ids)} | Embedded: {len(embedded_ids)} | Common: {len(common_ids)}")

if not common_ids:
    print("ERROR: No common cases found. Check your paths.")
    exit(1)


# 3. Rewrite BOTH manifests to ONLY contain the common IDs
def filter_manifest(path, keep_ids):
    tmp_path = path + ".tmp"
    with open(path) as f_in, open(tmp_path, "w") as f_out:
        for line in f_in:
            if line.strip():
                cid = int(json.loads(line)["case_id"])
                if cid in keep_ids:
                    f_out.write(line)
    os.replace(tmp_path, path)


filter_manifest(chunked_manifest, common_ids)
filter_manifest(embedded_manifest, common_ids)

# 4. Clear any failure logs (the load stage will block if these exist)
for p in [
    "/workspace/tmp/data/chunked/_failures.jsonl",
    "/workspace/tmp/data/embedded/_failures.jsonl",
]:
    if os.path.exists(p):
        open(p, "w").close()

print(f"SUCCESS: Both manifests synced to exactly {len(common_ids)} cases.")
print("You can now safely run the 'load' stage.")
