# Legal GraphRAG

**A targeted retrieval experiment for a large structured legal corpora.**

**Built as a retrieval-layer companion to** [RAG8s](https://github.com/Athithya-Sakthivel/RAG8s), this project isolates the impact of **structure-aware chunking** and **native citation graphs**. While RAG8s provides a complete end-to-end LLMOps pipeline for heterogeneous enterprise documents, Legal GraphRAG explores what happens when the corpus is highly connected, **breaking the 0.72 recall ceiling of fixed-size windowing** by treating legal citations as first-class graph edges.

| | RAG8s | Legal GraphRAG |
|---|---|---|
| **Scale** | 20 heterogenous files | **9,536 cases · 36K chunks · 45K citation edges** |
| **Scope** | E2E RAG platform (ingest → generate) | **Retrieval only** |
| **Chunking** | 512-token windows | **Parent–child, section-aligned** |
| **Graph** | None | **Native `CITES` edges, 1-hop** |
| **Routing** | Fixed hybrid pipeline | **Deterministic intent classifier** |
| **Storage** | Qdrant Cluster + S3 | **Single node PostgreSQL** |
| **Recall@5** | 0.72 | **0.91 average** |

**No LLM, no generation, no containerization.** This project isolates one question: how much does structure-aware chunking and graph-native routing improve retrieval over a large structured corpus?

---

## Why Fixed-Size Chunking Fails on Connected Legal Text

Legal research has three distinct query patterns, each requiring different retrieval strategies:

1. **"What did the court hold on qualified immunity?"** — Semantic search for legal reasoning
2. **"Which cases cite Roe v. Wade?"** — Exact citation lookup and graph traversal
3. **"Has Roe v. Wade been overruled?"** — Finding cases that cite it AND contain overruling language

A single retrieval strategy cannot handle all three well. Semantic search fails at citation lookup. Keyword search fails at conceptual questions. Graph traversal alone misses semantic context.

**The challenge:** Route each query to the cheapest sufficient path while maintaining **sub-250ms latency on CPU-only hardware**.

---

## Key Results

Evaluated on **150 stratified queries** across a **9,536-case corpus** from CourtListener.

| Query Type | N | Recall@10 | What This Means |
|---|---|---|---|
| **Holding** (semantic) | 50 | **88%** | Finds the right legal reasoning 88% of the time |
| **Lookup** (citation) | 50 | **96%** | Finds citing cases 96% of the time |
| **Validity** (overruling) | 30 | **95%** | Finds overruling cases 95% of the time |
| **Average** | 130 | **93%** | Across all query types |

**Latency:** Median **68ms**, 95th percentile **214ms** (CPU-only, no GPU).

**Infrastructure:** Single PostgreSQL database. No separate vector store, no graph database, no network round-trips between retrieval stages.

See full results: [`results/2026-11-12/eval_full.json`](results/2026-11-12/eval_full.json)

---

## Key Design Decisions

| Decision | Rationale | Trade-off |
| :--- | :--- | :--- |
| **Parent-child chunking** | Fixed-size chunks **split citations mid-sentence**. Section-aligned children (350 tokens) carry legal identity; parents (1,200 tokens) provide context without being embedded. | More complex ingestion. **Eliminates split-citation failures entirely.** |
| **Hybrid search** | Dense captures semantics but misses exact citations. Sparse captures exact matches but misses synonyms. **RRF fusion improves recall 15–20 points** over dense-only on legal text. | **2× retrieval latency.** Worth it for the recall gain. |
| **384-dim OpenVINO int8** | BGE-small quantized to int8 runs efficiently on CPU (**~35MB weights, sub-10ms inference**). Chunking and ingestion quality matter **far more** than embedding model choice for legal retrieval. | Slight accuracy loss vs. fp32. **Negligible** compared to structure-aware chunking gains. |
| **Deterministic routing** | Legal queries have predictable syntax. Regex detects intent at **100% accuracy with zero latency**, avoiding 200–500ms LLM router overhead and hallucination risk. | Cannot handle ambiguous queries. Acceptable for **precise legal terminology**. |
| **No cross-encoder reranker** | General-purpose rerankers **degrade legal retrieval** when first-stage is strong ([Pipitone & Alami](https://arxiv.org/abs/2408.10343)). No legal-native reranker exists with **<100ms CPU latency** for 20 passages. | Sacrifices 2–3 recall@5 points for **3s/query latency savings**. |
| **PostgreSQL-only** | pgvector + recursive CTEs + full-text search in a **single SQL statement**. Eliminates network round-trips. Scales to ~1M vectors on 8GB RAM; `pgvectorscale` provides seamless upgrade for tens of millions ([Timescale, 2024](https://github.com/timescale/pgvectorscale)). | Sacrifices horizontal scalability. Acceptable with the **clear scale-up path**. |
| **Batch indexing** | US case law changes periodically. Pipeline is **idempotent with fast skip**, safely re-runs every few hours without CDC or incremental embedding complexity. | Not suitable for real-time streams. Perfect for **daily/weekly/monthly legal updates**. |

---

# Architecture

The system is separated into two independent execution planes:

**[Batch Indexing Plane](src/indexing_pipeline/README.md)** — An **idempotent, fast-skip-on-resume** pipeline that ingests US case law opinions from CourtListener, detects **15 section types** (holding, analysis, facts, footnotes, …), splits them into a **parent–child hierarchy** (1,200-token parents stored but not embedded, 350-token children as the retrieval unit), generates **384-dim L2-normalized dense embeddings** via a local **OpenVINO int8** model and **tsvector** sparse representations, and upserts into **PostgreSQL** with **deterministic chunk IDs** and **native `CITES` citation edges** between cases. The pipeline is **safe to re-run every few hours** to catch newly published opinions without CDC or streaming pipeline complexity.

**[Online Retrieval Plane](src/retriever/README.md)** — A **sub-250ms p95, CPU-only** request path where a **deterministic regex router** classifies queries into **HOLDING** (semantic), **LOOKUP** (citation), or **VALIDITY** (overruling), dispatches each to the cheapest sufficient combination of retrieval arms — **dense HNSW** (pgvector, halfvec(384), relaxed-order iterative scan), **sparse tsvector** (ts_rank_cd), **exact citation / case-name match** (trigram fallback), and **1-hop recursive-CTE graph traversal** — fuses arms via **per-class Reciprocal Rank Fusion** with tunable k constants, applies **per-case diversity dedup**, and expands child chunks to their parent sections. No LLM, no cross-encoder reranker, no external vector database, no graph database — **one PostgreSQL instance composes vector, lexical, and graph retrieval in a single SQL statement**.


---

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/ea60fe63-fcc9-4cf5-9a2d-daf36972f0b3" />

---

## How It Works

### Holding Queries ("What did the court hold on X?")

**Path:** Dense search + sparse search → RRF fusion → parent expansion

**Why:** Semantic queries need both conceptual matching (dense) and exact statutory references (sparse). No graph traversal needed.

**Example:** Query "what did the court hold on qualified immunity" retrieves chunks from cases discussing qualified immunity, ranked by hybrid score.

### Lookup Queries ("Which cases cite X?")

**Path:** Exact match → resolve case ID → 1-hop graph traversal → parent expansion

**Why:** Citation lookup is deterministic. Exact match finds the case, graph traversal finds all citing cases. No semantic search needed.

**Example:** Query "which cases cite Roe v. Wade" resolves "Roe v. Wade" to case_id 12345, then traverses the citation graph to find all cases with `target_id = 12345`.

### Validity Queries ("Has X been overruled?")

**Path:** Exact match → resolve case ID → 1-hop graph traversal → dense search over citers for overruling language → RRF fusion

**Why:** Need both structural evidence (cases that cite it) and semantic evidence (cases that contain "overrule", "abrogate", "no longer good law"). Graph traversal finds citers, dense search filters for overruling language.

**Example:** Query "has Roe v. Wade been overruled" finds all cases citing Roe, then searches those cases for chunks containing "overruled" or "abrogated", returning Dobbs v. Jackson Women's Health Organization.

---

## Evaluation Methodology

### Gold Set Construction

**150 queries** generated deterministically from the live corpus:
- **Holding (50):** Use the actual holding text from a case as the query; gold answer is that case
- **Lookup (50):** "Which cases cite {case_name}?" — gold answer is all citing cases
- **Validity (30):** "Has {case_name} been overruled?" — gold answer is overruling cases
- **Negative control (20):** "Which cases cite {case_name}?" for cases with zero citers — gold answer is empty

**Stratified by court, year bucket, and citation in-degree** to ensure representative coverage.

### Metrics

- **Recall@5 / Recall@10:** Percentage of queries where the correct answer appears in the top 5 or top 10 results
- **Average recall:** Mean recall across all query types (weights each type equally, not by query count)
- **Bootstrap confidence intervals:** 95% CIs computed via bootstrap resampling (n=1000) to quantify uncertainty
- **Failure modes:** Every query categorized as success, classification error, resolution error, retrieval error, or ranking error

### Corpus Drift Guard

Each gold row embeds a **SHA-256 fingerprint** of the corpus state (chunker versions, chunk IDs, case IDs, edges). The evaluation runner **refuses to score** if the live corpus fingerprint diverges, preventing silent measurement against outdated gold.

---

## Quick Start

**Requirements:** Python 3.14+, PostgreSQL 15+ with pgvector, 8GB RAM, Docker (for database restore).

```bash
git clone https://github.com/Athithya-Sakthivel/legal-graphrag.git
cd legal-graphrag

# Run end-to-end evaluation (downloads pgvector backup, restores DB, runs 150 queries)
export HF_TOKEN=   # https://huggingface.co/settings/tokens/new?tokenType=read
export PG_PASS=e6a1c2e5bf8d  # or any password
bash src/utils/quickstart.sh
```

---


## Limitations

1. **Citation edges only.** No `OVERRULES` or `DISTINGUISHES` edge types, public citation data does not carry typed precedent edges. Validity detection is graph-seeded and prose-verified, not typed-edge traversed.

2. **Holding pool is narrow.** Only ~1.3% of opinions have a detected `holding` section. The 50-query holding eval samples 50 of 123 eligible cases — near-census of a section-detection-selected subset, not a random sample of holdings.

3. **Overruling detection is heuristic.** Substring match on `overrul|revers|vacat` stems produces false positives ("lower court reversed") and false negatives ("abrogated"). All gold rows are marked `verified=false`.

4. **No refusal mechanism.** Negative controls return lexical matches on the case's own text. Refusal belongs in a generation layer, which this project does not have.

5. **Single-node.** Optimized for correctness and latency at 10K–400K cases. No sharding, no replication, no horizontal scale path.

6. **Retrieval only.** Answer quality, faithfulness, and citation precision are unmeasured as there is no generation layer to measure.

---

## Non-Goals

- Legal advice
- Exhaustive precedent validation
- Multi-hop graph traversal (2+ hops)
- Non-US jurisdictions
- Real-time document ingestion
- Answer generation (no LLM, no generation layer)
- Microsoft-style GraphRAG (no LLM entity extraction, no community detection, no hierarchical summaries)

---

## Documentation

- [RAG Design](docs/rag/adaptive_graphrag.md) — routing, retrieval arms, fusion, evaluation gates
- [Chunking](docs/chunking.md) — segmentation, atomic spans, validation
- [Datasets](docs/datasets.md) — schema, artifacts, scale presets
- [Evaluation](src/evals/README.md) — harness, metrics, failure taxonomy
- [Embedder](docs/models/embedder.md) — BGE-small-en-v1.5 OpenVINO configuration
- [No Reranker](docs/models/no_reranker.md) — evidence against cross-encoder reranking
