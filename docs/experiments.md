# Retrieval experiments — measured record

## Scope and reproducibility

**MEASURED RESULT**, completed 2026-09-25T10:03:06.310847+00:00. This report is a snapshot of `data/benchmark.json`. Rerun `HF_HUB_OFFLINE=1 python evaluate.py --repeats 3` to produce a new raw report; latency varies by machine/load. The evaluator calculates comparison percentages from its own measurements.

- Machine: macOS-26.5.2-arm64-arm-64bit; processor reported as `arm`; CPU inference, 4 Torch threads. Exact chip identity was unavailable in the sandbox.
- Python 3.9.6; model revisions and dependency versions are recorded in the raw report.
- Dataset: 32 unique, manually labeled synthetic questions; 24 chunks from 24 fictional policy pages. This is a development fixture, not held-out production validation.
- 3 repetitions per question and configuration: 96 samples per row, 576 timed searches in randomized order (seed 42), after one warm-up per configuration.
- Dense/BM25 top-20, RRF k=60, final top-5. Chunk size 400, overlap approximately 60. Most pages fit one chunk; this is not a chunk-size optimization study.
- Quality: page-level Recall@5, truncated MRR@5, graded nDCG@5. Duplicate page hits get no additional credit. Repetitions do not multiply the independent query count.
- Time: complete warm retrieval from query embedding through final formatting; excludes model loading, indexing, HTTP transport, and queueing. This is not a browser response-time or throughput benchmark.
- Raw results include individual query indices, result IDs/scores, per-stage timings, actual candidate counts, dataset hash, and PDF hash. No query embeddings or cross-encoder scores are reused across configurations.

## Experiment 1: stage ablation

**Hypothesis:** Combining lexical and semantic rankings improves quality over dense-only; cross-encoder reranking improves fused ordering.

**Baseline:** Dense-only, top-20 candidate retrieval, final top-5.

**Change:** Compare BM25-only, RRF, and RRF followed by reranking.

**Dataset:** The 32-query synthetic fixture above, identical for every configuration.

**Metrics / Before / After:**

| Configuration | Candidates reranked | Recall@5 | MRR@5 | nDCG@5 | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0 | 1.0000 | 0.9792 | 0.9717 | 10.99 | 9.70 | 18.41 |
| bm25 | 0 | 0.9219 | 0.8542 | 0.8510 | 0.28 | 0.24 | 0.39 |
| rrf | 0 | 1.0000 | 0.9635 | 0.9555 | 14.69 | 10.20 | 19.81 |
| rerank_5 | 5 | 1.0000 | 0.9844 | 0.9796 | 72.90 | 66.16 | 117.72 |
| rerank_10 | 10 | 1.0000 | 0.9844 | 0.9771 | 153.49 | 119.03 | 273.83 |
| rerank_20 | 20 | 1.0000 | 0.9844 | 0.9771 | 263.16 | 224.94 | 506.96 |

**Percentage difference:** relative to dense-only nDCG@5: bm25: -12.42%; rrf: -1.67%; rerank_10: +0.56%.

**Tradeoff:** BM25 is extremely fast but misses some paraphrases. RRF adds little retrieval time but can move weak lexical matches upward. The cross-encoder adds substantial inference time and yields only a small gain over an already strong dense baseline.

**Conclusion:** This fixture does not support the claim that RRF alone beats dense retrieval. Its measured nDCG is lower. Reranking top-10 improves ranking slightly over dense-only, while all dense/hybrid configurations already reach full page Recall@5. These results explain the stages without guaranteeing that every stage helps on every corpus. No statistical significance is claimed for this small development set.

## Experiment 2.5: rerank top-5 versus top-20

**Hypothesis:** Reducing cross-encoder candidates lowers latency while preserving nDCG@5.

**Baseline:** Dense top-20 + BM25 top-20 → RRF → rerank top-20 → final top-5.

**Change:** Rerank only top-5; every other retrieval setting remains fixed.

**Dataset:** Same 32 questions, three repetitions, 96 samples per configuration. Actual reranked counts were 5–5 for the optimized configuration and 20–20 for the baseline.

**Metric / Before / After / Percentage difference:**

| Metric | Before: top-20 | After: top-5 | Difference |
|---|---:|---:|---:|
| MRR@5 | 0.984375 | 0.984375 | +0.00% |
| nDCG@5 | 0.977096 | 0.979605 | +0.26% |
| Mean full retrieval ms | 263.16 | 72.90 | 72.30% lower |
| P95 full retrieval ms | 506.96 | 117.72 | 76.78% lower |
| Mean reranking stage ms | 250.78 | 61.95 | 75.30% lower |

Quality retention = optimized nDCG / baseline nDCG = **100.26%**. A value above 100% means measured quality increased; it is not capped or relabeled as a probability.

**Tradeoff:** Lower candidate coverage can exclude relevant passages on other questions; this small corpus does not establish safety of the cutoff for larger HR collections.

**Conclusion:** Top-5 reduced mean measured retrieval and reranking latency on this fixture. The result supports this configuration comparison only; it is not a general speedup claim across hardware, document sizes, or concurrent workloads.

## Experiment 2.10: rerank top-10 versus top-20

**Hypothesis:** Reducing cross-encoder candidates lowers latency while preserving nDCG@5.

**Baseline:** Dense top-20 + BM25 top-20 → RRF → rerank top-20 → final top-5.

**Change:** Rerank only top-10; every other retrieval setting remains fixed.

**Dataset:** Same 32 questions, three repetitions, 96 samples per configuration. Actual reranked counts were 10–10 for the optimized configuration and 20–20 for the baseline.

**Metric / Before / After / Percentage difference:**

| Metric | Before: top-20 | After: top-10 | Difference |
|---|---:|---:|---:|
| MRR@5 | 0.984375 | 0.984375 | +0.00% |
| nDCG@5 | 0.977096 | 0.977096 | +0.00% |
| Mean full retrieval ms | 263.16 | 153.49 | 41.67% lower |
| P95 full retrieval ms | 506.96 | 273.83 | 45.99% lower |
| Mean reranking stage ms | 250.78 | 140.91 | 43.81% lower |

Quality retention = optimized nDCG / baseline nDCG = **100.00%**. A value above 100% means measured quality increased; it is not capped or relabeled as a probability.

**Tradeoff:** Lower candidate coverage can exclude relevant passages on other questions; this small corpus does not establish safety of the cutoff for larger HR collections.

**Conclusion:** Top-10 reduced mean measured retrieval and reranking latency on this fixture. The result supports this configuration comparison only; it is not a general speedup claim across hardware, document sizes, or concurrent workloads.

## Three defensible resume bullets

These bullets refer only to this synthetic benchmark and CPU setup:

- Built a local HR-policy retrieval engine combining FAISS, BM25, RRF, and cross-encoder reranking, achieving 0.9771 nDCG@5 versus 0.9717 for dense-only (0.56% relative improvement) on 32 manually labeled synthetic questions.
- Reduced mean cross-encoder reranking latency by 43.81% by scoring 10 instead of 20 fused candidates, retaining 100.00% of baseline nDCG@5 on the same fixture; full retrieval latency fell 41.67%.
- Measured 273.83 ms P95 warm retrieval latency, including query embedding through final ranking, across 32 synthetic evaluation questions and 96 timed trials using top-10 reranking on CPU.

## Verification and code review

Ten runnable tests pass: PDF extraction/overlap and blank PDFs; dense retrieval; BM25 and a one-chunk negative-score case; the RRF formula; bounded reranking and changed order; API upload/search/metrics including replacement/failure preservation/multiple documents; graded metrics and duplicate-page handling. Tests use real downloaded models. Browser upload/search and desktop/mobile layout were also inspected. The dependency consistency check passed. Upstream LibreSSL and FAISS deprecation warnings remain documented in README.

Every source file was inspected for unnecessary functions, wrappers, dependencies, duplicate logic, and dead code. Only the request-validation model is an application class; model loading, chunking, indexing, RRF, search, metrics, and evaluation each have concrete callers. The blank-PDF and BM25 edge cases are retained as checks, not production abstractions.

| Size measure | Count |
|---|---:|
| Python physical LOC, including tests/comments/blanks | 493 |
| Application + evaluation Python physical LOC | 358 |
| Test Python physical LOC | 135 |
| Nonblank, non-comment Python lines (includes docstrings) | 434 |
| Direct requirements | 10 (9 runtime/evaluation + 1 test) |
| Source files | 7 (4 implementation/evaluation Python + 1 test + HTML + CSS) |
| Evaluation questions | 32 |

## Unmeasured hypotheses

Larger chunks might retain useful context but increase inference cost or trigger truncation. More candidates might rescue relevant passages on harder or larger corpora. Incremental indexing might reduce upload time at scale. GPU inference might improve throughput. None of these were measured here and none is a project achievement.

## Exact-number answer checks — interface refinement

Ten additional questions were run through the same hybrid top-5 pipeline and verbatim answer selection. All ten passed assertions for the expected phrase, source page, and excerpt presence in a retrieved passage. These are regression checks on the fictional sample, not a production accuracy estimate.

| Question | Required phrase in the displayed answer | Page |
|---|---|---:|
| Annual leave per year | 20 working days | 1 |
| Casual leave per year | six days | 2 |
| Consecutive casual leave | two consecutive | 2 |
| Paid sick leave per year | ten paid sick leave days | 3 |
| Maternity leave duration | 26 weeks | 4 |
| Notice during probation | fifteen calendar days | 7 |
| Notice after confirmation | sixty calendar days | 8 |
| Work from home per week | two days per week | 9 |
| Travel claim deadline | thirty calendar days | 15 |
| Monthly internet allowance | 1500 currency units per month | 16 |

Run `HF_HUB_OFFLINE=1 python -m unittest test_system.RetrievalTests.test_exact_policy_numbers -v` to repeat. The browser also displayed the annual-leave answer with its page-1 citation. No answer values are hardcoded into the application; these expected values exist only in the test assertions.
