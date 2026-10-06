# Minimal HR Policy Retrieval

> **Current engineering and interview documentation:** See [the project guide](docs/project-guide.md) for the implemented Gemini answer pipeline, AWS architecture, measured resume bullets, limitations, and interview preparation. Some descriptions below predate generation and deployment; the guide identifies those historical differences.

A passage finder for HR policy PDFs. Upload a document, ask a question, and inspect ranked passages with document/page provenance and measured retrieval time. It does not generate answers or make policy decisions.

## Run

Python 3.9–3.12. The recorded run used Python 3.9.6 on macOS ARM64; Python 3.11 with OpenSSL is preferable to Apple's older system Python, which emits an urllib3/LibreSSL warning.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8765
```

Add a server-only `.env` file before starting:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

Run [supabase.sql](supabase.sql) once in the Supabase SQL editor. Open <http://127.0.0.1:8765>, upload `data/hr_policy.pdf` for the fictional demonstration, then try “Who handles POSH complaints?” Models download on first startup into `.cache/huggingface`; the server becomes ready after loading both. Once cached, set `HF_HUB_OFFLINE=1` to prevent model-update network checks. Model revisions are pinned in `retrieval.py`.

Use **one worker**. Supabase stores source PDFs, chunks, and normalized embeddings; the app rebuilds only its small local BM25 index at startup. Uploads with a new filename add documents; the same filename replaces that document. Failed ingestion leaves the local search state unchanged.

## Architecture

```text
PDF → text extraction → cleaning → page-aware chunks
                                    ↓
                       ┌────────────┴───────────┐
                       │                        │
                    Embeddings              BM25 index
                       │                        │
                Supabase pgvector                 │
                       └────────────┬───────────┘
                                    ↓
Query → query embedding / tokens → Dense top-20 + BM25 top-20
                                    ↓
                              RRF (k = 60)
                                    ↓
                         Candidate selection (10)
                                    ↓
                             Cross-encoder
                                    ↓
                         Final top-5 passages
```

Dense and BM25 are independent retrieval branches, executed sequentially in one process. No concurrency framework is needed for this local corpus.

## Why these stages?

- **Dense search:** Sentence Transformers `multi-qa-MiniLM-L6-cos-v1` represents semantic similarity, including paraphrases without exact word overlap. Normalized embeddings are stored and searched with Supabase pgvector cosine similarity.
- **BM25:** `rank-bm25` preserves exact terms such as POSH, gratuity, probation, and notice period. Unicode word tokenization lowercases text without stemming. Nonmatching chunks are excluded by token presence, not score sign: BM25 can be zero or negative in tiny corpora.
- **Hybrid retrieval:** mixes these signals. It is a hypothesis about coverage, not a guarantee of better rankings; compare the ablations below.
- **RRF:** sums `1 / (60 + rank)` across lists, starting ranks at 1. It avoids normalizing incompatible cosine/BM25 scores. A passage present in both lists gets two contributions.
- **Cross-encoder:** `ms-marco-MiniLM-L6-v2` jointly scores query/passage pairs. Only the top N fused candidates are scored. Its output is a raw relevance logit, **not a probability**. No stage can recover a relevant passage outside its candidate set.

## Chunking

Default: **400 model tokens, approximately 60-token overlap**. These values retain paragraph context while leaving room in the 512-token models for typical queries. Whole-word boundaries avoid changing token counts through partial WordPiece words. Original casing and wording are preserved after whitespace normalization. Chunks never span pages, so page citations remain exact; short pages remain short chunks.

Engineering API settings support 32–500 tokens and overlap below chunk size. Token overlap is approximate because boundaries stay on words. Very long single words exceeding the configured window are rejected with a request to increase the size. At large chunk sizes or long queries, the cross-encoder's 512-token pair limit truncates input; increasing chunk size can lose tail information. The default is a starting hypothesis, not a measured optimum. The fixture's short pages do **not** validate a 400-token optimum or long-document performance.

Each chunk keeps only `chunk_id`, `document`, `page` (1-based), and `text`. IDs derive from document name, bytes, page, and token offset.

## Latency and evaluation

```sh
HF_HUB_OFFLINE=1 python -m unittest -v
HF_HUB_OFFLINE=1 python evaluate.py --repeats 3
# A separate chunk-size experiment; keep the default benchmark intact:
HF_HUB_OFFLINE=1 python evaluate.py --chunk-size 200 --overlap 40 --output data/benchmark_200.json
```

Remove the offline setting if the models have not been downloaded yet. `httpx` is only needed for the API smoke test; the other nine direct dependencies support the application/evaluator.

The evaluator compares dense, BM25, RRF, and reranking top 5/10/20 with the **same corpus, queries, and first-stage limits**. Each trial recomputes its own retrieval pipeline: no cached query embeddings or reused cross-encoder scores. It warms each configuration once, then shuffles all trials with seed 42. Three repetitions over 32 distinct questions give 96 timing samples per configuration, 576 total. Repetitions improve timing sampling, not the number of independent relevance judgments.

`data/evaluation.json` contains manually authored graded judgments against the 24-page synthetic `data/hr_policy.pdf`. Grade 2 directly answers; grade 1 provides supporting context. Labels were written before measured ranking runs. The set includes exact terms, paraphrases, related policy distractors, and questions with multiple relevant pages. It is a small development fixture, not a held-out or production benchmark; unjudged pages count as irrelevant. Its fictional benefit amounts and policy conditions are not legal guidance.

All quality metrics use the first five returned **chunks**. A page earns credit only at its first occurrence, so overlapping chunks cannot inflate scores. Metrics are macro-averaged over questions:

- **Recall@5:** unique relevant pages found / all labeled relevant pages.
- **MRR@5:** reciprocal rank of the first relevant page, or zero if absent from the first five results. This is truncated MRR, not full-ranking MRR.
- **nDCG@5:** graded gain `2^grade - 1`, discounted by `log2(rank + 1)`, divided by ideal DCG.
- **Mean, P50, P95 latency:** empirical statistics over measured trials; P50/P95 use NumPy's default linear percentile interpolation.

`time.perf_counter()` measures query embedding, dense lookup, BM25, RRF, reranking, and total retrieval. The live dense lookup includes the Supabase request. Total includes validation and formatting inside retrieval, excludes startup/download, ingestion, and lock queueing. Disabled stages return zero. This is warm **server retrieval latency**, not browser end-to-end response time. Live first-query timings may be higher. The API serializes model/index access; these numbers are not a concurrency or throughput benchmark.

Every run writes `data/benchmark.json`: raw per-trial timings/rankings, summaries, candidate counts, comparisons, seed, model revisions, dependency versions, and corpus/dataset hashes. `/metrics` references that fixture report, **not the user's uploaded corpus**. Rerunning updates that report; the dated documentation below remains a recorded snapshot until deliberately refreshed.

## Measured results

Recorded 2026-09-25T10:03:06.310847+00:00: top-10 reranking achieved **0.9844 MRR@5**, **0.9771 nDCG@5**, and **273.83 ms P95** warm retrieval latency. Compared with top-20, mean full retrieval time decreased **41.67%**, retaining **100.00%** of nDCG@5 on the 32-question synthetic fixture.

See [the recorded experiment and measured resume bullets](docs/experiments.md). All numeric achievements there come from the saved run; no improvement percentage is hardcoded into the evaluator.

The latency reduction calculation is `(baseline_mean - optimized_mean) / baseline_mean * 100`. The baseline reranks 20 candidates. Quality retention is `optimized_nDCG / baseline_nDCG * 100`, which can exceed 100% if the smaller candidate set ranks better. The report distinguishes full retrieval time from the reranking stage alone.

**EXPECTED/HYPOTHESIZED:** lexical fusion may improve coverage on larger or unfamiliar HR corpora; smaller rerank pools may reduce time while preserving quality. Neither is guaranteed. In this fixture, RRF alone underperforms dense-only on ranking quality. Only the measured configurations and dataset support the recorded claims.

## API

The only data endpoints are `POST /upload`, `POST /search`, and `GET /metrics`; `/` and `/static/style.css` serve the interface.

```sh
curl -F 'file=@data/hr_policy.pdf' -F 'chunk_size=400' -F 'overlap=60' http://127.0.0.1:8765/upload
curl -H 'Content-Type: application/json' -d '{"query":"Who handles POSH complaints?","dense_k":20,"bm25_k":20,"candidate_count":10,"final_k":5,"rrf_k":60}' http://127.0.0.1:8765/search
curl http://127.0.0.1:8765/metrics
```

`/search` accepts `mode`: `dense`, `bm25`, `rrf`, or `rerank` (default). All ranking limits are integers from 1 to 100; `final_k` cannot exceed `candidate_count` in rerank mode. `candidate_count` is both the fused cutoff and rerank budget: separate identical settings would be redundant. Fewer candidates/results are returned when the corpus is smaller. A BM25 query with no matching tokens returns an empty list; dense/hybrid retrieval always produces nearest passages, even for out-of-domain questions.

The response contains `query`, `mode`, `score_type`, actual reranked `candidate_count`, `results`, and `latency`. Each result includes rank, text, document, page, score, and chunk ID. Timing keys are:

```text
query_embedding_ms, dense_search_ms, bm25_search_ms,
rrf_ms, rerank_ms, total_retrieval_ms
```

`/metrics` returns current document/chunk counts, statistics for up to the last 100 successful searches (cleared after upload), and the saved benchmark's metadata/summary if available. Live metrics may mix configurations; use `evaluate.py` for controlled comparisons.

Upload limits: 20 MB, 300 pages, two million extracted characters, 5,000 total indexed chunks. Malformed/encrypted/scanned PDFs are rejected; OCR is intentionally absent. Invalid uploads return 400 (413 for size), invalid search inputs 422, searching an empty index 409. PDF byte/page limits are practical local-demo guards, not a complete defense against hostile compressed files; keep this app bound to localhost with trusted documents.

## Files and tradeoffs

```text
app.py                 API, cached BM25 state, locking, UI routes
storage.py             small server-side Supabase REST client
retrieval.py           models, BM25, RRF, bounded reranking
ingest.py             PDF extraction and chunking
evaluate.py           metrics, repeated benchmarks, raw results
test_system.py        runnable ingestion/retrieval/API/metric checks
requirements.txt      nine runtime dependencies + httpx for tests
templates/index.html  accessible forms and small vanilla JavaScript
static/style.css      responsive layout
data/evaluation.json  questions and graded page judgments
data/hr_policy.pdf    fictional, inspectable test corpus
data/benchmark.json   generated measurements and raw trials
docs/decisions.md     concise engineering decision log
docs/experiments.md   hypotheses, results, limitations, resume bullets
```

No agents, chat history, frontend framework, background queue, or deployment infrastructure. Supabase pgvector stores persistent dense vectors; BM25 remains a small local index because its tokenized representation is fast to rebuild. FAISS remains only in the offline evaluator so its historical benchmark remains reproducible. Uploads embed the incoming document once; optimize incremental bulk ingestion only after measuring a larger corpus. HTML renders document text using `textContent`, never as executable markup. Form controls have labels, status announcements, keyboard focus, loading/error/empty states, and a mobile layout.

The one application class is the FastAPI request-validation model. There are no manager/service/repository wrappers. Two cached model loaders prevent repeated downloads/loading; the standalone RRF and quality functions keep the ranking formulas directly testable. On the tested macOS wheels, PyTorch must import before FAISS to avoid a native crash; the import order is intentional. Upstream LibreSSL and FAISS deprecation warnings were observed without failing the checks.

## Answer-first interface update

The interface now fixes hybrid RRF + reranking to five candidates and five supporting results, the best observed tradeoff in the saved 5/10/20 experiment. Chunking stays at 400 tokens / approximately 60 overlap; controls remain available only through the engineering API/evaluator. Method selectors, raw model scores, and the saved quality table are removed from the user interface.

Search now returns an `answer` containing a **verbatim selected excerpt**, document, and page. The existing cross-encoder selects a two-sentence window from up to 24 windows across the three highest-ranked passages. This is not generated synthesis, and it does not establish that a question is answerable; the interface explicitly labels it as matching policy wording and offers full passages for conditions/context. All returned supporting passages are collapsed, and timing appears last in a closed disclosure.

`answer_selection_ms` and `total_response_ms` include this extra selection pass separately. The saved benchmark and resume bullets measure the original retrieval stages only, **not** this new answer-selection stage. Historical LOC figures in the experiment snapshot predate this UI update. See the researched default rationale in `docs/decisions.md`.
