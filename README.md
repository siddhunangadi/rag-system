# HR Policy Finder

A small web app that helps you find answers inside HR-policy PDFs.

**Upload a policy → ask a question → read the answer and its cited pages.**

For example, upload an employee handbook and ask: **“How many annual leave days do I get?”** The app searches the uploaded text and asks Gemini to write a short answer from the passages it finds.

## What is in this folder?

```text
rag-system/
├── app.py                  Starts the app and handles browser requests
├── ingest.py               Reads a PDF and splits its pages into small passages
├── retrieval.py            Searches passages and creates an answer
├── storage.py              Reads and writes data in Supabase
│
├── templates/
│   └── index.html          The page you see in your browser, including its JavaScript
├── static/
│   └── style.css           The appearance and mobile layout of that page
│
├── data/
│   ├── hr_policy.pdf       A fictional PDF you can upload to try the app
│   └── evaluation.json     Questions and expected pages used to measure search quality
│
├── supabase.sql            Sets up the database and PDF storage
├── test_system.py          Checks that the application behaves correctly
├── evaluate.py             Measures and compares the search methods
├── requirements.txt        Lists the Python packages to install
├── Dockerfile              Packages the app so it can run in Docker
├── .dockerignore           Keeps unnecessary files out of the Docker build
├── .gitignore              Keeps local files and secrets out of Git
└── README.md               This guide
```

**Start reading the code in `app.py`.** It receives uploads and questions, then calls the other Python files. The sections below show those connections.

## What happens when I upload a PDF?

1. **`app.py` receives the file.** It checks the filename and upload size.
2. **`ingest.py` reads each page.** It breaks the text into smaller passages and keeps the document name and page number with each one.
3. **`app.py` creates a numerical representation of each passage.** This is called an *embedding*. It lets the app compare text by meaning.
4. **`storage.py` saves the data in Supabase.** The original PDF goes into file storage. The passages and their embeddings go into Supabase’s database.

The app also prepares a keyword search index in memory. Uploading another PDF with the **same filename** replaces that document.

## What happens when I ask a question?

1. **`templates/index.html` sends the question to `app.py`.**
2. **`app.py` calls `retrieval.py`.** It searches for passages in two ways: similar meaning through Supabase, and matching words through keyword search.
3. **`retrieval.py` combines and reorders the results.** A local model rereads the strongest question–passage pairs to choose the most relevant passages.
4. **Gemini writes the answer.** It receives up to five passages and returns the IDs of the passages it cites.
5. **The app checks those IDs and shows the answer.** The browser displays the cited document names and page numbers, plus the retrieved passages for you to review.

The app rejects missing or invalid citations instead of assigning a guessed source. A valid source ID makes an answer traceable; it does not prove the answer is correct.

## How to run it

You need **Python 3.9–3.12**, a **Supabase project with pgvector 0.8 or newer**, and a **Gemini API key**.

### 1. Install the packages

Download the project and install its packages. If you already have it, start from its folder.

```sh
git clone https://github.com/siddhunangadi/rag-system.git
cd rag-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Supabase

Open your Supabase project's **SQL Editor** and run [supabase.sql](supabase.sql). This creates the document tables, private PDF bucket, and vector search functions. Run it when setting up or upgrading the app.

### 3. Add the keys

Create `.env` beside `app.py`:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
GEMINI_API_KEY=your-gemini-api-key
```

Keep `.env` private. It is excluded from Git. Supabase is required for storage and semantic search; Gemini is required to generate answers.

### 4. Start the app

```sh
uvicorn app:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and upload `data/hr_policy.pdf` to try it. Copying a PDF into `data/` does **not** index it; upload it through the page.

The local search models download on first startup. Later startups reuse the downloaded models.

## How to check it

Run the application tests from the project folder:

```sh
python -m unittest -v
```

Tests use real local search models and replace Supabase and Gemini calls with test responses. They do not need live service credentials. After the models are cached, `HF_HUB_OFFLINE=1` lets these tests run without downloading models.

`evaluate.py` measures retrieval quality and speed; it does not test generated answers. Use a **separate Supabase project** containing only the PDFs listed in `data/evaluation.json`, uploaded with matching chunk settings. Point `.env` at that project, then run:

```sh
python evaluate.py --repeats 3
```

It checks the stored documents match the evaluation data, reads Supabase without uploading anything, and writes results to `data/benchmark.json`. That output is excluded from Git.

## Things to know

- Use unencrypted, text-based PDFs. Scanned images need OCR, which is not included. Each PDF can be up to 20 MB, 300 pages, and 2 million extracted characters; uploads are limited to 5,000 total passages across the collection.
- PDFs and passages remain in Supabase after a restart. The app rebuilds its in-memory keyword index on startup. Run one application process so uploads and searches stay consistent.
- The app uses one shared document collection and has no user login or per-user document separation.
- Gemini receives your question and the retrieved passage text. Embedding and reranking happen locally on CPU.
- Re-upload existing PDFs after changing chunk settings; a restart does not reprocess saved PDFs.
- Review the original policy wording for conditions and exceptions. Search can retrieve weak matches, and model answers can be wrong. This is a policy lookup tool, not legal advice.

<details>
<summary>Optional: search settings and model names</summary>

| Part | Implementation |
|---|---|
| PDF splitting | LangChain `RecursiveCharacterTextSplitter`; up to 500 characters, target overlap 100 characters; each page split separately |
| Embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`; 384 numbers per passage |
| Meaning search | Supabase PostgreSQL, pgvector, cosine similarity, and HNSW |
| Keyword search | BM25 |
| Combining rankings | Reciprocal Rank Fusion (RRF) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Answer generation | Gemini through `gemini-flash-lite-latest` |

Search takes up to 20 meaning matches and 20 keyword matches. The web page reranks up to five candidates; the API defaults to ten. Both return up to five passages by default. Similarity scores are not confidence percentages, and there is no minimum relevance threshold.

The chunk settings are a starting configuration, not a benchmarked optimum. The embedding model still tokenizes each resulting passage internally.

</details>

<details>
<summary><strong>Technical note: checking HNSW index use</strong></summary>

HNSW is an approximate nearest-neighbor index: it trades some recall for search speed. An index existing does not prove that PostgreSQL uses it. On a small table, a full scan can be cheaper.

An observed check on **2026-10-18** showed a sequential scan of 100 chunks followed by a top-N sort despite the index existing. The current `match_policy_chunks` function orders directly by cosine distance and applies these settings for the duration of the function:

- `enable_seqscan=off` prefers an index scan.
- `hnsw.ef_search=100` increases the initial candidate search breadth.
- `hnsw.iterative_scan=strict_order` continues scanning when dead tuples reduce the initial result set.

The settings are restored when the function returns. A follow-up on **2026-10-21** confirmed an HNSW scan in the underlying query with these settings. The live RPC returned 20 results for a limit of 20 and all 99 stored chunks for a limit of 100. These small-corpus checks are not a general performance benchmark.

To inspect the underlying SQL on your own data:

```sql
begin;
select vector_dims('[0]'::vector);
set local enable_seqscan = off;
set local hnsw.ef_search = 100;
set local hnsw.iterative_scan = strict_order;
explain (analyze, buffers, verbose)
select chunk_id, 1 - (embedding <=> (select embedding from policy_chunks limit 1)) as similarity
from policy_chunks
order by embedding <=> (select embedding from policy_chunks limit 1)
limit 20;
rollback;
```

Look for `Index Scan using policy_chunks_embedding_idx` on the outer search. The sample-vector subquery may scan a row to obtain its input. This checks the underlying SQL, not an application's full request. Recheck representative plans and retrieval quality as the corpus grows.

</details>
