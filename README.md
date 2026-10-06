# HR Policy Retrieval

A FastAPI app for searching uploaded HR-policy PDFs. It combines semantic search, BM25 keyword matching, reciprocal-rank fusion, and cross-encoder reranking. Results include document and page provenance.

## Run locally

Python 3.9–3.12 is supported.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8765
```

Create a server-only `.env` file:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

Run [supabase.sql](supabase.sql) once in the Supabase SQL editor. Then open <http://127.0.0.1:8765>, upload a PDF, and search it. The included `data/hr_policy.pdf` is a fictional demo document.

Models download on first startup. Afterward, set `HF_HUB_OFFLINE=1` to prevent model-update checks.

## Architecture

```text
PDF → text extraction → chunks → embeddings + BM25
                                  ↓
                   Supabase pgvector + local BM25 index
                                  ↓
                   RRF → cross-encoder → ranked passages
```

Uploaded documents are stored in Supabase. The service rebuilds its small local BM25 index at startup.

## Test and evaluate

```sh
HF_HUB_OFFLINE=1 python -m unittest -v
HF_HUB_OFFLINE=1 python evaluate.py --repeats 3
```

The evaluator uses the fictional test corpus and writes local benchmark output under `data/`; generated benchmark files are intentionally not committed.

## Project layout

```text
app.py                 API and UI routes
ingest.py              PDF extraction and chunking
retrieval.py           retrieval and reranking
storage.py             Supabase client
evaluate.py            offline evaluation
test_system.py         application tests
supabase.sql           database schema
templates/             HTML interface
static/                stylesheet
data/                  fictional test corpus and judgments
```
