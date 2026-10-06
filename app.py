"""Single-process local API. Run: uvicorn app:app --host 127.0.0.1"""
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
import json
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ingest import chunk_pdf
from retrieval import build_index, embedding_model, reranker, search, answer_from_llm
from storage import dense_search, enabled as cloud_enabled, load_chunks, save_document

ROOT = Path(__file__).parent
index = None
recent = deque(maxlen=100)
lock = Lock()


@asynccontextmanager
async def lifespan(app):
    embedding_model()
    reranker()
    global index
    if cloud_enabled():
        chunks = load_chunks()
        if chunks:
            index = build_index(chunks, embed=False)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: Literal['dense', 'bm25', 'rrf', 'rerank'] = 'rerank'
    dense_k: int = Field(default=20, ge=1, le=100)
    bm25_k: int = Field(default=20, ge=1, le=100)
    candidate_count: int = Field(default=10, ge=1, le=100)
    final_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=100)


@app.get('/')
def home():
    return FileResponse(ROOT / 'templates/index.html')


@app.get('/static/style.css')
def stylesheet():
    return FileResponse(ROOT / 'static/style.css')


@app.post('/upload')
def upload(file: UploadFile = File(...), chunk_size: int = Form(400), overlap: int = Form(60)):
    global index
    data = file.file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, 'PDF must be 20 MB or smaller.')
    document = Path(file.filename or 'policy.pdf').name[:200]
    if not document.lower().endswith('.pdf'):
        raise HTTPException(400, 'Upload a PDF file.')
    # ponytail: serialize this local demo; use immutable snapshots if concurrent throughput matters.
    with lock:
        try:
            incoming = chunk_pdf(data, document, embedding_model().tokenizer, chunk_size, overlap)
            previous = [c for c in index[0] if c['document'] != document] if index else []
            if len(previous) + len(incoming) > 5000:
                raise ValueError('The local index is limited to 5000 chunks.')
            replacement = build_index(previous + incoming, embed=not cloud_enabled())
            if cloud_enabled():
                vectors = embedding_model().encode([c['text'] for c in incoming], normalize_embeddings=True,
                                                    convert_to_numpy=True, show_progress_bar=False)
                save_document(document, data, incoming, vectors)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        index = replacement
        recent.clear()
    return dict(document=document, chunks=len(incoming), total_chunks=len(replacement[0]),
                chunk_size=chunk_size, overlap=overlap)


@app.post('/search')
def query(request: SearchRequest):
    with lock:
        if index is None:
            raise HTTPException(409, 'Upload a policy PDF first.')
        try:
            positions = {chunk['chunk_id']: i for i, chunk in enumerate(index[0])}
            remote = (lambda vector, limit: [(positions[row['chunk_id']], float(row['similarity']))
                                              for row in dense_search(vector, limit) if row['chunk_id'] in positions]) if cloud_enabled() else None
            result = search(index=index, dense_search=remote, **request.model_dump())
            started = perf_counter()
            result['answer'] = answer_from_llm(result['query'], result['results'])
            result['latency']['answer_generation_ms'] = (perf_counter() - started) * 1000
            result['latency']['total_response_ms'] = result['latency']['total_retrieval_ms'] + result['latency']['answer_generation_ms']
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        recent.append(result['latency']['total_retrieval_ms'])
        return result


@app.get('/metrics')
def metrics():
    with lock:
        totals = list(recent)
        report = ROOT / 'data/benchmark.json'
        evaluation = json.loads(report.read_text()) if report.exists() else None
        return dict(documents=len({c['document'] for c in index[0]}) if index else 0,
                    chunks=len(index[0]) if index else 0, recent_searches=len(totals),
                    latency_ms=dict(average=float(np.mean(totals)), p50=float(np.percentile(totals, 50)),
                                    p95=float(np.percentile(totals, 95))) if totals else None,
                    evaluation={key: evaluation[key] for key in ('metadata', 'summary')} if evaluation else None)
