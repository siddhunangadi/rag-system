"""Dense + lexical retrieval, RRF, then a bounded cross-encoder pass."""
import os
import json
from functools import lru_cache
from pathlib import Path
import re
from time import perf_counter
import numpy as np
from rank_bm25 import BM25Okapi
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache" / "huggingface"))
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import httpx
from storage import dense_search, setting

# A fixed CPU budget makes local benchmarks easier to reproduce.
torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '4')))
EMBEDDING_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
EMBEDDING_REVISION = 'b207367332321f8e44f96e224ef15bc607f4dbf0'
RERANK_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'


@lru_cache(maxsize=1)
def embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device='cpu')


@lru_cache(maxsize=1)
def reranker():
    return CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, device='cpu', max_length=512)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_index(chunks):
    if not chunks:
        raise ValueError('Upload a policy PDF before searching.')
    tokens = [tokenize(c['text']) for c in chunks]
    if not any(tokens):
        raise ValueError('The document has no searchable words.')
    return chunks, BM25Okapi(tokens)


def rrf(dense, lexical, k=60):
    scores = {}
    for ranking in (dense, lexical):
        for rank, (i, _) in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def search(query, index, mode='rerank', dense_k=20, bm25_k=20, candidate_count=10, final_k=5, rrf_k=60):
    started = perf_counter()
    query = query.strip()
    if not query or len(query) > 500:
        raise ValueError('Enter a question of 1–500 characters.')
    if mode not in ('dense', 'bm25', 'rrf', 'rerank'):
        raise ValueError('Unknown retrieval mode.')
    if not all(1 <= k <= 100 for k in (dense_k, bm25_k, candidate_count, final_k, rrf_k)):
        raise ValueError('Retrieval limits must be between 1 and 100.')
    if mode == 'rerank' and final_k > candidate_count:
        raise ValueError('final_k cannot exceed candidate_count.')
    chunks, lexical = index
    latency = dict.fromkeys(('query_embedding_ms', 'dense_search_ms', 'bm25_search_ms', 'rrf_ms', 'rerank_ms'), 0.0)
    ranked = []
    if mode != 'bm25':
        t = perf_counter()
        vector = embedding_model().encode([query], normalize_embeddings=True, show_progress_bar=False)
        latency['query_embedding_ms'] = (perf_counter() - t) * 1000
        t = perf_counter()
        positions = {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}
        ranked = [(positions[row['chunk_id']], float(row['similarity']))
                  for row in dense_search(vector[0].tolist(), min(dense_k, len(chunks)))
                  if row['chunk_id'] in positions]
        latency['dense_search_ms'] = (perf_counter() - t) * 1000
    if mode != 'dense':
        t = perf_counter()
        terms = tokenize(query)
        scores = lexical.get_scores(terms)
        # Matching terms can have zero or negative BM25 scores in very small corpora.
        lexical_ranked = [(int(i), float(scores[i])) for i in np.argsort(-scores, kind='stable')
                          if any(term in lexical.doc_freqs[i] for term in terms)][:bm25_k]
        latency['bm25_search_ms'] = (perf_counter() - t) * 1000
        if mode == 'bm25':
            ranked = lexical_ranked
    if mode in ('rrf', 'rerank'):
        t = perf_counter()
        ranked = rrf(ranked, lexical_ranked, rrf_k)
        latency['rrf_ms'] = (perf_counter() - t) * 1000
    actual_candidates = 0
    if mode == 'rerank':
        t = perf_counter()
        ranked = ranked[:candidate_count]
        actual_candidates = len(ranked)
        scores = reranker().predict([(query, chunks[i]['text']) for i, _ in ranked], show_progress_bar=False)
        ranked = sorted([(i, float(s)) for (i, _), s in zip(ranked, scores)], key=lambda pair: (-pair[1], pair[0]))
        latency['rerank_ms'] = (perf_counter() - t) * 1000
    results = [dict(chunks[i], rank=rank, score=score) for rank, (i, score) in enumerate(ranked[:final_k], 1)]
    latency['total_retrieval_ms'] = (perf_counter() - started) * 1000
    return dict(query=query, results=results, latency=latency, mode=mode, candidate_count=actual_candidates,
                score_type={'dense': 'cosine_similarity', 'bm25': 'bm25', 'rrf': 'rrf', 'rerank': 'cross_encoder_logit'}[mode])


def answer_from_llm(query, results):
    if not results:
        return None
    key = setting('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('Gemini is not configured.')
    passages = {f'S{i}': result for i, result in enumerate(results[:5], 1)}
    context = '\n\n'.join(f'{source_id}: {result["text"]}' for source_id, result in passages.items())
    not_found = "I couldn't find this in the policy."
    prompt = ('Answer the policy question in one or two short sentences. Use only the passages below. '
              'Return JSON with text and source_ids. Cite every passage supporting the answer using its source ID. '
              f'If the answer is absent, return text "{not_found}" and an empty source_ids list.\n\n'
              f'Question: {query}\n\n{context}')
    schema = {'type': 'OBJECT', 'properties': {'text': {'type': 'STRING'},
              'source_ids': {'type': 'ARRAY', 'items': {'type': 'STRING', 'enum': list(passages)}}},
              'required': ['text', 'source_ids']}
    try:
        response = httpx.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent',
                              params={'key': key}, json={'contents': [{'parts': [{'text': prompt}]}],
                              'generationConfig': {'maxOutputTokens': 300, 'temperature': 0,
                                                   'responseMimeType': 'application/json', 'responseSchema': schema}}, timeout=30)
    except httpx.RequestError:
        raise RuntimeError('Cannot reach Gemini. Try again later.') from None
    if response.is_error:
        raise RuntimeError(f'Gemini request failed ({response.status_code}).')
    try:
        answer = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
        text, source_ids = answer['text'], answer['source_ids']
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError('Gemini returned an invalid answer or missing source citations.') from None
    if (not isinstance(text, str) or not text.strip() or not isinstance(source_ids, list)
            or any(not isinstance(source_id, str) or source_id not in passages for source_id in source_ids)
            or (not source_ids and text.strip() != not_found)):
        raise RuntimeError('Gemini returned an invalid answer or missing source citations.')
    if text.strip() == not_found:
        source_ids = []
    sources = [dict(source_id=source_id, **{key: passages[source_id][key] for key in ('chunk_id', 'document', 'page')})
               for source_id in dict.fromkeys(source_ids)]
    return dict(text=text.strip(), sources=sources)
