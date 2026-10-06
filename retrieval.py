"""Dense + lexical retrieval, RRF, then a bounded cross-encoder pass."""
import os
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
from storage import setting
# Import after PyTorch: the reverse order crashed native inference on the tested macOS wheels.
try:
    import faiss
except ImportError:  # production uses Supabase pgvector instead of local FAISS
    faiss = None

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


def build_index(chunks, embed=True):
    if not chunks:
        raise ValueError('Upload a policy PDF before searching.')
    tokens = [tokenize(c['text']) for c in chunks]
    if not any(tokens):
        raise ValueError('The document has no searchable words.')
    index = None
    if embed:
        if faiss is None:
            raise RuntimeError('FAISS is required for local indexing; configure Supabase for production.')
        vectors = embedding_model().encode([c['text'] for c in chunks], normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
    return chunks, index, BM25Okapi(tokens)


def rrf(dense, lexical, k=60):
    scores = {}
    for ranking in (dense, lexical):
        for rank, (i, _) in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def search(query, index, mode='rerank', dense_k=20, bm25_k=20, candidate_count=10, final_k=5, rrf_k=60, dense_search=None):
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
    chunks, dense, lexical = index
    latency = dict.fromkeys(('query_embedding_ms', 'dense_search_ms', 'bm25_search_ms', 'rrf_ms', 'rerank_ms'), 0.0)
    ranked = []
    if mode != 'bm25':
        t = perf_counter()
        vector = embedding_model().encode([query], normalize_embeddings=True, show_progress_bar=False)
        latency['query_embedding_ms'] = (perf_counter() - t) * 1000
        t = perf_counter()
        if dense_search:
            ranked = dense_search(vector[0].tolist(), min(dense_k, len(chunks)))
        else:
            scores, ids = dense.search(vector, min(dense_k, len(chunks)))
            ranked = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]
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
    context = '\n\n'.join(f'Passage {i}: {result["text"]}' for i, result in enumerate(results[:5], 1))
    prompt = ('Answer the policy question in one or two short sentences. Use only the passages below. '
              'If the answer is absent, say: "I couldn\'t find this in the policy."\n\n'
              f'Question: {query}\n\n{context}')
    response = httpx.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent',
                          params={'key': key}, json={'contents': [{'parts': [{'text': prompt}]}],
                                                      'generationConfig': {'maxOutputTokens': 100, 'temperature': 0}}, timeout=30)
    if response.is_error:
        raise RuntimeError(f'Gemini request failed ({response.status_code}).')
    try:
        text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError('Gemini returned no answer.') from None
    return dict(text=text, document=results[0]['document'], page=results[0]['page'])
