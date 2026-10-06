"""Independent warm runs of each retrieval mode; no embedding/score reuse across modes."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import numpy as np
from ingest import chunk_pdf
from retrieval import EMBEDDING_MODEL, RERANK_MODEL, EMBEDDING_REVISION, RERANK_REVISION, build_index, embedding_model, reranker, search, torch

ROOT = Path(__file__).parent


def quality(results, relevant, k=5):
    labels = {(r['document'], r['page']): r['grade'] for r in relevant}
    seen, gains = set(), []
    for result in results[:k]:
        key = (result['document'], result['page'])
        gains.append(labels.get(key, 0) if key not in seen else 0)
        seen.add(key)
    recall = sum(g > 0 for g in gains) / len(labels)
    mrr = next((1 / rank for rank, g in enumerate(gains, 1) if g > 0), 0.0)
    dcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, 1))
    ideal = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(sorted(labels.values(), reverse=True)[:k], 1))
    return recall, mrr, dcg / ideal


def run(args):
    dataset = json.loads(args.dataset.read_text())
    chunks = []
    hashes = {}
    for document in dataset['documents']:
        data = (args.dataset.parent / document).read_bytes()
        hashes[document] = hashlib.sha256(data).hexdigest()
        chunks.extend(chunk_pdf(data, document, embedding_model().tokenizer, args.chunk_size, args.overlap))
    for item in dataset['queries']:
        if not item['relevant'] or any(r['grade'] not in (1, 2) for r in item['relevant']):
            raise ValueError('Every query needs positive relevance judgments with grades 1 or 2.')
        if not {(r['document'], r['page']) for r in item['relevant']} <= {(c['document'], c['page']) for c in chunks}:
            raise ValueError('Relevance judgment references a page absent from the corpus.')
    index = build_index(chunks)
    reranker()
    configs = [('dense', 'dense', 0), ('bm25', 'bm25', 0), ('rrf', 'rrf', 0)] + [(f'rerank_{n}', 'rerank', n) for n in (5, 10, 20)]
    for _, mode, n in configs:
        search(dataset['queries'][0]['query'], index, mode=mode, candidate_count=n or 10)
    jobs = [(repeat, i, config) for repeat in range(args.repeats) for i in range(len(dataset['queries'])) for config in configs]
    random.Random(42).shuffle(jobs)
    samples = []
    for repeat, i, (name, mode, n) in jobs:
        item = dataset['queries'][i]
        result = search(item['query'], index, mode=mode, candidate_count=n or 10)
        recall, mrr, ndcg = quality(result['results'], item['relevant'])
        samples.append(dict(config=name, repeat=repeat, query_index=i, recall_at_5=recall, mrr_at_5=mrr,
                            ndcg_at_5=ndcg, candidate_count=result['candidate_count'], latency=result['latency'],
                            results=[{key: r[key] for key in ('chunk_id', 'document', 'page', 'score')} for r in result['results']]))
        if len(samples) % len(dataset['queries']) == 0:
            print(f'{len(samples)}/{len(jobs)} timed searches', flush=True)
    summary = []
    for name, mode, n in configs:
        rows = [s for s in samples if s['config'] == name]
        times = [s['latency']['total_retrieval_ms'] for s in rows]
        row = dict(config=name, candidate_count=n, samples=len(rows),
                   actual_candidates_min=min(s['candidate_count'] for s in rows),
                   actual_candidates_max=max(s['candidate_count'] for s in rows))
        row.update({key: float(np.mean([s[key] for s in rows])) for key in ('recall_at_5', 'mrr_at_5', 'ndcg_at_5')})
        row.update(average_latency_ms=float(np.mean(times)), p50_latency_ms=float(np.percentile(times, 50)),
                   p95_latency_ms=float(np.percentile(times, 95)),
                   average_rerank_ms=float(np.mean([s['latency']['rerank_ms'] for s in rows])))
        summary.append(row)
    baseline = summary[-1]
    comparisons = [dict(candidate_count=row['candidate_count'],
                        latency_reduction_percentage=(baseline['average_latency_ms'] - row['average_latency_ms']) / baseline['average_latency_ms'] * 100,
                        rerank_latency_reduction_percentage=(baseline['average_rerank_ms'] - row['average_rerank_ms']) / baseline['average_rerank_ms'] * 100,
                        ndcg_retained_percentage=row['ndcg_at_5'] / baseline['ndcg_at_5'] * 100 if baseline['ndcg_at_5'] else None)
                   for row in summary[3:5]]
    metadata = dict(timestamp=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                    processor=platform.processor(), python=platform.python_version(), torch_threads=torch.get_num_threads(),
                    embedding_model=EMBEDDING_MODEL, rerank_model=RERANK_MODEL, device='cpu',
                    embedding_revision=EMBEDDING_REVISION, rerank_revision=RERANK_REVISION,
                    versions={name: version(name) for name in ('sentence-transformers', 'transformers', 'torch', 'faiss-cpu', 'numpy', 'rank-bm25', 'pypdf')},
                    chunk_size=args.chunk_size, overlap=args.overlap, chunks=len(chunks), query_count=len(dataset['queries']),
                    repeats=args.repeats, seed=42, dense_k=20, bm25_k=20, rrf_k=60, final_k=5,
                    dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(), corpus_sha256=hashes,
                    timing_scope='Warm single-request retrieval; includes query embedding; excludes loading, indexing, HTTP and queueing.')
    report = dict(metadata=metadata, summary=summary, comparisons=comparisons, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(dict(metadata=metadata, summary=summary, comparisons=comparisons), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/evaluation.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/benchmark.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=400)
    parser.add_argument('--overlap', type=int, default=60)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args)
