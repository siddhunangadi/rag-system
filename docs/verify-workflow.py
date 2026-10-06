"""Offline study trace: real retrieval, simulated Gemini; never writes Supabase.
Run from project root: .venv/bin/python docs/verify-workflow.py
"""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HR_LOCAL_ONLY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.chdir(ROOT)

import httpx
from fastapi.testclient import TestClient
import app
import retrieval

calls = []


def simulated_gemini(url, **kwargs):
    calls.append(dict(url=url, body=kwargs['json'], timeout=kwargs['timeout']))
    return httpx.Response(200, json={'candidates': [{'content': {'parts': [
        {'text': 'SIMULATED GEMINI RESPONSE — not an observed model answer.'}
    ]}}]})


with patch('retrieval.setting', return_value='offline-test-key'), patch('retrieval.httpx.post', side_effect=simulated_gemini):
    suite = unittest.defaultTestLoader.loadTestsFromName('test_system.ApiTests')
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    assert outcome.wasSuccessful(), 'Existing API regression failed'
    app.index = None
    app.recent.clear()
    with TestClient(app.app, raise_server_exceptions=False) as client:
        upload = client.post('/upload', files={'file': ('hr_policy.pdf', (ROOT / 'data/hr_policy.pdf').read_bytes(), 'application/pdf')})
        assert upload.status_code == 200
        body = {'query': 'How many annual leave days do I get per year?', 'mode': 'rerank', 'candidate_count': 5, 'final_k': 5}
        stages = {}
        original_rrf = retrieval.rrf
        model = retrieval.reranker()
        original_predict = model.predict

        def trace_rrf(dense, lexical, k):
            fused = original_rrf(dense, lexical, k)
            stages.update(dense=dense, bm25=lexical, fused=fused, rrf_k=k)
            return fused

        def trace_predict(pairs, **kwargs):
            scores = original_predict(pairs, **kwargs)
            stages.update(rerank_pairs=pairs, rerank_scores=scores.tolist())
            return scores

        with patch('retrieval.rrf', side_effect=trace_rrf), patch.object(model, 'predict', side_effect=trace_predict):
            result = client.post('/search', json=body)
        assert result.status_code == 200
        payload = result.json()
        assert payload['results'][0]['page'] == 1
        assert '20 working days' in payload['results'][0]['text']
        assert payload['answer']['text'].startswith('SIMULATED')
        assert payload['candidate_count'] == len(payload['results']) == 5
        assert calls[-1]['body']['contents'][0]['parts'][0]['text'].count('Passage ') == 5
        observed = {'scope': 'Real local PDF/models/FAISS/BM25/RRF/reranker/API; Gemini transport simulated; no browser or Supabase execution',
                    'request': body, 'upload': upload.json(), 'response': payload, 'stages': stages,
                    'gemini_request_without_credentials': calls[-1]}
        failures = {}
        for name, request in [('whitespace', {'query': '   '}), ('invalid_limit', {'query': 'leave', 'dense_k': 0}),
                              ('invalid_mode', {'query': 'leave', 'mode': 'unknown'}), ('cutoff', {'query': 'leave', 'candidate_count': 2})]:
            response = client.post('/search', json=request)
            assert response.status_code == 422
            failures[name] = {'status': response.status_code, 'body': response.json()}
        with patch('retrieval.setting', return_value=''):
            response = client.post('/search', json=body)
            assert response.status_code == 503
            failures['missing_gemini_key'] = {'status': 503, 'body': response.json()}
        for name, fake in [('gemini_http_error', httpx.Response(429)), ('missing_candidates', httpx.Response(200, json={}))]:
            with patch('retrieval.httpx.post', return_value=fake):
                response = client.post('/search', json=body)
                assert response.status_code == 503
                failures[name] = {'status': 503, 'body': response.json()}
        with patch('retrieval.httpx.post', side_effect=httpx.ReadTimeout('simulated timeout')):
            response = client.post('/search', json=body)
            assert response.status_code == 500
            failures['transport_timeout'] = {'status': 500, 'body': response.text, 'content_type': response.headers['content-type']}
        observed['failures'] = failures
        observed['metrics'] = client.get('/metrics').json()
        vector = retrieval.embedding_model().encode([body['query']], normalize_embeddings=True)
        observed['query_vector'] = {'shape': list(vector.shape), 'dtype': str(vector.dtype), 'first_8_values': vector[0][:8].tolist()}
        target = ROOT / 'docs/workflow-runtime-evidence.json'
        target.write_text(json.dumps(observed, indent=2) + '\n')
        print(json.dumps({'evidence': str(target), 'chunks': upload.json()['chunks'],
                          'pages': [r['page'] for r in payload['results']], 'failure_checks': len(failures),
                          'gemini': 'SIMULATED', 'vector_shape': list(vector.shape)}, indent=2))
