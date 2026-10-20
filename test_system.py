"""Small real-model regression suite; run with python -m unittest -v."""
import unittest
import io
import math
from unittest.mock import patch
import numpy as np
from pathlib import Path


class IngestionTests(unittest.TestCase):
    def test_pdf_and_overlap(self):
        from ingest import chunk_pdf
        chunks = chunk_pdf(Path('data/hr_policy.pdf').read_bytes(), 'hr_policy.pdf', 500, 100)
        self.assertGreater(len(chunks), 24)
        self.assertEqual(chunks[0]['page'], 1)
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))
        self.assertTrue(any(chunks[0]['text'][-n:] == chunks[1]['text'][:n] for n in range(20, 101)))
        self.assertTrue(all(len(c['text']) <= 500 for c in chunks))
        for size, overlap in [(0, 0), (64, 64), (64, -1)]:
            with self.assertRaises(ValueError):
                chunk_pdf(b'', 'bad.pdf', size, overlap)
        with self.assertRaises(ValueError):
            chunk_pdf(b'not a pdf', 'bad.pdf')
        from pypdf import PdfWriter
        writer, buffer = PdfWriter(), io.BytesIO()
        writer.add_blank_page(width=612, height=792)
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, 'No extractable text'):
            chunk_pdf(buffer.getvalue(), 'scan.pdf')


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ingest import chunk_pdf
        from retrieval import embedding_model, build_index
        chunks = chunk_pdf(Path('data/hr_policy.pdf').read_bytes(), 'hr_policy.pdf')
        cls.index = build_index(chunks)
        cls.vectors = embedding_model().encode([c['text'] for c in chunks], normalize_embeddings=True)

    def setUp(self):
        # Replace only the external vector service; embedding, BM25 and reranking stay real.
        remote = patch('retrieval.dense_search', side_effect=self.remote_search)
        remote.start()
        self.addCleanup(remote.stop)

    def remote_search(self, vector, limit):
        scores = self.vectors @ np.asarray(vector)
        return [{'chunk_id': self.index[0][i]['chunk_id'], 'similarity': float(scores[i])}
                for i in np.argsort(-scores)[:limit]]

    def test_dense(self):
        from retrieval import search
        result = search('Who handles workplace sexual harassment?', self.index, mode='dense')
        self.assertTrue(any(item['page'] == 11 for item in result['results']))
        self.assertEqual(len(result['results']), 5)
        self.assertGreater(result['latency']['query_embedding_ms'], 0)
        self.assertEqual(result['latency']['rerank_ms'], 0)

    def test_bm25(self):
        from retrieval import search
        result = search('POSH Internal Committee', self.index, mode='bm25')
        self.assertEqual(result['results'][0]['page'], 11)
        self.assertEqual(result['latency']['query_embedding_ms'], 0)
        self.assertEqual(search('zzzzunknownword', self.index, mode='bm25')['results'], [])

    def test_exact_policy_numbers(self):
        from retrieval import search
        cases = [
            ('How many annual leave days do employees get per year?', 1, '20 working days'),
            ('How many casual leave days do I get each year?', 2, 'six days'),
            ('How many consecutive casual leave days can I take?', 2, 'two consecutive'),
            ('How many paid sick leave days are available each year?', 3, 'ten paid sick leave days'),
            ('How many weeks of paid maternity leave are available?', 4, '26 weeks'),
            ('How many days notice must I give during probation?', 7, 'fifteen calendar days'),
            ('How many days notice must a confirmed employee give?', 8, 'sixty calendar days'),
            ('How many days per week can I work from home?', 9, 'two days per week'),
            ('Within how many days must I submit business travel expenses?', 15, 'thirty calendar days'),
            ('What is the monthly home internet reimbursement limit?', 16, '1500 currency units per month'),
        ]
        for question, page, expected in cases:
            with self.subTest(question=question):
                result = search(question, self.index, candidate_count=5)
                self.assertTrue(any(item['page'] == page and expected in ' '.join(item['text'].split()) for item in result['results']))

    def test_policy_conditions_and_exceptions_remain_retrievable(self):
        from retrieval import search
        cases = [
            ('Can I carry forward annual leave and when must I use it?', 1, ['five', 'March 31', 'expire']),
            ('Can casual leave be carried forward or paid out?', 2, ['cannot be carried forward or paid out']),
            ('Does probation finish automatically after six months?', 7, ['does not replace written confirmation']),
            ('Can employees work from another country?', 9, ['separate written approval', 'information security']),
            ('Do equipment-based roles qualify for work from home?', 9, ['physical equipment', 'may be excluded']),
        ]
        for question, page, phrases in cases:
            with self.subTest(question=question):
                result = search(question, self.index, candidate_count=5)
                text = ' '.join(' '.join(item['text'].split()) for item in result['results'] if item['page'] == page)
                for phrase in phrases:
                    self.assertIn(phrase, text)

    def test_empty_candidates_return_no_results(self):
        from retrieval import search
        with patch('retrieval.dense_search', return_value=[]):
            result = search('zzzzunknownword', self.index)
        self.assertEqual(result['results'], [])
        self.assertEqual(result['candidate_count'], 0)

    def test_bm25_single_chunk(self):
        from retrieval import build_index, search
        tiny = build_index([next(c for c in self.index[0] if c['page'] == 11)])
        result = search('POSH', tiny, mode='bm25')
        self.assertEqual(len(result['results']), 1)
        self.assertEqual(result['results'][0]['page'], 11)

    def test_rrf(self):
        from retrieval import rrf, search
        fused = rrf([(0, .9), (1, .8)], [(1, 20), (2, 10)], 60)
        self.assertEqual([i for i, _ in fused], [1, 0, 2])
        self.assertAlmostEqual(fused[0][1], 1/62 + 1/61)
        result = search('POSH complaint', self.index, mode='rrf')
        self.assertEqual(result['results'][0]['page'], 11)
        self.assertGreater(result['latency']['rrf_ms'], 0)

    def test_rerank(self):
        from retrieval import search
        query = 'What must I do to resign and complete my exit?'
        fused = search(query, self.index, mode='rrf', final_k=10)
        result = search(query, self.index, mode='rerank', candidate_count=10)
        self.assertGreater(result['latency']['rerank_ms'], 0)
        self.assertEqual(result['candidate_count'], 10)
        self.assertTrue({r['chunk_id'] for r in result['results']} <= {r['chunk_id'] for r in fused['results']})
        self.assertNotEqual([r['page'] for r in fused['results'][:5]], [r['page'] for r in result['results']])
        scores = [r['score'] for r in result['results']]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for options in [dict(candidate_count=2), dict(dense_k=0), dict(mode='invalid')]:
            with self.assertRaises(ValueError):
                search(query, self.index, **options)


class ApiTests(unittest.TestCase):
    def test_upload_search_metrics(self):
        from fastapi.testclient import TestClient
        from app import app
        with patch('app.config'), patch('app.load_chunks', return_value=[]), patch('app.save_document'), \
                patch('retrieval.dense_search', return_value=[]), \
                patch('app.answer_from_llm', return_value={'text': 'Contact the Internal Committee.',
                      'sources': [{'source_id': 'S1', 'chunk_id': 'fixture', 'document': 'hr_policy.pdf', 'page': 11}]}), \
                TestClient(app) as client:
            self.assertEqual(client.post('/search', json={'query': 'leave'}).status_code, 409)
            self.assertEqual(client.post('/upload', files={'file': ('bad.pdf', b'broken')}).status_code, 400)
            pdf = Path('data/hr_policy.pdf').read_bytes()
            response = client.post('/upload', files={'file': ('hr_policy.pdf', pdf, 'application/pdf')})
            self.assertEqual(response.status_code, 200, response.text)
            count = response.json()['chunks']
            self.assertGreater(count, 24)
            self.assertEqual(response.json()['chunk_size'], 500)
            self.assertEqual(response.json()['overlap'], 100)
            client.post('/upload', files={'file': ('hr_policy.pdf', pdf)})
            self.assertEqual(client.post('/upload', files={'file': ('hr_policy.pdf', b'broken')}).status_code, 400)
            self.assertEqual(client.get('/metrics').json()['chunks'], count)
            self.assertIn('HR Policy Finder', client.get('/').text)
            self.assertEqual(client.get('/static/style.css').status_code, 200)
            self.assertEqual(client.post('/search', json={'query': '   '}).status_code, 422)
            self.assertEqual(client.post('/search', json={'query': 'leave', 'dense_k': 0}).status_code, 422)
            response = client.post('/search', json={'query': 'Who handles POSH complaints?'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['results'][0]['page'], 11)
            self.assertEqual(response.json()['answer']['sources'][0]['page'], 11)
            self.assertTrue(response.json()['answer']['text'])
            self.assertGreater(response.json()['latency']['answer_generation_ms'], 0)
            self.assertEqual(client.get('/metrics').json()['recent_searches'], 1)
            self.assertEqual(client.post('/upload', files={'file': ('second.pdf', pdf)}).status_code, 200)
            self.assertEqual(client.get('/metrics').json()['documents'], 2)
            self.assertEqual(client.get('/metrics').json()['chunks'], 2 * count)
            with patch('app.save_document', side_effect=RuntimeError('Supabase request failed (503).')):
                self.assertEqual(client.post('/upload', files={'file': ('third.pdf', pdf)}).status_code, 503)
            self.assertEqual(client.get('/metrics').json()['documents'], 2)
            with patch('retrieval.dense_search', side_effect=RuntimeError('Cannot reach Supabase.')):
                self.assertEqual(client.post('/search', json={'query': 'leave'}).status_code, 503)

    def test_startup_requires_supabase(self):
        from fastapi.testclient import TestClient
        from app import app
        with patch('storage.setting', return_value=''), self.assertRaisesRegex(RuntimeError, 'SUPABASE_URL'):
            with TestClient(app):
                pass


class EvaluationTests(unittest.TestCase):
    def test_graded_metrics_and_duplicate_pages(self):
        from evaluate import quality
        labels = [{'document':'a.pdf', 'page':1, 'grade':2}, {'document':'a.pdf', 'page':2, 'grade':1}]
        one = {'document':'a.pdf', 'page':1}
        two = {'document':'a.pdf', 'page':2}
        self.assertEqual(quality([one, two], labels), (1.0, 1.0, 1.0))
        recall, mrr, ndcg = quality([two, two, one], labels)
        self.assertEqual((recall, mrr), (1.0, 1.0))
        self.assertAlmostEqual(ndcg, 2.5 / (3 + 1 / math.log2(3)))
        self.assertEqual(quality([], labels), (0.0, 0.0, 0.0))


class AnswerSourceTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {'chunk_id': 'a:1', 'document': 'leave.pdf', 'page': 1, 'text': 'Leave policy.'},
            {'chunk_id': 'b:7', 'document': 'benefits.pdf', 'page': 7, 'text': 'Internet allowance is 1500 per month.'},
        ]
        settings = patch('retrieval.setting', return_value='test-key')
        post = patch('retrieval.httpx.post')
        settings.start()
        self.post = post.start()
        self.addCleanup(settings.stop)
        self.addCleanup(post.stop)
        self.post.return_value.is_error = False

    def answer(self, payload):
        import json
        from retrieval import answer_from_llm
        self.post.return_value.json.return_value = {'candidates': [{'content': {'parts': [{'text': json.dumps(payload)}]}}]}
        return answer_from_llm('What is the internet allowance?', self.results)

    def test_answer_cites_non_first_passage(self):
        answer = self.answer({'text': '1500 per month.', 'source_ids': ['S2']})
        self.assertEqual(answer.get('sources'), [
            {'source_id': 'S2', 'chunk_id': 'b:7', 'document': 'benefits.pdf', 'page': 7}])
        self.assertEqual(answer['text'], '1500 per month.')
        self.assertNotIn('document', answer)

    def test_answer_cites_multiple_passages(self):
        answer = self.answer({'text': 'See both policies.', 'source_ids': ['S2', 'S1', 'S2']})
        self.assertEqual([s['page'] for s in answer.get('sources', [])], [7, 1])

    def test_invalid_or_missing_citations_are_rejected(self):
        payloads = [
            {'text': '1500 per month.', 'source_ids': ['S9']},
            {'text': '1500 per month.', 'source_ids': ['S2', 'S9']},
            {'text': '1500 per month.'},
            {'text': '1500 per month.', 'source_ids': []},
            {'text': '1500 per month.', 'source_ids': 'S2'},
            {'text': '1500 per month.', 'source_ids': [2]},
            {'text': '', 'source_ids': ['S2']},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                self.answer(payload)

    def test_unanswerable_question_has_no_sources(self):
        for source_ids in ([], ['S1']):
            with self.subTest(source_ids=source_ids):
                answer = self.answer({'text': "I couldn't find this in the policy.", 'source_ids': source_ids})
                self.assertEqual(answer, {'text': "I couldn't find this in the policy.", 'sources': []})

    def test_empty_context_does_not_generate_answer(self):
        from retrieval import answer_from_llm
        self.assertIsNone(answer_from_llm('Anything?', []))

    def test_malformed_model_response_is_rejected(self):
        from retrieval import answer_from_llm
        for response in ({}, {'candidates': [{'content': {'parts': [{'text': 'not json'}]}}]}):
            self.post.return_value.json.return_value = response
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                answer_from_llm('Anything?', self.results)


class StorageConfigurationTests(unittest.TestCase):
    def test_missing_supabase_configuration_is_an_error(self):
        from unittest.mock import patch
        from storage import config
        with patch('storage.setting', return_value=''):
            with self.assertRaisesRegex(RuntimeError, 'SUPABASE_URL'):
                config()

    def test_unavailable_supabase_has_clear_error(self):
        import httpx
        from storage import load_chunks
        with patch('storage.config', return_value=('https://example.supabase.co', 'test-key')), \
                patch('storage.httpx.request', side_effect=httpx.ConnectError('unreachable')):
            with self.assertRaisesRegex(RuntimeError, 'Cannot reach Supabase'):
                load_chunks()

    def test_dense_search_sends_vector_and_limit_to_supabase(self):
        import httpx
        from storage import dense_search
        def respond(method, url, **kwargs):
            self.assertEqual(method, 'POST')
            self.assertEqual(url, 'https://example.supabase.co/rest/v1/rpc/match_policy_chunks')
            self.assertEqual(kwargs['json'], {'query_embedding': '[0.1,0.2]', 'match_count': 20})
            return httpx.Response(200, json=[{'chunk_id': 'a:1', 'similarity': 0.9}])
        with patch('storage.config', return_value=('https://example.supabase.co', 'test-key')), \
                patch('storage.httpx.request', side_effect=respond):
            self.assertEqual(dense_search([0.1, 0.2], 20), [{'chunk_id': 'a:1', 'similarity': 0.9}])


class CharacterChunkTests(unittest.TestCase):
    def test_character_chunks_preserve_paragraphs_and_pages(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from ingest import chunk_pdf
        text = 'Annual leave is 20 days.\n\nUnused leave requires approval.\nExceptions apply during probation.'
        pages = [Mock(), Mock()]
        pages[0].extract_text.return_value = text
        pages[1].extract_text.return_value = 'A separate page.'
        with patch('ingest.PdfReader', return_value=SimpleNamespace(is_encrypted=False, pages=pages)):
            chunks = chunk_pdf(b'%PDF-', 'policy.pdf')
        self.assertEqual([c['text'] for c in chunks], [text, 'A separate page.'])
        self.assertEqual([c['page'] for c in chunks], [1, 2])
        self.assertTrue(all(c['document'] == 'policy.pdf' for c in chunks))
        self.assertEqual(len({c['chunk_id'] for c in chunks}), 2)

    def test_default_chunks_use_character_limit(self):
        from ingest import chunk_pdf
        chunks = chunk_pdf(Path('data/hr_policy.pdf').read_bytes(), 'hr_policy.pdf')
        self.assertGreater(len(chunks), 24)
        self.assertTrue(all(len(c['text']) <= 500 for c in chunks))


if __name__ == '__main__':
    unittest.main()
