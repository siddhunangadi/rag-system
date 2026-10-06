"""Small real-model regression suite; run with python -m unittest -v."""
import unittest
import io
import math
from pathlib import Path


class IngestionTests(unittest.TestCase):
    def test_pdf_and_overlap(self):
        from ingest import chunk_pdf
        from retrieval import embedding_model
        tokenizer = embedding_model().tokenizer
        chunks = chunk_pdf(Path('data/hr_policy.pdf').read_bytes(), 'hr_policy.pdf', tokenizer, 64, 12)
        self.assertGreater(len(chunks), 24)
        self.assertEqual(chunks[0]['page'], 1)
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))
        first = tokenizer.encode(chunks[0]['text'], add_special_tokens=False)
        second = tokenizer.encode(chunks[1]['text'], add_special_tokens=False)
        self.assertTrue(any(first[-n:] == second[:n] for n in range(12, 20)))
        self.assertTrue(all(len(tokenizer.encode(c['text'], add_special_tokens=False)) <= 64 for c in chunks))
        for size, overlap in [(0, 0), (64, 64), (64, -1)]:
            with self.assertRaises(ValueError):
                chunk_pdf(b'', 'bad.pdf', tokenizer, size, overlap)
        with self.assertRaises(ValueError):
            chunk_pdf(b'not a pdf', 'bad.pdf', tokenizer)
        from pypdf import PdfWriter, PdfReader
        writer, buffer = PdfWriter(), io.BytesIO()
        writer.add_blank_page(width=612, height=792)
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, 'No extractable text'):
            chunk_pdf(buffer.getvalue(), 'scan.pdf', tokenizer)
        writer, buffer = PdfWriter(), io.BytesIO()
        writer.add_page(PdfReader('data/hr_policy.pdf').pages[0])
        writer.write(buffer)
        chunks = chunk_pdf(buffer.getvalue(), 'one.pdf', tokenizer, 32, 31)
        self.assertTrue(all(len(tokenizer.encode(c['text'], add_special_tokens=False)) <= 32 for c in chunks))


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ingest import chunk_pdf
        from retrieval import embedding_model, build_index
        chunks = chunk_pdf(Path('data/hr_policy.pdf').read_bytes(), 'hr_policy.pdf', embedding_model().tokenizer)
        cls.index = build_index(chunks)

    def test_dense(self):
        from retrieval import search
        result = search('Who handles workplace sexual harassment?', self.index, mode='dense')
        self.assertEqual(result['results'][0]['page'], 11)
        self.assertEqual(len(result['results']), 5)
        self.assertGreater(result['latency']['query_embedding_ms'], 0)
        self.assertEqual(result['latency']['rerank_ms'], 0)

    def test_bm25(self):
        from retrieval import search
        result = search('POSH Internal Committee', self.index, mode='bm25')
        self.assertEqual(result['results'][0]['page'], 11)
        self.assertEqual(result['latency']['query_embedding_ms'], 0)
        self.assertEqual(search('zzzzunknownword', self.index, mode='bm25')['results'], [])

    def test_llm_answer_skips_empty_context(self):
        from retrieval import answer_from_llm
        self.assertIsNone(answer_from_llm('How much leave do I get?', []))

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
                self.assertTrue(any(item['page'] == page and expected in item['text'] for item in result['results']))

    def test_bm25_single_chunk(self):
        from retrieval import build_index, search
        tiny = build_index([self.index[0][10]])
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
        with TestClient(app) as client:
            self.assertEqual(client.post('/search', json={'query':'leave'}).status_code, 409)
            self.assertEqual(client.post('/upload', files={'file':('bad.pdf', b'broken')}).status_code, 400)
            pdf = Path('data/hr_policy.pdf').read_bytes()
            response = client.post('/upload', files={'file':('hr_policy.pdf', pdf, 'application/pdf')})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['chunks'], 24)
            client.post('/upload', files={'file':('hr_policy.pdf', pdf)})
            self.assertEqual(client.post('/upload', files={'file':('hr_policy.pdf', b'broken')}).status_code, 400)
            self.assertEqual(client.get('/metrics').json()['chunks'], 24)
            self.assertIn('HR Policy Finder', client.get('/').text)
            self.assertEqual(client.get('/static/style.css').status_code, 200)
            self.assertEqual(client.post('/search', json={'query':'   '}).status_code, 422)
            self.assertEqual(client.post('/search', json={'query':'leave', 'dense_k':0}).status_code, 422)
            response = client.post('/search', json={'query':'Who handles POSH complaints?'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['results'][0]['page'], 11)
            self.assertEqual(response.json()['answer']['page'], 11)
            self.assertTrue(response.json()['answer']['text'])
            self.assertGreater(response.json()['latency']['answer_generation_ms'], 0)
            self.assertEqual(client.get('/metrics').json()['recent_searches'], 1)
            self.assertEqual(client.post('/upload', files={'file':('second.pdf', pdf)}).status_code, 200)
            self.assertEqual(client.get('/metrics').json()['documents'], 2)
            self.assertEqual(client.get('/metrics').json()['chunks'], 48)


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


if __name__ == '__main__':
    unittest.main()
