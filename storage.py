"""Small Supabase REST client; credentials stay in the ignored .env file."""
import os
from pathlib import Path
from urllib.parse import quote
import httpx


def setting(name):
    values = {}
    path = Path(__file__).with_name('.env')
    if path.exists():
        values = dict(line.strip().split('=', 1) for line in path.read_text().splitlines()
                      if '=' in line and not line.lstrip().startswith('#'))
    return os.getenv(name, values.get(name, ''))


def config():
    url, key = setting('SUPABASE_URL').rstrip('/'), setting('SUPABASE_SERVICE_ROLE_KEY')
    if not url or not key:
        raise RuntimeError('Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before starting the app.')
    return url, key


def request(method, path, **kwargs):
    url, key = config()
    try:
        response = httpx.request(method, url + path, headers={'apikey': key, 'Authorization': f'Bearer {key}', **kwargs.pop('headers', {})}, timeout=30, **kwargs)
    except httpx.RequestError:
        raise RuntimeError('Cannot reach Supabase. Check the connection and SUPABASE_URL.') from None
    if response.is_error:
        raise RuntimeError(f'Supabase request failed ({response.status_code}).')
    return response


def load_chunks():
    chunks, start = [], 0
    while True:
        batch = request('GET', '/rest/v1/policy_chunks?select=chunk_id,document,page,text&order=document.asc,chunk_id.asc',
                        headers={'Range': f'{start}-{start + 999}'}).json()
        chunks.extend(batch)
        if len(batch) < 1000:
            return chunks
        start += len(batch)


def save_document(name, data, chunks, vectors):
    path = f"policy-pdfs/{quote(name, safe='')}"
    request('POST', f'/storage/v1/object/{path}', content=data,
            headers={'Content-Type': 'application/pdf', 'x-upsert': 'true'})
    rows = [dict(chunk, embedding='[' + ','.join(map(str, vector)) + ']') for chunk, vector in zip(chunks, vectors)]
    request('POST', '/rest/v1/rpc/replace_policy_document', json={'p_name': name, 'p_path': path, 'p_chunks': rows})


def dense_search(vector, limit):
    return request('POST', '/rest/v1/rpc/match_policy_chunks', json={
        'query_embedding': '[' + ','.join(map(str, vector)) + ']', 'match_count': limit}).json()
