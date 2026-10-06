"""Page-aware PDF extraction and overlapping tokenizer-sized chunks."""
import hashlib
import io
import re
from pypdf import PdfReader


def chunk_pdf(data, document, tokenizer, chunk_size=400, overlap=60):
    if not 32 <= chunk_size <= 500 or not 0 <= overlap < chunk_size:
        raise ValueError('Use 32–500 tokens and 0 <= overlap < chunk_size.')
    if not data.startswith(b'%PDF-'):
        raise ValueError('Not a valid PDF file.')
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted or len(reader.pages) > 300:
            raise ValueError('Use an unencrypted PDF with at most 300 pages.')
        pages = [re.sub(r'\s+', ' ', p.extract_text() or '').strip() for p in reader.pages]
    except Exception as exc:
        raise ValueError('Cannot read this PDF. Use an unencrypted, text-based PDF (max 300 pages).') from exc
    if sum(map(len, pages)) > 2_000_000:
        raise ValueError('PDF contains too much text (maximum 2 million characters).')
    prefix = hashlib.sha256(document.encode() + data).hexdigest()[:16]
    chunks = []
    for page, text in enumerate(pages, 1):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets, words = encoded['offset_mapping'], encoded.word_ids()
        start = 0
        while start < len(offsets):
            end = min(start + chunk_size, len(offsets))
            # Avoid cutting a WordPiece word: re-tokenizing fragments can increase token count.
            while end < len(offsets) and end > start and words[end] == words[end - 1]:
                end -= 1
            if end == start:
                raise ValueError('A word exceeds the chunk size; choose a larger chunk size.')
            chunks.append(dict(chunk_id=f'{prefix}:{page}:{start}', document=document, page=page,
                               text=text[offsets[start][0]:offsets[end - 1][1]]))
            if end == len(offsets):
                break
            next_start = max(start + 1, end - overlap)
            while next_start > start + 1 and words[next_start] == words[next_start - 1]:
                next_start -= 1
            while next_start < end and words[next_start] == words[start]:
                next_start += 1
            start = next_start
    if not chunks:
        raise ValueError('No extractable text. Scanned PDFs need OCR, which is not included.')
    return chunks
