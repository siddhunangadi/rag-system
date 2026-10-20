"""Page-aware PDF extraction and recursive character splitting."""
import hashlib
import io
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf(data, document, chunk_size=500, overlap=100):
    if not 32 <= chunk_size <= 500 or not 0 <= overlap < chunk_size:
        raise ValueError('Use 32–500 characters and 0 <= overlap < chunk_size.')
    if not data.startswith(b'%PDF-'):
        raise ValueError('Not a valid PDF file.')
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted or len(reader.pages) > 300:
            raise ValueError('Use an unencrypted PDF with at most 300 pages.')
        pages = [re.sub(r'[^\S\n]+', ' ', (p.extract_text() or '').replace('\r\n', '\n').replace('\r', '\n')).strip()
                 for p in reader.pages]
    except Exception as exc:
        raise ValueError('Cannot read this PDF. Use an unencrypted, text-based PDF (max 300 pages).') from exc
    if sum(map(len, pages)) > 2_000_000:
        raise ValueError('PDF contains too much text (maximum 2 million characters).')
    prefix = hashlib.sha256(document.encode() + data).hexdigest()[:16]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
    chunks = []
    for page, text in enumerate(pages, 1):
        for position, passage in enumerate(splitter.split_text(text)):
            chunks.append(dict(chunk_id=f'{prefix}:{page}:{position}', document=document, page=page, text=passage))
    if not chunks:
        raise ValueError('No extractable text. Scanned PDFs need OCR, which is not included.')
    return chunks
