from typing import List
import os
import logging

from pypdf import PdfReader
import docx  # python-docx

from ..config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def save_upload(file_bytes: bytes, filename: str) -> str:
    """
    Save raw file bytes to disk and return the saved path.
    """
    _, ext = os.path.splitext(filename)
    safe_name = filename.replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    base, ext = os.path.splitext(save_path)
    counter = 1
    # Avoid overwriting existing files
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return save_path


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text


def read_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_file_to_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in [".docx", ".doc"]:
        return read_docx(path)
    elif ext in [".txt", ".md"]:
        return read_txt(path)
    else:
        # Fallback: try as text
        return read_txt(path)


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Simple sliding-window character-based chunking.
    Uses configurable defaults from environment variables.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP
        
    chunks: List[str] = []
    start = 0
    length = len(text)
    # Safety cap to avoid runaway memory usage: if too many chunks are
    # being generated, stop early and return what we have. This prevents
    # infinite-loop or runaway-memory situations on malformed inputs.
    max_chunks = 10000
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            try:
                chunks.append(chunk)
            except MemoryError:
                logger.exception("MemoryError while appending chunk; returning partial chunks")
                return chunks
        # Safety: stop if we've built too many chunks
        if len(chunks) >= max_chunks:
            logger.warning("Reached max_chunks=%d; stopping chunking early", max_chunks)
            break
        # If we've reached the end of the text, stop looping to avoid
        # resetting start back to 0 (which causes an infinite loop for
        # short texts when overlap >= length).
        if end >= length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
