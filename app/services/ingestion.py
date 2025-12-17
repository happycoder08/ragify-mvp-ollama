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


def chunk_text_sections(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Section-aware chunking for plain text that preserves formatting and keeps
    numbered headings (e.g., "4. EMAIL SIGNATURE SETUP") together with their
    immediate bullet lines in the same chunk. Generalized; no question-specific logic.

    - Detect section starts via numbered headings, markdown headings, ALL-CAPS headings, or label lines.
    - Build sections by grouping content until the next heading.
    - For each section, keep the heading + contiguous bullet block in the first chunk.
    - Split remaining content with a sliding window while preserving exact formatting.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    import re
    # Keep line endings to preserve exact formatting
    lines = text.splitlines(keepends=True)

    def is_numbered_heading(line: str) -> bool:
        return re.match(r"^\s*\d+\.\s+\S", line) is not None

    def is_markdown_heading(line: str) -> bool:
        return re.match(r"^\s*#{1,6}\s+", line) is not None

    def is_label_heading(line: str) -> bool:
        l = line.strip()
        # Exclude bullets that end with colon (e.g., "- Required format:")
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", l):
            return False
        return len(l) <= 120 and l.endswith(":")

    def is_allcaps_heading(line: str) -> bool:
        l = line.strip()
        has_alpha = re.search(r"[A-Za-z]", l) is not None
        return has_alpha and l == l.upper() and 4 <= len(l) <= 120

    def is_heading(line: str) -> bool:
        return (
            is_numbered_heading(line)
            or is_markdown_heading(line)
            or is_label_heading(line)
            or is_allcaps_heading(line)
        )

    def is_bullet(line: str) -> bool:
        return re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line) is not None

    # Group into sections by heading detection
    sections: List[List[str]] = []
    buf: List[str] = []
    for line in lines:
        if is_heading(line) and buf:
            sections.append(buf)
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append(buf)

    # If no sections detected, fallback to simple sliding-window chunking
    if not sections or sum(len("".join(s)) for s in sections) < max(1, len(text) // 2):
        return chunk_text(text, chunk_size, overlap)

    chunks: List[str] = []
    for sec_lines in sections:
        sec_text = "".join(sec_lines)
        
        # If section fits in chunk_size, keep it as-is
        if len(sec_text) <= chunk_size:
            if sec_text.strip():
                chunks.append(sec_text)
            continue

        # Section exceeds chunk_size: split while preserving heading+bullets in first chunk
        # Find heading line (first line of section)
        heading_idx = 0
        # Find contiguous bullet block immediately following the heading
        # Include ALL lines that belong to bullets (including indented content)
        bullet_end_idx = heading_idx + 1
        while bullet_end_idx < len(sec_lines):
            current_line = sec_lines[bullet_end_idx]
            # Continue if line is a bullet OR indented content under a bullet
            if is_bullet(current_line):
                bullet_end_idx += 1
            elif current_line.strip() == "":
                # Empty line: check if next line is a bullet to keep section together
                if bullet_end_idx + 1 < len(sec_lines) and is_bullet(sec_lines[bullet_end_idx + 1]):
                    bullet_end_idx += 1
                else:
                    break
            elif current_line.startswith("  ") or current_line.startswith("\t"):
                # Indented content belongs to previous bullet
                bullet_end_idx += 1
            else:
                # Non-indented, non-bullet line: end of bullet block
                break
        
        # Collect heading + ALL contiguous bullets (with their indented content) into first chunk
        # This guarantees heading+bullet block stays together (even if > chunk_size)
        first_chunk_lines = sec_lines[:bullet_end_idx]
        first_chunk_text = "".join(first_chunk_lines)
        chunks.append(first_chunk_text)
        logger.debug(
            "Section heading+bullets chunk: %d lines, %d chars (heading + %d bullet lines)",
            len(first_chunk_lines),
            len(first_chunk_text),
            bullet_end_idx - heading_idx - 1
        )

        # Remaining content after bullets: apply sliding-window chunking
        remainder = "".join(sec_lines[bullet_end_idx:])
        if remainder.strip():
            start = 0
            L = len(remainder)
            step = max(1, chunk_size - overlap)
            while start < L:
                end = min(start + chunk_size, L)
                chunk = remainder[start:end].strip()
                if chunk:
                    chunks.append(chunk + "\n")
                if end >= L:
                    break
                start = max(0, end - overlap)

    return chunks
