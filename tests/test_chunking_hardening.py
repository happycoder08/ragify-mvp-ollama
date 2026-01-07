import pytest
from app.services.ingestion import chunk_text, _is_strict_header, _is_merge_candidate, _harden_chunks

def test_is_strict_header():
    # Strict headers (structure only, length ignored)
    assert _is_strict_header("INTRODUCTION") is True
    assert _is_strict_header("1. Overview") is True
    assert _is_strict_header("Section A: Details") is True
    assert _is_strict_header("----------------") is True
    assert _is_strict_header("THIS IS A HEADER\nAND IT IS UPPERCASE") is True
    assert _is_strict_header("This Is A Title\nWith Multiple Lines") is True
    
    # Not strict headers
    assert _is_strict_header("This is a normal sentence.") is False
    assert _is_strict_header("HEADER\nThis is content.") is False
    assert _is_strict_header("1. Header\n- Bullet point") is False

def test_is_merge_candidate():
    # Short text -> True
    assert _is_merge_candidate("Short sentence.") is True
    
    # Long strict header -> True
    long_header = "HEADER\n" * 20 # > 120 chars
    assert _is_merge_candidate(long_header) is True
    
    # Long content -> False
    long_content = "This is a normal sentence. " * 10 # > 120 chars
    assert _is_merge_candidate(long_content) is False

def test_harden_chunks_merge():
    chunks = [
        "HEADER ONE",
        "This is the content for header one."
    ]
    # Both are merge candidates (short).
    # They should be merged into one buffer.
    # Since buffer is not strict header at end, it should be kept.
    hardened = _harden_chunks(chunks)
    assert len(hardened) == 1
    assert hardened[0] == "HEADER ONE\nThis is the content for header one."

def test_harden_chunks_multiple_headers():
    chunks = [
        "HEADER ONE",
        "SUBHEADER A",
        "Content starts here."
    ]
    # All short -> merged.
    hardened = _harden_chunks(chunks)
    assert len(hardened) == 1
    assert hardened[0] == "HEADER ONE\nSUBHEADER A\nContent starts here."

def test_harden_chunks_drop_trailing_header():
    chunks = [
        "Content chunk is long enough to be kept on its own. " * 5, # > 120 chars
        "TRAILING HEADER"
    ]
    # Chunk 1: Not merge candidate. Added to merged_chunks.
    # Chunk 2: Merge candidate (short). Added to buffer.
    # End: Buffer is "TRAILING HEADER". Strict header? Yes. Dropped.
    
    hardened = _harden_chunks(chunks)
    assert len(hardened) == 1
    assert hardened[0] == "Content chunk is long enough to be kept on its own. " * 5

def test_harden_chunks_keep_trailing_content():
    chunks = [
        "Content chunk is long enough to be kept on its own. " * 5,
        "Short trailing content."
    ]
    # Chunk 1: Kept.
    # Chunk 2: Buffered.
    # End: Buffer is "Short trailing content.". Strict header? False. Kept.
    
    hardened = _harden_chunks(chunks)
    assert len(hardened) == 2
    assert hardened[1] == "Short trailing content."

def test_harden_chunks_mixed():
    chunks = [
        "HEADER 1",
        "Content 1 is long enough to be kept on its own. " * 5,
        "HEADER 2",
        "Content 2 is long enough to be kept on its own. " * 5,
        "TRAILING HEADER"
    ]
    # HEADER 1 (short) -> Buffer
    # Content 1 (long) -> Prepend Buffer -> "HEADER 1\nContent 1..." -> Merged
    # HEADER 2 (short) -> Buffer
    # Content 2 (long) -> Prepend Buffer -> "HEADER 2\nContent 2..." -> Merged
    # TRAILING HEADER (short) -> Buffer
    # End -> Buffer is strict header -> Dropped
    
    hardened = _harden_chunks(chunks)
    assert len(hardened) == 2
    assert hardened[0].startswith("HEADER 1\nContent 1")
    assert hardened[1].startswith("HEADER 2\nContent 2")

def test_chunk_text_integration():
    text = "This is some content. " * 10 + "\n\nTRAILING HEADER"
    # Ensure chunk size is small enough to split
    chunks = chunk_text(text, chunk_size=50, overlap=0)
    
    # The last chunk should NOT be "TRAILING HEADER"
    assert not any("TRAILING HEADER" == c for c in chunks)
    
    # Test merging
    text2 = "HEADER\n" + "Content " * 20
    chunks2 = chunk_text(text2, chunk_size=50, overlap=0)
    
    # "HEADER" should be merged
    assert not any(c == "HEADER" for c in chunks2)
    assert any(c.startswith("HEADER\n") for c in chunks2)
