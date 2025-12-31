from app.services.rag_service import ChunkHit, _apply_header_reranking, _dedupe_results, _dedupe_by_header

def test_chunkhit_atomic_alignment():
    # Setup: 3 distinct chunks
    hits = [
        ChunkHit(chunk_id="A", doc="TextA", meta={"source_file":"fileA", "header":"H1"}, dist=0.3),
        ChunkHit(chunk_id="B", doc="TextB", meta={"source_file":"fileB", "header":"H2"}, dist=0.1),
        ChunkHit(chunk_id="C", doc="TextC", meta={"source_file":"fileC", "header":"H1"}, dist=0.2),
    ]
    # Shuffle order by reranking (header match on 'H1')
    reranked = _apply_header_reranking(hits, "H1?")
    # Dedupe by header (max 1 per header)
    deduped = _dedupe_by_header(reranked, max_per_header=1)
    # Dedupe by chunk_id (should not drop any, all unique)
    final = _dedupe_results(deduped)
    # Build a lookup for original doc/chunk_id/source_file
    original = {(h.chunk_id, h.meta["source_file"]): h.doc for h in hits}
    # Assert atomic alignment: doc text must match chunk_id and source_file
    for h in final:
        key = (h.chunk_id, h.meta["source_file"])
        assert h.doc == original[key], f"Alignment broken for {key}: got {h.doc}, expected {original[key]}"

def test_chunkhit_alignment_parallel_array_break():
    # This would break if helpers used parallel arrays or index-based logic
    hits = [
        ChunkHit(chunk_id="X", doc="TextX", meta={"source_file":"fileX", "header":"H1"}, dist=0.5),
        ChunkHit(chunk_id="Y", doc="TextY", meta={"source_file":"fileY", "header":"H2"}, dist=0.4),
        ChunkHit(chunk_id="Z", doc="TextZ", meta={"source_file":"fileZ", "header":"H3"}, dist=0.3),
    ]
    # Intentionally reverse the list to simulate reordering
    reversed_hits = list(reversed(hits))
    # Apply helpers in sequence
    out = _dedupe_results(_dedupe_by_header(_apply_header_reranking(reversed_hits, "H2")))
    # Check atomic alignment
    for h in out:
        # The doc must always match the chunk_id and source_file from the original
        found = False
        for orig in hits:
            if h.chunk_id == orig.chunk_id and h.meta["source_file"] == orig.meta["source_file"]:
                assert h.doc == orig.doc, f"Mismatch: {h.chunk_id} {h.meta['source_file']} {h.doc} != {orig.doc}"
                found = True

        assert found, f"Returned chunk {h.chunk_id} {h.meta['source_file']} not in original set"
