import pytest

from app.services import rag_service


@pytest.mark.asyncio
async def test_rerank_dedup_selects_unique_chunks():
    """Near-identical chunks should collapse to a single selection, letting the next distinct chunk through."""

    # Five near-duplicates plus one distinct chunk
    base = "Arrive at 8:00 AM at the main reception on the 3rd floor."
    variants = [
        base,
        base.replace("reception", "Reception"),
        base.replace("8:00 AM", "8 am"),
        base + "  ",
        "   " + base,
    ]
    distinct = "Orientation starts at 9:00 AM in the auditorium."

    docs = variants + [distinct]
    metas = [{"source_file": f"doc{i}.txt", "chunk": i} for i in range(len(docs))]
    dists = [200.0, 201.0, 202.0, 203.0, 204.0, 250.0]
    ids = [f"id_{i}" for i in range(len(docs))]

    # Pretend retrieval already happened; run dedupe helper directly
    results = list(zip(docs, metas, dists))
    deduped, deduped_ids = rag_service._dedupe_results(results, ids)

    # Expect first variant retained, others dropped, distinct kept
    assert len(deduped) == 2
    assert deduped_ids[0] == "id_0"
    assert deduped_ids[1] == "id_5"
    assert deduped[0][0].startswith("Arrive at 8:00 AM")
    assert "Orientation starts" in deduped[1][0]

    # Ensure ordering is deterministic (original order preserved)
    assert deduped_ids == ["id_0", "id_5"]
