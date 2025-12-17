"""Debug the time question failure."""

from app.services.rag_service import _compute_grounding_gate

selected_chunks = [
    (
        "Onboarding starts at 9:00 AM on your first day. "
        "Plan to arrive 15 minutes early for check-in.",
        {"chunk": 0},
        0.3
    )
]
chunk_ids = ["chunk_0"]

should_proceed, reason, lines, score = _compute_grounding_gate(
    "when does onboarding start?", selected_chunks, chunk_ids
)

print(f"should_proceed: {should_proceed}")
print(f"reason: {reason}")
print(f"lines: {lines}")
print(f"score: {score}")
print(f"\nChunk text: '{selected_chunks[0][0]}'")
