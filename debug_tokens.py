"""Debug token overlap."""

from app.services.rag_service import _tokenize_and_filter

question = "what time is onboarding scheduled?"
chunk = "Onboarding starts at 9:00 AM on your first day. The onboarding schedule includes orientation sessions."

q_tokens = set(_tokenize_and_filter(question))
chunk_tokens = set(_tokenize_and_filter(chunk))

print(f"Question tokens: {sorted(q_tokens)}")
print(f"Chunk tokens: {sorted(chunk_tokens)}")
print(f"Overlap: {sorted(q_tokens & chunk_tokens)}")
print(f"Overlap count: {len(q_tokens & chunk_tokens)}")
