"""
Test camera/video synonym support in eval.
"""
import sys
sys.path.insert(0, 'c:/Users/sergi/Documents/RAGify/ragify-mvp-ollama')

from eval.run_eval import normalize_synonyms

print("\n" + "="*70)
print("TEST: Camera/Video Synonym Normalization")
print("="*70)

# Test camera/video synonym support
test_cases = [
    ("Video on for team meetings", ["camera", "video"]),
    ("Camera required for standups", ["camera", "video"]),
    ("Meeting policy requires video", ["camera", "video"]),
    ("Turn on your camera for calls", ["camera", "video"]),
]

all_passed = True
for text, expected_terms in test_cases:
    normalized = normalize_synonyms(text)
    print(f"\nOriginal: {text}")
    print(f"Normalized: {normalized}")
    has_all = all(term in normalized for term in expected_terms)
    print(f"Contains all {expected_terms}: {has_all}")
    if not has_all:
        print(f"  FAIL: Missing terms:", [t for t in expected_terms if t not in normalized])
        all_passed = False
    else:
        print("  PASS")

print("\n" + "="*70)
if all_passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("="*70)

# Test the evidence snippet extraction logic inline
print("\n" + "="*70)
print("TEST: Evidence Snippet Extraction (inline logic)")
print("="*70)

import re

def extract_snippet(chunk_text: str, max_chars: int = 400) -> str:
    """Test implementation of evidence snippet extraction."""
    lines = chunk_text.split('\n')
    if not lines:
        return chunk_text[:max_chars]
    
    # Find first non-empty line
    first_line = ""
    for line in lines:
        if line.strip():
            first_line = line.strip()
            break
    
    is_header = (
        first_line.endswith(':') or
        first_line.endswith(')') or
        (len(first_line) > 3 and first_line == first_line.upper() and any(c.isalpha() for c in first_line))
    )
    
    if not is_header:
        return chunk_text[:max_chars]
    
    # Collect header + up to 3 bullets
    snippet_lines = []
    bullet_count = 0
    total_chars = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        is_bullet = re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', line) is not None
        
        if len(snippet_lines) == 0:  # Header
            snippet_lines.append(line_stripped)
            total_chars += len(line_stripped)
        elif is_bullet and bullet_count < 3:
            snippet_lines.append(line_stripped)
            total_chars += len(line_stripped)
            bullet_count += 1
            if total_chars >= max_chars:
                break
        elif bullet_count >= 3:
            break
    
    return '\n'.join(snippet_lines)

# Test header with bullets
chunk1 = """   WHAT TO ASK YOUR MANAGER:
   - "What does success look like in my first 30 days?"
   - "Who are the key people I should connect with?"
   - "What's the best way to reach you if I have questions?"
   - "Are there any team norms or unwritten rules I should know?"
"""

snippet1 = extract_snippet(chunk1)
print(f"\nTest 1: Header with bullets")
print(f"Input (first 150 chars): {chunk1[:150]}")
print(f"\nExtracted snippet:")
print(snippet1)
print(f"\nContains 'success': {'success' in snippet1.lower()}")
print(f"Contains '30 days': {'30 days' in snippet1.lower()}")
if 'success' in snippet1.lower() and '30 days' in snippet1.lower():
    print("PASS: Evidence includes manager question with 'success' and '30 days'")
else:
    print("FAIL: Missing required keywords")

print("\n" + "="*70)
