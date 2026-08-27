import re
import string
from typing import Set, Tuple

def answer_supported_by_evidence(answer: str, evidence_text: str) -> bool:
    """
    Validate that an answer is grounded in the provided evidence.
    Stronger numeric/time validation to prevent 'any digit overlap' false positives.
    """

    # Rule 1: Check for exact refusal phrase
    if "The document does not specify this." in answer:
        return True

    def normalize(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    # --- New: time extraction + normalization ---
    def extract_times(text: str) -> Set[str]:
        """
        Extract times and normalize into canonical forms like:
        - "8am" -> "08:00am"
        - "8:00 AM" -> "08:00am"
        - "12 pm" -> "12:00pm"
        Also extracts 24h times like "19:00" as "19:00" (optional handling).
        """
        t = text.lower()

        results: Set[str] = set()

        # 1) 12-hour formats: "8am", "8 am", "8:00am", "8:00 am"
        # capture hour, optional minute, am/pm
        for m in re.finditer(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', t):
            hh = int(m.group(1))
            mm = int(m.group(2) or "00")
            mer = m.group(3)
            # clamp sanity
            if 1 <= hh <= 12 and 0 <= mm <= 59:
                results.add(f"{hh:02d}:{mm:02d}{mer}")

        # 2) 24-hour format: "19:00"
        for m in re.finditer(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', t):
            hh = int(m.group(1))
            mm = int(m.group(2))
            results.add(f"{hh:02d}:{mm:02d}")  # keep as 24h

        return results

    answer_times = extract_times(answer)
    evidence_times = extract_times(evidence_text)

    # If the answer contains any time, require at least one exact time match.
    # (You can tighten further: require ALL answer times exist in evidence.)
    if answer_times:
        # Exact match on canonical time forms
        if answer_times & evidence_times:
            pass
        else:
            return False

    # --- New: numeric validation that's not "any digit anywhere" ---
    # If answer contains numbers but not times, require a meaningful number match.
    # Exclude common trivial numbers that create tons of false positives.
    TRIVIAL_NUMS = {"0", "1", "2", "3", "4", "5", "10"}  # tune as needed
    answer_numbers = set(re.findall(r'\b\d+\b', answer))
    evidence_numbers = set(re.findall(r'\b\d+\b', evidence_text))

    if answer_numbers and not answer_times:
        meaningful = {n for n in answer_numbers if n not in TRIVIAL_NUMS}
        # If answer only has trivial numbers, fall back to lexical overlap.
        if meaningful:
            if not (meaningful & evidence_numbers):
                return False

    # Rule 3: Tokenize and remove stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }

    answer_norm = normalize(answer)
    evidence_norm = normalize(evidence_text)

    answer_tokens = set(t for t in answer_norm.split() if t and t not in STOPWORDS)
    evidence_tokens = set(t for t in evidence_norm.split() if t and t not in STOPWORDS)

    # Rule 4b: For non-numeric answers, require at least K=2 content tokens overlap
    K = 2
    overlap_count = len(answer_tokens & evidence_tokens)

    return overlap_count >= K
