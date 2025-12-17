"""
Answer validation module.

Provides post-LLM validation to ensure generated answers are grounded in evidence.
"""

import re
import string


def answer_supported_by_evidence(answer: str, evidence_text: str) -> bool:
    """
    Validate that an answer is grounded in the provided evidence.
    Fast deterministic check using lexical overlap and pattern matching.
    
    Args:
        answer: The generated answer to validate
        evidence_text: The evidence text (context) used to generate the answer
    
    Returns:
        True if answer is supported by evidence, False otherwise
    
    Rules:
        1. Exact refusal phrase "The document does not specify this." → True
        2. Normalize both texts (lowercase, remove punctuation, collapse whitespace)
        3. Tokenize and remove stopwords
        4. Require K=2 content tokens from answer in evidence OR
           at least one numeric/time pattern match if answer contains digits/times
        5. Otherwise → False
    """
    # Rule 1: Check for exact refusal phrase
    if "The document does not specify this." in answer:
        return True
    
    # Rule 2: Normalize - lowercase, remove punctuation, collapse whitespace
    def normalize(text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Collapse whitespace
        text = ' '.join(text.split())
        return text
    
    answer_norm = normalize(answer)
    evidence_norm = normalize(evidence_text)
    
    # Rule 3: Tokenize and remove stopwords
    # Small built-in stopword set (common English words with little semantic value)
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    answer_tokens = set(t for t in answer_norm.split() if t and t not in STOPWORDS)
    evidence_tokens = set(t for t in evidence_norm.split() if t and t not in STOPWORDS)
    
    # Rule 4a: Check if answer contains numeric/time patterns
    has_digit = bool(re.search(r'\d', answer))
    has_time = bool(re.search(r'\d{1,2}:\d{2}', answer))
    has_ampm = bool(re.search(r'\b(?:am|pm)\b', answer.lower()))
    
    if has_digit or has_time or has_ampm:
        # Answer contains numeric/time info - check if at least one pattern appears in evidence
        # Extract all numbers from answer
        answer_numbers = set(re.findall(r'\b\d+\b', answer))
        evidence_numbers = set(re.findall(r'\b\d+\b', evidence_text))
        
        # Extract time patterns (HH:MM)
        answer_times = set(re.findall(r'\d{1,2}:\d{2}', answer))
        evidence_times = set(re.findall(r'\d{1,2}:\d{2}', evidence_text))
        
        # At least one number or time must match
        if (answer_numbers & evidence_numbers) or (answer_times & evidence_times):
            return True
        else:
            # Numeric/time pattern in answer but not in evidence - likely hallucination
            return False
    
    # Rule 4b: For non-numeric answers, require at least K=2 content tokens overlap
    K = 2
    overlap_count = len(answer_tokens & evidence_tokens)
    
    if overlap_count >= K:
        return True
    
    # Rule 5: Failed all checks
    return False
