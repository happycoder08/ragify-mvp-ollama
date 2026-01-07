# app/schemas/clarification.py
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any


class ClarificationType(str, Enum):
    TIMEFRAME = "TIMEFRAME"
    DOCUMENT = "DOCUMENT"
    SCOPE = "SCOPE"
    LOCATION = "LOCATION"
    ROLE = "ROLE"
    OTHER = "OTHER"


MAX_OPTIONS: int = 6


def build_question(c_type: ClarificationType, options: Optional[List[str]] = None) -> str:
    opts = [o for o in (options or []) if isinstance(o, str) and o.strip()]
    if c_type == ClarificationType.TIMEFRAME:
        return f"Which timeframe should I use: {' or '.join(opts)}?" if opts else "Which timeframe should I use?"
    if c_type == ClarificationType.DOCUMENT:
        return f"Which document should I use: {' or '.join(opts)}?" if opts else "Which document should I use?"
    if c_type == ClarificationType.SCOPE:
        return f"Which scope do you mean: {' or '.join(opts)}?" if opts else "What scope do you mean?"
    if c_type == ClarificationType.LOCATION:
        return f"Which location: {' or '.join(opts)}?" if opts else "Which location?"
    if c_type == ClarificationType.ROLE:
        return f"Which role/team: {' or '.join(opts)}?" if opts else "Which role/team?"
    return "Can you clarify what you mean?"


def normalize_options(options: Optional[List[str]]) -> Optional[List[str]]:
    if not options:
        return None
    clean: List[str] = []
    seen = set()
    for o in options:
        if not isinstance(o, str):
            continue
        s = o.strip()
        if not s or s in seen:
            continue
        clean.append(s)
        seen.add(s)
        if len(clean) >= MAX_OPTIONS:
            break
    return clean or None


def build_clarification_payload(
    c_type: ClarificationType,
    options: Optional[List[str]] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    norm_opts = normalize_options(options)
    q = question.strip() if isinstance(question, str) and question.strip() else build_question(c_type, norm_opts)
    return {
        "needs_clarification": True,
        "clarification": {
            "type": c_type.value,
            "question": q,
            **({"options": norm_opts} if norm_opts else {}),
        },
        "pipeline_marker": "CLARIFICATION_REQUIRED",
        "refused": False,
        "answer": q,
        "evidence": [],
        "sources": [],
    }
