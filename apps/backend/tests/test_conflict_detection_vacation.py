import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ChunkHit:
    chunk_id: str
    doc: str
    meta: Dict[str, Any] = field(default_factory=dict)
    snippet: str = "" 

def _detect_numeric_conflict(question: str, evidence_items: List[Any]) -> Any:
    import re
    
    # 1. Check intent (strict heuristic)
    q_lower = question.lower()
    if "vacation" not in q_lower:
        return None
    if not any(k in q_lower for k in ["days", "per year", "per calendar year"]):
        return None
        
    # 2. Extract numbers from evidence
    patterns = [
        re.compile(r"receive\s+(\d{1,2})\s+vacation\s+days", re.IGNORECASE),
        re.compile(r"(\d{1,2})\s+vacation\s+days\s+per", re.IGNORECASE)
    ]
    
    conflict_values = [] 
    seen_values = set()
    source_files_map = {} 
    
    for item in evidence_items:
        snippet = getattr(item, "doc", "") 
        chunk_id = getattr(item, "chunk_id", "")
        meta = getattr(item, "meta", {})
        source_file = meta.get("source_file") or meta.get("filename")
        
        if not source_file and chunk_id:
             parts = chunk_id.rsplit("_", 1)
             if len(parts) >= 2:
                 source_file = parts[0]
        
        if not source_file:
            continue

        found_in_chunk = set()
        for pat in patterns:
            matches = pat.findall(snippet)
            for m in matches:
                try:
                    val = int(m)
                    if 1 <= val <= 60:
                        found_in_chunk.add(val)
                except ValueError:
                    pass
        
        for val in found_in_chunk:
            conflict_values.append({
                "value": val,
                "source_file": source_file,
                "chunk_id": chunk_id
            })
            seen_values.add(val)
            if source_file not in source_files_map:
                source_files_map[source_file] = set()
            source_files_map[source_file].add(val)

    # 3. Check for conflicts
    if len(seen_values) < 2:
        return None
        
    if len(source_files_map) < 2:
        return None
        
    all_vals = set()
    for v_set in source_files_map.values():
        all_vals.update(v_set)
        
    if len(all_vals) < 2:
        return None

    # 4. Extract candidate year options
    years = set()
    year_pattern = re.compile(r"(202[0-9])")
    for fname in source_files_map.keys():
        ym = year_pattern.search(fname)
        if ym:
            years.add(ym.group(1))
            
    options = sorted(list(years))

    return {
        "pipeline_marker": "CLARIFICATION_REQUIRED",
        "clarification": {
            "type": "TIMEFRAME",
            "question": f"Which policy year should I use: {' or '.join(options)}?" if len(options) > 1 else "Which policy year should I use?",
            "options": options
        },
        "conflict_detected": True,
        "conflict_kind": "VACATION_DAYS_YEAR",
        "conflict_values": conflict_values
    }

def test_vacation_conflict():
    q = "How many vacation days do full-time employees receive per year?"
    
    # Case 1: Conflict
    ev1 = ChunkHit(
        chunk_id="Benefits_Policy_2025.txt_0",
        doc="Full-time employees receive 15 vacation days per calendar year.",
        meta={"source_file": "Benefits_Policy_2025.txt"}
    )
    ev2 = ChunkHit(
        chunk_id="Benefits_Policy_2026.txt_0",
        doc="Full-time employees receive 20 vacation days per calendar year.",
        meta={"source_file": "Benefits_Policy_2026.txt"}
    )
    
    res = _detect_numeric_conflict(q, [ev1, ev2])
    assert res is not None
    assert res["pipeline_marker"] == "CLARIFICATION_REQUIRED"
    assert res["clarification"]["options"] == ["2025", "2026"]
    assert len(res["conflict_values"]) == 2

    # Case 2: No Conflict (same values)
    ev3 = ChunkHit(
        chunk_id="Benefits_Policy_2026_v2.txt_0",
        doc="Full-time employees receive 15 vacation days per calendar year.",
        meta={"source_file": "Benefits_Policy_2026_v2.txt"}
    )
    res = _detect_numeric_conflict(q, [ev1, ev3])
    assert res is None

    # Case 3: No Conflict (same file)
    ev4 = ChunkHit(
        chunk_id="Benefits_Policy_2025.txt_1",
        doc="You get 20 vacation days per year.", 
        meta={"source_file": "Benefits_Policy_2025.txt"}
    )
    res = _detect_numeric_conflict(q, [ev1, ev4])
    assert res is None

    # Case 4: Irrelevant question
    res = _detect_numeric_conflict("What is the dress code?", [ev1, ev2])
    assert res is None
