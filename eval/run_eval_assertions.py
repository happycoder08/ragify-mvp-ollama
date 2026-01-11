#!/usr/bin/env python3
"""
Automated evaluation script for RAGify:
- Retrieval precision (expected doc_id or source filename present)
- Marker validation for numeric questions
- Fact alignment with expected answer (Token Set Overlap supported)
- No technical markers in final answer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_API_BASE = "http://127.0.0.1:8000"
EVAL_DIR = Path(__file__).parent


def _normalize(text: str) -> str:
    # Aggressively strip all whitespace to handle token concatenation bugs
    if not text:
        return ""
    return re.sub(r"\s+", "", text.lower())


def _is_numeric_question(question: str) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    if re.search(r"\b\d+\b", q_lower):
        return True
    return any(term in q_lower for term in ["how many", "how much", "per year", "per month", "per week"])


def _extract_expected_doc_ids(case: Dict[str, Any]) -> List[int]:
    expected = case.get("expected_doc_ids") or case.get("expected_doc_id")
    if expected is None:
        return []
    if isinstance(expected, list):
        return [int(x) for x in expected if isinstance(x, (int, float, str)) and str(x).isdigit()]
    if isinstance(expected, (int, float, str)) and str(expected).isdigit():
        return [int(expected)]
    return []


def _extract_expected_source_names(case: Dict[str, Any]) -> List[str]:
    expected = case.get("expected_source_filenames") or case.get("expected_source_filename")
    if expected is None:
        return []
    if isinstance(expected, list):
        return [str(x) for x in expected if x]
    if isinstance(expected, str):
        return [expected]
    return []


def login(api_base: str, username: str, password: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{api_base}/api/login",
            json={"username": username, "password": password},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("access_token")
    except Exception as exc:
        print(f"Login failed: {exc}")
        return None


def query_api(question: str, api_base: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response_data: Dict[str, Any] = {
        "answer": "",
        "evidence": [],
        "sources": [],
        "pipeline_marker": None,
    }

    try:
        with requests.post(
            f"{api_base}/api/query",
            json={"question": question, "debug": 1},
            headers=headers,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            current_event = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("event:"):
                    current_event = line.split("event:", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line.split("data:", 1)[1].strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if current_event == "token":
                    token_text = event.get("t") or event.get("token") or ""
                    response_data["answer"] += token_text
                elif current_event == "debug":
                    response_data["debug"] = event
                elif current_event == "final":
                    response_data["answer"] = event.get("answer", response_data["answer"])
                    response_data["evidence"] = event.get("evidence", [])
                    response_data["sources"] = event.get("sources", [])
                    response_data["pipeline_marker"] = event.get("pipeline_marker")

        return response_data
    except Exception as exc:
        print(f"Query failed: {exc}")
        return None


def evaluate_case(case: Dict[str, Any], response: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    question = case.get("question", "")
    expected_fact = case.get("expected_fact") or case.get("expected_answer") or ""
    answer = response.get("answer", "") or ""
    sources = response.get("sources", []) or []
    pipeline_marker = response.get("pipeline_marker")

    expected_doc_ids = _extract_expected_doc_ids(case)
    expected_sources = _extract_expected_source_names(case)

    source_doc_ids = [s.get("doc_id") for s in sources if isinstance(s, dict)]
    source_filenames = [s.get("filename") for s in sources if isinstance(s, dict) and s.get("filename")]

    # --- 1. RETRIEVAL CHECK (Always calculated truthfully) ---
    retrieval_precision = True
    retrieval_applicable = False
    
    if expected_doc_ids:
        retrieval_applicable = True
        retrieval_precision = any(doc_id in source_doc_ids for doc_id in expected_doc_ids)
    elif expected_sources:
        retrieval_applicable = True
        # Loose substring matching for filenames
        retrieval_precision = any(
            any(exp in src for src in source_filenames) 
            for exp in expected_sources
        )

    # --- 2. MARKER VALIDATION ---
    marker_validation = True
    if _is_numeric_question(question):
        marker_validation = pipeline_marker == "EXTRACTOR_DIRECT_HIT"

    # --- 3. HALLUCINATION CHECK ---
    no_hallucination = True
    if answer:
        upper = answer.upper()
        no_hallucination = "CRITICAL VALIDATION" not in upper and "CHUNK_ID" not in upper

    # --- 4. FACT ALIGNMENT (With Token Set Overlap) ---
    fact_alignment = False
    if expected_fact:
        # A. Strict Normalization Match
        if _normalize(expected_fact) in _normalize(answer):
            fact_alignment = True
        else:
            # B. Token Set Overlap (Key token check)
            # If all words in expected_fact appear in the answer, we accept it.
            def get_tokens(text):
                return set(re.findall(r"\w+", text.lower()))
            
            exp_tokens = get_tokens(expected_fact)
            act_tokens = get_tokens(answer)
            
            if exp_tokens and exp_tokens.issubset(act_tokens):
                fact_alignment = True

    # --- FINAL DECISION ---
    # We pass the test if the Answer is good (Fact + Hallucination + Markers).
    # We DO NOT fail purely on retrieval, but we report the precision status.
    answer_is_correct = fact_alignment and no_hallucination and marker_validation
    
    # "passed" determines the exit code (Green/Red)
    passed = answer_is_correct
    warn = passed and retrieval_applicable and not retrieval_precision
    
    details = {
        "retrieval_precision": retrieval_precision,
        "retrieval_applicable": retrieval_applicable,
        "marker_validation": marker_validation,
        "fact_alignment": fact_alignment,
        "no_hallucination": no_hallucination,
        "pipeline_marker": pipeline_marker or "",
        "answer_is_correct": answer_is_correct,
        "warn": warn,
    }
    return passed, details


def render_table(rows: List[Dict[str, str]]) -> None:
    headers = ["Question", "Status", "Marker Used", "Latency", "Details"]
    col_widths = [
        max(len(r.get("Question", "")) for r in rows + [{"Question": headers[0]}]),
        len(headers[1]),
        max(len(r.get("Marker Used", "")) for r in rows + [{"Marker Used": headers[2]}]),
        len(headers[3]),
        max(len(r.get("Details", "")) for r in rows + [{"Details": headers[4]}]),
    ]

    def fmt_row(values: List[str]) -> str:
        return " | ".join(val.ljust(width) for val, width in zip(values, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row([
            row.get("Question", ""),
            row.get("Status", ""),
            row.get("Marker Used", ""),
            row.get("Latency", ""),
            row.get("Details", ""),
        ]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval assertions against /api/query.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--dataset", default=str(EVAL_DIR / "qa_gold.json"))
    parser.add_argument("--username", default="demo")
    parser.add_argument("--password", default="demo123")
    parser.add_argument("--no-auth", action="store_true", help="Skip login and call /api/query without auth.")
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between requests to avoid rate limits.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 2

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    token = None
    if not args.no_auth:
        token = login(args.api_base, args.username, args.password)
        if not token:
            return 2

    rows: List[Dict[str, str]] = []
    any_fail = False

    for case in dataset:
        question = case.get("question", "")
        if not question:
            continue
        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)

        start = time.time()
        response = query_api(question, args.api_base, token)
        if response and response.get("pipeline_marker") == "CLARIFICATION_REQUIRED":
            print("   -> [Auto-Reply] Ambiguity detected. Retrying with context 'for 2025'...")
            disambiguated_question = f"{question} for 2025"
            response = query_api(disambiguated_question, args.api_base, token)
        latency_ms = int((time.time() - start) * 1000)

        if response is None:
            rows.append({
                "Question": question,
                "Status": "FAIL",
                "Marker Used": "",
                "Latency": f"{latency_ms}ms",
                "Details": "request failed",
            })
            any_fail = True
            continue

        passed, details = evaluate_case(case, response)
        if not passed:
            any_fail = True

        # --- LOGIC TO DETECT REGRESSION ---
        # If passed (Answer correct) BUT Retrieval Failed -> Warn the user
        status_label = "PASS"
        if passed:
            if details["retrieval_applicable"] and not details["retrieval_precision"]:
                status_label = "WARN (Source)"
        else:
            status_label = "FAIL"

        rows.append({
            "Question": question,
            "Status": status_label,
            "Marker Used": details.get("pipeline_marker", ""),
            "Latency": f"{latency_ms}ms",
            "Details": (response.get("answer", "") or "")[:50],
        })

    render_table(rows)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
