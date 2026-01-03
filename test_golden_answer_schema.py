"""
Golden tests for AnswerSchema validation.

Tests grouped by AnswerSchema with assertions for:
- output shape matches schema
- citations present
- refusal only when appropriate

Do NOT assert exact wording. Assert structure + citations.
"""

import pytest
import re
from app.schemas.query import AnswerSchema
from app.services.rag_service import _validate_response_by_schema


# FACT_SINGLE golden tests - question -> expected schema + sample valid response
FACT_SINGLE_GOLDEN = [
    {
        "question": "What time do I arrive my first day?",
        "expected_schema": AnswerSchema.FACT_SINGLE,
        "sample_response": "You arrive at 8:00 AM (CHUNK_ID=example_1)",
    },
    {
        "question": "When is team lunch?",
        "expected_schema": AnswerSchema.FACT_SINGLE,
        "sample_response": "Team lunch is at 12:00 PM (CHUNK_ID=example_2)",
    },
    {
        "question": "Where is the main reception?",
        "expected_schema": AnswerSchema.FACT_SINGLE,
        "sample_response": "The main reception is on the first floor (CHUNK_ID=example_3)",
    },
    {
        "question": "Who is my manager?",
        "expected_schema": AnswerSchema.FACT_SINGLE,
        "sample_response": "Your manager is John Smith (CHUNK_ID=example_4)",
    },
    {
        "question": "What is the wifi password?",
        "expected_schema": AnswerSchema.FACT_SINGLE,
        "sample_response": "The wifi password is CompanyWiFi2024 (CHUNK_ID=example_5)",
    },
]


# CHECKLIST_PROCEDURE golden tests
CHECKLIST_PROCEDURE_GOLDEN = [
    {
        "question": "How do I set up my email signature?",
        "expected_schema": AnswerSchema.CHECKLIST_PROCEDURE,
        "sample_response": "1. Open Outlook and go to settings. (CHUNK_ID=example_1)\n2. Click on signature. (CHUNK_ID=example_1)\n3. Enter your information. (CHUNK_ID=example_1)",
    },
    {
        "question": "What documents do I need to bring?",
        "expected_schema": AnswerSchema.CHECKLIST_PROCEDURE,
        "sample_response": "1. Government-issued ID. (CHUNK_ID=example_2)\n2. Social security card. (CHUNK_ID=example_2)\n3. Birth certificate. (CHUNK_ID=example_2)",
    },
    {
        "question": "How do I get my badge?",
        "expected_schema": AnswerSchema.CHECKLIST_PROCEDURE,
        "sample_response": "1. Go to HR on first day. (CHUNK_ID=example_3)\n2. Bring your ID. (CHUNK_ID=example_3)\n3. Have your photo taken. (CHUNK_ID=example_3)",
    },
    {
        "question": "What are the first day steps?",
        "expected_schema": AnswerSchema.CHECKLIST_PROCEDURE,
        "sample_response": "1. Arrive at 8 AM. (CHUNK_ID=example_4)\n2. Check in at reception. (CHUNK_ID=example_4)\n3. Attend orientation. (CHUNK_ID=example_4)",
    },
    {
        "question": "How do I access the building?",
        "expected_schema": AnswerSchema.CHECKLIST_PROCEDURE,
        "sample_response": "1. Use your badge at the entrance. (CHUNK_ID=example_5)\n2. Enter through the main door. (CHUNK_ID=example_5)\n3. Take the elevator to your floor. (CHUNK_ID=example_5)",
    },
]


# POLICY_EXCERPT golden tests
POLICY_EXCERPT_GOLDEN = [
    {
        "question": "What is the dress code policy?",
        "expected_schema": AnswerSchema.POLICY_EXCERPT,
        "sample_response": "- Business casual attire is required (CHUNK_ID=example_1)\n- No jeans or sneakers (CHUNK_ID=example_1)\n- Shirts must be tucked in (CHUNK_ID=example_1)",
    },
    {
        "question": "What are the parking rules?",
        "expected_schema": AnswerSchema.POLICY_EXCERPT,
        "sample_response": "- Employees may park in designated areas only (CHUNK_ID=example_2)\n- Visitor parking is not permitted (CHUNK_ID=example_2)\n- Parking permits are required (CHUNK_ID=example_2)",
    },
    {
        "question": "What is the vacation policy?",
        "expected_schema": AnswerSchema.POLICY_EXCERPT,
        "sample_response": "- Employees accrue 10 days of vacation per year (CHUNK_ID=example_3)\n- Vacation must be approved in advance (CHUNK_ID=example_3)\n- Maximum carryover is 15 days (CHUNK_ID=example_3)",
    },
    {
        "question": "What are the IT security guidelines?",
        "expected_schema": AnswerSchema.POLICY_EXCERPT,
        "sample_response": "- All passwords must be changed quarterly (CHUNK_ID=example_4)\n- Do not share login credentials (CHUNK_ID=example_4)\n- Report suspicious emails immediately (CHUNK_ID=example_4)",
    },
    {
        "question": "What is the code of conduct?",
        "expected_schema": AnswerSchema.POLICY_EXCERPT,
        "sample_response": "- Maintain professional behavior at all times (CHUNK_ID=example_5)\n- Respect colleagues and clients (CHUNK_ID=example_5)\n- Follow company policies (CHUNK_ID=example_5)",
    },
]


# BOOLEAN_SPECIFIED golden tests
BOOLEAN_SPECIFIED_GOLDEN = [
    {
        "question": "Do I need to bring ID on first day?",
        "expected_schema": AnswerSchema.BOOLEAN_SPECIFIED,
        "sample_response": "Yes — you need to bring government-issued ID (CHUNK_ID=example_1)",
    },
    {
        "question": "Is there free parking?",
        "expected_schema": AnswerSchema.BOOLEAN_SPECIFIED,
        "sample_response": "No — parking requires a permit (CHUNK_ID=example_2)",
    },
    {
        "question": "Is there a dress code?",
        "expected_schema": AnswerSchema.BOOLEAN_SPECIFIED,
        "sample_response": "Yes — business casual is required (CHUNK_ID=example_3)",
    },
    {
        "question": "Do I get a company laptop?",
        "expected_schema": AnswerSchema.BOOLEAN_SPECIFIED,
        "sample_response": "Yes — laptops are provided on first day (CHUNK_ID=example_4)",
    },
    {
        "question": "Can I work from home?",
        "expected_schema": AnswerSchema.BOOLEAN_SPECIFIED,
        "sample_response": "No the document does not specify this.",
    },
]


# NOT_FOUND_EXPLICIT golden tests
NOT_FOUND_EXPLICIT_GOLDEN = [
    {
        "question": "What is the meaning of life?",
        "expected_schema": AnswerSchema.NOT_FOUND_EXPLICIT,
        "sample_response": "The document does not specify this.",
    },
    {
        "question": "How do I time travel?",
        "expected_schema": AnswerSchema.NOT_FOUND_EXPLICIT,
        "sample_response": "The document does not specify this.",
    },
    {
        "question": "What is the weather like on Mars?",
        "expected_schema": AnswerSchema.NOT_FOUND_EXPLICIT,
        "sample_response": "The document does not specify this.",
    },
    {
        "question": "How do I become invisible?",
        "expected_schema": AnswerSchema.NOT_FOUND_EXPLICIT,
        "sample_response": "The document does not specify this.",
    },
    {
        "question": "What is the secret recipe?",
        "expected_schema": AnswerSchema.NOT_FOUND_EXPLICIT,
        "sample_response": "The document does not specify this.",
    },
]


@pytest.mark.parametrize("case", FACT_SINGLE_GOLDEN)
def test_golden_fact_single(case):
    _run_schema_test(case)


@pytest.mark.parametrize("case", CHECKLIST_PROCEDURE_GOLDEN)
def test_golden_checklist_procedure(case):
    _run_schema_test(case)


@pytest.mark.parametrize("case", POLICY_EXCERPT_GOLDEN)
def test_golden_policy_excerpt(case):
    _run_schema_test(case)


@pytest.mark.parametrize("case", BOOLEAN_SPECIFIED_GOLDEN)
def test_golden_boolean_specified(case):
    _run_schema_test(case)


@pytest.mark.parametrize("case", NOT_FOUND_EXPLICIT_GOLDEN)
def test_golden_not_found_explicit(case):
    _run_schema_test(case)


def _run_schema_test(case):
    """Test schema validation with sample responses."""
    expected_schema = case["expected_schema"]
    sample_response = case["sample_response"]
    
    # Test response validation
    is_valid = _validate_response_by_schema(sample_response, expected_schema)
    assert is_valid, f"Sample response validation failed for {expected_schema}: {sample_response}"
    
    # Test citations are present (except for NOT_FOUND_EXPLICIT and canonical refusals)
    if expected_schema != AnswerSchema.NOT_FOUND_EXPLICIT:
        # For BOOLEAN_SPECIFIED, citations are not required if using canonical refusal
        if not (expected_schema == AnswerSchema.BOOLEAN_SPECIFIED and "the document does not specify this" in sample_response):
            assert "(CHUNK_ID=" in sample_response, f"No citations in sample response for {expected_schema}"
    
    # Test structure assertions
    if expected_schema == AnswerSchema.FACT_SINGLE:
        # Should be single sentence with citation
        # Remove citation and count sentences
        response_no_citation = re.sub(r'\s*\(CHUNK_ID=[^)]+\)$', '', sample_response).strip()
        sentences = re.split(r'[.!?]+', response_no_citation)
        sentences = [s.strip() for s in sentences if s.strip()]
        assert len(sentences) == 1, f"FACT_SINGLE should have exactly 1 sentence: {sample_response}"
        
    elif expected_schema == AnswerSchema.CHECKLIST_PROCEDURE:
        # Should have numbered list format
        lines = [line.strip() for line in sample_response.split('\n') if line.strip()]
        assert len(lines) >= 2, f"CHECKLIST_PROCEDURE should have multiple lines: {sample_response}"
        assert all(line.startswith(tuple(f"{i}." for i in range(1, 10))) for line in lines), f"CHECKLIST_PROCEDURE should be numbered: {sample_response}"
        
    elif expected_schema == AnswerSchema.POLICY_EXCERPT:
        # Should have bullet-like format (multiple lines with citations)
        lines = [line.strip() for line in sample_response.split('\n') if line.strip()]
        assert len(lines) >= 2, f"POLICY_EXCERPT should have multiple lines: {sample_response}"
        assert all(line.startswith('-') for line in lines), f"POLICY_EXCERPT should start with bullets: {sample_response}"
        
    elif expected_schema == AnswerSchema.BOOLEAN_SPECIFIED:
        # Should start with Yes/No
        assert sample_response.startswith(('Yes', 'No')), f"BOOLEAN_SPECIFIED should start with Yes/No: {sample_response}"
        # If it's "No" with canonical refusal, no citation needed
        if "the document does not specify this" in sample_response:
            pass  # No citation required for canonical refusal
        else:
            # Otherwise must have citation
            assert "(CHUNK_ID=" in sample_response, f"BOOLEAN_SPECIFIED with info should have citation: {sample_response}"
        
    elif expected_schema == AnswerSchema.NOT_FOUND_EXPLICIT:
        # Should be exact canonical refusal
        assert sample_response == "The document does not specify this.", f"NOT_FOUND_EXPLICIT should be canonical refusal: {sample_response}"