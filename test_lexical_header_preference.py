import pytest

from app.services import rag_service


def test_header_overlap_prefers_arrival_over_lunch():
    question = "What time should I arrive?"
    arrive_doc = "ARRIVE AT THE OFFICE (8:00 AM)\nPlease arrive by 8:00 AM at main reception."
    lunch_doc = "TEAM LUNCH (12:00 PM - 1:00 PM)\nMeet the team for lunch."

    arrive_score = rag_service._lexical_overlap_score(question, arrive_doc)
    lunch_score = rag_service._lexical_overlap_score(question, lunch_doc)

    assert arrive_score > lunch_score
