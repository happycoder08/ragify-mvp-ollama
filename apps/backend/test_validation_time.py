import pytest
from app.services import validation

@pytest.mark.parametrize("answer,evidence,expected", [
    ("You should arrive at 7:00 PM.", "ARRIVE AT THE OFFICE (8:00 AM)", False),
    ("Arrive at 8:00 AM.", "ARRIVE AT THE OFFICE (8:00 AM)", True),
    ("Arrive at 8:00 AM.", "ARRIVE AT THE OFFICE (8 AM)", True),
    ("Arrive at 8 AM.", "ARRIVE AT THE OFFICE (8:00 AM)", True),
    ("Arrive at 8am.", "ARRIVE AT THE OFFICE (8:00 AM)", True),
    ("Arrive at 8:00 am.", "ARRIVE AT THE OFFICE (8am)", True),
    ("Arrive at 19:00.", "ARRIVE AT THE OFFICE (8:00 AM)", False),
    ("Arrive at 8:00 AM.", "ARRIVE AT THE OFFICE (8:00 AM) and 9:00 AM", True),
    ("Arrive at 9:00 AM.", "ARRIVE AT THE OFFICE (8:00 AM) and 9:00 AM", True),
    ("Arrive at 10:00 AM.", "ARRIVE AT THE OFFICE (8:00 AM) and 9:00 AM", False),
])
def test_time_validation(answer, evidence, expected):
    assert validation.answer_supported_by_evidence(answer, evidence) == expected
