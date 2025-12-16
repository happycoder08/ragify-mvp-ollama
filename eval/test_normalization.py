"""
Unit tests for text normalization functions used in eval scoring.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from run_eval import normalize_answer, normalize_times, normalize_numbers


def test_normalize_answer():
    """Test basic text normalization."""
    assert normalize_answer("  Hello   World  ") == "hello world"
    assert normalize_answer("UPPERCASE") == "uppercase"
    assert normalize_answer("Multiple   Spaces") == "multiple spaces"
    print("✓ normalize_answer tests passed")


def test_normalize_times():
    """Test time format normalization."""
    # AM/PM formats
    assert normalize_times("8:00 AM") == "8am"
    assert normalize_times("8 am") == "8am"
    assert normalize_times("08:00 AM") == "8am"
    assert normalize_times("8:00am") == "8am"
    assert normalize_times("3:00 PM") == "3pm"
    assert normalize_times("03:00 PM") == "3pm"
    
    # 24-hour formats
    assert normalize_times("08:00") == "8"
    assert normalize_times("13:00") == "13"
    
    # Mixed text
    assert normalize_times("arrive at 8:00 AM") == "arrive at 8am"
    assert normalize_times("between 10 am and 3 PM") == "between 10am and 3pm"
    
    print("✓ normalize_times tests passed")


def test_normalize_numbers():
    """Test numeric format normalization."""
    # Currency symbols
    assert normalize_numbers("$1500") == "1500"
    assert normalize_numbers("£2000") == "2000"
    assert normalize_numbers("€500") == "500"
    assert normalize_numbers("¥1000") == "1000"
    
    # Commas in numbers
    assert normalize_numbers("1,500") == "1500"
    assert normalize_numbers("10,000") == "10000"
    assert normalize_numbers("1,234,567") == "1234567"
    
    # Combined
    assert normalize_numbers("$1,500") == "1500"
    assert normalize_numbers("£2,000.50") == "2000.50"
    assert normalize_numbers("salary is $50,000") == "salary is 50000"
    
    # Edge cases
    assert normalize_numbers("no numbers here") == "no numbers here"
    assert normalize_numbers("15 days") == "15 days"
    
    print("✓ normalize_numbers tests passed")


def test_combined_normalization():
    """Test combined normalization pipeline."""
    text = "Arrive at 8:00 AM with $1,500"
    
    # Apply all normalizations
    normalized = normalize_answer(text)
    normalized = normalize_times(normalized)
    normalized = normalize_numbers(normalized)
    
    # Should match these patterns
    assert "8am" in normalized
    assert "1500" in normalized
    assert "$" not in normalized
    assert "," not in normalized
    
    print("✓ combined normalization tests passed")


def test_keyword_matching():
    """Test realistic keyword matching scenarios."""
    # Time matching
    text1 = "You should arrive at 8:00 AM on your first day"
    norm1 = normalize_times(normalize_answer(text1))
    assert "8am" in norm1
    
    # Number matching
    text2 = "You get 15 vacation days"
    norm2 = normalize_numbers(normalize_answer(text2))
    assert "15" in norm2
    
    # Currency matching
    text3 = "The reimbursement limit is $1,500 per year"
    norm3 = normalize_numbers(normalize_answer(text3))
    assert "1500" in norm3
    
    print("✓ keyword matching tests passed")


if __name__ == "__main__":
    print("\nRunning normalization unit tests...\n")
    test_normalize_answer()
    test_normalize_times()
    test_normalize_numbers()
    test_combined_normalization()
    test_keyword_matching()
    print("\n✅ All tests passed!")
