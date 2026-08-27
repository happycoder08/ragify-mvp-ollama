"""
Test SSE (Server-Sent Events) format for /api/query endpoint.

Verifies that the streaming response follows SSE specification:
- Content-Type: text/event-stream
- Event format: "event: <name>\ndata: <json>\n\n"
- Event types: debug, token, final, error
"""

import pytest
from test_integration import parse_sse_events


def test_sse_parser():
    """Test the SSE parser with various formats."""
    # Simple single event
    sse_text = 'event: token\ndata: {"t":"hello"}\n\n'
    events = parse_sse_events(sse_text)
    assert len(events) == 1
    assert events[0]["event"] == "token"
    assert events[0]["data"] == {"t": "hello"}
    
    # Multiple events
    sse_text = '''event: debug
data: {"evidence_count":5}

event: token
data: {"t":"hello"}

event: token
data: {"t":" world"}

event: final
data: {"answer":"hello world","refused":false}

'''
    events = parse_sse_events(sse_text)
    assert len(events) == 4
    assert events[0]["event"] == "debug"
    assert events[1]["event"] == "token"
    assert events[2]["event"] == "token"
    assert events[3]["event"] == "final"
    assert events[3]["data"]["answer"] == "hello world"


def test_sse_multiline_data():
    """Test SSE parser with multi-line data."""
    sse_text = '''event: final
data: {
data:   "answer": "multi-line",
data:   "refused": false
data: }

'''
    events = parse_sse_events(sse_text)
    assert len(events) == 1
    assert events[0]["event"] == "final"
    # Multi-line JSON should be parsed correctly
    assert isinstance(events[0]["data"], dict)
    assert events[0]["data"]["answer"] == "multi-line"


def test_sse_event_types():
    """Test that all expected SSE event types are recognized."""
    event_types = ["debug", "token", "final", "error"]
    
    for event_type in event_types:
        sse_text = f'event: {event_type}\ndata: {{"test":"data"}}\n\n'
        events = parse_sse_events(sse_text)
        assert len(events) == 1
        assert events[0]["event"] == event_type
        assert events[0]["data"] == {"test": "data"}


def test_sse_empty_events():
    """Test SSE parser with empty or malformed input."""
    # Empty string
    events = parse_sse_events("")
    assert events == []
    
    # Only whitespace
    events = parse_sse_events("\n\n\n")
    assert events == []
    
    # Event without data
    events = parse_sse_events("event: test\n\n")
    assert events == []


def test_sse_json_parsing():
    """Test that SSE data is properly parsed as JSON."""
    # Valid JSON
    sse_text = 'event: test\ndata: {"key":"value","num":42}\n\n'
    events = parse_sse_events(sse_text)
    assert events[0]["data"]["key"] == "value"
    assert events[0]["data"]["num"] == 42
    
    # Invalid JSON (should keep as string)
    sse_text = 'event: test\ndata: not valid json\n\n'
    events = parse_sse_events(sse_text)
    assert isinstance(events[0]["data"], str)
    assert events[0]["data"] == "not valid json"
