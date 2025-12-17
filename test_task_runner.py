"""
Unit tests for TaskRunner abstraction.

Tests InlineTaskRunner and BackgroundTaskRunner behavior.
"""

import pytest
import asyncio
from fastapi import BackgroundTasks
from app.services.task_runner import (
    InlineTaskRunner,
    BackgroundTaskRunner,
    create_inline_task_runner,
    create_background_task_runner,
)


# ============================================================================
# Test InlineTaskRunner
# ============================================================================

def test_inline_task_runner_sync_function():
    """Test InlineTaskRunner executes sync functions immediately."""
    runner = InlineTaskRunner()
    
    # Track execution
    results = []
    
    def sync_task(value):
        results.append(value)
    
    # Submit task
    runner.submit(sync_task, "test-value")
    
    # Should execute immediately
    assert len(results) == 1
    assert results[0] == "test-value"


def test_inline_task_runner_async_function():
    """Test InlineTaskRunner executes async functions immediately."""
    runner = InlineTaskRunner()
    
    # Track execution
    results = []
    
    async def async_task(value):
        results.append(value)
        await asyncio.sleep(0.01)  # Simulate async work
        results.append(value + "-done")
    
    # Submit task
    runner.submit(async_task, "test-value")
    
    # Should execute immediately (blocks until complete)
    assert len(results) == 2
    assert results[0] == "test-value"
    assert results[1] == "test-value-done"


def test_inline_task_runner_with_args_and_kwargs():
    """Test InlineTaskRunner handles positional and keyword arguments."""
    runner = InlineTaskRunner()
    
    results = []
    
    def task_with_args(a, b, c=None, d=None):
        results.append((a, b, c, d))
    
    runner.submit(task_with_args, "arg1", "arg2", c="kwarg1", d="kwarg2")
    
    assert len(results) == 1
    assert results[0] == ("arg1", "arg2", "kwarg1", "kwarg2")


def test_inline_task_runner_propagates_exceptions():
    """Test InlineTaskRunner propagates exceptions to caller."""
    runner = InlineTaskRunner()
    
    def failing_task():
        raise ValueError("Task failed!")
    
    with pytest.raises(ValueError, match="Task failed!"):
        runner.submit(failing_task)


def test_inline_task_runner_multiple_tasks():
    """Test InlineTaskRunner executes multiple tasks in order."""
    runner = InlineTaskRunner()
    
    execution_order = []
    
    def task1():
        execution_order.append(1)
    
    def task2():
        execution_order.append(2)
    
    def task3():
        execution_order.append(3)
    
    runner.submit(task1)
    runner.submit(task2)
    runner.submit(task3)
    
    assert execution_order == [1, 2, 3]


# ============================================================================
# Test BackgroundTaskRunner
# ============================================================================

def test_background_task_runner_adds_task():
    """Test BackgroundTaskRunner delegates to FastAPI BackgroundTasks."""
    background_tasks = BackgroundTasks()
    runner = BackgroundTaskRunner(background_tasks)
    
    # Track if task was added
    task_was_added = False
    original_add_task = background_tasks.add_task
    
    def mock_add_task(func, *args, **kwargs):
        nonlocal task_was_added
        task_was_added = True
        # Verify correct arguments passed
        assert func.__name__ == "test_task"
        assert args == ("arg1", "arg2")
        assert kwargs == {"key": "value"}
        original_add_task(func, *args, **kwargs)
    
    background_tasks.add_task = mock_add_task
    
    def test_task(a, b, key=None):
        pass
    
    runner.submit(test_task, "arg1", "arg2", key="value")
    
    assert task_was_added, "Task should be added to BackgroundTasks"


def test_background_task_runner_with_async_function():
    """Test BackgroundTaskRunner handles async functions."""
    background_tasks = BackgroundTasks()
    runner = BackgroundTaskRunner(background_tasks)
    
    async def async_task(value):
        return value
    
    # Should not raise exception
    runner.submit(async_task, "test-value")
    
    # Verify task was added
    assert len(background_tasks.tasks) == 1


def test_background_task_runner_with_sync_function():
    """Test BackgroundTaskRunner handles sync functions."""
    background_tasks = BackgroundTasks()
    runner = BackgroundTaskRunner(background_tasks)
    
    def sync_task(value):
        return value
    
    # Should not raise exception
    runner.submit(sync_task, "test-value")
    
    # Verify task was added
    assert len(background_tasks.tasks) == 1


# ============================================================================
# Test Factory Functions
# ============================================================================

def test_create_inline_task_runner():
    """Test factory function creates InlineTaskRunner."""
    runner = create_inline_task_runner()
    
    assert isinstance(runner, InlineTaskRunner)
    
    # Verify it works
    results = []
    runner.submit(lambda: results.append("worked"))
    assert results == ["worked"]


def test_create_background_task_runner():
    """Test factory function creates BackgroundTaskRunner."""
    background_tasks = BackgroundTasks()
    runner = create_background_task_runner(background_tasks)
    
    assert isinstance(runner, BackgroundTaskRunner)
    assert runner.background_tasks is background_tasks
    
    # Verify it works
    runner.submit(lambda: None)
    assert len(background_tasks.tasks) == 1


# ============================================================================
# Test Protocol Compliance
# ============================================================================

def test_inline_task_runner_has_submit_method():
    """Verify InlineTaskRunner has submit method."""
    runner = InlineTaskRunner()
    assert hasattr(runner, "submit")
    assert callable(runner.submit)


def test_background_task_runner_has_submit_method():
    """Verify BackgroundTaskRunner has submit method."""
    background_tasks = BackgroundTasks()
    runner = BackgroundTaskRunner(background_tasks)
    assert hasattr(runner, "submit")
    assert callable(runner.submit)


# ============================================================================
# Integration Tests
# ============================================================================

def test_inline_runner_simulates_indexing_workflow():
    """
    Simulate document indexing workflow with InlineTaskRunner.
    Verifies that indexing completes immediately (no polling needed).
    """
    runner = InlineTaskRunner()
    
    # Simulate document state
    document_status = {"status": "pending"}
    
    def index_document(doc_id):
        # Simulate indexing work
        document_status["status"] = "indexing"
        # ... do work ...
        document_status["status"] = "indexed"
    
    # Submit indexing task
    runner.submit(index_document, doc_id=123)
    
    # With InlineTaskRunner, status should be "indexed" immediately
    assert document_status["status"] == "indexed"


def test_background_runner_simulates_indexing_workflow():
    """
    Simulate document indexing workflow with BackgroundTaskRunner.
    Verifies that task is scheduled but not executed immediately.
    """
    background_tasks = BackgroundTasks()
    runner = BackgroundTaskRunner(background_tasks)
    
    # Simulate document state
    document_status = {"status": "pending"}
    
    def index_document(doc_id):
        document_status["status"] = "indexed"
    
    # Submit indexing task
    runner.submit(index_document, doc_id=123)
    
    # With BackgroundTaskRunner, status should still be "pending"
    assert document_status["status"] == "pending"
    
    # Task is scheduled but not executed
    assert len(background_tasks.tasks) == 1


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
