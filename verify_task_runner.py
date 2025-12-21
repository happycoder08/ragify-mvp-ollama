"""
Verify TaskRunner abstraction works correctly.

Demonstrates:
- InlineTaskRunner executes immediately
- BackgroundTaskRunner delegates to FastAPI BackgroundTasks
"""

from fastapi import BackgroundTasks
from app.services.task_runner import (
    InlineTaskRunner,
    BackgroundTaskRunner,
    create_inline_task_runner,
    create_background_task_runner,
)


def main():
    print("=" * 60)
    print("TaskRunner Abstraction Verification")
    print("=" * 60)
    
    # Test 1: InlineTaskRunner
    print("\n1. InlineTaskRunner (immediate execution)")
    print("-" * 60)
    
    inline_runner = create_inline_task_runner()
    
    execution_log = []
    
    def task1():
        execution_log.append("task1")
    
    def task2():
        execution_log.append("task2")
    
    async def async_task():
        execution_log.append("async_task")
    
    print("Submitting tasks...")
    inline_runner.submit(task1)
    inline_runner.submit(task2)
    inline_runner.submit(async_task)
    
    print(f"✓ All tasks executed immediately: {execution_log}")
    assert execution_log == ["task1", "task2", "async_task"]
    
    # Test 2: BackgroundTaskRunner
    print("\n2. BackgroundTaskRunner (deferred execution)")
    print("-" * 60)
    
    background_tasks = BackgroundTasks()
    background_runner = create_background_task_runner(background_tasks)
    
    execution_log = []
    
    def background_task():
        execution_log.append("background_task")
    
    print("Submitting task to background...")
    background_runner.submit(background_task)
    
    print(f"✓ Task scheduled (not executed yet): {len(background_tasks.tasks)} task(s) pending")
    assert len(execution_log) == 0  # Not executed yet
    assert len(background_tasks.tasks) == 1  # Scheduled
    
    # Test 3: Runtime Integration
    print("\n3. Runtime Integration")
    print("-" * 60)
    
    from app.runtime import build_test_runtime
    
    runtime = build_test_runtime()
    
    print(f"Test runtime task_runner type: {type(runtime.task_runner).__name__}")
    assert isinstance(runtime.task_runner, InlineTaskRunner)
    
    # Verify immediate execution
    results = []
    runtime.task_runner.submit(lambda: results.append("executed"))
    assert results == ["executed"]
    print("✓ Test runtime uses InlineTaskRunner (immediate execution)")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ All TaskRunner Tests Passed!")
    print("=" * 60)
    
    print("\nSummary:")
    print("  - InlineTaskRunner: Executes tasks immediately (tests)")
    print("  - BackgroundTaskRunner: Defers tasks to FastAPI (production)")
    print("  - Test runtime: Uses InlineTaskRunner instance")
    print("  - Production runtime: Uses factory (create_background_task_runner)")
    print("\nUsage in production:")
    print("  task_runner = runtime.task_runner(background_tasks)")
    print("  task_runner.submit(process_document, doc_id, tenant_id, ...)")
    print("=" * 60)


if __name__ == "__main__":
    main()
