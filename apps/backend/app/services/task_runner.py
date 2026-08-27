"""
TaskRunner abstraction for background task execution.

Provides a clean interface for submitting background tasks that can be:
- BackgroundTaskRunner: FastAPI background tasks (async, fire-and-forget)
- InlineTaskRunner: Immediate execution (for tests, predictable state)

This abstraction enables:
1. Testable task execution (inline = immediate, predictable)
2. Clean separation of concerns (no direct BackgroundTasks dependency)
3. Easy mocking and testing of async workflows
"""

import logging
from typing import Protocol, Any, Callable
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


class TaskRunner(Protocol):
    """
    Interface for task execution strategies.
    
    Implementations can choose when and how to execute tasks:
    - Background: fire-and-forget async execution
    - Inline: immediate synchronous execution
    - Queued: deferred execution with job queue
    """
    
    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Submit a task for execution.
        
        Args:
            func: Callable to execute (can be sync or async)
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
        
        Returns:
            None: Task submission is fire-and-forget
        """
        ...


class BackgroundTaskRunner:
    """
    Production task runner using FastAPI BackgroundTasks.
    
    Tasks are executed asynchronously after the HTTP response is sent.
    This is the default behavior in production for non-blocking operations.
    
    Example:
        background_tasks = BackgroundTasks()
        runner = BackgroundTaskRunner(background_tasks)
        runner.submit(index_document, file_id="123", tenant_id="acme")
    """
    
    def __init__(self, background_tasks: BackgroundTasks):
        """
        Initialize with FastAPI BackgroundTasks instance.
        
        Args:
            background_tasks: FastAPI BackgroundTasks from request dependency
        """
        self.background_tasks = background_tasks
    
    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Submit task to FastAPI background execution.
        
        Task will run asynchronously after the response is sent.
        If func is async, it will be awaited. If sync, it runs in thread pool.
        
        Args:
            func: Callable to execute (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self.background_tasks.add_task(func, *args, **kwargs)
        logger.debug(
            "Submitted background task: %s (args=%s, kwargs=%s)",
            func.__name__,
            args,
            kwargs
        )


class InlineTaskRunner:
    """
    Test task runner that executes tasks immediately (inline).
    
    Tasks run synchronously in the same context as the caller.
    This makes tests predictable and eliminates race conditions.
    
    Use this in tests when you need:
    - Deterministic state after task submission
    - No async polling or waiting
    - Immediate verification of task effects
    
    Example:
        runner = InlineTaskRunner()
        runner.submit(index_document, file_id="123", tenant_id="acme")
        # Task has completed by this point
        assert document_is_indexed("123")
    """
    
    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Execute task immediately in current thread/context.
        
        If func is async, it will be run in a dedicated thread with its own event loop.
        If func is sync, it will be called directly.
        
        Args:
            func: Callable to execute (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Note:
            Exceptions are propagated to caller (no silent failures)
        """
        import asyncio
        import inspect
        import threading
        from concurrent.futures import Future
        
        logger.debug(
            "Executing inline task: %s (args=%s, kwargs=%s)",
            func.__name__,
            args,
            kwargs
        )
        
        try:
            if inspect.iscoroutinefunction(func):
                # Async function: run in separate thread to avoid event loop conflicts
                future: Future = Future()
                
                def run_in_thread():
                    """Run async function in dedicated thread with its own event loop"""
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(func(*args, **kwargs))
                        loop.close()
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()  # Wait for completion
                
                # Get result or raise exception
                future.result()
            else:
                # Sync function: call directly
                func(*args, **kwargs)
            
            logger.debug("Inline task completed: %s", func.__name__)
            
        except Exception as e:
            logger.error(
                "Inline task failed: %s - %s: %s",
                func.__name__,
                type(e).__name__,
                str(e),
                exc_info=True
            )
            # Re-raise so tests can catch failures
            raise


def create_background_task_runner(background_tasks: BackgroundTasks) -> BackgroundTaskRunner:
    """
    Factory function to create BackgroundTaskRunner.
    
    Args:
        background_tasks: FastAPI BackgroundTasks dependency
    
    Returns:
        BackgroundTaskRunner instance
    """
    return BackgroundTaskRunner(background_tasks)


def create_inline_task_runner() -> InlineTaskRunner:
    """
    Factory function to create InlineTaskRunner.
    
    Returns:
        InlineTaskRunner instance for immediate execution
    """
    return InlineTaskRunner()
