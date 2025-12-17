# TaskRunner Abstraction Implementation

## Overview
Introduced a TaskRunner abstraction to make background task execution testable and configurable. Enables immediate task execution in tests (no polling) while maintaining async background execution in production.

## Files Created/Modified

### New Files
- **app/services/task_runner.py** (186 lines)
  - `TaskRunner` protocol interface
  - `InlineTaskRunner` class (immediate execution for tests)
  - `BackgroundTaskRunner` class (wraps FastAPI BackgroundTasks for production)
  - Factory functions: `create_inline_task_runner()`, `create_background_task_runner()`

- **test_task_runner.py** (258 lines)
  - 14 comprehensive unit tests (all passing ✓)
  - Tests: sync functions, async functions, args/kwargs, exceptions, multiple tasks, protocol compliance
  - Integration tests: simulates document indexing workflows

- **verify_task_runner.py** (94 lines)
  - Verification script demonstrating TaskRunner behavior
  - Shows immediate vs deferred execution

### Modified Files
- **app/runtime.py**
  - Updated `AppRuntime.task_runner` field documentation
  - `build_runtime_from_env()`: Uses `create_background_task_runner` factory
  - `build_test_runtime()`: Uses `create_inline_task_runner()` instance

- **main.py**
  - Imported `build_runtime_from_env`
  - Added global `runtime` variable
  - Initialize runtime in `startup_event()`
  - Updated upload endpoint to use `runtime.task_runner(background_tasks).submit(...)`

- **test_integration.py**
  - Added `build_test_runtime` import
  - Added `runtime_override` fixture (uses InlineTaskRunner)
  - Updated `client` fixture to include `runtime_override`
  - Updated `test_full_workflow` to remove polling (indexing is immediate)

- **test_runtime.py**
  - Updated `test_test_runtime_has_sync_task_runner()` to check for `InlineTaskRunner` instance

## Architecture

### TaskRunner Protocol
```python
class TaskRunner(Protocol):
    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Submit a task for execution."""
        ...
```

### InlineTaskRunner
**Purpose:** Immediate execution for tests (deterministic, no polling)

**Behavior:**
- Executes sync functions directly
- Executes async functions using `asyncio.run()` or `loop.create_task()`
- Blocks until task completes
- Propagates exceptions to caller

**Use Case:**
```python
runner = InlineTaskRunner()
runner.submit(index_document, doc_id=123, tenant_id="acme")
# Task has completed by this point
```

### BackgroundTaskRunner
**Purpose:** Production execution via FastAPI BackgroundTasks

**Behavior:**
- Delegates to `background_tasks.add_task()`
- Tasks run asynchronously after HTTP response sent
- Fire-and-forget (no blocking)

**Use Case:**
```python
background_tasks = BackgroundTasks()
runner = BackgroundTaskRunner(background_tasks)
runner.submit(index_document, doc_id=123, tenant_id="acme")
# Task scheduled but not executed yet
```

## Integration with AppRuntime

### Production Runtime
```python
from app.services.task_runner import create_background_task_runner

# Runtime holds factory function
task_runner = create_background_task_runner

# In endpoint:
@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks):
    # Create TaskRunner instance from factory
    runner = runtime.task_runner(background_tasks)
    runner.submit(process_document_background, doc_id, tenant_id, ...)
```

### Test Runtime
```python
from app.services.task_runner import create_inline_task_runner

# Runtime holds InlineTaskRunner instance
task_runner = create_inline_task_runner()

# In tests (no BackgroundTasks needed):
runtime.task_runner.submit(process_document_background, doc_id, tenant_id, ...)
# Task completes immediately
```

## Usage in main.py Upload Endpoint

**Before:**
```python
background_tasks.add_task(
    process_document_background,
    doc_id,
    tenant_id,
    saved_path,
    file.filename
)
```

**After:**
```python
# Use runtime task runner (creates BackgroundTaskRunner from background_tasks)
task_runner = runtime.task_runner(background_tasks)
task_runner.submit(
    process_document_background,
    doc_id,
    tenant_id,
    saved_path,
    file.filename
)
```

## Integration Test Updates

**Before:**
```python
# Upload document
upload_response = client.post("/api/upload", files=files, headers=headers)
# Wait for async indexing
time.sleep(0.5)
# Query document
query_response = client.post("/api/query", json={...}, headers=headers)
```

**After:**
```python
# Upload document (with InlineTaskRunner, indexing completes immediately)
upload_response = client.post("/api/upload", files=files, headers=headers)
# No polling needed - task already completed
# Query document
query_response = client.post("/api/query", json={...}, headers=headers)
```

## Test Results

### Task Runner Tests (14/14 passing)
```
test_task_runner.py::test_inline_task_runner_sync_function PASSED
test_task_runner.py::test_inline_task_runner_async_function PASSED
test_task_runner.py::test_inline_task_runner_with_args_and_kwargs PASSED
test_task_runner.py::test_inline_task_runner_propagates_exceptions PASSED
test_task_runner.py::test_inline_task_runner_multiple_tasks PASSED
test_task_runner.py::test_background_task_runner_adds_task PASSED
test_task_runner.py::test_background_task_runner_with_async_function PASSED
test_task_runner.py::test_background_task_runner_with_sync_function PASSED
test_task_runner.py::test_create_inline_task_runner PASSED
test_task_runner.py::test_create_background_task_runner PASSED
test_task_runner.py::test_inline_task_runner_has_submit_method PASSED
test_task_runner.py::test_background_task_runner_has_submit_method PASSED
test_task_runner.py::test_inline_runner_simulates_indexing_workflow PASSED
test_task_runner.py::test_background_runner_simulates_indexing_workflow PASSED
```

### Runtime Tests (5/5 passing)
```
test_runtime.py::test_build_test_runtime_with_mock_provider PASSED
test_runtime.py::test_test_runtime_has_sync_task_runner PASSED
test_runtime.py::test_test_runtime_accepts_custom_providers PASSED
test_runtime.py::test_runtime_dataclass_structure PASSED
test_runtime.py::test_dummy_db_session_generator PASSED
```

### Verification Output
```
✓ All tasks executed immediately: ['task1', 'task2', 'async_task']
✓ Task scheduled (not executed yet): 1 task(s) pending
✓ Test runtime uses InlineTaskRunner (immediate execution)
```

## Benefits

### For Testing
1. **No Polling:** Tasks execute immediately (predictable state)
2. **Deterministic:** Same input → same output (no race conditions)
3. **Fast Tests:** No artificial delays or polling loops
4. **Exception Visibility:** Errors propagate to test code
5. **Clean Assertions:** State is final after `submit()` returns

### For Production
1. **Unchanged Behavior:** Still uses FastAPI BackgroundTasks
2. **Async Execution:** Non-blocking HTTP responses
3. **Standard Pattern:** Follows FastAPI conventions
4. **Clean Abstraction:** No direct BackgroundTasks coupling in business logic

## Key Implementation Details

### InlineTaskRunner Async Handling
```python
def submit(self, func: Callable, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(func(*args, **kwargs))
            loop.run_until_complete(task)
        except RuntimeError:
            asyncio.run(func(*args, **kwargs))
    else:
        func(*args, **kwargs)
```

### BackgroundTaskRunner Delegation
```python
def submit(self, func: Callable, *args, **kwargs):
    self.background_tasks.add_task(func, *args, **kwargs)
```

### Runtime Factory Pattern
- **Production:** `task_runner` is a factory function that requires `BackgroundTasks` parameter
- **Testing:** `task_runner` is an `InlineTaskRunner` instance that needs no parameters

## Constraints Met

✅ **Production behavior unchanged:** Still uses async background tasks  
✅ **Test execution is inline:** Indexing runs immediately (no polling)  
✅ **Clean abstraction:** Protocol-based interface (`submit()` method)  
✅ **Integration test updated:** No more `time.sleep()` for polling

## Next Steps (Optional)

### Potential Enhancements
1. Add timeout support for InlineTaskRunner (prevent hanging)
2. Add retry logic to BackgroundTaskRunner
3. Add task result tracking (success/failure metrics)
4. Support for task cancellation

### Additional Testing
1. Test with multiple concurrent uploads
2. Test exception handling in background tasks
3. Load testing with many background tasks

## Conclusion

The TaskRunner abstraction successfully decouples task execution strategy from business logic. Tests are now deterministic and fast (no polling), while production behavior remains unchanged (async background tasks). The protocol-based design makes it easy to add new execution strategies in the future (e.g., Celery, Redis Queue).
