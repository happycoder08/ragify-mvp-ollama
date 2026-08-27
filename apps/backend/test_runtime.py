"""
Unit tests for AppRuntime dependency container.

Verifies that runtime can be built with different configurations,
especially for CI/testing scenarios.
"""

import os
import pytest
from app.runtime import AppRuntime, build_test_runtime, build_runtime_from_env


def test_build_test_runtime_with_mock_provider():
    """
    Test that AppRuntime can be built for testing with mock LLM provider
    and without HTTP client.
    
    This validates the CI testing scenario where no external dependencies
    (Ollama, OpenAI, PostgreSQL) are available.
    """
    # Set environment to use mock provider
    original_llm = os.environ.get("LLM_PROVIDER")
    original_embed = os.environ.get("EMBEDDING_PROVIDER")
    
    try:
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
        
        # Build test runtime
        runtime = build_test_runtime()
        
        # Verify runtime configuration
        assert isinstance(runtime, AppRuntime)
        assert runtime.db_enabled == False, "Database should be disabled in test runtime"
        assert runtime.http_client is None, "HTTP client should be None for mock providers"
        assert runtime.llm_provider is not None, "LLM provider should be initialized"
        assert runtime.embedding_provider is not None, "Embedding provider should be initialized"
        assert runtime.task_runner is not None, "Task runner should be initialized"
        
        # Verify providers are mock implementations
        assert "Mock" in type(runtime.llm_provider).__name__, "Should use MockLLMProvider"
        
        print("✓ Test runtime built successfully with mock provider and no HTTP client")
        
    finally:
        # Restore original environment
        if original_llm:
            os.environ["LLM_PROVIDER"] = original_llm
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        
        if original_embed:
            os.environ["EMBEDDING_PROVIDER"] = original_embed
        elif "EMBEDDING_PROVIDER" in os.environ:
            del os.environ["EMBEDDING_PROVIDER"]


def test_test_runtime_has_sync_task_runner():
    """
    Verify that test runtime uses InlineTaskRunner for immediate execution.
    
    This ensures tasks run immediately in tests rather than being
    deferred to background.
    """
    from app.services.task_runner import InlineTaskRunner
    
    original_provider = os.environ.get("LLM_PROVIDER")
    
    try:
        os.environ["LLM_PROVIDER"] = "mock"
        runtime = build_test_runtime()
        
        # Task runner should be InlineTaskRunner instance
        assert isinstance(runtime.task_runner, InlineTaskRunner), \
            "Test runtime should use InlineTaskRunner"
        
        # Verify it has submit method
        assert hasattr(runtime.task_runner, "submit"), \
            "Task runner should have submit method"
        
        # Verify it executes immediately
        results = []
        runtime.task_runner.submit(lambda: results.append("executed"))
        assert results == ["executed"], \
            "InlineTaskRunner should execute tasks immediately"
        
        print("✓ Test runtime has InlineTaskRunner for immediate execution")
        
    finally:
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


def test_test_runtime_accepts_custom_providers():
    """
    Test that build_test_runtime accepts custom provider instances.
    
    This allows tests to inject specific mock implementations.
    """
    # Create a simple mock provider
    class CustomMockProvider:
        async def generate_stream(self, *args, **kwargs):
            yield "test"
    
    custom_llm = CustomMockProvider()
    custom_embed = CustomMockProvider()
    
    # Build runtime with custom providers
    runtime = build_test_runtime(
        llm_provider=custom_llm,
        embedding_provider=custom_embed
    )
    
    # Verify custom providers were used
    assert runtime.llm_provider is custom_llm, "Should use provided LLM provider"
    assert runtime.embedding_provider is custom_embed, "Should use provided embedding provider"
    
    print("✓ Test runtime accepts custom provider instances")


def test_runtime_dataclass_structure():
    """
    Verify AppRuntime has expected attributes.
    """
    original_provider = os.environ.get("LLM_PROVIDER")
    
    try:
        os.environ["LLM_PROVIDER"] = "mock"
        runtime = build_test_runtime()
        
        # Check all expected attributes exist
        required_attrs = [
            'db_enabled',
            'get_db_session',
            'http_client',
            'llm_provider',
            'embedding_provider',
            'task_runner'
        ]
        
        for attr in required_attrs:
            assert hasattr(runtime, attr), f"Runtime should have '{attr}' attribute"
        
        print("✓ AppRuntime has all required attributes")
        
    finally:
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


def test_dummy_db_session_generator():
    """
    Test that default test runtime provides a working db session generator.
    """
    import asyncio
    
    original_provider = os.environ.get("LLM_PROVIDER")
    
    try:
        os.environ["LLM_PROVIDER"] = "mock"
        runtime = build_test_runtime()
        
        # Should be able to call get_db_session as async context manager
        async def check_session():
            async with runtime.get_db_session() as session:
                # Default dummy session yields None
                assert session is None, "Default test session should yield None"
        
        # Run the async function
        asyncio.run(check_session())
        
        print("✓ Dummy database session generator works")
        
    finally:
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


if __name__ == "__main__":
    # Run tests
    test_build_test_runtime_with_mock_provider()
    test_test_runtime_has_sync_task_runner()
    test_test_runtime_accepts_custom_providers()
    test_runtime_dataclass_structure()
    test_dummy_db_session_generator()
    
    print("\n✓ All runtime tests passed!")
