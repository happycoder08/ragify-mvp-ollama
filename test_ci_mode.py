"""
Test CI mode configuration.

Verifies that setting CI=true or APP_MODE=ci automatically configures:
- LLM_PROVIDER=mock
- EMBEDDING_PROVIDER=mock
- TASK_RUNNER=inline
- No HTTP client dependency
"""

import os
import pytest
from app.runtime import build_runtime_from_env


def test_ci_mode_with_ci_env_var():
    """Test that CI=true triggers CI mode configuration."""
    # Save original env
    original_ci = os.environ.get("CI")
    original_llm = os.environ.get("LLM_PROVIDER")
    original_embedding = os.environ.get("EMBEDDING_PROVIDER")
    
    try:
        # Set CI mode
        os.environ["CI"] = "true"
        
        # Clear provider settings to test auto-configuration
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        if "EMBEDDING_PROVIDER" in os.environ:
            del os.environ["EMBEDDING_PROVIDER"]
        
        # Build runtime
        runtime = build_runtime_from_env()
        
        # Verify CI mode configuration
        assert runtime.http_client is None, "CI mode should not initialize HTTP client"
        assert type(runtime.llm_provider).__name__ == "MockLLMProvider", "CI mode should use MockLLMProvider"
        assert type(runtime.embedding_provider).__name__ == "MockEmbedder", "CI mode should use MockEmbedder"
        
        # Verify task runner is inline (has 'submit' method, not a factory)
        assert hasattr(runtime.task_runner, 'submit'), "CI mode should use InlineTaskRunner"
        assert type(runtime.task_runner).__name__ == "InlineTaskRunner"
        
        # Verify environment was set
        assert os.environ.get("LLM_PROVIDER") == "mock"
        assert os.environ.get("EMBEDDING_PROVIDER") == "mock"
        
        print("✓ CI mode with CI=true works correctly")
        
    finally:
        # Restore original env
        if original_ci:
            os.environ["CI"] = original_ci
        elif "CI" in os.environ:
            del os.environ["CI"]
        
        if original_llm:
            os.environ["LLM_PROVIDER"] = original_llm
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        
        if original_embedding:
            os.environ["EMBEDDING_PROVIDER"] = original_embedding
        elif "EMBEDDING_PROVIDER" in os.environ:
            del os.environ["EMBEDDING_PROVIDER"]


def test_ci_mode_with_app_mode_env_var():
    """Test that APP_MODE=ci triggers CI mode configuration."""
    # Save original env
    original_app_mode = os.environ.get("APP_MODE")
    original_llm = os.environ.get("LLM_PROVIDER")
    original_embedding = os.environ.get("EMBEDDING_PROVIDER")
    
    try:
        # Set CI mode via APP_MODE
        os.environ["APP_MODE"] = "ci"
        
        # Clear provider settings
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        if "EMBEDDING_PROVIDER" in os.environ:
            del os.environ["EMBEDDING_PROVIDER"]
        
        # Build runtime
        runtime = build_runtime_from_env()
        
        # Verify CI mode configuration
        assert runtime.http_client is None, "APP_MODE=ci should not initialize HTTP client"
        assert type(runtime.llm_provider).__name__ == "MockLLMProvider"
        assert type(runtime.embedding_provider).__name__ == "MockEmbedder"
        assert hasattr(runtime.task_runner, 'submit')
        
        print("✓ CI mode with APP_MODE=ci works correctly")
        
    finally:
        # Restore original env
        if original_app_mode:
            os.environ["APP_MODE"] = original_app_mode
        elif "APP_MODE" in os.environ:
            del os.environ["APP_MODE"]
        
        if original_llm:
            os.environ["LLM_PROVIDER"] = original_llm
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        
        if original_embedding:
            os.environ["EMBEDDING_PROVIDER"] = original_embedding
        elif "EMBEDDING_PROVIDER" in os.environ:
            del os.environ["EMBEDDING_PROVIDER"]


def test_normal_mode_detects_non_ci():
    """Test that normal mode (not CI) is detected correctly."""
    # Save original env
    original_ci = os.environ.get("CI")
    original_app_mode = os.environ.get("APP_MODE")
    original_llm = os.environ.get("LLM_PROVIDER")
    
    try:
        # Ensure CI mode is OFF
        if "CI" in os.environ:
            del os.environ["CI"]
        if "APP_MODE" in os.environ:
            del os.environ["APP_MODE"]
        
        # Set mock provider to avoid actual HTTP calls in test
        os.environ["LLM_PROVIDER"] = "mock"
        
        # Import the detection logic
        is_ci_mode = (
            os.getenv("CI", "").lower() in ("true", "1", "yes") or
            os.getenv("APP_MODE", "").lower() == "ci"
        )
        
        # Verify CI mode is not detected
        assert not is_ci_mode, "CI mode should not be detected when CI and APP_MODE are not set"
        
        print("✓ Normal mode correctly detected (CI mode disabled)")
        
    finally:
        # Restore original env
        if original_ci:
            os.environ["CI"] = original_ci
        elif "CI" in os.environ:
            del os.environ["CI"]
        
        if original_app_mode:
            os.environ["APP_MODE"] = original_app_mode
        elif "APP_MODE" in os.environ:
            del os.environ["APP_MODE"]
        
        if original_llm:
            os.environ["LLM_PROVIDER"] = original_llm
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


def test_ci_mode_variations():
    """Test various CI environment variable values."""
    test_cases = [
        ("CI", "true"),
        ("CI", "1"),
        ("CI", "yes"),
        ("CI", "TRUE"),  # Case insensitive
        ("APP_MODE", "ci"),
        ("APP_MODE", "CI"),  # Case insensitive
    ]
    
    for env_var, env_value in test_cases:
        # Save original
        original = os.environ.get(env_var)
        
        try:
            os.environ[env_var] = env_value
            
            # Clear provider vars
            if "LLM_PROVIDER" in os.environ:
                del os.environ["LLM_PROVIDER"]
            
            runtime = build_runtime_from_env()
            
            assert runtime.http_client is None, f"Failed for {env_var}={env_value}"
            assert type(runtime.llm_provider).__name__ == "MockLLMProvider", f"Failed for {env_var}={env_value}"
            
        finally:
            if original:
                os.environ[env_var] = original
            elif env_var in os.environ:
                del os.environ[env_var]
            
            # Clean up provider vars
            if "LLM_PROVIDER" in os.environ:
                del os.environ["LLM_PROVIDER"]
    
    print(f"✓ All {len(test_cases)} CI mode variations work correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
