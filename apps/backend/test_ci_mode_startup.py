"""
Integration test: Verify app starts successfully in CI mode.

Tests that the FastAPI app can boot without Ollama/OpenAI when CI=true.
"""

import os
import pytest
from fastapi.testclient import TestClient


def test_app_boots_in_ci_mode():
    """Test that the app starts successfully in CI mode without external dependencies."""
    # Save original environment
    original_ci = os.environ.get("CI")
    original_llm = os.environ.get("LLM_PROVIDER")
    original_embedding = os.environ.get("EMBEDDING_PROVIDER")
    
    try:
        # Enable CI mode
        os.environ["CI"] = "true"
        
        # Clear any provider overrides
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        if "EMBEDDING_PROVIDER" in os.environ:
            del os.environ["EMBEDDING_PROVIDER"]
        
        # Import app (this triggers startup event)
        from main import app
        
        # Create test client (simulates app startup)
        with TestClient(app) as client:
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "ok"
            assert data["mock_mode"] is True, "CI mode should enable mock mode"
            
            print("✓ App boots successfully in CI mode")
            print(f"  Health check: {data}")
            
    finally:
        # Restore original environment
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


def test_app_boots_with_app_mode_ci():
    """Test that the app starts with APP_MODE=ci."""
    # Save original environment
    original_app_mode = os.environ.get("APP_MODE")
    original_llm = os.environ.get("LLM_PROVIDER")
    
    try:
        # Enable CI mode via APP_MODE
        os.environ["APP_MODE"] = "ci"
        
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        
        # Import app
        from main import app
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["mock_mode"] is True
            
            print("✓ App boots successfully with APP_MODE=ci")
            
    finally:
        if original_app_mode:
            os.environ["APP_MODE"] = original_app_mode
        elif "APP_MODE" in os.environ:
            del os.environ["APP_MODE"]
        
        if original_llm:
            os.environ["LLM_PROVIDER"] = original_llm
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
