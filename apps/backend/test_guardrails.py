"""
Test script for guardrails and rate limiting.
Demonstrates file validation, size limits, and rate limiting behavior.
"""
import requests
import json
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def get_token():
    """Login and get JWT token."""
    response = requests.post(
        f"{BASE_URL}/api/login",
        json={"username": "test", "password": "test123"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(response.text)
        sys.exit(1)

def get_guardrails(token):
    """Get tenant guardrail configuration."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/guardrails", headers=headers)
    if response.status_code == 200:
        config = response.json()
        print("\n📋 Guardrail Configuration:")
        print(f"  Max file size: {config['max_file_size_mb']} MB")
        print(f"  Max files per request: {config['max_files_per_request']}")
        print(f"  Allowed extensions: {', '.join(config['allowed_extensions'])}")
        print(f"  Max requests/minute: {config['max_requests_per_minute']}")
        print(f"  Max requests/hour: {config['max_requests_per_hour']}")
        print(f"  Max upload MB/hour: {config['max_upload_mb_per_hour']}")
        print(f"  LLM timeout: {config['llm_timeout_seconds']}s")
        print(f"  Upload timeout: {config['upload_timeout_seconds']}s")
        return config
    else:
        print(f"❌ Failed to get guardrails: {response.status_code}")
        return None

def get_rate_limit_status(token):
    """Get current rate limit usage."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/rate-limit-status", headers=headers)
    if response.status_code == 200:
        usage = response.json()
        print("\n📊 Rate Limit Status:")
        print(f"  Requests last minute: {usage['requests_last_minute']}/{usage['limits']['max_requests_per_minute']}")
        print(f"  Requests last hour: {usage['requests_last_hour']}/{usage['limits']['max_requests_per_hour']}")
        print(f"  Upload MB last hour: {usage['uploads_mb_last_hour']}/{usage['limits']['max_upload_mb_per_hour']}")
        return usage
    else:
        print(f"❌ Failed to get rate limit status: {response.status_code}")
        return None

def test_invalid_extension(token):
    """Test uploading a file with invalid extension."""
    print("\n[TEST 1] Invalid file extension")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a fake .exe file
    files = {"files": ("malware.exe", b"fake content", "application/octet-stream")}
    
    response = requests.post(
        f"{BASE_URL}/api/upload",
        headers=headers,
        files=files
    )
    
    if response.status_code == 400:
        print(f"✅ Correctly rejected: {response.json()['detail']}")
        return True
    else:
        print(f"❌ Should have rejected invalid extension (got {response.status_code})")
        return False

def test_file_too_large(token, max_size_mb):
    """Test uploading a file that's too large."""
    print(f"\n[TEST 2] File too large (>{max_size_mb} MB)")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a file larger than the limit
    large_content = b"X" * (max_size_mb * 1024 * 1024 + 1000)
    files = {"files": ("large.txt", large_content, "text/plain")}
    
    response = requests.post(
        f"{BASE_URL}/api/upload",
        headers=headers,
        files=files
    )
    
    if response.status_code == 413:
        print(f"✅ Correctly rejected: {response.json()['detail']}")
        return True
    else:
        print(f"❌ Should have rejected large file (got {response.status_code})")
        return False

def test_too_many_files(token, max_files):
    """Test uploading too many files at once."""
    print(f"\n[TEST 3] Too many files (>{max_files})")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create more files than the limit
    files = [
        ("files", (f"file{i}.txt", b"content", "text/plain"))
        for i in range(max_files + 1)
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/upload",
        headers=headers,
        files=files
    )
    
    if response.status_code == 400:
        print(f"✅ Correctly rejected: {response.json()['detail']}")
        return True
    else:
        print(f"❌ Should have rejected too many files (got {response.status_code})")
        return False

def test_rate_limiting(token, max_per_minute):
    """Test rate limiting by making many requests."""
    print(f"\n[TEST 4] Rate limiting ({max_per_minute} requests/minute)")
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"Making {max_per_minute + 2} rapid requests...")
    
    for i in range(max_per_minute + 2):
        response = requests.post(
            f"{BASE_URL}/api/query",
            headers=headers,
            json={"question": f"Test query {i}", "mode": "fast"}
        )
        
        if response.status_code == 429:
            print(f"✅ Rate limit triggered after {i} requests: {response.json()['detail']}")
            return True
        elif response.status_code != 200:
            print(f"⚠️ Request {i} failed with {response.status_code}: {response.text[:100]}")
        
        time.sleep(0.1)  # Small delay between requests
    
    print(f"⚠️ Rate limit not triggered after {max_per_minute + 2} requests (may need more)")
    return False

def test_valid_upload(token):
    """Test a valid upload to ensure guardrails don't block legitimate requests."""
    print("\n[TEST 5] Valid upload (should succeed)")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a valid small text file
    content = b"This is a test document for guardrail validation.\n" * 10
    files = {"files": ("test_valid.txt", content, "text/plain")}
    
    response = requests.post(
        f"{BASE_URL}/api/upload",
        headers=headers,
        files=files
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Upload succeeded: {result.get('indexed_chunks', 0)} chunks indexed")
        return True
    else:
        print(f"❌ Valid upload failed: {response.status_code}")
        print(response.text)
        return False

def main():
    print("=" * 70)
    print("Testing Guardrails and Rate Limiting")
    print("=" * 70)
    
    # Login
    print("\n[SETUP] Login")
    token = get_token()
    
    # Get guardrail configuration
    config = get_guardrails(token)
    if not config:
        print("❌ Cannot proceed without guardrail config")
        return
    
    # Get initial rate limit status
    get_rate_limit_status(token)
    
    # Run tests
    results = []
    
    results.append(("Invalid extension", test_invalid_extension(token)))
    results.append(("File too large", test_file_too_large(token, config["max_file_size_mb"])))
    results.append(("Too many files", test_too_many_files(token, config["max_files_per_request"])))
    results.append(("Valid upload", test_valid_upload(token)))
    results.append(("Rate limiting", test_rate_limiting(token, config["max_requests_per_minute"])))
    
    # Show final rate limit status
    get_rate_limit_status(token)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All guardrail tests passed!")
    else:
        print("⚠️ Some tests failed. Check configuration and implementation.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
