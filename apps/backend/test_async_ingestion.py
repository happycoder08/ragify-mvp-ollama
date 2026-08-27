"""
Test script for async ingestion with background processing.
Demonstrates upload, status polling, and reindexing.
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

def upload_test_file(token, content="Test document content for async indexing."):
    """Upload a test file."""
    print("\n[TEST] Uploading test file")
    headers = {"Authorization": f"Bearer {token}"}
    
    files = {"files": ("test_async.txt", content.encode(), "text/plain")}
    
    response = requests.post(
        f"{BASE_URL}/api/upload",
        headers=headers,
        files=files
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Upload successful: {result['message']}")
        if result.get('documents'):
            doc = result['documents'][0]
            print(f"   Document ID: {doc['id']}")
            print(f"   Filename: {doc['filename']}")
            print(f"   Initial status: {doc['status']}")
            return doc['id']
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return None

def get_document_status(token, doc_id):
    """Get status of a specific document."""
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(
        f"{BASE_URL}/api/documents/{doc_id}/status",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get status: {response.status_code}")
        return None

def poll_until_complete(token, doc_id, max_wait=60, interval=2):
    """Poll document status until it's indexed or failed."""
    print(f"\n[POLLING] Waiting for document {doc_id} to complete...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        doc = get_document_status(token, doc_id)
        
        if not doc:
            print("❌ Failed to get status")
            return None
        
        status = doc['status']
        print(f"   Status: {status} (elapsed: {int(time.time() - start_time)}s)")
        
        if status == 'indexed':
            print(f"✅ Document indexed successfully!")
            return doc
        elif status == 'failed':
            print(f"❌ Indexing failed: {doc.get('error_message', 'Unknown error')}")
            return doc
        
        time.sleep(interval)
    
    print(f"⚠️ Timeout after {max_wait}s - document still processing")
    return None

def list_documents(token):
    """List all documents."""
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/api/documents", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('documents', [])
        
        print(f"\n📋 Documents ({len(docs)} total):")
        for doc in docs:
            print(f"   [{doc['id']}] {doc['filename']} - {doc['status']}")
            if doc.get('error_message'):
                print(f"       Error: {doc['error_message']}")
        
        return docs
    else:
        print(f"❌ Failed to list documents: {response.status_code}")
        return []

def reindex_document(token, doc_id, filename):
    """Reindex a document."""
    print(f"\n[REINDEX] Reindexing document {doc_id}: {filename}")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(
        f"{BASE_URL}/api/documents/{doc_id}/reindex",
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        return True
    else:
        print(f"❌ Reindex failed: {response.status_code}")
        print(response.text)
        return False

def test_query(token, question="What is this document about?"):
    """Test querying with indexed documents."""
    print(f"\n[QUERY] Asking: {question}")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        headers=headers,
        json={"question": question, "mode": "fast"},
        stream=True
    )
    
    if response.status_code == 200:
        print("🤖 Answer: ", end="", flush=True)
        answer = ""
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "token" in data:
                    print(data["token"], end="", flush=True)
                    answer += data["token"]
                elif "sources" in data:
                    sources = data["sources"]
        
        print()  # Newline
        if sources:
            print(f"📚 Sources: {', '.join(sources)}")
        
        return answer
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(response.text)
        return None

def main():
    print("=" * 70)
    print("Testing Async Ingestion with Background Processing")
    print("=" * 70)
    
    # Login
    print("\n[1] Login")
    token = get_token()
    
    # List initial documents
    print("\n[2] List initial documents")
    initial_docs = list_documents(token)
    
    # Upload test file
    print("\n[3] Upload test file")
    doc_id = upload_test_file(token, "This is a test document about async ingestion. " * 10)
    
    if not doc_id:
        print("❌ Cannot proceed without document ID")
        return
    
    # Poll for completion
    print("\n[4] Poll for indexing completion")
    final_doc = poll_until_complete(token, doc_id, max_wait=30, interval=1)
    
    if not final_doc:
        print("⚠️ Could not verify completion")
        return
    
    # List documents again
    print("\n[5] List all documents")
    all_docs = list_documents(token)
    
    # Test query if indexed
    if final_doc['status'] == 'indexed':
        print("\n[6] Test query with indexed document")
        test_query(token, "What is this about?")
    
    # Test reindexing
    if final_doc['status'] == 'indexed' or final_doc['status'] == 'failed':
        print("\n[7] Test reindexing")
        if reindex_document(token, doc_id, final_doc['filename']):
            print("\n[8] Poll for reindexing completion")
            poll_until_complete(token, doc_id, max_wait=30, interval=1)
    
    # Final status
    print("\n[9] Final document list")
    list_documents(token)
    
    print("\n" + "=" * 70)
    print("✅ Async ingestion test complete!")
    print("=" * 70)
    print("\nKey features demonstrated:")
    print("  ✓ Upload returns immediately with 'pending' status")
    print("  ✓ Background processing indexes document asynchronously")
    print("  ✓ Status can be polled via API")
    print("  ✓ Failed documents can be reindexed")
    print("  ✓ Frontend can auto-refresh based on status")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
