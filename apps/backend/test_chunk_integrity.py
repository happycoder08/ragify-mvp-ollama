"""
Test chunk integrity: verify that EMAIL SIGNATURE SETUP chunk contains both
the heading and critical details like "Arial" and "10pt".
"""
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_signature_chunk_integrity():
    """
    Query the chunk store to verify the EMAIL SIGNATURE SETUP chunk
    contains the heading along with format details (Arial, 10pt).
    """
    print("\n" + "="*70)
    print("CHUNK INTEGRITY TEST: Email Signature Section")
    print("="*70)
    
    # Login to get token
    login_resp = requests.post(f"{BASE_URL}/api/login", json={
        "username": "demo",
        "password": "demo123"
    })
    assert login_resp.status_code == 200, f"Login failed: {login_resp.status_code}"
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Query for email signature with debug=1 to get all selected chunks
    query_resp = requests.post(
        f"{BASE_URL}/api/query",
        headers=headers,
        json={
            "question": "How do I set up my email signature?",
            "mode": "full",
            "debug": 1
        },
        stream=True
    )
    assert query_resp.status_code == 200, f"Query failed: {query_resp.status_code}"
    
    # Parse NDJSON response - first line should be debug object
    chunks_data = None
    for line in query_resp.iter_lines():
        if line:
            try:
                data = json.loads(line.decode('utf-8'))
                if "debug" in data:
                    debug_obj = data["debug"]
                    if "selected_chunks" in debug_obj:
                        chunks_data = debug_obj["selected_chunks"]
                        break
            except json.JSONDecodeError as e:
                continue
    
    assert chunks_data is not None, "No debug chunks found in response"
    
    print(f"\nRetrieved {len(chunks_data)} chunks")
    
    # Print all chunks to see what we got
    for i, chunk in enumerate(chunks_data):
        print(f"\n--- Chunk {i} ---")
        print(f"  ID: {chunk.get('id', 'N/A')}")
        print(f"  Header: {chunk.get('header', 'N/A')[:80]}")
        print(f"  Snippet: {chunk.get('snippet', 'N/A')[:150]}")
    
    # Search for chunk(s) containing signature-related content in headers
    signature_chunks = []
    for chunk in chunks_data:
        header = chunk.get("header", "").lower()
        if "signature" in header:
            signature_chunks.append(chunk)
    
    print(f"\nFound {len(signature_chunks)} chunks with 'signature' in header")
    
    # For signature chunk, verify the full content in vectorstore
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='vectorstore'))
    coll = client.get_collection('documents_default')
    
    # Get ALL chunks and find the one with signature
    all_results = coll.get()
    all_docs = all_results.get('documents', [])
    all_ids = all_results.get('ids', [])
    
    sig_chunk_in_db = None
    sig_chunk_id = None
    for i, doc in enumerate(all_docs):
        if 'signature' in doc.lower():
            sig_chunk_in_db = doc
            sig_chunk_id = all_ids[i]
            break
    
    if sig_chunk_in_db:
        full_lower = sig_chunk_in_db.lower()
        has_signature = "signature" in full_lower
        has_arial = "arial" in full_lower
        has_10pt = "10pt" in full_lower or "10 pt" in full_lower
        
        print(f"\n  Found signature chunk in vectorstore:")
        print(f"  ID: {sig_chunk_id}")
        print(f"  Length: {len(sig_chunk_in_db)} chars")
        print(f"    - Has 'signature': {has_signature}")
        print(f"    - Has 'arial': {has_arial}")
        print(f"    - Has '10pt': {has_10pt}")
        
        if has_signature and has_arial and has_10pt:
            # Also verify it was retrieved in the query
            retrieved = any('signature' in c.get('header', '').lower() for c in chunks_data)
            if retrieved:
                print(f"\nPASS: Email signature chunk contains all keywords AND was retrieved by query")
                print(f"\nFull chunk content:\n{sig_chunk_in_db}")
                found_complete_chunk = True
            else:
                print(f"\nWARNING: Chunk has all keywords but was NOT in top 5 retrieved chunks")
                found_complete_chunk = False
        else:
            print(f"\nFAIL: Chunk missing keywords")
            found_complete_chunk = False
    else:
        print(f"\nFAIL: No chunk with 'signature' found in vectorstore")
        found_complete_chunk = False
    
    if not found_complete_chunk:
        print(f"\nFAIL: No chunk contains all three keywords: 'signature', 'arial', '10pt'")
        print("\nThis indicates the EMAIL SIGNATURE SETUP heading was chunked separately")
        print("from the format details. The section-aware chunking should keep them together.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("TEST PASSED")
    print("="*70)

if __name__ == "__main__":
    try:
        test_signature_chunk_integrity()
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
