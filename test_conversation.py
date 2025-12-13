"""
Test script for conversation support.
Demonstrates creating conversations, adding messages, and querying with context.
"""
import requests
import json
import sys

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

def create_conversation(token, title):
    """Create a new conversation."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{BASE_URL}/api/conversations",
        headers=headers,
        json={"title": title}
    )
    if response.status_code == 200:
        conv = response.json()
        print(f"✅ Created conversation: {conv['title']} (ID: {conv['id']})")
        return conv
    else:
        print(f"❌ Failed to create conversation: {response.status_code}")
        print(response.text)
        return None

def list_conversations(token):
    """List all conversations."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/conversations", headers=headers)
    if response.status_code == 200:
        convs = response.json()
        print(f"\n📋 Found {len(convs)} conversations:")
        for conv in convs:
            print(f"  - {conv['title']} (ID: {conv['id']}, Messages: {conv['message_count']})")
        return convs
    else:
        print(f"❌ Failed to list conversations: {response.status_code}")
        return []

def get_conversation(token, conv_id):
    """Get conversation with messages."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/conversations/{conv_id}", headers=headers)
    if response.status_code == 200:
        conv = response.json()
        print(f"\n💬 Conversation: {conv['title']}")
        print(f"Messages: {len(conv.get('messages', []))}")
        for msg in conv.get("messages", []):
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  [{role}] {content}")
        return conv
    else:
        print(f"❌ Failed to get conversation: {response.status_code}")
        return None

def query_with_conversation(token, question, conversation_id=None):
    """Query with optional conversation context."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "question": question,
        "mode": "fast"
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    print(f"\n🤔 Query: {question}")
    if conversation_id:
        print(f"   (with conversation context: {conversation_id})")
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        headers=headers,
        json=payload,
        stream=True
    )
    
    if response.status_code == 200:
        print("🤖 Answer: ", end="", flush=True)
        full_answer = ""
        sources = []
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "token" in data:
                    print(data["token"], end="", flush=True)
                    full_answer += data["token"]
                elif "sources" in data:
                    sources = data["sources"]
        
        print()  # Newline
        if sources:
            print(f"📚 Sources: {', '.join(sources)}")
        
        return full_answer, sources
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(response.text)
        return None, []

def delete_conversation(token, conv_id):
    """Delete a conversation."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(f"{BASE_URL}/api/conversations/{conv_id}", headers=headers)
    if response.status_code == 200:
        print(f"✅ Deleted conversation {conv_id}")
        return True
    else:
        print(f"❌ Failed to delete conversation: {response.status_code}")
        return False

def main():
    print("=" * 60)
    print("Testing Conversation Support")
    print("=" * 60)
    
    # Login
    print("\n[1] Login")
    token = get_token()
    
    # List existing conversations
    print("\n[2] List existing conversations")
    list_conversations(token)
    
    # Create a new conversation
    print("\n[3] Create new conversation")
    conv = create_conversation(token, "Test Chat - Company Policies")
    if not conv:
        print("❌ Cannot proceed without conversation")
        return
    
    conv_id = conv["id"]
    
    # First query - no context yet
    print("\n[4] First query (establishing context)")
    query_with_conversation(
        token,
        "What are the remote work policies?",
        conversation_id=conv_id
    )
    
    # Second query - with context from first
    print("\n[5] Second query (using conversation context)")
    query_with_conversation(
        token,
        "Are there any exceptions to these policies?",
        conversation_id=conv_id
    )
    
    # Third query - testing context continuity
    print("\n[6] Third query (testing multi-turn context)")
    query_with_conversation(
        token,
        "How do I request an exception?",
        conversation_id=conv_id
    )
    
    # Get conversation with all messages
    print("\n[7] Get conversation with messages")
    get_conversation(token, conv_id)
    
    # List conversations again (should show message count)
    print("\n[8] List conversations (verify message count)")
    list_conversations(token)
    
    # Clean up
    print("\n[9] Clean up (delete conversation)")
    response = input("Delete test conversation? (y/n): ").lower()
    if response == 'y':
        delete_conversation(token, conv_id)
        list_conversations(token)
    
    print("\n" + "=" * 60)
    print("✅ Conversation test complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
