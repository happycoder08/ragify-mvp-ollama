import requests
import sys

API_URL = "http://localhost:8000"
USERNAME = "demo"
PASSWORD = "demo123"
QUESTION = "What time do I arrive my first day"
TENANT_ID = "default"

def get_token():
    resp = requests.post(f"{API_URL}/api/login", json={"username": USERNAME, "password": PASSWORD})
    resp.raise_for_status()
    return resp.json()["access_token"]

def query_with_debug(token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "tenant_id": TENANT_ID,
        "question": QUESTION,
        "debug": 1,
        "stream": False
    }
    resp = requests.post(f"{API_URL}/api/query", json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


def main():
    try:
        token = get_token()
        result = query_with_debug(token)
        with open("debug.json", "w", encoding="utf-8") as f:
            import json
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Debug output written to debug.json")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
