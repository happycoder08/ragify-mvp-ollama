import requests
import json

API_URL = "http://localhost:8000"
USERNAME = "demo"
PASSWORD = "demo123"
QUESTION = "What time should I arrive on my first day?"
TENANT_ID = "default"

def get_token():
    resp = requests.post(f"{API_URL}/api/login", json={"username": USERNAME, "password": PASSWORD})
    resp.raise_for_status()
    return resp.json()["access_token"]

def sse_events(resp):
    """
    Minimal SSE parser:
    yields dicts: {"event": <name>, "data": <string>}
    """
    event_name = None
    data_lines = []

    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()

        # blank line = dispatch event
        if line == "":
            if event_name and data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # ignore other SSE fields (id:, retry:)

def query_debug(token):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"tenant_id": TENANT_ID, "question": QUESTION, "debug": 1}

    with requests.post(f"{API_URL}/api/query", json=payload, headers=headers, stream=True) as resp:
        resp.raise_for_status()

        debug_obj = None
        final_obj = None

        for evt in sse_events(resp):
            if evt["event"] == "debug":
                debug_obj = json.loads(evt["data"])
            elif evt["event"] == "final":
                final_obj = json.loads(evt["data"])
                break  # final means done

        if not final_obj:
            raise RuntimeError("Did not receive a final SSE event.")

        return debug_obj, final_obj

def main():
    token = get_token()
    debug_obj, final_obj = query_debug(token)

    if debug_obj:
        with open("debug.json", "w", encoding="utf-8") as f:
            json.dump(debug_obj, f, indent=2, ensure_ascii=False)

    with open("final.json", "w", encoding="utf-8") as f:
        json.dump(final_obj, f, indent=2, ensure_ascii=False)

    print("Wrote debug.json and final.json")
    print("Final refused:", final_obj.get("refused"), "reason:", final_obj.get("refusal_reason"))

if __name__ == "__main__":
    main()
