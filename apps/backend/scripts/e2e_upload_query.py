"""Live end-to-end upload and query check.

Run from the repository root while the backend is running:
    python apps/backend/scripts/e2e_upload_query.py

For a deterministic local check, start the backend with ``LLM_PROVIDER=mock``.
"""

from __future__ import annotations

import argparse
import sys
import time

import requests


SAMPLE_TEXT = """RAGify integration test document

The onboarding team meets in Room 42 at 10:30 AM on the first day.
The support contact is integration@example.com.
"""


def wait_for_indexed(client: requests.Session, base_url: str, document_id: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        response = client.get(f"{base_url}/api/documents/{document_id}/status")
        response.raise_for_status()
        document = response.json()
        last_status = document.get("status")
        if last_status == "indexed":
            return document
        if last_status == "failed":
            raise RuntimeError(f"Document indexing failed: {document.get('error_message')}")
        time.sleep(1)
    raise TimeoutError(f"Document {document_id} did not reach indexed status; last status={last_status!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default="demo")
    parser.add_argument("--password", default="demo123")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    with requests.Session() as client:
        login = client.post(
            f"{base_url}/api/login",
            json={"username": args.username, "password": args.password},
        )
        login.raise_for_status()
        client.headers["Authorization"] = f"Bearer {login.json()['access_token']}"

        upload = client.post(
            f"{base_url}/api/upload",
            files={"files": ("e2e_integration.txt", SAMPLE_TEXT, "text/plain")},
        )
        upload.raise_for_status()
        upload_payload = upload.json()
        documents = upload_payload.get("documents", [])
        if not documents or not documents[0].get("id"):
            raise AssertionError(f"Upload did not return a document ID: {upload_payload}")

        document_id = documents[0]["id"]
        indexed = wait_for_indexed(client, base_url, document_id, args.timeout)

        query = client.post(
            f"{base_url}/api/query",
            json={
                "question": "Where does the onboarding team meet and at what time?",
                "doc_ids": [document_id],
                "mode": "full",
                "debug": 1,
                "stream": False,
            },
        )
        query.raise_for_status()
        payload = query.json()
        if payload.get("refused"):
            raise AssertionError(f"Query unexpectedly refused: {payload}")
        if not payload.get("sources"):
            raise AssertionError(f"Query returned no sources: {payload}")
        if not payload.get("evidence"):
            raise AssertionError(f"Query returned no evidence: {payload}")

        print(f"PASS: document {document_id} reached {indexed['status']}")
        print(f"PASS: query returned {len(payload['sources'])} source(s) and {len(payload['evidence'])} evidence item(s)")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"FAIL: backend request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
