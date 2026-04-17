"""Quick smoke test against a running inference server."""
from __future__ import annotations

import os
import sys

import httpx


def main(url: str = "http://localhost:8000") -> int:
    token = os.environ.get("API_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi in three words."},
        ],
        "max_new_tokens": 32,
        "temperature": 0.0,
    }
    r = httpx.post(f"{url}/v1/generate", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    print(r.json())
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
