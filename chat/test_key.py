#!/usr/bin/env python3
"""Quick test: does the API key in .env actually work?"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
key = os.getenv("ANTHROPIC_API_KEY", "")
print(f"Key length: {len(key)}")
print(f"Key starts: {key[:15]}")
print(f"Key ends: {key[-10:]}")

import urllib.request

req = urllib.request.Request(
    "https://api.anthropic.com/v1/messages",
    data=json.dumps(
        {"model": "claude-sonnet-4-6", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
    ).encode(),
    headers={
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
)
try:
    resp = urllib.request.urlopen(req)
    print(f"SUCCESS: {resp.read().decode()[:100]}")
except Exception as e:
    print(f"FAILED: {e}")
    if hasattr(e, "read"):
        print(e.read().decode()[:200])
