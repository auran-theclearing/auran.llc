#!/usr/bin/env python3
"""Write the correct API key from Secrets Manager into .env"""

import json

import boto3

sm = boto3.client("secretsmanager")
key = json.loads(sm.get_secret_value(SecretId="auran/anthropic-api-key")["SecretString"])["api_key"]

with open("/opt/auran-chat/.env") as f:
    lines = f.readlines()

with open("/opt/auran-chat/.env", "w") as f:
    for line in lines:
        if line.startswith("ANTHROPIC_API_KEY="):
            f.write(f"ANTHROPIC_API_KEY={key}\n")
        else:
            f.write(line)

print(f"Wrote key: len={len(key)} starts={key[:15]} ends={key[-10:]}")
