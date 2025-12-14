#!/usr/bin/env python
"""Test documents endpoint"""
import requests
import json
import time

time.sleep(2)

BASE_URL = 'http://localhost:8000'

# Login
login_resp = requests.post(f'{BASE_URL}/api/login', json={'username': 'demo', 'password': 'demo123'})
print(f'Login: {login_resp.status_code}')

if login_resp.status_code != 200:
    print(f'Login failed: {login_resp.text}')
    exit(1)

token = login_resp.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Get documents
docs_resp = requests.get(f'{BASE_URL}/api/documents', headers=headers)
print(f'Documents: {docs_resp.status_code}')
docs = docs_resp.json()

print(json.dumps(docs, indent=2))
