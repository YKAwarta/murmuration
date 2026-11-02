#!/usr/bin/env sh
# Start the API with auto-reload from the repository root (the web app proxies /api to port 8081).
cd "$(dirname "$0")/.." && exec uvicorn api.main:app --host 127.0.0.1 --port 8081 --reload
