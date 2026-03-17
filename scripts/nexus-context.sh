#!/usr/bin/env bash
# Fetch Nexus project context for Claude Code session injection
# Falls back silently if Nexus isn't running
PROJECT_PATH="${1:-$(pwd)}"
NEXUS_URL="${NEXUS_URL:-http://localhost:8000}"
response=$(curl -s --max-time 3 "${NEXUS_URL}/context/${PROJECT_PATH}" 2>/dev/null)
[ $? -eq 0 ] && [ -n "$response" ] && \
    echo "$response" | python -c "import sys,json;print(json.load(sys.stdin).get('context_block',''))" 2>/dev/null
