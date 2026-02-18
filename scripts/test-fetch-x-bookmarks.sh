#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Load .env.local if present (gitignored; copy from your GitHub secrets)
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi

# Trim whitespace/newlines (same as workflow)
X_REFRESH_TOKEN=$(printf '%s' "${X_REFRESH_TOKEN:-}" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
X_CLIENT_ID=$(printf '%s' "${X_CLIENT_ID:-}" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
X_CLIENT_SECRET=$(printf '%s' "${X_CLIENT_SECRET:-}" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
X_ACCESS_TOKEN=$(printf '%s' "${X_ACCESS_TOKEN:-}" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

if [ -n "$X_REFRESH_TOKEN" ] && [ -n "$X_CLIENT_ID" ] && [ -n "$X_CLIENT_SECRET" ]; then
  echo "Refreshing X access token..."
  # macOS base64 doesn't have -w 0; strip newline for header
  basic=$(printf '%s' "${X_CLIENT_ID}:${X_CLIENT_SECRET}" | base64 | tr -d '\n')
  printf '%s' "$X_REFRESH_TOKEN" > /tmp/refresh_token.txt
  resp=$(curl -sS -X POST 'https://api.x.com/2/oauth2/token' \
    -H "Authorization: Basic $basic" \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode 'grant_type=refresh_token' \
    --data-urlencode 'refresh_token@/tmp/refresh_token.txt')
  token=$(echo "$resp" | jq -r '.access_token // empty')
  if [ -z "$token" ]; then
    echo "Failed to refresh token: $resp"
    exit 1
  fi
  export X_ACCESS_TOKEN="$token"
  echo "Refreshed access token."
else
  if [ -z "$X_ACCESS_TOKEN" ]; then
    echo "Error: Set X_ACCESS_TOKEN or (X_REFRESH_TOKEN + X_CLIENT_ID + X_CLIENT_SECRET) in .env.local or env."
    echo "Create .env.local with the same values as your GitHub Actions secrets and run again."
    exit 1
  fi
  echo "Using existing X_ACCESS_TOKEN."
fi

export X_USERNAME="${X_USERNAME:-gauri__gupta}"
export MAX_BOOKMARKS="${MAX_BOOKMARKS:-5}"
npm run fetch-x-bookmarks
