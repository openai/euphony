#!/usr/bin/env bash

set -euo pipefail

# Resolve the repository root from the script location so the script works even
# when it is launched from another directory.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Allow callers to override the ports without editing the script.
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8020}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

BACKEND_PID=""
FRONTEND_PID=""

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 1
  fi
}

cleanup() {
  # Shut both child processes down when the user presses Ctrl+C or when the
  # script exits for any other reason. `kill` can fail if a process already
  # stopped, so ignore those cases quietly.
  if [[ -n "${FRONTEND_PID}" ]]; then
    kill "${FRONTEND_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BACKEND_PID}" ]]; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}

require_command uv
require_command npm

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

printf 'Starting backend on http://%s:%s\n' "${BACKEND_HOST}" "${BACKEND_PORT}"
uv run python -m uvicorn fastapi-main:app \
  --app-dir server \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" &
BACKEND_PID="$!"

printf 'Starting frontend on http://%s:%s\n' "${FRONTEND_HOST}" "${FRONTEND_PORT}"
npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" &
FRONTEND_PID="$!"

printf '\n'
printf 'Euphony backend:  http://%s:%s\n' "${BACKEND_HOST}" "${BACKEND_PORT}"
printf 'Euphony frontend: http://%s:%s\n' "${FRONTEND_HOST}" "${FRONTEND_PORT}"
printf 'Codex sessions:   http://%s:%s/sessions.html\n' "${BACKEND_HOST}" "${BACKEND_PORT}"
printf '\nPress Ctrl+C to stop both processes.\n\n'

# `wait -n` is unavailable in the older Bash 3.2 that ships with macOS, so use
# a small polling loop that exits as soon as either child process stops. The
# subsequent `wait` call still captures the real exit status of that process.
while true; do
  if [[ -n "${BACKEND_PID}" ]] && ! kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
    wait "${BACKEND_PID}"
    exit $?
  fi

  if [[ -n "${FRONTEND_PID}" ]] && ! kill -0 "${FRONTEND_PID}" >/dev/null 2>&1; then
    wait "${FRONTEND_PID}"
    exit $?
  fi

  sleep 1
done
