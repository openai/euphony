#!/usr/bin/env bash

set -euo pipefail

# Resolve the repository root from the script location so the script works even
# when it is launched from another directory.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the requested run mode up front. The default is a production-style
# local launch that only starts the backend server against the existing `dist/`
# build. Passing `dev` as the first positional argument, or setting MODE=dev,
# switches to the full hot-reload workflow with both backend and frontend.
MODE="${MODE:-${1:-prod}}"

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
  # Shut child processes down when the user presses Ctrl+C or when the script
  # exits for any other reason. `kill` can fail if a process already stopped,
  # so ignore those cases quietly.
  if [[ -n "${FRONTEND_PID}" ]]; then
    kill "${FRONTEND_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BACKEND_PID}" ]]; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}

normalize_mode() {
  local value="${1:-prod}"
  case "${value}" in
    prod|production)
      printf 'prod'
      ;;
    dev|development)
      printf 'dev'
      ;;
    *)
      printf 'Unsupported mode: %s\nExpected one of: prod, dev\n' "${value}" >&2
      exit 1
      ;;
  esac
}

wait_for_process_exit() {
  local pid="$1"
  local label="$2"
  local exit_code="0"

  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
    # Capture the child exit status explicitly so `set -e` does not terminate
    # the script before we can print which background task stopped.
    if wait "${pid}"; then
      exit_code="0"
    else
      exit_code="$?"
    fi
    printf '\n%s stopped with exit code %s.\n' "${label}" "${exit_code}" >&2
    exit "${exit_code}"
  fi
}

require_command uv

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

MODE="$(normalize_mode "${MODE}")"

# Production mode serves the already-built frontend from FastAPI, so warn
# early if the caller has not built the assets yet. This keeps the default
# startup fast while still making the missing prerequisite obvious.
if [[ "${MODE}" == "prod" ]] && [[ ! -f "${ROOT_DIR}/dist/index.html" ]]; then
  printf 'Missing built frontend assets in dist/. Run `npm run build` first or start with `./start.sh dev`.\n' >&2
  exit 1
fi

if [[ "${MODE}" == "dev" ]]; then
  require_command npm
fi

# Build the uvicorn argument list incrementally so production mode starts a
# plain backend process while development mode adds file watching cleanly.
BACKEND_COMMAND=(
  uv run python -m uvicorn fastapi-main:app
  --app-dir server
  --host "${BACKEND_HOST}"
  --port "${BACKEND_PORT}"
)

if [[ "${MODE}" == "dev" ]]; then
  BACKEND_COMMAND+=(--reload)
fi

printf 'Starting backend on http://%s:%s\n' "${BACKEND_HOST}" "${BACKEND_PORT}"
"${BACKEND_COMMAND[@]}" &
BACKEND_PID="$!"

if [[ "${MODE}" == "dev" ]]; then
  # Pass the backend URL into Vite so the frontend dev server still talks to
  # the matching API port when callers override BACKEND_PORT.
  printf 'Starting frontend dev server on http://%s:%s\n' "${FRONTEND_HOST}" "${FRONTEND_PORT}"
  VITE_EUPHONY_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/" \
    npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" &
  FRONTEND_PID="$!"
fi

printf '\n'
printf 'Mode:             %s\n' "${MODE}"
printf 'Euphony backend:  http://%s:%s\n' "${BACKEND_HOST}" "${BACKEND_PORT}"
printf 'Codex sessions:   http://%s:%s/sessions.html\n' "${BACKEND_HOST}" "${BACKEND_PORT}"
if [[ "${MODE}" == "dev" ]]; then
  printf 'Euphony frontend: http://%s:%s\n' "${FRONTEND_HOST}" "${FRONTEND_PORT}"
else
  printf 'Euphony frontend: served by backend from `dist/`\n'
fi
printf '\nPress Ctrl+C to stop all started processes.\n\n'

# `wait -n` is unavailable in the older Bash 3.2 that ships with macOS, so use
# a small polling loop that exits as soon as any child process stops. The
# subsequent `wait` call still captures the real exit status of that process.
while true; do
  wait_for_process_exit "${BACKEND_PID}" "Backend server"
  wait_for_process_exit "${FRONTEND_PID}" "Frontend dev server"

  sleep 1
done
