import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi.testclient import TestClient


def load_server_module():
    module_path = Path(__file__).resolve().parents[1] / "server" / "fastapi-main.py"
    module_name = f"euphony_fastapi_main_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_session(path: Path, events: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def test_codex_sessions_endpoints_list_filter_and_serve_file(
    monkeypatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / ".codex"
    sessions_dir = codex_home / "sessions"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    older_session_path = sessions_dir / "2026" / "04" / "23" / "older.jsonl"
    newer_session_path = sessions_dir / "2026" / "04" / "24" / "newer.jsonl"
    search_session_path = sessions_dir / "2026" / "04" / "25" / "searchable.jsonl"

    write_session(
        older_session_path,
        [
            {
                "timestamp": "2026-04-23T09:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "session-older",
                    "timestamp": "2026-04-23T09:00:00Z",
                    "cwd": "/tmp/older",
                },
            },
            {
                "timestamp": "2026-04-23T09:01:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Fix the broken dashboard layout",
                },
            },
            {
                "timestamp": "2026-04-23T09:02:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "Inspecting the dashboard styles.",
                },
            },
            {
                "timestamp": "2026-04-23T09:05:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Ship the dashboard patch",
                },
            },
            {
                "timestamp": "2026-04-23T09:06:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "The dashboard patch is ready. "
                    + ("A" * 400),
                },
            },
        ],
    )

    write_session(
        newer_session_path,
        [
            {
                "timestamp": "2026-04-24T12:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "session-newer",
                    "timestamp": "2026-04-24T12:00:00Z",
                    "cwd": "/tmp/newer",
                },
            },
            {
                "timestamp": "2026-04-24T12:01:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Draft release notes for the April ship",
                },
            },
            {
                "timestamp": "2026-04-24T12:03:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "Reviewing release-note highlights.",
                },
            },
        ],
    )

    write_session(
        search_session_path,
        [
            {
                "timestamp": "2026-04-25T07:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "session-searchable",
                    "timestamp": "2026-04-25T07:00:00Z",
                    "cwd": "/tmp/searchable",
                },
            },
            {
                "timestamp": "2026-04-25T07:01:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Investigate the nightly regression",
                },
            },
            {
                "timestamp": "2026-04-25T07:02:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "Triaging a telemetry anomaly in the overnight build.",
                },
            },
            {
                "timestamp": "2026-04-25T07:04:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Prepare the mitigation summary",
                },
            },
            {
                "timestamp": "2026-04-25T07:05:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "Mitigation summary drafted for the nightly regression.",
                },
            },
        ],
    )

    module = load_server_module()
    client = TestClient(module.fastapi_app)

    response = client.get("/codex-sessions/")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total"] == 3
    assert payload["matchedCount"] == 3
    assert [item["session_id"] for item in payload["data"]] == [
        "session-searchable",
        "session-newer",
        "session-older",
    ]
    assert payload["data"][2]["first_prompt"] == "Fix the broken dashboard layout"
    assert payload["data"][2]["last_prompt"] == "Ship the dashboard patch"
    assert payload["data"][2]["last_response"].startswith(
        "The dashboard patch is ready."
    )
    assert payload["data"][2]["last_response"].endswith("...")
    assert payload["data"][2]["last_response_time"] == "2026-04-23T09:06:00Z"

    filtered_response = client.get(
        "/codex-sessions/",
        params={"firstPromptFilter": "release", "lastPromptFilter": "april"},
    )
    assert filtered_response.status_code == 200
    filtered_payload = filtered_response.json()
    assert filtered_payload["matchedCount"] == 1
    assert filtered_payload["data"][0]["session_id"] == "session-newer"

    response_filtered_response = client.get(
        "/codex-sessions/",
        params={"responseFilter": "highlights"},
    )
    assert response_filtered_response.status_code == 200
    response_filtered_payload = response_filtered_response.json()
    assert response_filtered_payload["matchedCount"] == 1
    assert response_filtered_payload["data"][0]["session_id"] == "session-newer"

    exact_search_response = client.get(
        "/codex-sessions/",
        params={"searchQuery": "mitigation summary", "searchMode": "exact"},
    )
    assert exact_search_response.status_code == 200
    exact_search_payload = exact_search_response.json()
    assert exact_search_payload["matchedCount"] == 1
    assert exact_search_payload["data"][0]["session_id"] == "session-searchable"

    fuzzy_search_response = client.get(
        "/codex-sessions/",
        params={"searchQuery": "relese", "searchMode": "fuzzy"},
    )
    assert fuzzy_search_response.status_code == 200
    fuzzy_search_payload = fuzzy_search_response.json()
    assert fuzzy_search_payload["matchedCount"] == 1
    assert fuzzy_search_payload["data"][0]["session_id"] == "session-newer"

    full_text_search_response = client.get(
        "/codex-sessions/",
        params={"searchQuery": "telemetry anomaly", "searchMode": "full_text"},
    )
    assert full_text_search_response.status_code == 200
    full_text_search_payload = full_text_search_response.json()
    assert full_text_search_payload["matchedCount"] == 1
    assert full_text_search_payload["data"][0]["session_id"] == "session-searchable"

    file_response = client.get(
        "/codex-session-file/",
        params={"relative_path": "2026/04/23/older.jsonl"},
    )
    assert file_response.status_code == 200
    assert "session-older" in file_response.text
