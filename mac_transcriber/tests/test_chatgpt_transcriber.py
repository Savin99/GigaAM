import asyncio
import json

import httpx
import pytest

from chatgpt_transcriber.config import Settings, load_settings
from chatgpt_transcriber.server import build_server
from chatgpt_transcriber.service_client import (
    GigaAMServiceClient,
    OpenAIFile,
    is_transcript_ready,
    job_id_for_file_id,
    public_status,
)
from mac_transcriber.chatgpt_auth import derive_chatgpt_bridge_api_key


def _settings(tmp_path):
    return Settings(
        service_url="http://gigaam.private",
        bridge_api_key="test-key",
        max_batch_files=10,
        max_file_bytes=1024,
        download_timeout_seconds=10,
        service_timeout_seconds=10,
        temp_root=tmp_path / "tmp",
        allowed_download_host_suffixes=("oaiusercontent.com",),
        host="127.0.0.1",
        port=18110,
    )


def test_batch_tool_declares_chatgpt_file_array(tmp_path):
    server = build_server(_settings(tmp_path))
    tool = server._tool_manager._tools["transcribe_audio_batch"]

    assert tool.meta == {  # ChatGPT receives this as MCP tool `_meta`.
        "openai/fileParams": ["audio_files"],
        "openai/toolInvocation/invoking": "Queuing audio for GigaAM transcription…",
        "openai/toolInvocation/invoked": "Audio queued for GigaAM transcription",
    }
    params = tool.parameters
    assert params["properties"]["audio_files"]["type"] == "array"
    assert params["properties"]["audio_files"]["maxItems"] == 10
    file_schema = params["$defs"]["OpenAIFile"]
    assert file_schema["required"] == ["download_url", "file_id"]
    assert set(file_schema["properties"]) == {
        "download_url",
        "file_id",
        "mime_type",
        "file_name",
    }


def test_submit_streams_download_then_creates_transcript_only_job(tmp_path):
    settings = _settings(tmp_path)
    file = OpenAIFile(
        download_url="https://files.oaiusercontent.com/audio-1",
        file_id="file_audio_1",
        mime_type="audio/mp4",
        file_name="Диктофон.m4a",
    )
    expected_job_id = job_id_for_file_id(file.file_id)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "gigaam.private" and request.method == "GET":
            return httpx.Response(404, json={"detail": "Meeting not found"})
        if request.url.host == "files.oaiusercontent.com":
            return httpx.Response(200, content=b"audio-bytes")
        if request.url.host == "gigaam.private" and request.method == "POST":
            assert request.url.path == "/chatgpt/meetings"
            assert b'transcript_only' not in request.content
            assert b'processing_mode' not in request.content
            assert expected_job_id.encode() in request.content
            return httpx.Response(200, json={"id": expected_job_id, "status": "uploaded"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = GigaAMServiceClient(settings, transport=httpx.MockTransport(handler))
    submitted = asyncio.run(client.submit_audio_file(file))

    assert submitted.job_id == expected_job_id
    assert submitted.file_name == "Диктофон.m4a"
    assert submitted.reused_existing_job is False
    assert [request.method for request in requests] == ["GET", "GET", "POST"]
    assert requests[1].url.host == "files.oaiusercontent.com"
    assert requests[2].headers["authorization"] == "Bearer test-key"
    assert requests[2].url.path == "/chatgpt/meetings"


def test_bridge_settings_derive_an_internal_bearer_from_tunnel_key(tmp_path, monkeypatch):
    env_file = tmp_path / "tunnel.env"
    control_plane_key = "test-control-plane-key"
    env_file.write_text(
        "IGNORED_VALUE=must-not-be-loaded\n"
        f"CONTROL_PLANE_API_KEY={control_plane_key}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHATGPT_TRANSCRIBER_ENV_FILE", str(env_file))
    monkeypatch.setenv("MAC_TRANSCRIBER_API_KEY", "operator-key")

    settings = load_settings()

    assert settings.bridge_api_key == derive_chatgpt_bridge_api_key(control_plane_key)
    assert settings.bridge_api_key != control_plane_key
    assert settings.bridge_api_key != "operator-key"


def test_bridge_settings_fail_fast_without_tunnel_key(tmp_path, monkeypatch):
    env_file = tmp_path / "empty.env"
    env_file.write_text("OTHER_KEY=value\n", encoding="utf-8")
    monkeypatch.setenv("CHATGPT_TRANSCRIBER_ENV_FILE", str(env_file))

    with pytest.raises(ValueError, match="bridge credential is missing"):
        load_settings()


def test_status_exposes_ready_transcript_without_traceback():
    raw_status = {
        "id": "job-1",
        "status": "blocked_on_ai",
        "phase": "blocked_on_ai",
        "progress": 0.9,
        "message": "AI report unavailable",
        "traceback": "private stack trace",
        "artifacts": ["transcript_md", "transcript_json"],
        "metadata": {"source_filename": "voice.m4a"},
        "result": {"segments": 42},
    }

    assert is_transcript_ready(raw_status)
    assert public_status(raw_status) == {
        "job_id": "job-1",
        "state": "transcript_ready",
        "backend_state": "blocked_on_ai",
        "phase": "blocked_on_ai",
        "progress": 0.9,
        "message": "AI report unavailable",
        "transcript_ready": True,
        "source_filename": "voice.m4a",
        "segments": 42,
    }


def test_transcript_only_artifacts_skip_report_and_memory(tmp_path):
    from mac_transcriber import asr

    artifacts = tmp_path / "artifacts"
    asr.write_artifacts(
        output_dir=artifacts,
        meeting_id="chatgpt-job",
        title="Диктофон",
        source_filename="voice.m4a",
        model_name="v3_e2e_rnnt",
        tracks=[asr.TrackSpec(path=tmp_path / "voice.m4a", speaker="Speaker")],
        segments=[
            asr.Segment(
                speaker="Speaker",
                track="voice.m4a",
                start=0,
                end=1.5,
                text="Тестовая расшифровка.",
            )
        ],
        elapsed_seconds=2.5,
        include_report=False,
    )

    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    assert (artifacts / "transcript.md").exists()
    assert (artifacts / "transcript.json").exists()
    assert not (artifacts / "report.md").exists()
    assert not (artifacts / "manifest.json").exists()
    assert summary["report_status"] == "not_requested"
