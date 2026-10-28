import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from mac_transcriber.archive import sha256_file, write_meeting_manifest
from mac_transcriber import service
from mac_transcriber.reporting import ReportQuotaError, ReportUnavailableError


@pytest.fixture(autouse=True)
def _isolate_chatgpt_bridge_auth(monkeypatch):
    """Keep the developer's local tunnel credential out of ordinary unit tests."""

    monkeypatch.setattr(service, "CHATGPT_BRIDGE_API_KEY", "")


def test_process_meeting_blocks_on_quota_without_writing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)

    def raise_quota(**_kwargs):
        raise ReportQuotaError("OpenAI API returned HTTP 429: insufficient_quota")

    monkeypatch.setattr(service, "transcribe_meeting", raise_quota)

    meeting_dir = tmp_path / "meetings" / "m1"
    meeting_dir.mkdir(parents=True)
    service._process_meeting("m1")

    status = json.loads((meeting_dir / "status.json").read_text(encoding="utf-8"))
    # Нет денег -> пауза, а не failed и не completed; сырой отчёт не пишется.
    assert status["status"] == "blocked_on_quota"
    assert not (meeting_dir / "artifacts" / "report.md").exists()


def test_process_meeting_queues_when_ai_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)

    def raise_unavailable(**_kwargs):
        raise ReportUnavailableError("OpenAI API unavailable: Connection error.")

    monkeypatch.setattr(service, "transcribe_meeting", raise_unavailable)

    meeting_dir = tmp_path / "meetings" / "m2"
    meeting_dir.mkdir(parents=True)
    service._process_meeting("m2")

    status = json.loads((meeting_dir / "status.json").read_text(encoding="utf-8"))
    # API недоступен -> очередь, а не failed/completed; сырой local-отчёт не пишется.
    assert status["status"] == "blocked_on_ai"
    assert not (meeting_dir / "artifacts" / "report.md").exists()


def test_interrupted_meeting_ids_include_uploaded_and_processing_only(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    meetings = tmp_path / "meetings"
    for meeting_id, status in [
        ("uploaded", "uploaded"),
        ("processing", "processing"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("blocked", "blocked_on_ai"),
    ]:
        meeting_dir = meetings / meeting_id
        (meeting_dir / "input").mkdir(parents=True)
        (meeting_dir / "input" / "audio.m4a").write_bytes(b"audio")
        (meeting_dir / "status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )

    assert service._interrupted_meeting_ids() == ["processing", "uploaded"]


@pytest.mark.parametrize("enabled,transcript_only,recovers", [(True, False, True), (False, False, False), (True, True, False)])
def test_uncertain_scribe_submission_only_uses_opted_in_local_recovery(
    tmp_path, monkeypatch, enabled, transcript_only, recovers
):
    from mac_transcriber.scribe import ScribeReconciliationRequired

    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setenv("MAC_TRANSCRIBER_SCRIBE_LOCAL_RECOVERY", "1" if enabled else "0")
    meeting = tmp_path / "meetings" / "recovery"
    (meeting / "input").mkdir(parents=True)
    (meeting / "input/metadata.json").write_text(json.dumps({"transcript_only": transcript_only}))

    def unknown(**kwargs):
        raise ScribeReconciliationRequired("Cloud submission outcome unknown")

    recovered = []

    def local(path):
        recovered.append(path)
        service._write_status(path, "blocked_on_ai", segments_total=5, segments_done=5)

    monkeypatch.setattr(service, "transcribe_meeting", unknown)
    monkeypatch.setattr(service, "_recover_scribe_locally", local)
    service._process_meeting("recovery")
    status = json.loads((meeting / "status.json").read_text())
    assert bool(recovered) is recovers
    assert status["status"] == ("blocked_on_ai" if recovers else "failed")


def test_app_lifespan_starts_resume_worker(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            assert target is service._resume_interrupted_meetings
            assert name == "resume-interrupted-meetings"
            assert daemon is True

        def start(self):
            started.append(True)

    monkeypatch.delenv("MAC_TRANSCRIBER_RESUME_ON_STARTUP", raising=False)
    monkeypatch.setattr(service, "Thread", FakeThread)

    async def run_lifespan():
        async with service.app_lifespan(service.app):
            pass

    asyncio.run(run_lifespan())
    assert started == [True]


def test_app_lifespan_can_disable_resume_worker(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_RESUME_ON_STARTUP", "0")

    def fail_thread(**_kwargs):
        raise AssertionError("resume worker must stay disabled")

    monkeypatch.setattr(service, "Thread", fail_thread)

    async def run_lifespan():
        async with service.app_lifespan(service.app):
            pass

    asyncio.run(run_lifespan())


def test_regenerate_report_from_transcript_skips_asr(tmp_path, monkeypatch):
    from mac_transcriber import asr, reporting

    monkeypatch.setenv("MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL", " ")
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("MAC_TRANSCRIBER_DATABASE_URL", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_POSTGRES_DSN", raising=False)

    mdir = tmp_path / "meetings" / "m1"
    (mdir / "input").mkdir(parents=True)
    (mdir / "artifacts").mkdir()
    (mdir / "input" / "metadata.json").write_text(
        json.dumps(
            {"meeting_id": "m1", "title": "Sync", "source_filename": "audio.m4a"}
        ),
        encoding="utf-8",
    )
    (mdir / "artifacts" / "transcript.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": "S0001",
                    "start": 0.0,
                    "end": 5.0,
                    "speaker": "A",
                    "track": "a.m4a",
                    "text": "Договорились сделать отчёт.",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake(*, payload, api_key):
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "X",
                        "confidence": 0.9,
                        "rationale": "t",
                    },
                    "overview": "AI отчёт из готового транскрипта.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Решения"],
                            "rationale": "t",
                        }
                    ],
                    "timeline": [],
                    "decisions": [
                        {
                            "title": "D",
                            "text": "Договорились сделать отчёт.",
                            "citations": ["S0001"],
                        }
                    ],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake)

    artifacts = asr.regenerate_report_from_transcript(mdir)

    # Реальный AI-отчёт (не local), транскрипт не тронут.
    assert artifacts.generated_by != "local"
    assert (mdir / "artifacts" / "report.md").exists()
    transcript = json.loads(
        (mdir / "artifacts" / "transcript.json").read_text(encoding="utf-8")
    )
    assert transcript[0]["segment_id"] == "S0001"


def test_transcribe_segments_supports_transcribe_only_model(tmp_path, monkeypatch):
    from mac_transcriber import asr

    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"fake")
    segment = asr.Segment(
        start=0.0,
        end=1.0,
        speaker="SPEAKER_00",
        track="audio.m4a",
        wav=wav,
    )

    class TranscribeOnlyModel:
        def transcribe(self, path):
            assert path == str(wav)
            return "  Проверка   fallback. "

    monkeypatch.setattr(
        asr,
        "load_model",
        lambda *, model_name, cache_dir, device: TranscribeOnlyModel(),
    )

    result = asr.transcribe_segments(
        [segment],
        model_name="v3_e2e_rnnt",
        cache_dir="/tmp/cache",
        device="cpu",
        batch_size=4,
    )

    assert result[0].text == "Проверка fallback."
    assert result[0].wav is None


def test_load_local_env_file_overrides_report_model(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "MAC_TRANSCRIBER_REPORT_MODEL=openrouter:google/gemini-3.1-pro-preview",
                "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=openrouter:openai/gpt-5.5",
                "OPENROUTER_API_KEY=sk-or-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_MODEL", "gpt-5.5")
    # Регистрируем все ключи из файла через monkeypatch, чтобы teardown их
    # восстановил и тест не протекал в глобальный os.environ для соседних тестов.
    monkeypatch.setenv("MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL", "placeholder")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")

    service._load_local_env_file(env_file)

    assert (
        service.os.environ["MAC_TRANSCRIBER_REPORT_MODEL"]
        == "openrouter:google/gemini-3.1-pro-preview"
    )
    assert (
        service.os.environ["MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL"]
        == "openrouter:openai/gpt-5.5"
    )
    assert service.os.environ["OPENROUTER_API_KEY"] == "sk-or-secret"


def test_create_meeting_persists_form_title(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        data={"title": "Проектный синк"},
        files={"file": ("recording.m4a", b"fake audio", "audio/mp4")},
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    metadata = json.loads(
        (tmp_path / "meetings" / meeting_id / "input" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["title"] == "Проектный синк"


def test_create_meeting_persists_transcript_only_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        data={"transcript_only": "true"},
        files={"file": ("recording.m4a", b"fake audio", "audio/mp4")},
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    metadata = json.loads(
        (tmp_path / "meetings" / meeting_id / "input" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["transcript_only"] is True


def test_chatgpt_bridge_can_access_only_its_scoped_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "operator-token")
    monkeypatch.setattr(service, "CHATGPT_BRIDGE_API_KEY", "bridge-token")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)
    client = TestClient(service.app)
    operator_headers = {"Authorization": "Bearer operator-token"}
    bridge_headers = {"Authorization": "Bearer bridge-token"}

    operator_id = str(uuid.uuid4())
    operator_response = client.post(
        "/meetings",
        headers=operator_headers,
        data={"request_id": operator_id, "transcript_only": "false"},
        files={"file": ("operator.m4a", b"operator audio", "audio/mp4")},
    )
    assert operator_response.status_code == 200

    chatgpt_id = str(uuid.uuid4())
    bridge_response = client.post(
        "/chatgpt/meetings",
        headers=bridge_headers,
        data={"request_id": chatgpt_id, "title": "Диктофон"},
        files={"file": ("voice.m4a", b"chatgpt audio", "audio/mp4")},
    )
    assert bridge_response.status_code == 200
    metadata = json.loads(
        (tmp_path / "meetings" / chatgpt_id / "input" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["ingress"] == "chatgpt_mcp"
    assert metadata["language_code"] == "ru"
    assert metadata["processing_mode"] == "simple"
    assert metadata["transcript_only"] is True
    assert metadata["participants"] == []
    assert metadata["zoom_participant_tracks"] == []

    assert client.get("/meetings", headers=bridge_headers).status_code == 401
    assert (
        client.get(
            f"/chatgpt/meetings/{operator_id}/status", headers=bridge_headers
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/meetings/{operator_id}/status", headers=operator_headers
        ).status_code
        == 200
    )

    transcript_path = tmp_path / "meetings" / chatgpt_id / "artifacts" / "transcript.md"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text("# Расшифровка\n\nТекст", encoding="utf-8")
    (transcript_path.parent / "report.md").write_text(
        "private report", encoding="utf-8"
    )

    status = client.get(
        f"/chatgpt/meetings/{chatgpt_id}/status", headers=bridge_headers
    )
    assert status.status_code == 200
    assert status.json()["transcript_ready"] is True
    assert set(status.json()) == {
        "id",
        "status",
        "phase",
        "progress",
        "message",
        "source_filename",
        "segments",
        "transcript_ready",
    }
    transcript = client.get(
        f"/chatgpt/meetings/{chatgpt_id}/transcript", headers=bridge_headers
    )
    assert transcript.status_code == 200
    assert transcript.text == "# Расшифровка\n\nТекст"
    assert (
        client.get(
            f"/meetings/{chatgpt_id}/artifacts/report_md", headers=bridge_headers
        ).status_code
        == 401
    )

    retry = client.post(
        "/chatgpt/meetings",
        headers=bridge_headers,
        data={"request_id": chatgpt_id},
        files={"file": ("voice.m4a", b"second upload", "audio/mp4")},
    )
    assert retry.status_code == 200
    collision = client.post(
        "/chatgpt/meetings",
        headers=bridge_headers,
        data={"request_id": operator_id},
        files={"file": ("voice.m4a", b"collision", "audio/mp4")},
    )
    assert collision.status_code == 409


def test_process_meeting_skips_report_for_transcript_only_job(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    captured = {}

    def fake_transcribe(**kwargs):
        captured.update(kwargs)
        return {"segments": 1, "tracks": 1}

    monkeypatch.setattr(service, "transcribe_meeting", fake_transcribe)
    meeting_dir = tmp_path / "meetings" / "m-transcript-only"
    input_dir = meeting_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "metadata.json").write_text(
        json.dumps({"transcript_only": True}), encoding="utf-8"
    )

    service._process_meeting("m-transcript-only")

    assert captured["include_report"] is False


def test_create_meeting_request_id_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    processed = []
    monkeypatch.setattr(
        service, "_process_meeting", lambda meeting_id: processed.append(meeting_id)
    )
    request_id = str(uuid.uuid4())
    client = TestClient(service.app)

    first = client.post(
        "/meetings",
        data={"request_id": request_id, "title": "Retry-safe sync"},
        files={"file": ("recording.m4a", b"first audio", "audio/mp4")},
    )
    second = client.post(
        "/meetings",
        data={"request_id": request_id, "title": "Duplicate retry"},
        files={"file": ("recording.m4a", b"second audio", "audio/mp4")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == request_id
    assert second.json()["id"] == request_id
    assert processed == [request_id]
    input_dir = tmp_path / "meetings" / request_id / "input"
    assert (input_dir / "audio.m4a").read_bytes() == b"first audio"
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_request_id"] == request_id


def test_create_meeting_stages_on_linked_meetings_filesystem(tmp_path, monkeypatch):
    import errno
    from pathlib import Path

    root = tmp_path / "local"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "meetings").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(service, "ROOT", root)
    monkeypatch.setattr(service, "API_KEY", "")
    processed = []
    monkeypatch.setattr(service, "_process_meeting", processed.append)
    original_rename = Path.rename

    def same_volume_rename(source, target):
        if source.resolve().is_relative_to(external) != target.resolve().is_relative_to(
            external
        ):
            raise OSError(errno.EXDEV, "Cross-device link")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", same_volume_rename)
    request_id = str(uuid.uuid4())
    response = TestClient(service.app).post(
        "/meetings",
        data={"request_id": request_id},
        files={"file": ("recording.m4a", b"audio", "audio/mp4")},
    )
    assert response.status_code == 200
    assert processed == [request_id]
    assert (external / request_id / "input" / "audio.m4a").read_bytes() == b"audio"
    assert not list((external / ".uploads").iterdir())


def test_create_meeting_rejects_non_uuid_request_id(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    client = TestClient(service.app)

    response = client.post(
        "/meetings",
        data={"request_id": "../../unsafe"},
        files={"file": ("recording.m4a", b"audio", "audio/mp4")},
    )

    assert response.status_code == 400
    assert not (tmp_path / "meetings").exists()


def test_create_meeting_derives_title_from_zoom_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        files={
            "file": (
                "Быстрый созвон - -tech-leads - 2026-06-15 10-02.m4a",
                b"fake audio",
                "audio/mp4",
            )
        },
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    metadata = json.loads(
        (tmp_path / "meetings" / meeting_id / "input" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["title"] == "Быстрый созвон - tech-leads"


def test_create_meeting_persists_zoom_participant_tracks(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        data={
            "processing_mode": "zoom_participant_tracks",
            "participants": "Alice,Bob",
            "zoom_participant_tracks_json": json.dumps(
                [{"speaker_name": "Alice"}, {"speaker_name": "Bob"}]
            ),
        },
        files=[
            ("file", ("audio.m4a", b"mixed archive", "audio/mp4")),
            ("zoom_participant_files", ("alice.m4a", b"alice audio", "audio/mp4")),
            ("zoom_participant_files", ("bob.m4a", b"bob audio", "audio/mp4")),
        ],
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    input_dir = tmp_path / "meetings" / meeting_id / "input"
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))

    assert (input_dir / "audio.m4a").read_bytes() == b"mixed archive"
    assert (input_dir / "participants" / "01.m4a").read_bytes() == b"alice audio"
    assert (input_dir / "participants" / "02.m4a").read_bytes() == b"bob audio"
    assert metadata["processing_mode"] == "zoom_participant_tracks"
    assert metadata["participants"] == ["Alice", "Bob"]
    assert metadata["zoom_participant_tracks"] == [
        {"speaker_name": "Alice"},
        {"speaker_name": "Bob"},
    ]


def test_manifest_json_artifact_is_available(tmp_path):
    meeting_dir = tmp_path / "meeting-with-manifest"
    artifacts_dir = meeting_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (artifacts_dir / "context_pack.json").write_text("{}\n", encoding="utf-8")
    (artifacts_dir / "youtrack_context.json").write_text("{}\n", encoding="utf-8")

    assert service.ARTIFACTS["manifest_json"] == ("manifest.json", "application/json")
    assert service.ARTIFACTS["context_pack_json"] == (
        "context_pack.json",
        "application/json",
    )
    assert service.ARTIFACTS["youtrack_context_json"] == (
        "youtrack_context.json",
        "application/json",
    )
    assert "manifest_json" in service._available_artifacts(meeting_dir)
    assert "context_pack_json" in service._available_artifacts(meeting_dir)
    assert "youtrack_context_json" in service._available_artifacts(meeting_dir)


def test_terminal_status_refreshes_existing_manifest_and_syncs_memory(
    tmp_path,
    monkeypatch,
):
    meeting_dir = tmp_path / "meeting-final-status"
    input_dir = meeting_dir / "input"
    artifacts_dir = meeting_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    metadata = {
        "meeting_id": "meeting-final-status",
        "title": "Final status",
        "source_filename": "audio.m4a",
    }
    result = {"meeting_id": "meeting-final-status", "segments": 1}
    (input_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (meeting_dir / "status.json").write_text(
        json.dumps({"status": "processing"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_meeting_manifest(meeting_dir, metadata, result)
    sync_calls = []

    def fake_sync(sync_meeting_dir, manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status_entry = next(
            entry
            for entry in manifest["files"]
            if entry["relative_path"] == "status.json"
        )
        assert status_entry["sha256"] == sha256_file(sync_meeting_dir / "status.json")
        sync_calls.append(
            {
                "meeting_dir": sync_meeting_dir,
                "manifest_path": manifest_path,
                "status_hash": status_entry["sha256"],
            }
        )
        return "db down"

    monkeypatch.setattr(service, "sync_meeting_memory", fake_sync, raising=False)

    service._write_status(
        meeting_dir,
        "completed",
        phase="completed",
        progress=1.0,
        result={"segments": 1},
    )

    assert len(sync_calls) == 1
    assert sync_calls[0]["meeting_dir"] == meeting_dir
    assert sync_calls[0]["manifest_path"] == artifacts_dir / "manifest.json"
    status = json.loads((meeting_dir / "status.json").read_text(encoding="utf-8"))
    assert status["memory_sync_error"] == "db down"
    manifest = json.loads((artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
    status_entry = next(
        entry for entry in manifest["files"] if entry["relative_path"] == "status.json"
    )
    assert sync_calls[0]["status_hash"] != status_entry["sha256"]
    assert status_entry["sha256"] == sha256_file(meeting_dir / "status.json")
    assert manifest["metadata"] == metadata
    assert manifest["result"] == result


def test_scribe_options_round_trip_and_invalid_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda *args: None)
    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        files={"file": ("test.m4a", b"audio", "audio/mp4")},
        data={"microphone_mode": "single", "terms_hint": "LangGraph; Holocron"},
    )
    assert response.status_code == 200
    metadata = json.loads(
        (
            tmp_path / "meetings" / response.json()["id"] / "input/metadata.json"
        ).read_text()
    )
    assert metadata["microphone_mode"] == "single"
    assert metadata["terms_hint"] == "LangGraph; Holocron"
    invalid = client.post(
        "/meetings",
        files={"file": ("test.m4a", b"audio", "audio/mp4")},
        data={"microphone_mode": "guess"},
    )
    assert invalid.status_code == 422
