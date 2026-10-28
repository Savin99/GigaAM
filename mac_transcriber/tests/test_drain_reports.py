"""Тесты на чистые helper-функции захардённого дрейнера отчётов.

Тестируемый модуль: scripts/drain_reports_via_claude.py. Покрываются только
deterministic helper'ы, не требующие сети/launchd/claude/субпроцессов:
_needs_disk_upload, _mark_disk_uploaded, _health_status, _attempt_count,
_bump_attempt и файловый лок _acquire_lock (fcntl.flock).

Все записи — только во tmp_path.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# scripts/ не является пакетом (нет __init__.py), поэтому грузим модуль по пути.
# Импорт сам выполняет sys.path.insert(repo_root/scripts) из тела модуля и
# подтягивает reprocess_blocked + fcntl без сети/ключей.
_DRAIN_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "drain_reports_via_claude.py"
)
_spec = importlib.util.spec_from_file_location("drain_reports_via_claude", _DRAIN_PATH)
drain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drain)


def test_youtrack_agent_instructions_are_packaged():
    assert drain.YOUTRACK_INSTRUCTIONS.exists()


def test_targeted_drain_does_not_process_or_reupload_other_meetings(tmp_path, monkeypatch, capsys):
    import mac_transcriber

    service = SimpleNamespace(ROOT=tmp_path)
    monkeypatch.setitem(sys.modules, "mac_transcriber.service", service)
    monkeypatch.setattr(mac_transcriber, "service", service, raising=False)

    meetings = tmp_path / "meetings"
    target = meetings / "target"
    other = meetings / "other"
    reupload = meetings / "reupload"
    for d in (target, other, reupload):
        (d / "artifacts").mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"status": "completed"}))
        (d / "artifacts" / "report.md").write_text("report")
    monkeypatch.setattr(drain, "_load_env_files", lambda paths: None)
    monkeypatch.setattr(drain.rb, "_blocked_meetings", lambda root: [other, target])
    monkeypatch.setattr(drain.rb, "_needs_regen_meetings", lambda root, excluded: [])
    assert drain.main(["--meeting-id", "target", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "[blocked] target" in output
    assert "[blocked] other" not in output
    assert "[reupload] reupload" not in output


def _make_meeting(tmp_path: Path, status: dict | None = None) -> Path:
    """Создаёт каталог встречи с artifacts/ и (опционально) status.json."""
    meeting_dir = tmp_path / "meeting-001"
    (meeting_dir / "artifacts").mkdir(parents=True)
    if status is not None:
        (meeting_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return meeting_dir


def _read_status_file(meeting_dir: Path) -> dict:
    return json.loads((meeting_dir / "status.json").read_text(encoding="utf-8"))


# --- классификация исхода генерации: ok | limit | fail -----------------------
# Ключевое для «кончились лимиты»: НЕТ файла -> "limit" (транзиентно, без штрафа),
# файл есть но кривой -> "fail" (детерминированно, под кап), валидный -> "ok".


def test_classify_valid_payload_is_ok():
    assert drain._classify_generation(payload_valid=True, payload_exists=True) == "ok"


def test_classify_missing_payload_is_limit():
    # claude не отработал (лимит/авторизация/сеть) — файла нет вообще.
    assert (
        drain._classify_generation(payload_valid=False, payload_exists=False) == "limit"
    )


def test_classify_present_but_invalid_is_fail():
    # claude отработал, но выдал брак — это детерминированно, идёт под кап.
    assert (
        drain._classify_generation(payload_valid=False, payload_exists=True) == "fail"
    )


def test_payload_is_valid_variants(tmp_path: Path):
    missing = tmp_path / "nope.json"
    assert drain._payload_is_valid(missing) is False

    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert drain._payload_is_valid(broken) is False

    empty_sections = tmp_path / "empty.json"
    empty_sections.write_text(json.dumps({"adaptive_sections": []}), encoding="utf-8")
    assert drain._payload_is_valid(empty_sections) is False

    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"adaptive_sections": [{"title": "x", "items": []}]}),
        encoding="utf-8",
    )
    assert drain._payload_is_valid(good) is True


def test_json_object_has_key_requires_list(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"intents": []}), encoding="utf-8")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"intents": {}}), encoding="utf-8")

    assert drain._json_object_has_key(good, "intents") is True
    assert drain._json_object_has_key(wrong, "intents") is False


def test_prepare_youtrack_context_can_be_explicitly_disabled(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "agent_input.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"segment_id": "S0001", "text": "Обсудили задачу."}
                ]
            }
        ),
        encoding="utf-8",
    )

    context = drain._prepare_youtrack_context(
        work=work,
        model="claude-opus-5",
        enabled=False,
    )

    assert context["status"] == "disabled"
    enriched = json.loads((work / "agent_input.json").read_text(encoding="utf-8"))
    assert enriched["youtrack_context"]["status"] == "disabled"


# --- выбор Claude-модели -------------------------------------------------


def test_resolve_claude_model_reads_env_loaded_after_argparse(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_CLAUDE_MODEL", " opus ")

    assert drain._resolve_claude_model(None) == "opus"


def test_resolve_claude_model_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_CLAUDE_MODEL", "sonnet")

    assert drain._resolve_claude_model(" claude-opus-4-8 ") == "claude-opus-4-8"


# --- _needs_disk_upload ---------------------------------------------------


def test_needs_disk_upload_true_when_report_present_and_not_uploaded(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})
    (meeting_dir / "artifacts" / "report.md").write_text("# report\n", encoding="utf-8")

    assert drain._needs_disk_upload(meeting_dir) is True


def test_needs_disk_upload_false_after_uploaded_flag(tmp_path):
    meeting_dir = _make_meeting(
        tmp_path, status={"meeting_id": "m-1", "report_disk_uploaded": True}
    )
    (meeting_dir / "artifacts" / "report.md").write_text("# report\n", encoding="utf-8")

    assert drain._needs_disk_upload(meeting_dir) is False


def test_needs_disk_upload_false_without_report_md(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})
    # report.md отсутствует — заливать нечего.

    assert drain._needs_disk_upload(meeting_dir) is False


# --- _mark_disk_uploaded --------------------------------------------------


def test_mark_disk_uploaded_sets_flag_and_preserves_fields(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "keep-me"})
    (meeting_dir / "artifacts" / "report.md").write_text("# report\n", encoding="utf-8")
    assert drain._needs_disk_upload(meeting_dir) is True

    drain._mark_disk_uploaded(meeting_dir)

    status = _read_status_file(meeting_dir)
    assert status["report_disk_uploaded"] is True
    # Остальные поля не потёрты.
    assert status["meeting_id"] == "keep-me"
    # Идемпотентно: теперь заливка больше не нужна.
    assert drain._needs_disk_upload(meeting_dir) is False


# --- _health_status -------------------------------------------------------


def test_health_status_reads_failed(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    (meeting_dir / "artifacts" / "report_health.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )

    assert drain._health_status(meeting_dir) == "failed"


def test_health_status_reads_degraded(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    (meeting_dir / "artifacts" / "report_health.json").write_text(
        json.dumps({"status": "degraded"}), encoding="utf-8"
    )

    assert drain._health_status(meeting_dir) == "degraded"


def test_health_status_empty_when_missing(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    # report_health.json отсутствует.

    assert drain._health_status(meeting_dir) == ""


def test_health_status_empty_on_broken_json(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    (meeting_dir / "artifacts" / "report_health.json").write_text(
        "{ not valid json", encoding="utf-8"
    )

    # Не падает, возвращает "".
    assert drain._health_status(meeting_dir) == ""


# --- _attempt_count / _bump_attempt --------------------------------------


def test_attempt_count_starts_at_zero(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})

    assert drain._attempt_count(meeting_dir) == 0


def test_bump_attempt_increments_and_persists(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})

    assert drain._bump_attempt(meeting_dir) == 1
    assert drain._attempt_count(meeting_dir) == 1

    assert drain._bump_attempt(meeting_dir) == 2
    assert drain._attempt_count(meeting_dir) == 2

    # Значение персистится в status.json.
    status = _read_status_file(meeting_dir)
    assert status["report_attempt_count"] == 2
    # Прочие поля не потёрты.
    assert status["meeting_id"] == "m-1"


# --- _acquire_lock (fcntl.flock) -----------------------------------------


def test_acquire_lock_is_exclusive_and_releases(tmp_path):
    lock_path = tmp_path / "drain.lock"

    fd1 = drain._acquire_lock(lock_path)
    fd2 = None
    fd3 = None
    try:
        # Первый вызов берёт лок.
        assert fd1 is not None

        # Второй вызов на тот же путь (пока первый держится) — None.
        fd2 = drain._acquire_lock(lock_path)
        assert fd2 is None

        # Освобождаем первый лок.
        fd1.close()
        fd1 = None

        # Теперь лок снова берётся.
        fd3 = drain._acquire_lock(lock_path)
        assert fd3 is not None
    finally:
        for fd in (fd1, fd2, fd3):
            if fd is not None:
                fd.close()


def test_evidence_error_gets_one_repair_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source={'evidence_contract_version':1,'segments':[{'segment_id':'S0001','text':'Сделаю MR сегодня-завтра.'}]}
    (tmp_path/'agent_input.json').write_text(json.dumps(source))
    calls=[]
    def generate(cmd, **kwargs):
        calls.append(cmd)
        quote='Неверная цитата из соседней реплики.' if len(calls)==1 else source['segments'][0]['text']
        payload={'adaptive_sections':[{'title':'Раздел'}],'action_items':[{'title':'MR','text':'Подготовить MR','certainty':'confirmed','citations':['S0001'],'evidence':[{'segment_id':'S0001','quote':quote}]}]}
        (tmp_path/'ai_payload.json').write_text(json.dumps(payload))
        return SimpleNamespace(stdout='',stderr='',returncode=0)
    monkeypatch.setattr(drain.subprocess,'run',generate)
    assert drain._generate_payload(tmp_path,'')=='ok'
    assert len(calls)==2
    assert 'evidence_feedback.json' in ' '.join(calls[1])
