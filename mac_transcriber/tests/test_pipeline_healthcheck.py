"""Тесты на чистую логику health-check (blocked_too_long, run_checks).

scripts/ не пакет — грузим модуль по пути (как в test_drain_reports.py).
Сетевые проверки (healthz/claude) монкипатчим, время инжектим — без сети/субпроцессов.
"""

import importlib.util
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

_HC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pipeline_healthcheck.py"
_spec = importlib.util.spec_from_file_location("pipeline_healthcheck", _HC_PATH)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)

NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


def _meeting(root: Path, name: str, status: str, updated_at: str | None) -> None:
    d = root / name
    d.mkdir(parents=True)
    payload: dict = {"status": status}
    if updated_at is not None:
        payload["updated_at"] = updated_at
    (d / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_blocked_too_long_flags_only_old_blocked(tmp_path: Path):
    root = tmp_path / "meetings"
    root.mkdir()
    # свежий blocked (10 мин назад) — не застрял
    _meeting(root, "fresh", "blocked_on_ai", (NOW - timedelta(minutes=10)).isoformat())
    # старый blocked (5 ч назад) — застрял
    _meeting(root, "stale", "blocked_on_ai", (NOW - timedelta(hours=5)).isoformat())
    # completed старый — не считается
    _meeting(root, "done", "completed", (NOW - timedelta(hours=9)).isoformat())

    stuck = hc.blocked_too_long(root, max_hours=3, now=NOW)
    assert stuck == ["stale"]


def test_blocked_without_timestamp_is_flagged(tmp_path: Path):
    root = tmp_path / "meetings"
    root.mkdir()
    _meeting(root, "no_ts", "blocked_on_quota", None)
    assert hc.blocked_too_long(root, max_hours=3, now=NOW) == ["no_ts"]


def test_run_checks_fails_when_stuck(tmp_path, monkeypatch):
    root = tmp_path / "meetings"
    root.mkdir()
    _meeting(root, "stale", "blocked_on_ai", (NOW - timedelta(hours=5)).isoformat())
    monkeypatch.setattr(hc, "_meetings_root", lambda: root)
    monkeypatch.setattr(hc, "check_healthz", lambda port: (True, "healthz ok"))
    monkeypatch.setattr(hc, "check_claude", lambda: (True, "claude 2.1.169"))
    monkeypatch.delenv("MAC_TRANSCRIBER_HEALTHCHECK_STUCK_HOURS", raising=False)

    ok, lines = hc.run_checks(NOW)
    assert ok is False
    assert any("stuck blocked" in ln for ln in lines)


def test_run_checks_fails_when_relay_unreachable(tmp_path, monkeypatch):
    root = tmp_path / "meetings"
    root.mkdir()
    monkeypatch.setattr(hc, "_meetings_root", lambda: root)
    monkeypatch.setattr(hc, "check_healthz", lambda port: (True, "healthz ok"))
    monkeypatch.setattr(hc, "check_relay", lambda: (False, "relay unreachable"))
    monkeypatch.setattr(hc, "check_claude", lambda: (True, "claude 2.1.169"))
    monkeypatch.setenv("MAC_TRANSCRIBER_RELAY_HEALTHCHECK", "1")

    ok, lines = hc.run_checks(NOW)
    assert ok is False
    assert "relay unreachable" in lines


def test_run_checks_ok_when_healthy(tmp_path, monkeypatch):
    root = tmp_path / "meetings"
    root.mkdir()
    _meeting(root, "done", "completed", NOW.isoformat())
    monkeypatch.setattr(hc, "_meetings_root", lambda: root)
    monkeypatch.setattr(hc, "check_healthz", lambda port: (True, "healthz ok"))
    monkeypatch.setattr(hc, "check_relay", lambda: (True, "relay ok"))
    monkeypatch.setattr(hc, "check_claude", lambda: (True, "claude 2.1.169"))

    ok, lines = hc.run_checks(NOW)
    assert ok is True
    assert any("queue ok" in ln for ln in lines)


def test_relay_health_does_not_hide_worker_auth_failure(monkeypatch):
    responses = iter([
        subprocess.CompletedProcess([], 0, stdout='{"status":"ok"}', stderr=""),
        subprocess.CompletedProcess([], 1, stdout="AUTH_STATUS_CHECK=401", stderr=""),
    ])
    monkeypatch.setattr(hc.subprocess, "run", lambda *a, **kw: next(responses))
    assert hc.check_relay() == (
        False, "relay ok; worker auth failed: 401 (shared key mismatch)"
    )


def test_relay_checks_worker_authorization(monkeypatch):
    responses = iter([
        subprocess.CompletedProcess([], 0, stdout='{"status":"ok"}', stderr=""),
        subprocess.CompletedProcess([], 0, stdout="AUTH_STATUS_CHECK=404", stderr=""),
    ])
    monkeypatch.setattr(hc.subprocess, "run", lambda *a, **kw: next(responses))
    assert hc.check_relay() == (True, "relay ok; worker auth ok")


def test_check_claude_version_only_when_backend_not_claude(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="2.1.186\n", stderr="")

    monkeypatch.setattr(hc.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_BACKEND", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_CLAUDE_PROMPT_HEALTHCHECK", raising=False)

    ok, msg = hc.check_claude()

    assert ok is True
    assert msg == "claude 2.1.186"
    assert calls == [["/usr/local/bin/claude", "--version"]]


def test_check_claude_prompt_required_for_claude_backend(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.186\n", stderr="")
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="API Error: 401 Invalid authentication credentials"
        )

    monkeypatch.setattr(hc.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", "claude")
    monkeypatch.setenv("MAC_TRANSCRIBER_CLAUDE_HEALTHCHECK_ATTEMPTS", "1")

    ok, msg = hc.check_claude()

    assert ok is False
    assert "claude prompt rc=1" in msg
    assert calls == [
        ["/usr/local/bin/claude", "--version"],
        ["/usr/local/bin/claude", "-p", "ping", "--output-format", "json"],
    ]


def test_check_claude_reports_expired_oauth_without_json_tail(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.186", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps({
            "is_error": True,
            "result": "Failed to authenticate: OAuth session expired and could not be refreshed",
            "uuid": "irrelevant-trailing-metadata",
        }), stderr="")

    monkeypatch.setattr(hc.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", "claude")
    assert hc.check_claude() == (
        False, "claude OAuth session expired; run claude auth login"
    )


def test_check_claude_reports_missing_login_without_json_tail(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.251", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps({
            "is_error": True,
            "result": "Not logged in · Please run /login",
            "uuid": "irrelevant-trailing-metadata",
        }), stderr="Client.listTools() called but server does not advertise tools capability")

    monkeypatch.setattr(hc.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", "claude")
    assert hc.check_claude() == (
        False, "claude not logged in; run claude auth login"
    )


def test_check_claude_retries_transient_prompt_timeout(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.186\n", stderr="")
        if len([call for call in calls if call[1] == "-p"]) == 1:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"type":"result","is_error":false,"result":"pong"}\n',
            stderr="",
        )

    monkeypatch.setattr(hc.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", "claude")
    monkeypatch.setenv("MAC_TRANSCRIBER_CLAUDE_HEALTHCHECK_ATTEMPTS", "2")

    ok, msg = hc.check_claude()

    assert ok is True
    assert msg == "claude 2.1.186; prompt ok"
    assert calls == [
        ["/usr/local/bin/claude", "--version"],
        ["/usr/local/bin/claude", "-p", "ping", "--output-format", "json"],
        ["/usr/local/bin/claude", "-p", "ping", "--output-format", "json"],
    ]


def test_telegram_alert_waits_and_deduplicates_until_recovery(tmp_path, monkeypatch):
    state_path = tmp_path / "healthcheck-alert.json"
    sent = []

    def fake_telegram(token, chat_id, text):
        sent.append((token, chat_id, text))
        return True

    monkeypatch.setattr(hc, "_telegram", fake_telegram)

    kwargs = {
        "summary": "relay bad",
        "token": "token",
        "chat_id": "chat",
        "state_path": state_path,
        "failure_threshold": 2,
    }
    assert hc.handle_telegram_alert(ok=False, **kwargs) is False
    assert sent == []
    assert hc.handle_telegram_alert(ok=False, **kwargs) is True
    assert len(sent) == 1
    assert hc.handle_telegram_alert(ok=False, **kwargs) is False
    assert len(sent) == 1

    assert hc.handle_telegram_alert(ok=True, **kwargs) is False
    assert hc.handle_telegram_alert(ok=False, **kwargs) is False
    assert len(sent) == 1


def test_telegram_alert_retries_when_delivery_failed(tmp_path, monkeypatch):
    state_path = tmp_path / "healthcheck-alert.json"
    attempts = []

    def fake_telegram(token, chat_id, text):
        attempts.append(text)
        return len(attempts) > 1

    monkeypatch.setattr(hc, "_telegram", fake_telegram)
    kwargs = {
        "ok": False,
        "summary": "relay bad",
        "token": "token",
        "chat_id": "chat",
        "state_path": state_path,
        "failure_threshold": 1,
    }
    assert hc.handle_telegram_alert(**kwargs) is False
    assert hc.handle_telegram_alert(**kwargs) is True
    assert hc.handle_telegram_alert(**kwargs) is False
    assert len(attempts) == 2
