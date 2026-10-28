#!/usr/bin/env python3
"""Health-check пайплайна mac_transcriber для внешнего монитора (dead-man's-switch).

Запускается launchd'ом каждые ~10 мин. Проверяет живость пайплайна и шлёт
heartbeat на внешний монитор (Uptime Kuma / Healthchecks / ntfy — что угодно).
Главная идея: если Mac уснул / джоб умер / claude разлогинен — пинг НЕ уходит,
и монитор бьёт тревогу по grace-периоду. При явных проблемах шлём down с причиной.

Проверки:
  1. транскрайбер /healthz отвечает {"status":"ok"};
  2. нет встреч, застрявших в blocked_* дольше N часов (claude сломан/лимит/разлогин/кап);
  3. claude CLI отвечает на --version, а в claude-backend режиме ещё и на
     короткий headless prompt (нужен дренажу).
  4. опционально: relay на VPS реально достаёт Mac через reverse SSH tunnel.

ENV (в .env.local):
  MAC_TRANSCRIBER_HEALTHCHECK_URL       push-URL монитора (без него — no-op).
  MAC_TRANSCRIBER_HEALTHCHECK_FAIL_URL  опц. URL для down (Healthchecks: <URL>/fail).
                                        Если не задан — шлём на основной URL ?status=down.
  MAC_TRANSCRIBER_HEALTHCHECK_STUCK_HOURS  порог «застряло» (по умолч. 3).
  MAC_TRANSCRIBER_PORT                  порт транскрайбера (по умолч. 18003).
  MAC_TRANSCRIBER_RELAY_HEALTHCHECK     1/true/yes — проверять VPS relay.
  MAC_TRANSCRIBER_RELAY_URL             URL relay на VPS (по умолч.
                                        http://127.0.0.1:18013/healthz).
  MAC_TRANSCRIBER_TUNNEL_TARGET         SSH target VPS (по умолч.
                                        root@111.88.224.176).
  MAC_TRANSCRIBER_TUNNEL_IDENTITY_FILE  SSH key для VPS.
  MAC_TRANSCRIBER_CLAUDE_PROMPT_HEALTHCHECK
                                        1/true/yes — проверять claude -p даже
                                        если backend не claude.
  MAC_TRANSCRIBER_CLAUDE_HEALTHCHECK_ATTEMPTS
                                        число попыток claude -p (по умолч. 2).
  MAC_TRANSCRIBER_TELEGRAM_FAILURE_THRESHOLD
                                        сколько последовательных сбоев нужно
                                        до Telegram-алерта (по умолч. 2).
  MAC_TRANSCRIBER_HEALTHCHECK_STATE_PATH
                                        опц. путь к локальному состоянию
                                        дедупликации Telegram-алертов.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED = ("blocked_on_ai", "blocked_on_quota")


def _load_env_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _meetings_root() -> Path:
    root = os.environ.get("MAC_TRANSCRIBER_ROOT") or str(
        REPO_ROOT / ".local" / "mac_transcriber"
    )
    return Path(root) / "meetings"


def _parse_dt(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def blocked_too_long(meetings_root: Path, max_hours: float, now: datetime) -> list[str]:
    """ID встреч, застрявших в blocked_* дольше max_hours (по updated_at|created_at)."""
    stuck: list[str] = []
    for status_path in meetings_root.glob("*/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") not in BLOCKED:
            continue
        stamp = _parse_dt(
            str(status.get("updated_at") or status.get("created_at") or "")
        )
        if stamp is None:
            stuck.append(status_path.parent.name)  # нет метки — считаем подозрительной
            continue
        if (now - stamp).total_seconds() > max_hours * 3600:
            stuck.append(status_path.parent.name)
    return stuck


def check_healthz(port: int) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            body = resp.read().decode("utf-8", "replace")
        ok = resp.status == 200 and '"ok"' in body
        return ok, "healthz ok" if ok else f"healthz bad: {resp.status} {body[:60]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"healthz unreachable: {exc}"[:80]


def check_claude() -> tuple[bool, str]:
    claude = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not claude:
        return False, "claude not on PATH"
    try:
        res = subprocess.run(
            [claude, "--version"], capture_output=True, text=True, timeout=20
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"claude --version error: {exc}"[:80]
    if res.returncode != 0:
        return False, f"claude --version rc={res.returncode}"
    version_msg = f"claude {res.stdout.strip()[:30]}"
    backend = (os.environ.get("MAC_TRANSCRIBER_REPORT_BACKEND") or "").strip().lower()
    if backend != "claude" and not _env_truthy("MAC_TRANSCRIBER_CLAUDE_PROMPT_HEALTHCHECK"):
        return True, version_msg

    prompt_cmd = [claude, "-p", "ping", "--output-format", "json"]
    timeout = int(os.environ.get("MAC_TRANSCRIBER_CLAUDE_HEALTHCHECK_TIMEOUT", "60"))
    attempts = max(
        1, int(os.environ.get("MAC_TRANSCRIBER_CLAUDE_HEALTHCHECK_ATTEMPTS", "2"))
    )
    last_error = ""
    for _attempt in range(attempts):
        try:
            prompt = subprocess.run(
                prompt_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_error = f"claude prompt timeout after {timeout}s"
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = f"claude prompt error: {type(exc).__name__}: {exc}"[:100]
            continue
        try:
            payload = json.loads(prompt.stdout or "")
        except (ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        result = str(payload.get("result") or "")
        if "OAuth session expired" in result:
            return False, "claude OAuth session expired; run claude auth login"
        if "not logged in" in result.lower():
            return False, "claude not logged in; run claude auth login"
        if prompt.returncode == 0 and not payload.get("is_error"):
            return True, f"{version_msg}; prompt ok"
        tail = result or ((prompt.stdout or "") + " " + (prompt.stderr or "")).strip()
        last_error = f"claude prompt rc={prompt.returncode}: {tail[-100:]}"
    return False, last_error


def check_relay() -> tuple[bool, str]:
    """Проверить relay с точки зрения VPS, а не локального Mac."""
    target = os.environ.get("MAC_TRANSCRIBER_TUNNEL_TARGET", "root@111.88.224.176")
    relay_url = os.environ.get(
        "MAC_TRANSCRIBER_RELAY_URL", "http://127.0.0.1:18013/healthz"
    )
    identity = os.environ.get(
        "MAC_TRANSCRIBER_TUNNEL_IDENTITY_FILE",
        str(Path.home() / ".ssh" / "valamis_vps_ed25519"),
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    if identity and Path(identity).exists():
        command += ["-i", identity, "-o", "IdentitiesOnly=yes"]
    command += [target, f"curl -fsS -m8 {relay_url!r}"]

    try:
        res = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except Exception as exc:  # noqa: BLE001
        return False, f"relay unreachable: {exc}"[:100]
    body = (res.stdout or "") + (res.stderr or "")
    if res.returncode == 0 and '"ok"' in res.stdout:
        # A public health endpoint can succeed while every real job fails with 401.
        # Use the worker's own credential in-place; never return it over SSH.
        probe = """import os, urllib.request, urllib.error
request = urllib.request.Request(
    os.environ['TRANSCRIPTION_SERVICE_URL'] + '/meetings/no-such-meeting/status',
    headers={'Authorization': 'Bearer ' + os.environ['TRANSCRIPTION_SERVICE_API_KEY']})
try:
    urllib.request.urlopen(request, timeout=8)
except urllib.error.HTTPError as exc:
    print('AUTH_STATUS_CHECK=' + str(exc.code))
    raise SystemExit(0 if exc.code == 404 else 1)
raise SystemExit(1)
"""
        worker_command = command[:-1] + [
            "cd /root/projects/slack_zoom && docker compose exec -T worker python -c "
            + shlex.quote(probe)
        ]
        try:
            worker = subprocess.run(
                worker_command, capture_output=True, text=True, timeout=20
            )
        except Exception as exc:
            return False, f"relay ok; worker auth check failed: {type(exc).__name__}"
        if worker.returncode == 0 and "AUTH_STATUS_CHECK=404" in worker.stdout:
            return True, "relay ok; worker auth ok"
        if "AUTH_STATUS_CHECK=401" in worker.stdout:
            return False, "relay ok; worker auth failed: 401 (shared key mismatch)"
        return False, "relay ok; worker auth check failed"
    return False, f"relay bad: rc={res.returncode} {body.strip()[:80]}"


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def run_checks(now: datetime) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    h_ok, h_msg = check_healthz(int(os.environ.get("MAC_TRANSCRIBER_PORT", "18003")))
    lines.append(h_msg)
    ok = ok and h_ok

    if _env_truthy("MAC_TRANSCRIBER_RELAY_HEALTHCHECK"):
        r_ok, r_msg = check_relay()
        lines.append(r_msg)
        ok = ok and r_ok

    max_hours = float(os.environ.get("MAC_TRANSCRIBER_HEALTHCHECK_STUCK_HOURS", "3"))
    stuck = blocked_too_long(_meetings_root(), max_hours, now)
    if stuck:
        ok = False
        lines.append(f"stuck blocked >{max_hours:g}h: {len(stuck)} ({stuck[0]})")
    else:
        lines.append("queue ok (no long-blocked)")

    c_ok, c_msg = check_claude()
    lines.append(c_msg)
    ok = ok and c_ok

    return ok, lines


def _ping(url: str, params: dict[str, str]) -> None:
    full = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(full, timeout=10) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"healthcheck ping failed: {exc}", file=sys.stderr)


def _telegram(token: str, chat_id: str, text: str) -> bool:
    """Алерт в Telegram (Bot API). Молчит, если токен/чат не заданы."""
    if not (token and chat_id):
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            resp.read()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"telegram alert failed: {exc}", file=sys.stderr)
        return False


def _alert_state_path() -> Path:
    configured = os.environ.get("MAC_TRANSCRIBER_HEALTHCHECK_STATE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / ".local" / "mac_transcriber" / "healthcheck_alert_state.json"


def _load_alert_state(path: Path) -> dict[str, object]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"consecutive_failures": 0, "alert_sent": False}
    if not isinstance(state, dict):
        return {"consecutive_failures": 0, "alert_sent": False}
    return state


def _save_alert_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def handle_telegram_alert(
    *,
    ok: bool,
    summary: str,
    token: str,
    chat_id: str,
    state_path: Path,
    failure_threshold: int,
) -> bool:
    """Send one alert per incident, after N consecutive failed checks."""
    state = _load_alert_state(state_path)
    if ok:
        _save_alert_state(
            state_path, {"consecutive_failures": 0, "alert_sent": False}
        )
        return False

    failures = max(0, int(state.get("consecutive_failures", 0))) + 1
    alert_sent = bool(state.get("alert_sent", False))
    delivered = False
    if (
        failures >= max(1, failure_threshold)
        and not alert_sent
        and token
        and chat_id
    ):
        delivered = _telegram(
            token, chat_id, f"🔴 Mac-пайплайн отчётов: {summary[:350]}"
        )
        alert_sent = delivered

    _save_alert_state(
        state_path,
        {"consecutive_failures": failures, "alert_sent": alert_sent},
    )
    return delivered


def main(argv: list[str] | None = None) -> int:
    _load_env_files([REPO_ROOT / ".env.local"])
    url = os.environ.get("MAC_TRANSCRIBER_HEALTHCHECK_URL", "").strip()
    tg_token = os.environ.get("MAC_TRANSCRIBER_TELEGRAM_TOKEN", "").strip()
    tg_chat = os.environ.get("MAC_TRANSCRIBER_TELEGRAM_CHAT_ID", "").strip()

    now = datetime.now(UTC)
    ok, lines = run_checks(now)
    summary = "; ".join(lines)
    print(f"[{'OK' if ok else 'FAIL'}] {summary}")

    # Telegram — после нескольких последовательных сбоев и один раз на инцидент.
    # Успешная проверка сбрасывает состояние, но не создаёт лишний heartbeat.
    threshold = int(
        os.environ.get("MAC_TRANSCRIBER_TELEGRAM_FAILURE_THRESHOLD", "2")
    )
    handle_telegram_alert(
        ok=ok,
        summary=summary,
        token=tg_token,
        chat_id=tg_chat,
        state_path=_alert_state_path(),
        failure_threshold=threshold,
    )

    # Heartbeat-URL (Uptime Kuma / Healthchecks): up при здоровье, down при сбое.
    if url:
        if ok:
            _ping(url, {"status": "up", "msg": "OK"})
        else:
            fail_url = os.environ.get(
                "MAC_TRANSCRIBER_HEALTHCHECK_FAIL_URL", ""
            ).strip()
            if fail_url:
                _ping(fail_url, {"msg": summary[:300]})
            else:
                _ping(url, {"status": "down", "msg": summary[:300]})

    if not url and not (tg_token and tg_chat):
        print("монитор не настроен (нет URL и Telegram) — no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
