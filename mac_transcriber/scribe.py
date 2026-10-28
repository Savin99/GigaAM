"""ElevenLabs batch ASR adapter; downstream meeting artifacts stay unchanged."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from mac_transcriber.persistence import file_lock, save_json

import httpx

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_KEYTERMS = Path(__file__).with_name("scribe_keyterms.json")


def meeting_keyterms(metadata: dict) -> list[str]:
    configured = os.environ.get("MAC_TRANSCRIBER_SCRIBE_KEYTERMS_FILE")
    path = Path(configured) if configured else DEFAULT_KEYTERMS
    defaults = json.loads(path.read_text(encoding="utf-8"))
    return keyterms(keyterms(metadata.get("terms_hint")) + keyterms(defaults))


def microphone_mode(track, index: int, metadata: dict) -> str:
    entries = metadata.get("zoom_participant_tracks") or []
    entry = entries[index] if index < len(entries) else {}
    mode = entry.get("microphone_mode") or metadata.get("microphone_mode")
    mode = mode or "auto"
    if mode not in {"auto", "single", "room", "unknown"}:
        raise ScribeError("microphone_mode must be auto, single, room or unknown")
    return mode


def infer_microphone(words: list[dict]) -> dict:
    """Conservative timing heuristic, not biometric identification.

    Retain weak secondary IDs as anonymous speech even when the dominant voice
    qualifies for a Zoom label. Never delete or merge a possible brief speaker.
    """
    stats = {}
    previous = None
    previous_end = 0.0
    for word in sorted(words, key=lambda w: w["start"]):
        sid = word.get("speaker_id") or "unknown"
        item = stats.setdefault(sid, {"words": 0, "seconds": 0.0, "turns": 0})
        item["words"] += 1
        item["seconds"] += max(0.0, word["end"] - max(word["start"], item.get("last_end", 0)))
        item["last_end"] = max(item.get("last_end", 0), word["end"])
        if sid != previous or word["start"] - previous_end > 1.5:
            item["turns"] += 1
        previous, previous_end = sid, word["end"]
    for item in stats.values():
        item.pop("last_end")
        item["seconds"] = round(item["seconds"], 3)
    stable = [sid for sid, v in stats.items() if sid != "unknown"
              and v["words"] >= 8 and v["seconds"] >= 3
              and (v["turns"] >= 2 or v["seconds"] >= 10)]
    mode, reason, dominant = "unknown", "insufficient_or_ambiguous_speech", None
    if len(stable) >= 2:
        mode, reason = "room", "multiple_sustained_speaker_ids"
    elif stats:
        candidate = max(stats, key=lambda sid: stats[sid]["seconds"])
        main = stats[candidate]
        others = [v for sid, v in stats.items() if sid != candidate]
        weak_only = all(v["words"] <= 2 and v["seconds"] <= 0.8 for v in others)
        fraction = sum(v["seconds"] for v in others) / max(main["seconds"], 0.001)
        if candidate != "unknown" and main["words"] >= 20 and main["seconds"] >= 8 and weak_only and fraction <= 0.01:
            mode, reason, dominant = "single", "one_sustained_voice_only_tiny_secondary_ids", candidate
    return {"mode": mode, "reason": reason, "dominant_id": dominant,
            "speakers": stats, "method": "scribe_word_timing_v1",
            "identity": "inferred_from_zoom_connection_not_verified"}


class ScribeError(RuntimeError):
    """Safe to expose in meeting status (never includes provider response bodies)."""


class ScribeCacheMiss(ScribeError):
    """Offline recovery needs local ASR; a cloud request must not be sent."""


class ScribeReconciliationRequired(ScribeError):
    """The paid submission may have completed; never repeat it blindly."""


_save_json = save_json


def _socket_options() -> list[tuple[int, int, int]]:
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle_option = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    if idle_option is not None:
        options.append((socket.IPPROTO_TCP, idle_option, 20))
    if hasattr(socket, "TCP_KEEPINTVL"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10))
    return options


def keyterms(value: object) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,;\n]", value)
    if not isinstance(value, list):
        return []
    result = []
    for term in value:
        if not isinstance(term, str):
            continue
        term = " ".join(term.split())
        if (
            term
            and len(term) < 50
            and len(term.split()) <= 5
            and not re.search(r"[<>{}\[\]\\]", term)
            and term not in result
        ):
            result.append(term)
    return result[:1000]


class ScribeRetryableError(ScribeError):
    """No paid submission was accepted; a bounded later retry is safe."""


class ScribePermanentError(ScribeError):
    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = code


def sanitized_location(value: str) -> str:
    """Keep routing diagnostics, never credentials, query strings or fragments."""
    parsed = urlsplit(urljoin(ENDPOINT, value))
    host = parsed.hostname or ""
    path = re.sub(r"/[A-Za-z0-9_-]{80,}(?=/|$)", "/[redacted]", parsed.path)
    return urlunsplit((parsed.scheme, host, path[:240], "", ""))


def _provider_error_code(response) -> str:
    try:
        detail = response.json().get("detail", {})
        value = detail.get("code") or detail.get("status") if isinstance(detail, dict) else ""
    except (ValueError, AttributeError):
        return ""
    return value if isinstance(value, str) and re.fullmatch(r"[a-z_]{1,80}", value) else ""


def _recover_response(client, journal: Path, cached: Path, previous: dict, key: str) -> dict:
    transcription_id = previous.get("transcription_id")
    if not isinstance(transcription_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", transcription_id):
        raise ScribeReconciliationRequired(
            "Scribe submission outcome requires reconciliation; no retrievable transcription ID; audio was not resubmitted"
        )
    try:
        response = client.get(
            ENDPOINT + "/transcripts/" + transcription_id,
            headers={"xi-api-key": key},
        )
        if response.status_code != 200:
            raise ScribeReconciliationRequired(
                f"Scribe saved result lookup returned HTTP {response.status_code}; audio was not resubmitted"
            )
        payload = response.json()
        validate_response(payload)
    except (httpx.TransportError, ValueError, ScribeError) as exc:
        if isinstance(exc, ScribeReconciliationRequired):
            raise
        raise ScribeReconciliationRequired("Scribe saved result lookup unavailable; audio was not resubmitted") from None
    _save_json(cached, payload)
    _save_json(journal, {**previous, "state": "completed", "reconciled_at": datetime.now(UTC).isoformat()})
    return payload


def request_transcript(
    path: Path, *, terms: list[str], cache_dir: Path, single_speaker: bool = False,
    cache_only: bool = False,
) -> dict:
    data = {
        "model_id": "scribe_v2", "language_code": "ru", "diarize": "true",
        "timestamps_granularity": "word", "tag_audio_events": "false", "no_verbatim": "false",
    }
    if terms:
        data["keyterms"] = terms
    if single_speaker:
        data["num_speakers"] = "1"
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
    with path.open("rb") as audio:
        for chunk in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(chunk)
    cached = cache_dir / f"{digest.hexdigest()}.json"
    # The cache and journal must be checked again after taking the process lock.
    with file_lock(cached.with_suffix(".lock"), blocking=True):
        return _request_locked(path, data, cached, cache_only=cache_only)


def _request_locked(path: Path, data: dict, cached: Path, *, cache_only: bool) -> dict:
    cache_invalid = False
    if cached.exists():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            validate_response(payload)
            return payload
        except (ValueError, ScribeError):
            cache_invalid = True
    if cache_only:
        raise ScribeCacheMiss("No validated cached Scribe response; cloud submission disabled")
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise ScribePermanentError("ELEVENLABS_API_KEY is missing; configure it in .env.local", code="missing_api_key")
    journal = cached.with_suffix(".request.json")
    try:
        previous = json.loads(journal.read_text()) if journal.exists() else {}
    except ValueError:
        raise ScribeReconciliationRequired("Scribe journal is invalid; audio was not resubmitted") from None
    timeout = httpx.Timeout(3600.0, connect=30.0, write=300.0, pool=30.0)
    transport = httpx.HTTPTransport(socket_options=_socket_options(), retries=0)
    # Never redirect a POST or send xi-api-key to a different host.
    with httpx.Client(timeout=timeout, transport=transport, follow_redirects=False) as client:
        if cache_invalid or previous.get("state") in {"submitting", "unknown", "response_received", "completed"}:
            return _recover_response(client, journal, cached, previous, key)
        if previous.get("state") == "permanent_failure":
            raise ScribePermanentError(
                "Scribe submission blocked by saved permanent failure: " + previous.get("failure_code", "provider_error"),
                code=previous.get("failure_code", "provider_error"),
            )
        for attempt in range(3):
            info = {
                "audio_file": path.name, "cache_key": cached.stem,
                "state": "submitting", "attempt": attempt + 1,
                "started_at": datetime.now(UTC).isoformat(),
            }
            _save_json(journal, info)
            try:
                with path.open("rb") as audio:
                    with client.stream(
                        "POST", ENDPOINT, headers={"xi-api-key": key}, data=data,
                        files={"file": (path.name, audio, "application/octet-stream")},
                    ) as response:
                        info.update(state="response_received", http_status=response.status_code,
                                    request_id=response.headers.get("x-trace-id") or response.headers.get("request-id") or response.headers.get("x-request-id"))
                        if 300 <= response.status_code < 400:
                            info["redirect_location"] = sanitized_location(response.headers.get("location", ""))
                        _save_json(journal, info)
                        response.read()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError) as exc:
                info.update(state="not_submitted", transport_error=type(exc).__name__)
                _save_json(journal, info)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise ScribeRetryableError("Scribe connection could not be established; no audio submitted; safe retry limit reached") from None
            except httpx.TransportError as exc:
                info.update(state="unknown", transport_error=type(exc).__name__, failed_at=datetime.now(UTC).isoformat())
                _save_json(journal, info)
                raise ScribeReconciliationRequired(
                    f"Scribe {type(exc).__name__}; submission outcome unknown; request journal saved"
                ) from None
            code = _provider_error_code(response) if response.status_code != 200 else ""
            if response.status_code == 429 and code not in {"quota_exceeded", "insufficient_quota", "insufficient_credits", "insufficient_balance", "credit_balance_exhausted"}:
                info.update(state="retryable_rejection", failure_code="rate_limited")
                _save_json(journal, info)
                if attempt < 2:
                    try:
                        delay = float(response.headers.get("retry-after", str(2 ** attempt)))
                    except ValueError:
                        delay = 2 ** attempt
                    time.sleep(min(max(delay, 1), 30))
                    continue
                raise ScribeRetryableError("Scribe rate limit reached; safe retry limit exhausted")
            if response.status_code != 200:
                if response.status_code >= 500 or response.status_code == 408:
                    info.update(state="unknown", failure_code="provider_outcome_unknown")
                    _save_json(journal, info)
                    raise ScribeReconciliationRequired(f"Scribe HTTP {response.status_code}; processing outcome unknown; audio was not resubmitted")
                failure = "quota_exhausted" if response.status_code in {402, 429} else "authorization" if response.status_code in {401, 403} else "redirect" if 300 <= response.status_code < 400 else "invalid_input" if response.status_code in {400, 413, 415, 422} else "provider_error"
                info.update(state="permanent_failure", failure_code=failure, provider_error_code=code)
                _save_json(journal, info)
                detail = "; redirect target: " + info["redirect_location"] if failure == "redirect" else ""
                raise ScribePermanentError(f"Scribe HTTP {response.status_code}: {failure}{detail}", code=failure)
            try:
                payload = response.json()
                validate_response(payload)
            except (ValueError, ScribeError):
                info.update(state="unknown", failure_code="invalid_provider_response")
                _save_json(journal, info)
                raise ScribeReconciliationRequired("Scribe returned an invalid completed response; audio was not resubmitted") from None
            _save_json(cached, payload)
            info.update(state="completed", transcription_id=payload.get("transcription_id"), completed_at=datetime.now(UTC).isoformat())
            _save_json(journal, info)
            return payload
    raise ScribeRetryableError("Scribe safe retry limit exhausted")


def validate_response(payload: object) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("words"), list):
        raise ScribeError("Scribe response is missing word timestamps")
    words = payload["words"]
    if payload.get("text", "").strip() and not any(
        w.get("type") == "word" for w in words if isinstance(w, dict)
    ):
        raise ScribeError("Scribe returned text without timed words")
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str):
            raise ScribeError("Scribe returned an invalid word")
        if word.get("type") != "word":
            continue
        start, end = word.get("start"), word.get("end")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
        ):
            raise ScribeError("Scribe returned invalid word timestamps")


def transcribe_tracks(
    *, tracks, metadata: dict, cache_dir: Path, progress_callback=None, diagnostics=None,
    cache_only: bool = False,
):
    from mac_transcriber.asr import (
        Segment,
        _is_placeholder_speaker,
        speaker_tracks_from_segments_multi,
        normalize_text,
        dedupe_segments,
    )
    from mac_transcriber.glossary import apply_glossary

    segments = []
    speaker_offset = 0
    terms = meeting_keyterms(metadata)
    modes = [microphone_mode(track, i, metadata) for i, track in enumerate(tracks)]
    for index, track in enumerate(tracks):
        if progress_callback:
            progress_callback(index, len(tracks))
        payload = request_transcript(
            track.path,
            terms=terms,
            cache_dir=cache_dir,
            single_speaker=modes[index] == "single",
            **({"cache_only": True} if cache_only else {}),
        )
        words = [word for word in payload["words"] if word.get("type") == "word"]
        speaker_ids = dict.fromkeys(word.get("speaker_id") or "unknown" for word in words)
        inferred = infer_microphone(words) if modes[index] == "auto" else {"mode": modes[index], "reason": "explicit_meeting_override"}
        if diagnostics is not None:
            diagnostics.append({"track": track.path.name, "zoom_name": track.speaker,
                                "requested_mode": modes[index], **inferred})
        names = {}
        named = not _is_placeholder_speaker(track.speaker)
        if modes[index] == "single" and named:
            names = dict.fromkeys(speaker_ids, track.speaker)
        elif modes[index] == "auto" and inferred["mode"] == "single" and named:
            names[inferred["dominant_id"]] = track.speaker
        for sp in speaker_ids:
            if sp not in names:
                speaker_offset += 1
                label = f"Speaker {speaker_offset}"
                if modes[index] == "auto" and inferred["mode"] == "room" and named:
                    label = f"{track.speaker} / {label}"
                names[sp] = label
        # Explicit spacing tokens preserve the provider's punctuation and word boundaries.
        explicit_spacing = any(w.get("type") == "spacing" for w in payload["words"])
        current = None
        pending_space = ""
        for word in payload["words"]:
            if word.get("type") == "spacing":
                pending_space += word["text"]
                continue
            if word.get("type") != "word":
                continue
            speaker = names[word.get("speaker_id") or "unknown"]
            if (
                current is None
                or current.speaker != speaker
                or word["start"] - current.end > 1.5
                or word["end"] - current.start > 20
            ):
                current = Segment(
                    speaker=speaker,
                    track=track.path.name,
                    start=word["start"],
                    end=word["end"],
                    text=word["text"],
                )
                segments.append(current)
            else:
                separator = pending_space if explicit_spacing else " "
                current.text += separator + word["text"]
                current.end = max(current.end, word["end"])
            pending_space = ""
        if progress_callback:
            progress_callback(index + 1, len(tracks))
    for segment in segments:
        segment.text = apply_glossary(normalize_text(segment.text))
    segments = dedupe_segments(segments)
    return segments, speaker_tracks_from_segments_multi(tracks, segments) or tracks
