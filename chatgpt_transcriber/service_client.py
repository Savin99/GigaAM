"""Safe, small client for the private GigaAM meeting API."""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from chatgpt_transcriber.config import Settings


JOB_NAMESPACE = uuid.UUID("3d808cd3-0196-4c6a-8ae1-c6fae4d93019")
SUPPORTED_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".m4b",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


class OpenAIFile(BaseModel):
    """The exact file-object shape required by ChatGPT Apps file params."""

    download_url: str
    file_id: str
    mime_type: str = ""
    file_name: str = ""


class BridgeError(RuntimeError):
    """An error that is safe to return to ChatGPT without internal details."""


@dataclass(frozen=True)
class SubmittedJob:
    job_id: str
    file_name: str
    state: str
    reused_existing_job: bool


def job_id_for_file_id(file_id: str) -> str:
    if not file_id.strip():
        raise BridgeError("The attached file has no file identifier.")
    return str(uuid.uuid5(JOB_NAMESPACE, file_id))


def validate_job_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, ValueError) as exc:
        raise BridgeError("The transcription job identifier is invalid.") from exc


def is_transcript_ready(status: dict[str, object]) -> bool:
    explicit = status.get("transcript_ready")
    if isinstance(explicit, bool):
        return explicit
    artifacts = status.get("artifacts")
    return isinstance(artifacts, list) and "transcript_md" in artifacts


def public_status(status: dict[str, object]) -> dict[str, object]:
    """Limit status output to fields useful to a ChatGPT user."""

    metadata = status.get("metadata")
    result = status.get("result")
    transcript_ready = is_transcript_ready(status)
    return {
        "job_id": str(status.get("id") or ""),
        "state": "transcript_ready" if transcript_ready else str(status.get("status") or "unknown"),
        "backend_state": str(status.get("status") or "unknown"),
        "phase": str(status.get("phase") or ""),
        "progress": _safe_progress(status.get("progress")),
        "message": _safe_message(status.get("message")),
        "transcript_ready": transcript_ready,
        "source_filename": (
            str(metadata.get("source_filename") or "")
            if isinstance(metadata, dict)
            else ""
        ),
        "segments": result.get("segments") if isinstance(result, dict) else None,
    }


class GigaAMServiceClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def submit_audio_file(
        self,
        file: OpenAIFile,
        *,
        title: str = "",
        language_code: str = "ru",
    ) -> SubmittedJob:
        file_name = _safe_file_name(file.file_name, file.file_id)
        _validate_audio_file(file, file_name, self.settings)
        job_id = job_id_for_file_id(file.file_id)

        existing = await self.get_status(job_id, missing_ok=True)
        if existing is not None:
            return SubmittedJob(
                job_id=job_id,
                file_name=file_name,
                state=str(existing.get("status") or "uploaded"),
                reused_existing_job=True,
            )

        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="chatgpt-audio-", dir=self.settings.temp_root
        ) as temp_dir:
            audio_path = Path(temp_dir) / "audio.bin"
            await self._download_audio(file.download_url, audio_path)
            await self._create_meeting(
                audio_path,
                file_name=file_name,
                mime_type=file.mime_type or "application/octet-stream",
                title=title.strip(),
                language_code=language_code,
                request_id=job_id,
            )

        return SubmittedJob(
            job_id=job_id,
            file_name=file_name,
            state="uploaded",
            reused_existing_job=False,
        )

    async def get_status(
        self, job_id: str, *, missing_ok: bool = False
    ) -> dict[str, object] | None:
        safe_job_id = validate_job_id(job_id)
        async with self._service_http_client() as client:
            response = await client.get(f"/chatgpt/meetings/{safe_job_id}/status")
        if response.status_code == 404 and missing_ok:
            return None
        if response.status_code == 404:
            raise BridgeError("This transcription job was not found.")
        _raise_for_service_error(response, "Could not read the transcription status.")
        payload = _json_object(response, "The transcription service returned invalid status data.")
        return payload

    async def get_transcript(self, job_id: str) -> str | None:
        safe_job_id = validate_job_id(job_id)
        status = await self.get_status(safe_job_id)
        if status is None or not is_transcript_ready(status):
            return None
        async with self._service_http_client() as client:
            response = await client.get(
                f"/chatgpt/meetings/{safe_job_id}/transcript"
            )
        if response.status_code == 404:
            return None
        _raise_for_service_error(response, "Could not retrieve the transcript.")
        return response.text

    async def _download_audio(self, download_url: str, target: Path) -> None:
        _validate_download_url(download_url, self.settings)
        timeout = httpx.Timeout(self.settings.download_timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            async with client.stream("GET", download_url) as response:
                if response.is_redirect:
                    raise BridgeError("The attached-file download URL redirected unexpectedly.")
                _raise_for_service_error(response, "ChatGPT could not provide the attached audio.")
                declared_size = response.headers.get("content-length")
                if declared_size and declared_size.isdigit() and int(declared_size) > self.settings.max_file_bytes:
                    raise BridgeError("This audio file exceeds the configured size limit.")
                received = 0
                with target.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        received += len(chunk)
                        if received > self.settings.max_file_bytes:
                            raise BridgeError("This audio file exceeds the configured size limit.")
                        output.write(chunk)
        if target.stat().st_size == 0:
            raise BridgeError("The attached audio file is empty.")

    async def _create_meeting(
        self,
        audio_path: Path,
        *,
        file_name: str,
        mime_type: str,
        title: str,
        language_code: str,
        request_id: str,
    ) -> None:
        data = {
            "request_id": request_id,
        }
        if title:
            data["title"] = title
        with audio_path.open("rb") as audio_handle:
            async with self._service_http_client() as client:
                response = await client.post(
                    "/chatgpt/meetings",
                    data=data,
                    files={"file": (file_name, audio_handle, mime_type)},
                )
        _raise_for_service_error(response, "Could not queue the audio for transcription.")
        payload = _json_object(response, "The transcription service returned invalid job data.")
        if str(payload.get("id") or "") != request_id:
            raise BridgeError("The transcription service returned an unexpected job identifier.")

    def _service_http_client(self) -> httpx.AsyncClient:
        headers = {}
        if self.settings.bridge_api_key:
            headers["Authorization"] = f"Bearer {self.settings.bridge_api_key}"
        return httpx.AsyncClient(
            base_url=self.settings.service_url,
            headers=headers,
            timeout=httpx.Timeout(self.settings.service_timeout_seconds),
            transport=self.transport,
        )


def _validate_audio_file(file: OpenAIFile, file_name: str, settings: Settings) -> None:
    if not file.download_url.strip():
        raise BridgeError("The attached file has no download URL.")
    mime_type = file.mime_type.strip().lower()
    extension = Path(file_name).suffix.lower()
    if not mime_type.startswith("audio/") and extension not in SUPPORTED_EXTENSIONS:
        raise BridgeError("Please attach an audio recording (for example .m4a, .mp3, .wav, or .webm).")
    _validate_download_url(file.download_url, settings)


def _validate_download_url(download_url: str, settings: Settings) -> None:
    parsed = urlparse(download_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise BridgeError("ChatGPT provided an invalid audio download URL.")
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in settings.allowed_download_host_suffixes):
        raise BridgeError("The attached-file download host is not allowed by this bridge.")


def _safe_file_name(value: str, fallback_file_id: str) -> str:
    name = Path(value).name.strip()
    return name or f"recording-{fallback_file_id[:12]}.m4a"


def _raise_for_service_error(response: httpx.Response, message: str) -> None:
    if response.is_success:
        return
    if response.status_code == 401:
        raise BridgeError("The private transcription service rejected the bridge credentials.")
    if response.status_code == 413:
        raise BridgeError("The private transcription service rejected this file as too large.")
    raise BridgeError(message)


def _json_object(response: httpx.Response, message: str) -> dict[str, object]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise BridgeError(message) from exc
    if not isinstance(payload, dict):
        raise BridgeError(message)
    return payload


def _safe_progress(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, round(float(value), 3)))
    return None


def _safe_message(value: object) -> str:
    return str(value or "")[:300]
