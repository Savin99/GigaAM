"""Configuration for the ChatGPT-to-GigaAM MCP bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from mac_transcriber.chatgpt_auth import (
    DEFAULT_TUNNEL_ENV_FILE,
    load_chatgpt_bridge_api_key,
)


DEFAULT_ALLOWED_DOWNLOAD_HOST_SUFFIXES = (
    "chatgpt.com",
    "openai.com",
    "oaiusercontent.com",
)


@dataclass(frozen=True)
class Settings:
    service_url: str
    bridge_api_key: str
    max_batch_files: int
    max_file_bytes: int
    download_timeout_seconds: float
    service_timeout_seconds: float
    temp_root: Path
    allowed_download_host_suffixes: tuple[str, ...]
    host: str
    port: int


def load_settings() -> Settings:
    tunnel_env_file = _tunnel_env_file()

    service_url = os.environ.get(
        "CHATGPT_TRANSCRIBER_SERVICE_URL", "http://127.0.0.1:18003"
    ).strip()
    parsed_service_url = urlparse(service_url)
    if parsed_service_url.scheme not in {"http", "https"} or not parsed_service_url.netloc:
        raise ValueError("CHATGPT_TRANSCRIBER_SERVICE_URL must be an absolute HTTP URL")

    bridge_api_key = load_chatgpt_bridge_api_key(tunnel_env_file)
    if not bridge_api_key:
        raise ValueError(
            "The ChatGPT bridge credential is missing. "
            "Create the GigaAM tunnel key and save it in the selected local env file."
        )

    return Settings(
        service_url=service_url.rstrip("/"),
        bridge_api_key=bridge_api_key,
        max_batch_files=_positive_int("CHATGPT_TRANSCRIBER_MAX_BATCH_FILES", 10),
        max_file_bytes=_positive_int(
            "CHATGPT_TRANSCRIBER_MAX_FILE_BYTES", 1_073_741_824
        ),
        download_timeout_seconds=_positive_float(
            "CHATGPT_TRANSCRIBER_DOWNLOAD_TIMEOUT_SECONDS", 900.0
        ),
        service_timeout_seconds=_positive_float(
            "CHATGPT_TRANSCRIBER_SERVICE_TIMEOUT_SECONDS", 120.0
        ),
        temp_root=Path(
            os.environ.get(
                "CHATGPT_TRANSCRIBER_TEMP_ROOT", ".local/chatgpt_transcriber/tmp"
            )
        ).expanduser(),
        allowed_download_host_suffixes=_host_suffixes_from_env(),
        host=os.environ.get("CHATGPT_TRANSCRIBER_HOST", "127.0.0.1").strip()
        or "127.0.0.1",
        port=_positive_int("CHATGPT_TRANSCRIBER_PORT", 18110),
    )


def _tunnel_env_file() -> Path:
    """Return the private OpenAI tunnel-key file without loading it into env."""

    raw_path = os.environ.get("CHATGPT_TRANSCRIBER_ENV_FILE", "").strip()
    if not raw_path:
        return DEFAULT_TUNNEL_ENV_FILE
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ValueError(f"CHATGPT_TRANSCRIBER_ENV_FILE does not exist: {path}")
    return path


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _host_suffixes_from_env() -> tuple[str, ...]:
    raw = os.environ.get("CHATGPT_TRANSCRIBER_ALLOWED_DOWNLOAD_HOSTS", "").strip()
    values = raw.split(",") if raw else list(DEFAULT_ALLOWED_DOWNLOAD_HOST_SUFFIXES)
    suffixes = tuple(
        value.strip().lower().lstrip("*.")
        for value in values
        if value.strip().lstrip("*.")
    )
    if not suffixes:
        raise ValueError("CHATGPT_TRANSCRIBER_ALLOWED_DOWNLOAD_HOSTS must not be empty")
    return suffixes
