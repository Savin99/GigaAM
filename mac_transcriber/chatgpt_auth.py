"""Narrow local credential for the ChatGPT transcription bridge.

The OpenAI Secure MCP Tunnel needs a Control Plane API key.  The local GigaAM
service must never receive that raw key over HTTP, so both local processes derive
the same opaque, bridge-only bearer credential from it instead.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


CONTROL_PLANE_API_KEY_NAME = "CONTROL_PLANE_API_KEY"
DEFAULT_TUNNEL_ENV_FILE = (
    Path(__file__).resolve().parents[1] / ".local" / "gigaam-chatgpt-tunnel.env"
)
_BRIDGE_DERIVATION_CONTEXT = b"gigaam-chatgpt-bridge/v1"


def derive_chatgpt_bridge_api_key(control_plane_api_key: str) -> str:
    """Return an opaque local bearer without exposing the OpenAI API key."""

    normalized = control_plane_api_key.strip()
    if not normalized:
        return ""
    return hmac.new(
        normalized.encode("utf-8"),
        _BRIDGE_DERIVATION_CONTEXT,
        hashlib.sha256,
    ).hexdigest()


def load_chatgpt_bridge_api_key(env_file: Path | None = None) -> str:
    """Derive the bridge credential from one named local env-file value only."""

    value = load_control_plane_api_key(env_file)
    return derive_chatgpt_bridge_api_key(value)


def load_control_plane_api_key(env_file: Path | None = None) -> str:
    """Read only the tunnel runtime key from its dedicated local env file.

    This is intentionally narrow: the launchd helper needs this one value to put it
    into the official tunnel client's process environment, but must not source or
    import any other local shell settings.
    """

    return _read_named_env_value(
        env_file or DEFAULT_TUNNEL_ENV_FILE,
        CONTROL_PLANE_API_KEY_NAME,
    )


def _read_named_env_value(path: Path, expected_name: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == expected_name:
            return value.strip().strip('"').strip("'")
    return ""
