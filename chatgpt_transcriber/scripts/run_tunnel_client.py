"""Start the official tunnel-client without sourcing a shell env file.

The control-plane key is read from the one dedicated local env file and placed only
in the official tunnel-client process environment.  It is never forwarded to the
GigaAM service or printed by this helper.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_transcriber.chatgpt_auth import (  # noqa: E402
    DEFAULT_TUNNEL_ENV_FILE,
    load_control_plane_api_key,
)


DEFAULT_TUNNEL_BINARY = (
    REPO_ROOT
    / ".local/chatgpt_transcriber/tunnel-client/v0.0.13/tunnel-client"
)
DEFAULT_PROFILE_FILE = (
    REPO_ROOT
    / ".local/chatgpt_transcriber/tunnel-runtime-profiles/gigaam-chatgpt.yaml"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official OpenAI Secure MCP Tunnel client for GigaAM."
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_TUNNEL_ENV_FILE)
    parser.add_argument("--profile-file", type=Path, default=DEFAULT_PROFILE_FILE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_TUNNEL_BINARY)
    parser.add_argument(
        "--wait-for-mcp-url",
        default="http://127.0.0.1:18110/mcp",
        help="Local MCP endpoint whose TCP listener must be ready before the tunnel runs.",
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=60.0,
        help="Maximum time to wait for the local MCP listener.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env_file = args.env_file.expanduser()
    profile_file = args.profile_file.expanduser()
    binary = args.binary.expanduser()

    if not profile_file.is_file():
        raise SystemExit(f"Tunnel profile is missing: {profile_file}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"Tunnel client is missing or not executable: {binary}")

    _wait_for_mcp_listener(args.wait_for_mcp_url, args.startup_wait_seconds)

    control_plane_api_key = load_control_plane_api_key(env_file)
    if not control_plane_api_key:
        raise SystemExit(
            "CONTROL_PLANE_API_KEY is missing from the selected local tunnel env file."
        )

    child_env = dict(os.environ)
    child_env["CONTROL_PLANE_API_KEY"] = control_plane_api_key
    os.execvpe(
        str(binary),
        [str(binary), "run", "--profile-file", str(profile_file)],
        child_env,
    )


def _wait_for_mcp_listener(url: str, timeout_seconds: float) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit("--wait-for-mcp-url must be an absolute HTTP URL")
    if timeout_seconds <= 0:
        raise SystemExit("--startup-wait-seconds must be positive")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((parsed.hostname, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"Local MCP listener did not become ready: {url}")


if __name__ == "__main__":
    main()
