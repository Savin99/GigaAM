#!/usr/bin/env bash
set -euo pipefail

# Installs only the two ChatGPT/GigaAM user agents.  Neither generated plist
# contains the OpenAI runtime key; both reference the fixed ignored env file.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
chatgpt_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$chatgpt_dir/.." && pwd)"
launchd_source_dir="$chatgpt_dir/launchd"
launch_agents_dir="$HOME/Library/LaunchAgents"
log_dir="$HOME/Library/Logs/gigaam-chatgpt"
domain="gui/$(id -u)"
launchctl_bin="/bin/launchctl"
plutil_bin="/usr/bin/plutil"

tunnel_env_file="$repo_root/.local/gigaam-chatgpt-tunnel.env"
profile_file="$repo_root/.local/chatgpt_transcriber/tunnel-runtime-profiles/gigaam-chatgpt.yaml"
tunnel_binary="$repo_root/.local/chatgpt_transcriber/tunnel-client/v0.0.13/tunnel-client"
python_bin="$repo_root/.venv/bin/python"
runner="$chatgpt_dir/scripts/run_tunnel_client.py"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 66
  fi
}

require_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    echo "Missing or non-executable $label: $path" >&2
    exit 69
  fi
}

install_agent() {
  local label="$1"
  local source_path="$launchd_source_dir/$label.plist.example"
  local target_path="$launch_agents_dir/$label.plist"
  local service="$domain/$label"

  require_file "$source_path" "LaunchAgent template"
  cp "$source_path" "$target_path"
  chmod 600 "$target_path"

  "$plutil_bin" -lint "$target_path" >/dev/null
  "$launchctl_bin" bootout "$service" >/dev/null 2>&1 || true
  "$launchctl_bin" enable "$service" >/dev/null 2>&1 || true
  # Recent macOS versions may reject bootstrap for a previously registered
  # legacy LaunchAgent even after bootout. The compatible load path registers
  # the same user-owned plist without touching unrelated agents.
  if ! "$launchctl_bin" bootstrap "$domain" "$target_path"; then
    "$launchctl_bin" load -w "$target_path"
  fi
  "$launchctl_bin" kickstart -k "$service" >/dev/null 2>&1 || true
  echo "Installed and started $label"
}

require_executable "$launchctl_bin" "launchctl"
require_executable "$plutil_bin" "plutil"
require_executable "$python_bin" "project Python"
require_executable "$tunnel_binary" "official tunnel-client"
require_file "$tunnel_env_file" "dedicated tunnel env file"
require_file "$profile_file" "tunnel profile"
require_file "$runner" "tunnel runner"

mkdir -p "$launch_agents_dir" "$log_dir" "$repo_root/.local/chatgpt_transcriber/tmp"
chmod 700 "$log_dir" "$repo_root/.local/chatgpt_transcriber" "$repo_root/.local/chatgpt_transcriber/tmp"
chmod 600 "$tunnel_env_file" "$profile_file"
for log_name in service.out.log service.err.log bridge.out.log bridge.err.log tunnel.out.log tunnel.err.log; do
  touch "$log_dir/$log_name"
  chmod 600 "$log_dir/$log_name"
done

install_agent "com.ilyaqa.gigaam-chatgpt-service"
install_agent "com.ilyaqa.gigaam-chatgpt-bridge"
install_agent "com.ilyaqa.gigaam-chatgpt-tunnel"

echo "LaunchAgents: $launch_agents_dir"
echo "Logs: $log_dir"
