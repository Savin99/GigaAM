#!/usr/bin/env bash
set -euo pipefail

# Start the reverse SSH tunnel used by the VPS relay. Before binding the remote
# port, clear stale sshd listeners left by a broken previous tunnel.

tunnel_identity_file="${MAC_TRANSCRIBER_TUNNEL_IDENTITY_FILE:-$HOME/.ssh/valamis_vps_ed25519}"
tunnel_remote_forward="${MAC_TRANSCRIBER_TUNNEL_REMOTE_FORWARD:-127.0.0.1:18013:127.0.0.1:18003}"
tunnel_target="${MAC_TRANSCRIBER_TUNNEL_TARGET:-root@111.88.224.176}"

remote_bind="${tunnel_remote_forward%%:*}"
rest="${tunnel_remote_forward#*:}"
remote_port="${rest%%:*}"

ssh_base=(
  "/usr/bin/ssh"
  "-o" "BatchMode=yes"
  "-o" "ConnectTimeout=10"
  "-o" "StrictHostKeyChecking=accept-new"
)

if [[ -f "$tunnel_identity_file" ]]; then
  ssh_base+=("-i" "$tunnel_identity_file" "-o" "IdentitiesOnly=yes")
fi
if [[ -n "${MAC_TRANSCRIBER_TUNNEL_BIND_INTERFACE:-}" ]]; then
  ssh_base+=("-o" "BindInterface=$MAC_TRANSCRIBER_TUNNEL_BIND_INTERFACE")
fi

cleanup_remote_forward() {
  [[ "$remote_port" =~ ^[0-9]+$ ]] || return 0
  "${ssh_base[@]}" "$tunnel_target" \
    "ss -ltnp 'sport = :$remote_port' 2>/dev/null | awk '/sshd/ {print \$NF}' | sed -n 's/.*pid=\\([0-9][0-9]*\\).*/\\1/p' | sort -u | xargs -r kill" \
    >/dev/null 2>&1 || true
}

cleanup_remote_forward

exec "${ssh_base[@]}" \
  "-N" \
  "-o" "ExitOnForwardFailure=yes" \
  "-o" "ServerAliveInterval=30" \
  "-o" "ServerAliveCountMax=3" \
  "-R" "$tunnel_remote_forward" \
  "$tunnel_target"
