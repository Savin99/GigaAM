# GigaAM Transcriber for ChatGPT Web

This is a private, tool-only ChatGPT MCP app. It lets a user attach several Voice
Memos recordings in a ChatGPT Web message, sends each recording to the existing
private GigaAM service as a separate asynchronous job, and returns timestamped
Markdown transcript chunks to the same conversation.

It deliberately has no custom widget. ChatGPT's supported file-parameter contract
passes an array of attached files directly to `transcribe_audio_batch`, which is the
lowest-friction way to handle a batch of `.m4a` recordings.

```text
ChatGPT Web attachments
        |
        v
private MCP bridge (/mcp)
        |  one scoped job per recording
        v
private GigaAM service (/chatgpt/meetings)
        |
        v
transcript.md -> get_transcript -> ChatGPT conversation
```

## What the app does

- `transcribe_audio_batch(audio_files[])` accepts from one to ten audio attachments.
  It downloads every temporary ChatGPT file URL during the tool call, then creates one
  retry-safe GigaAM job per file.
- `get_transcription_status(job_ids[])` reports actual progress. A job is usable as
  soon as `transcript_ready` is true.
- `get_transcript(job_id, offset, limit)` returns Markdown directly to ChatGPT, so the
  conversation can summarize, correct, search, or otherwise work with the recording.

The existing GigaAM worker intentionally processes audio one at a time. A batch is
therefore accepted at once but transcribed serially; the bridge never promises false
parallel processing.

The bridge sends `transcript_only=true` to GigaAM. That skips report/PDF generation,
meeting-memory lookup, and AI enrichment. This keeps personal audio transcription
separate from the Slack/Zoom reporting pipeline and avoids its cost/privacy boundary.

## Local setup

Install dependencies once in the project virtual environment:

```bash
uv pip install --python .venv/bin/python -r mac_transcriber/requirements.txt
```

Start the existing private GigaAM service first, normally on `127.0.0.1:18003`. Then
start this bridge in a second terminal:

```bash
export CHATGPT_TRANSCRIBER_SERVICE_URL=http://127.0.0.1:18003
export CHATGPT_TRANSCRIBER_PORT=18110
export CHATGPT_TRANSCRIBER_ENV_FILE="$PWD/.local/gigaam-chatgpt-tunnel.env"
.venv/bin/python -m chatgpt_transcriber.server
```

The selected env file contains the dedicated OpenAI tunnel key. The bridge and service
derive an opaque local bearer from that key, so the raw OpenAI key is never forwarded
to `127.0.0.1:18003` and the bridge never reads the legacy
`MAC_TRANSCRIBER_API_KEY`. Its contents are never printed or returned through MCP
tools.

The bridge binds to `127.0.0.1` by default. For a VPS sidecar, set
`CHATGPT_TRANSCRIBER_SERVICE_URL` to its private GigaAM relay URL and keep the MCP
server itself behind a private tunnel or authenticated HTTPS proxy.

### Durable macOS operation

After the dedicated tunnel key, the official `tunnel-client` binary, and its local
profile have been created, install the three user LaunchAgents:

```bash
bash chatgpt_transcriber/scripts/install_launchd.sh
```

This keeps a dedicated GigaAM service, the local MCP bridge, and the official outbound
tunnel alive after login and restarts. The dedicated service uses its own local data
directory and port `18103`, so it does not modify the existing Slack/Zoom service or
its old tunnel. The installer writes no API key into a plist: the bridge reads only
the named key from `.local/gigaam-chatgpt-tunnel.env`, and the tunnel runner puts that
key only into the official tunnel-client process environment. Check the local services
without dumping launchd environment variables:

```bash
launchctl list | rg 'com\.ilyaqa\.gigaam-chatgpt'
curl -fsS http://127.0.0.1:18003/healthz
curl -fsS http://127.0.0.1:18103/healthz
```

## Connect in ChatGPT Web

1. In ChatGPT, open **Settings → Security and login** and enable **Developer mode**.
2. Expose the local `/mcp` endpoint through an OpenAI Secure MCP Tunnel or a temporary
   HTTPS tunnel. Do not expose the underlying GigaAM service itself.
3. Open **ChatGPT Plugins**, press the plus button, and add the HTTPS MCP URL ending
   in `/mcp`.
4. In a new chat, attach several `.m4a` files and say: “Транскрибируй все эти записи
   отдельно, по-русски.”
5. When the app reports job IDs, ask “проверь статус расшифровок”; once ready, ask
   ChatGPT to retrieve a transcript and work with it.

Long recordings are returned in Markdown chunks. ChatGPT can request the next chunk
automatically when it needs more text.

## Security and operating limits

- The bridge accepts only HTTPS download URLs on ChatGPT/OpenAI file hosts by default.
  Override `CHATGPT_TRANSCRIBER_ALLOWED_DOWNLOAD_HOSTS` only when the ChatGPT runtime
  documents another required file host.
- Default caps are ten files per batch and 1 GiB per file. Override with
  `CHATGPT_TRANSCRIBER_MAX_BATCH_FILES` and `CHATGPT_TRANSCRIBER_MAX_FILE_BYTES`.
- A deterministic UUID derived from the ChatGPT `file_id` makes retried tool calls
  idempotent without persisting temporary download URLs.
- The bridge uses separate `/chatgpt/*` service routes. Its credential can create only
  transcript-only dictaphone jobs and read only Markdown transcripts created through
  that ingress; it cannot list meetings, access Slack/Zoom jobs, reports, PDFs, or
  other artifacts.
- This first version is for one private Developer Mode connection. It does not have
  OAuth user tenancy, so do not publish it or share its endpoint. A multi-user/public
  version needs OAuth 2.1, per-user job ownership, retention/TTL policy, and a stable
  HTTPS deployment.

## Verification

```bash
.venv/bin/python -m pytest -q \
  mac_transcriber/tests/test_service.py \
  mac_transcriber/tests/test_chatgpt_transcriber.py

npx @modelcontextprotocol/inspector@latest
```

In MCP Inspector choose **Streamable HTTP** and enter
`http://127.0.0.1:18110/mcp`. Verify that the `transcribe_audio_batch` input schema
advertises `audio_files` as an array and has `openai/fileParams: ["audio_files"]`.

Current OpenAI references used for this implementation:

- https://developers.openai.com/plugins/reference
- https://developers.openai.com/plugins/build/mcp-server
- https://developers.openai.com/plugins/deploy/connect-chatgpt
