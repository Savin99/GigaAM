# Mac Transcriber

Local GigaAM service that implements the `slack_zoom` `meeting_mvp` transcriber API.

The service lives on the Mac because GigaAM/RNNT is too heavy for the small VPS. The
VPS keeps Slack, Zoom, storage, Redis/RQ, and polling. This service receives the audio
bundle, runs `v3_e2e_rnnt`, applies the local glossary, and exposes transcript
artifacts back to `slack_zoom`.

The launchd agent includes `/opt/homebrew/bin` in `PATH` so GigaAM audio loading can
find Homebrew `ffmpeg`.

When Zoom provides separate participant tracks, those tracks remain the source of
speaker labels. When only one mixed audio file is available, the service first tries
local speaker diarization with `pyannote/speaker-diarization-community-1`, then sends
speaker turns to GigaAM for recognition. If pyannote is not installed, the Hugging Face
token is missing, or diarization fails, the service falls back to the original
single-speaker segmentation.

## Architecture

```text
Slack/Zoom -> VPS slack_zoom -> Docker host relay -> SSH reverse tunnel -> Mac transcriber
                                             ^                              |
                                             |                              v
                                     artifacts polling              GigaAM RNNT
```

Current ports:

- Mac local service: `127.0.0.1:18003`
- SSH reverse tunnel on VPS: `127.0.0.1:18013 -> Mac 127.0.0.1:18003`
- VPS Docker bridge relay: `172.22.0.1:18004 -> 127.0.0.1:18013`
- `slack_zoom` containers use `http://host.docker.internal:18004`

The transcriber is not exposed publicly.

## Manual Run

### ElevenLabs Scribe v2

To replace local recognition and diarization, set these in the ignored `.env.local`
at the repository root before restarting the transcriber LaunchAgent:

```dotenv
MAC_TRANSCRIBER_MODEL=scribe_v2
ELEVENLABS_API_KEY=your-private-api-key
```

The service loads this file at startup. The default remains `v3_e2e_rnnt` until
configured; rollback is setting `MAC_TRANSCRIBER_MODEL=v3_e2e_rnnt` and restarting
the transcriber after the current meeting finishes. The tunnel and VPS need no changes.

Scribe receives each complete input track using Russian, word timestamps, diarization,
and verbatim text, with audio event tags disabled. Microphone mode is inferred independently for each meeting; sustained single
voices use Zoom labels and shared microphones use connection / Speaker labels.
See "Microphone modes and Scribe terminology" below for configuration.
`metadata.terms_hint` (a list or comma/semicolon/newline-separated terms) is passed as
validated `keyterms`. The existing glossary, artifact/report generation, PDF, and Slack
contract remain in place. No automatic fallback to GigaAM occurs on API errors.

Validated provider responses, including raw word timings, are saved under each
meeting's `scribe_cache/`, outside the exposed artifact routes. Cache keys include
the audio content and request options, avoiding another paid upload on retry.
HTTP 429 is retried up to three attempts; ambiguous timeouts are reported rather than
automatically resubmitted. Each participant track is a separate billable request.
Audio is sent to ElevenLabs with its standard retention policy.

Requests also have a `*.request.json` journal with timestamps, provider request ID
(when received), and transport error type. An interrupted or ambiguous submission
blocks another upload until an operator reconciles it with the provider. Preserve
the journal as recovery evidence before explicitly clearing it for a retry.

For uncertain cloud submissions, `MAC_TRANSCRIBER_SCRIBE_LOCAL_RECOVERY=1` enables
recovery from validated Scribe caches plus local GigaAM for missing tracks. Set
`MAC_TRANSCRIBER_LOCAL_MODEL_CACHE` to the existing GigaAM cache if needed (default
`~/.cache/gigaam`). Recovery never submits more cloud ASR requests or downloads
models. Each track has a resumable checkpoint and its engine is recorded in
`asr_metadata.json`; local speaker labels identify Zoom connections. This option
does not change the transcript-only ChatGPT bridge workflow.

To recover one already failed meeting, run
`.venv/bin/python mac_transcriber/scripts/recover_scribe_locally.py <meeting-dir> --publish`.
Only a complete transcript is published and queued for the normal report. Keep the
original audio, request journals, and `local_recovery/` checkpoints for the audit.

The Scribe connection enables TCP keepalive during long requests. If VPN routing
breaks the private SSH tunnel, `MAC_TRANSCRIBER_TUNNEL_BIND_INTERFACE=en1` in the
tunnel LaunchAgent environment can bind SSH to the physical interface. Verify the
interface and remote health before resuming jobs.

After ASR recovery, generate a deferred report for just that meeting with
`.venv/bin/python mac_transcriber/scripts/drain_reports_via_claude.py --meeting-id <id>`.
This also restricts reuploads to the selected meeting.

API reference: https://elevenlabs.io/docs/api-reference/speech-to-text/convert

### Local GigaAM

```bash
uv pip install --python .venv/bin/python -r mac_transcriber/requirements.txt
export HF_TOKEN="hf_..."
MAC_TRANSCRIBER_API_KEY=change-me \
MAC_TRANSCRIBER_ROOT=/Users/ilyaqa/Projects/GigaAM/.local/mac_transcriber \
.venv/bin/python -m uvicorn mac_transcriber.service:app --host 127.0.0.1 --port 18003
```

Health check:

```bash
curl -fsS http://127.0.0.1:18003/healthz
```

## Run With launchd

```bash
uv pip install --python .venv/bin/python -r mac_transcriber/requirements.txt
```

For single-file speaker diarization, accept the Hugging Face terms for
`pyannote/speaker-diarization-community-1`, create a read token, and keep it in
`HF_TOKEN` when installing/running the LaunchAgent. The default diarization device is
CPU; override with `MAC_TRANSCRIBER_DIARIZATION_DEVICE` if needed.

#### In-room shared microphone (per-track diarization)

When some participants share a single room microphone, their Zoom track holds several
voices but is labelled with one name. Set `MAC_TRANSCRIBER_DIARIZE_TRACKS=1` to run
diarization *inside each participant track*: a track with one real voice keeps its Zoom
name (clean remote participants are untouched), a track with two or more real voices is
split into `Speaker 1`, `Speaker 2`, … numbered across the whole meeting. Requires the
same `HF_TOKEN`.

A track counts as a shared microphone only if it has **≥2 clusters above
`MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S`** (default `20` seconds). This guard is
essential: pyannote emits tiny (1–9 s) spurious clusters even on a clean single-speaker
track, and without the floor that noise was counted as a second speaker and fragmented
clean all-remote tracks into bogus `Speaker N`. On real meetings the gap is wide (noise
≤9 s vs real voices ≥160 s), so the floor cleanly separates the two cases and makes this
safe to enable globally. Optional hard cap per track:
`MAC_TRANSCRIBER_DIARIZE_TRACKS_MAX_SPEAKERS`. It runs the full pyannote pipeline once per
track (~7 min/73-min track on MPS). Speaker labels stay anonymous — mapping `Speaker N`
to real names (voice enrollment) is a separate step.

### Speech segmentation backend (VAD)

`build_segments` splits each track (Zoom participant tracks and the single-file
diarization fallback) into speech chunks before GigaAM. Two detectors are available via
`MAC_TRANSCRIBER_VAD`:

- `silero` (default): `silero-vad` neural detector (in `requirements.txt`). On meeting
  audio it fragments less, keeps speech boundaries inside sentences intact, and captures
  slightly more speech at comparable speed. Validated across the existing multi-track
  meetings: cross-track speech overlap is the same or lower than `rms` on all but one
  meeting (no systematic ghost/duplicate turns on other speakers' mics). Use `rms` to
  roll back if a specific recording shows cross-track bleed.
- `rms`: legacy energy-threshold detector, no extra dependency. Opt-out fallback.

```env
# Default is silero; set this only to roll back to the energy detector.
MAC_TRANSCRIBER_VAD=rms
```

This only changes *where* each track is cut; segment timestamps stay absolute (sample
position in the source), so chronological order across tracks is unaffected. Speaker
labels are unchanged: in multi-track mode the speaker is the track, not the VAD. The
pyannote diarization path is not affected. If the `silero-vad` package is missing, the
service logs a warning and falls back to `rms`.

A/B comparison of the two detectors on existing meetings:

```bash
.venv/bin/python mac_transcriber/scripts/ab_vad_eval.py --max-meetings 4 --limit-seconds 600
```

Install and start the user LaunchAgents:

```bash
export HF_TOKEN="hf_..."
export MAC_TRANSCRIBER_API_KEY="<same-shared-secret-as-vps>"
mac_transcriber/scripts/install_launchd.sh
```

You can also pass the key as the first argument:

```bash
mac_transcriber/scripts/install_launchd.sh "$MAC_TRANSCRIBER_API_KEY"
```

The installer copies the plist examples into `~/Library/LaunchAgents`, writes logs under
`~/Library/Logs/slack_zoom`, replaces the placeholder API key in the generated plist, then
bootstraps and kickstarts the agents:

- `com.slack-zoom.gigaam-transcriber`
- `com.slack-zoom.gigaam-tunnel`
- `com.slack-zoom.gigaam-reprocess` — periodic queue drainer (every 15 min via
  `StartInterval`). Reprocesses meetings parked as `blocked_on_quota`/`blocked_on_ai`
  (AI was unavailable) once the API is back. Carries no secrets — it reads them from
  `.env.local` via `--env-file`. Drains the queue on its own; no manual step needed.

Do not put real secrets in `mac_transcriber/launchd/*.example`, this README, or git. The
generated plist in `~/Library/LaunchAgents` contains the local shared secret.

Check service state:

```bash
launchctl print "gui/$(id -u)/com.slack-zoom.gigaam-transcriber"
launchctl print "gui/$(id -u)/com.slack-zoom.gigaam-tunnel"
curl -fsS http://127.0.0.1:18003/healthz
```

Check logs:

```bash
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-transcriber.err.log
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-tunnel.err.log
```

Uninstall:

```bash
mac_transcriber/scripts/uninstall_launchd.sh
```

## VPS Settings

`/root/projects/slack_zoom/.env` on the VPS should contain:

```env
TRANSCRIPTION_SERVICE_ADAPTER=meeting_mvp
TRANSCRIPTION_SERVICE_URL=http://host.docker.internal:18004
TRANSCRIPTION_SERVICE_API_KEY=<same-key>
```

After changing `.env`, recreate the containers:

```bash
cd /root/projects/slack_zoom
docker compose up -d --force-recreate app worker
```

The VPS relay is a systemd service:

```bash
systemctl status slack-zoom-mac-transcriber-relay.service
```

The relay binds only to the Docker bridge (`172.22.0.1:18004`) and forwards to the
loopback SSH reverse tunnel (`127.0.0.1:18013`).

## Endpoints

- `GET /healthz`
- `POST /meetings`
- `GET /meetings/{id}/status`
- `GET /meetings/{id}/artifacts/{kind}`

## External Service Contract

External systems should treat this service as an async transcription/report job API.
The service stores generated files locally and exposes them through artifact endpoints.

Authentication, when `MAC_TRANSCRIBER_API_KEY` is set:

```http
Authorization: Bearer <api-key>
```

Recommended flow:

1. Create a meeting job with `POST /meetings`.
2. Poll `GET /meetings/{id}/status` until `status == "completed"` or `status == "failed"`.
3. Download `summary_json`.
4. Send `summary_json.slack.text` as the Slack message body.
5. Upload only files listed in `summary_json.slack.files` by downloading each `kind` from
   `GET /meetings/{id}/artifacts/{kind}`.

Create a job:

```bash
curl -fsS -X POST "$TRANSCRIPTION_SERVICE_URL/meetings" \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  -F "file=@/absolute/path/to/recording.m4a" \
  -F "title=Meeting title" \
  -F "language_code=ru" \
  -F "processing_mode=simple"
```

For a raw transcript without AI report, PDF, or meeting-memory enrichment, add the
following form field. The ChatGPT MCP bridge uses this mode so personal dictaphone
files remain separate from the reporting pipeline:

```bash
-F "transcript_only=true"
```

`title` is optional but recommended. If it is omitted, the service derives a display
title from the uploaded filename and strips common Zoom date/time suffixes instead of
using the internal UUID.

Response:

```json
{"id":"<meeting-id>","status":"uploaded"}
```

Poll status:

```bash
curl -fsS \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  "$TRANSCRIPTION_SERVICE_URL/meetings/<meeting-id>/status"
```

Important status fields:

- `status`: `uploaded`, `processing`, `completed`, or `failed`
- `phase`: current pipeline phase
- `progress`: `0..1`
- `artifacts`: available artifact kinds
- `report_health_status`: `ok`, `degraded`, or `failed`
- `report_alerts`: report generation warnings that should be visible to operators

Download summary for Slack:

```bash
curl -fsS \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  "$TRANSCRIPTION_SERVICE_URL/meetings/<meeting-id>/artifacts/summary_json"
```

`summary_json.slack` is the Slack handoff contract:

```json
{
  "slack": {
    "text": "*Meeting title*\nDate · duration\n...",
    "files": [
      {"kind":"report_pdf","filename":"report.pdf","mime_type":"application/pdf"},
      {"kind":"report_md","filename":"report.md","mime_type":"text/markdown"},
      {"kind":"transcript_md","filename":"transcript.md","mime_type":"text/markdown"}
    ]
  }
}
```

Do not upload JSON files to Slack. `coverage_json`, `report_health`, and `report_json`
are machine/debug artifacts only.

Download each Slack attachment:

```bash
curl -fsS \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  -o report.pdf \
  "$TRANSCRIPTION_SERVICE_URL/meetings/<meeting-id>/artifacts/report_pdf"
```

For Zoom participant tracks, send `zoom_participant_files` as repeated multipart fields
and optionally send `participants` as comma-separated display names.

## Rich Reports

The transcriber writes a cited protocol report next to the raw transcript:

- `report.json`: structured source of truth for report sections and segment citations
- `report.md`: human-readable Markdown protocol report
- `report.html`: bundled Jinja protocol template used for PDF rendering
- `report.typ`: legacy Typst source kept as a diagnostic/export artifact
- `report.pdf`: generated from `report.html` with WeasyPrint when PDF generation is enabled

The full transcript stays in `transcript.md`. The PDF intentionally omits the full
transcript and coverage audit; it includes protocol sections, citations, and
artifact/source notes. Coverage details stay in `coverage.json`.

Manual run with a local deterministic report:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm" \
  --report
```

Manual run with OpenAI report enrichment and PDF compilation:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm" \
  --ai-report \
  --pdf
```

Service mode enables OpenAI report enrichment and PDF generation by default in the
provided local/launchd starters. Provide `OPENAI_API_KEY` or `.env.local`, and install
Playwright browsers. If AI or PDF generation fails, the job still writes fallback
artifacts and marks the report as `degraded` in `report_health.json`.

```env
MAC_TRANSCRIBER_REPORT_MODE=ai
MAC_TRANSCRIBER_REPORT_MODEL=gpt-5.5
MAC_TRANSCRIBER_REPORT_PDF=1
MAC_TRANSCRIBER_AI_CHUNK_SIZE=80
MAC_TRANSCRIBER_AI_SYNTHESIS_CHUNK_LIMIT=24
MAC_TRANSCRIBER_AI_SYNTHESIS_BATCH_SIZE=4
# Optional: retry thin current-report extraction with a stronger model before memory enrichment.
# MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=gpt-5.5
# OpenRouter can be used as a primary model by prefixing the model id:
# MAC_TRANSCRIBER_REPORT_MODEL=openrouter:google/gemini-3.1-pro-preview
# MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=openrouter:openai/gpt-5.5
# Optional OpenRouter reserve when the OpenAI report request fails:
# OPENROUTER_API_KEY=sk-or-v1-...
# MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL=openai/gpt-4o-mini
# MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS=12000
# Optional when playwright is not on PATH:
# MAC_TRANSCRIBER_PLAYWRIGHT=/opt/homebrew/bin/playwright
```

Supported artifacts:

- `manifest_json`
- `context_pack_json`
- `youtrack_context_json`
- `transcript_md`
- `transcript_json`
- `summary_json`
- `report_md`
- `report_json`
- `report_html`
- `report_health`
- `report_typ`
- `report_pdf`
- `segments_tsv`
- `speaker_track_map`

### Read-only YouTrack reconciliation

When the Claude report backend is used, the scheduled drain performs two additional
bounded Opus passes before the final report. The first pass extracts transcript-cited
search intents, Python retrieves real candidates through the local YouTrack MCP
endpoint, and the second pass classifies candidate links as `confirmed`, `possible`,
or `unmatched`. Python rejects invented issue IDs and segment citations.

The durable snapshot is written to `youtrack_context.json`, included in the report
artifacts, and uploaded with the report to Yandex Disk. YouTrack failure is fail-open:
the report still completes and `report_health.json` records a degraded enrichment
check. Only `search_issues` and `get_issue` are allowed; no tracker write operation is
available to this pipeline.

Credentials are read from `YOUTRACK_URL`/`YOUTRACK_TOKEN` when present, otherwise from
the local `mcp_servers.youtrack.env` section in `~/.codex/config.toml`. They are never
copied into prompts or artifacts. Optional controls:

```env
MAC_TRANSCRIBER_YOUTRACK_ENRICHMENT=1
MAC_TRANSCRIBER_YOUTRACK_PROJECT=CH
MAC_TRANSCRIBER_YOUTRACK_VERIFY_TLS=1
MAC_TRANSCRIBER_YOUTRACK_TIMEOUT=20
MAC_TRANSCRIBER_YOUTRACK_CLAUDE_TIMEOUT=600
MAC_TRANSCRIBER_YOUTRACK_EFFORT=high
```

Slack message text is stored in `summary_json.slack.text`. Upload only the files
listed in `summary_json.slack.files`.

## Local Meeting Memory

The Mac transcriber is the owner of meeting memory. Slack/Zoom on the VPS remains an
orchestrator: it sends recordings, polls status, downloads final artifacts, and posts
Slack messages. It does not need direct database access.

Every completed job writes `artifacts/manifest.json`. The manifest records meeting
metadata, result summary, and an inventory of input/generated files with relative path,
size, sha256, and timestamps. Keep it with the meeting artifacts; it is the audit index
for later reprocessing.

Optional Postgres memory is enabled only when a local database URL is configured:

```env
MAC_TRANSCRIBER_DATABASE_URL=postgresql://gigaam:gigaam@127.0.0.1:5432/gigaam_memory
```

`MAC_TRANSCRIBER_POSTGRES_DSN` is accepted as a fallback variable. When configured, the
service fetches a small `context_pack.json` from prior meetings before report
generation, then upserts the current meeting, artifacts, transcript segments, report
facts, and citation links into Postgres after artifact generation. The schema reserves
tables for future `pgvector` embedding chunks and report context links. If Postgres is
unavailable, the job still completes; `summary.json` gets `context_pack_error` or
`memory_sync_error`, and the final manifest is rewritten so recorded hashes match the
files on disk.

The database should be local to the Mac for this architecture. Back up both Postgres
and `MAC_TRANSCRIBER_ROOT`; Postgres is the queryable memory, while the artifact folder
is the durable file archive.

OpenAI embeddings can be enabled with the existing `OPENAI_API_KEY`. The default model
is `text-embedding-3-small`, stored in the reserved `embedding_chunks.embedding
vector(1536)` column:

```bash
.venv/bin/python mac_transcriber/scripts/backfill_embeddings.py --env-file .env.local
```

Useful options:

```bash
.venv/bin/python mac_transcriber/scripts/backfill_embeddings.py \
  --env-file .env.local \
  --meeting-id <meeting-id> \
  --model text-embedding-3-small \
  --batch-size 64
```

The embedding backfill rebuilds chunks for selected meetings from transcript segments
and extracted report facts. New memory syncs also attempt a best-effort embedding sync
when both `MAC_TRANSCRIBER_DATABASE_URL` and `OPENAI_API_KEY` are configured.

### Semantic prior-context retrieval

Prior-meeting context is retrieved by the **meaning of the current meeting**, not just
its title: the semantic (embedding) search query is built from the current transcript,
so context is still found when the title/source filename is generic (e.g. `audio.m4a`).
Token search and the recency fallback continue to use the title for cheap exact matches.

Optionally drop weakly-related embedding hits by setting a maximum cosine distance
(`0`–`2`; lower is stricter). Unset means no threshold. A good starting point:

```env
# Discard prior-context chunks whose cosine distance to the current meeting exceeds this.
MAC_TRANSCRIBER_CONTEXT_MAX_DISTANCE=0.6
```

### Reasoning effort

For gpt-5 / o-series report models the reasoning depth of the report-shaping passes
(direct report, chunked synthesis, memory enrichment) can be raised for higher quality
at the cost of latency/tokens. Levels: `minimal` < `low` < `medium` < `high`. Defaults
are `medium` (report/synthesis) and `low` (enrichment); per-chunk notes stay `low`.

```env
# Override reasoning depth of the report-shaping passes (omit to keep defaults).
MAC_TRANSCRIBER_REASONING_EFFORT=high
```

### Out-of-funds handling (AI quota)

If the AI provider rejects a request for billing/quota reasons (HTTP 429
`insufficient_quota` / "check your plan and billing"), the report is **not** allowed to
fall back to the raw local keyword dump. Instead the meeting is parked with status
`blocked_on_quota`: the transcript is kept, no (garbage) report is written, and the job
is not marked `completed` or `failed`. After you top up the balance, drain the backlog
**sequentially** (report-only — the saved transcript is reused, ASR is NOT re-run) — it
stops at the first meeting that hits the quota again:

```bash
.venv/bin/python mac_transcriber/scripts/reprocess_blocked.py            # drain backlog
.venv/bin/python mac_transcriber/scripts/reprocess_blocked.py --dry-run  # just list the queue
```

## Verification

Local Mac:

```bash
curl -fsS http://127.0.0.1:18003/healthz
launchctl list | rg 'com\.slack-zoom\.gigaam'
```

VPS to Mac tunnel:

```bash
ssh root@111.88.224.176 'curl -fsS http://127.0.0.1:18013/healthz'
```

`slack_zoom-worker` to Mac through the relay:

```bash
ssh root@111.88.224.176 'cd /root/projects/slack_zoom && docker compose exec -T worker python - <<'"'"'PY'"'"'
import os
import urllib.error
import urllib.request

base = os.environ["TRANSCRIPTION_SERVICE_URL"]
key = os.environ["TRANSCRIPTION_SERVICE_API_KEY"]
print(urllib.request.urlopen(base + "/healthz", timeout=5).read().decode())
req = urllib.request.Request(
    base + "/meetings/no-such-meeting/status",
    headers={"Authorization": "Bearer " + key},
)
try:
    urllib.request.urlopen(req, timeout=5)
except urllib.error.HTTPError as exc:
    print("AUTH_STATUS_CHECK=" + str(exc.code))
PY'
```

Expected result: `{"status":"ok"}` and `AUTH_STATUS_CHECK=404`.

## Security Notes

- Keep `MAC_TRANSCRIBER_API_KEY` out of git and chat logs.
- Do not bind the Mac service or VPS relay to public `0.0.0.0`.
- `launchctl print` can reveal environment variables, including the API key.
- Raw audio and generated artifacts are stored under `MAC_TRANSCRIBER_ROOT`; keep that
  path out of git.
- The current VPS SSH configuration is key-only; password login is disabled.

## Troubleshooting

Check Mac logs:

```bash
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-transcriber.err.log
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-tunnel.err.log
```

Check VPS relay and ports:

```bash
ssh root@111.88.224.176 'systemctl status slack-zoom-mac-transcriber-relay.service --no-pager'
ssh root@111.88.224.176 'ss -ltnp | grep -E ":18013|:18004"'
```

If the VPS reboots, `slack_zoom` containers should restart automatically with
`restart=unless-stopped`, and the relay is enabled through systemd.

### Microphone modes and Scribe terminology

Scribe uses `scribe_keyterms.json` by default; per-meeting `terms_hint` is merged
with it. Override the dictionary with `MAC_TRANSCRIBER_SCRIBE_KEYTERMS_FILE` (JSON
list, `[]` disables defaults). Keyterms incur ElevenLabs' additional usage charge.

`POST /meetings` accepts optional `terms_hint` and `microphone_mode` form fields.
Each entry in `zoom_participant_tracks_json` can also include `microphone_mode`:

- `single`: explicitly confirmed personal microphone. Request `num_speakers=1`
  and retain the Zoom participant name, including brief utterances.
- `room`: shared microphone. Keep all detected anonymous speakers, without a
  duration threshold or assignment of the Zoom account's name to a voice.
- `unknown`: keep anonymous labels; do not infer a person's identity.

Mode priority: per-track metadata, explicit meeting mode, then `auto` (default).
Saved mappings by Zoom name are no longer consulted, because rooms change.
`auto` uses Scribe word timings: two IDs with at least 8 words and 3 seconds each,
plus repeated turns or 10 seconds of speech, count as a shared microphone.
A dominant ID needs at least 20 words and 8 seconds to inherit the Zoom name;
secondary IDs must each have at most 2 words / 0.8 seconds and together at most
1% of dominant speech. Their speech remains anonymous, never merged or dropped.
Other cases remain unknown. Room labels are `Zoom connection / Speaker N`.
These are conservative heuristics, not verified voice identities; real voices can
be split or merged by Scribe. `asr_metadata_json` records the inferred mode,
reason and per-ID statistics so thresholds can be audited on real recordings.

New Claude reports require verbatim source evidence for formal decisions/tasks.
Quotes must match every cited segment. Proposals are retained as open questions,
not commitments. A failed evidence check gets one regeneration attempt and blocks
publication if still invalid. This verifies provenance, not semantic entailment;
the report author also checks meaning, owner, deadline and certainty. The complete
citation set is rendered in decisions/tasks (previously only the first was shown).
