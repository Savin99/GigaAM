"""A tool-only ChatGPT MCP server for batch GigaAM transcription.

The MCP process is intentionally lightweight: it downloads the temporary ChatGPT
file URL immediately, forwards the audio to the already-running private GigaAM
service, then exposes job status and Markdown transcript chunks back to ChatGPT.
"""

from __future__ import annotations

from typing import Annotated, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from chatgpt_transcriber.config import Settings, load_settings
from chatgpt_transcriber.service_client import (
    BridgeError,
    GigaAMServiceClient,
    OpenAIFile,
    public_status,
)


SERVER_INSTRUCTIONS = """
GigaAM Transcriber accepts one or several audio files attached in ChatGPT and
transcribes each file as a separate asynchronous job using the private GigaAM
service. Use transcribe_audio_batch when the user attaches recordings and asks
for transcription. Do not say a transcript is ready until get_transcription_status
reports transcript_ready. When it is ready, call get_transcript and use the
returned Markdown directly for summaries, edits, analysis, or other work. A
blocked report status can still have a ready transcript, but this bridge requests
transcript-only jobs so reports are normally not involved.
""".strip()


def build_server(
    settings: Settings | None = None,
    *,
    client: GigaAMServiceClient | None = None,
) -> FastMCP:
    settings = settings or load_settings()
    service_client = client or GigaAMServiceClient(settings)
    server = FastMCP(
        "GigaAM Transcriber",
        instructions=SERVER_INSTRUCTIONS,
        host=settings.host,
        port=settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
    )

    @server.tool(
        name="transcribe_audio_batch",
        title="Transcribe attached audio recordings",
        description=(
            "Use this when the user has attached one or more audio recordings and "
            "wants a GigaAM transcription. Each file becomes a separate queued job."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={
            "openai/fileParams": ["audio_files"],
            "openai/toolInvocation/invoking": "Queuing audio for GigaAM transcription…",
            "openai/toolInvocation/invoked": "Audio queued for GigaAM transcription",
        },
        structured_output=True,
    )
    async def transcribe_audio_batch(
        audio_files: Annotated[
            list[OpenAIFile], Field(min_length=1, max_length=10)
        ],
        title_prefix: Annotated[str, Field(max_length=160)] = "",
        language_code: Literal["ru"] = "ru",
    ) -> dict[str, object]:
        if len(audio_files) > settings.max_batch_files:
            raise BridgeError(
                f"This bridge accepts at most {settings.max_batch_files} audio files per batch."
            )

        jobs: list[dict[str, object]] = []
        for index, audio_file in enumerate(audio_files, start=1):
            title = _title_for_file(title_prefix, audio_file.file_name, index, len(audio_files))
            try:
                job = await service_client.submit_audio_file(
                    audio_file,
                    title=title,
                    language_code=language_code,
                )
            except BridgeError as exc:
                jobs.append(
                    {
                        "file_name": audio_file.file_name or f"attachment-{index}",
                        "state": "not_queued",
                        "error": str(exc),
                    }
                )
            except httpx.HTTPError:
                jobs.append(
                    {
                        "file_name": audio_file.file_name or f"attachment-{index}",
                        "state": "not_queued",
                        "error": "The bridge could not reach a required service.",
                    }
                )
            else:
                jobs.append(
                    {
                        "job_id": job.job_id,
                        "file_name": job.file_name,
                        "state": job.state,
                        "reused_existing_job": job.reused_existing_job,
                    }
                )

        queued_count = sum("job_id" in job for job in jobs)
        return {
            "queued_count": queued_count,
            "requested_count": len(audio_files),
            "jobs": jobs,
            "next_step": (
                "Use get_transcription_status with the returned job IDs. "
                "When transcript_ready is true, use get_transcript to read the Markdown."
            ),
        }

    @server.tool(
        name="get_transcription_status",
        title="Check transcription status",
        description=(
            "Use this when the user asks whether one or more GigaAM transcription "
            "jobs are ready, queued, or failed."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        meta={
            "openai/toolInvocation/invoking": "Checking transcription status…",
            "openai/toolInvocation/invoked": "Transcription status checked",
        },
        structured_output=True,
    )
    async def get_transcription_status(
        job_ids: Annotated[list[str], Field(min_length=1, max_length=10)]
    ) -> dict[str, object]:
        statuses: list[dict[str, object]] = []
        for job_id in job_ids:
            try:
                status = await service_client.get_status(job_id)
            except BridgeError as exc:
                statuses.append({"job_id": job_id, "state": "unavailable", "error": str(exc)})
            except httpx.HTTPError:
                statuses.append(
                    {
                        "job_id": job_id,
                        "state": "unavailable",
                        "error": "The bridge could not reach the transcription service.",
                    }
                )
            else:
                assert status is not None
                statuses.append(public_status(status))
        return {"jobs": statuses}

    @server.tool(
        name="get_transcript",
        title="Read a completed transcript",
        description=(
            "Use this when a GigaAM transcription job is transcript_ready and you need "
            "the Markdown text to summarize, edit, analyze, or otherwise work with it."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        meta={
            "openai/toolInvocation/invoking": "Retrieving transcript…",
            "openai/toolInvocation/invoked": "Transcript retrieved",
        },
    )
    async def get_transcript(
        job_id: str,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=60_000)] = 50_000,
    ) -> str:
        try:
            transcript = await service_client.get_transcript(job_id)
        except BridgeError as exc:
            return str(exc)
        except httpx.HTTPError:
            return "The bridge could not reach the transcription service."
        if transcript is None:
            return (
                "The transcript is not ready yet. Use get_transcription_status "
                "for this job before requesting it again."
            )
        if offset >= len(transcript):
            return "The requested transcript offset is beyond the end of the transcript."

        chunk = transcript[offset : offset + limit]
        next_offset = offset + len(chunk)
        if next_offset < len(transcript):
            return (
                f"<!-- Transcript chunk: characters {offset}-{next_offset - 1}. "
                f"Call get_transcript with offset={next_offset} for the next chunk. -->\n\n"
                f"{chunk}"
            )
        return chunk

    return server


def _title_for_file(prefix: str, file_name: str, index: int, total: int) -> str:
    prefix = prefix.strip()
    if not prefix:
        return ""
    return prefix if total == 1 else f"{prefix} — {index}"


def main() -> None:
    """Start only after the local bridge credential has been validated."""

    build_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
