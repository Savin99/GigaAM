#!/usr/bin/env python3
"""Recover a saved meeting without any new cloud ASR submission.

Reuse validated Scribe caches and transcribe remaining tracks using cached GigaAM.
Every track is checkpointed. Original provider journals and input stay intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audio_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_dir", type=Path)
    parser.add_argument("--model-cache", type=Path, default=Path.home() / ".cache/gigaam")
    parser.add_argument("--publish", action="store_true", help="Publish complete artifacts and queue the normal report")
    args = parser.parse_args(argv)
    meeting = args.meeting_dir.resolve()
    if args.publish:
        status = json.loads((meeting / "status.json").read_text())
        asr_metadata = meeting / "artifacts/asr_metadata.json"
        already_published = (
            status.get("status") == "blocked_on_ai"
            and asr_metadata.exists()
            and json.loads(asr_metadata.read_text()).get("model_name") == "local_recovery"
        )
        if status.get("status") == "completed" or already_published:
            print("Meeting already published; artifacts and report status left intact")
            return 0
    work = meeting / "local_recovery"
    work.mkdir(exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MAC_TRANSCRIBER_DIARIZATION"] = "0"
    from mac_transcriber import asr, scribe
    import torch

    torch.set_num_threads(2)
    metadata = json.loads((meeting / "input/metadata.json").read_text())
    tracks = asr._track_specs(input_dir=meeting / "input", metadata=metadata)
    all_segments = []
    engines = []
    started = time.time()
    for index, track in enumerate(tracks):
        checkpoint = work / f"track-{index + 1:02d}.json"
        digest = audio_digest(track.path)
        one_metadata = {**metadata, "zoom_participant_tracks": (metadata.get("zoom_participant_tracks") or [])[index:index + 1]}
        if checkpoint.exists():
            result = json.loads(checkpoint.read_text())
            if result["audio_sha256"] != digest:
                raise ValueError("Recovery checkpoint does not match input audio")
            segments = [asr.Segment(**item) for item in result["segments"]]
            engine = result["engine"]
        else:
            print(f"track={index + 1}/{len(tracks)} phase=cache_check", flush=True)
            try:
                segments, _ = scribe.transcribe_tracks(
                    tracks=[track], metadata=one_metadata,
                    cache_dir=meeting / "scribe_cache", cache_only=True,
                )
                engine = "scribe_v2_cached"
            except scribe.ScribeCacheMiss:
                cache = args.model_cache.expanduser().resolve()
                for filename in ["v3_e2e_rnnt.ckpt", "v3_e2e_rnnt_tokenizer.model"]:
                    if not (cache / filename).is_file():
                        raise RuntimeError("Local GigaAM model cache is incomplete; no download attempted")
                # Names identify Zoom connections, not verified voice identities.
                local_track = asr.TrackSpec(track.path, f"{track.speaker} (дорожка Zoom)")
                print(f"track={index + 1}/{len(tracks)} phase=local_vad", flush=True)
                segments = asr.build_segments([local_track])
                print(f"track={index + 1}/{len(tracks)} phase=local_asr segments={len(segments)}", flush=True)
                segments = asr.transcribe_segments(
                    segments, model_name="v3_e2e_rnnt", cache_dir=str(cache),
                    device="cpu", batch_size=2,
                    progress_callback=lambda done, total: print(
                        f"track={index + 1} segments={done}/{total}", flush=True
                    ) if done % 50 == 0 or done == total else None,
                )
                engine = "gigaam_v3_e2e_rnnt_local"
            save_json(checkpoint, {"audio_sha256": digest, "engine": engine, "segments": [asdict(item) for item in segments]})
        print(f"track={index + 1}/{len(tracks)} phase=checkpoint segments={len(segments)} engine={engine}", flush=True)
        all_segments.extend(segments)
        engines.append({"track": track.path.name, "engine": engine, "audio_sha256": digest})

    all_segments = asr.dedupe_segments(all_segments)
    if not all_segments:
        raise RuntimeError("Recovery produced no speech; nothing published")
    output = work / "artifacts"
    output.mkdir(exist_ok=True)
    asr.write_artifacts(
        output_dir=output, meeting_id=meeting.name,
        title=asr.resolve_meeting_title(metadata, meeting_dir=meeting),
        source_filename=metadata.get("source_filename") or "audio.m4a",
        model_name="scribe_v2_cached+gigaam_v3_e2e_rnnt_local_recovery",
        tracks=tracks, segments=all_segments, elapsed_seconds=time.time() - started,
        include_report=False,
    )
    save_json(output / "asr_metadata.json", {"model_name": "local_recovery", "cloud_submissions": 0, "tracks": engines})
    if args.publish:
        from mac_transcriber import service
        original = meeting / "artifacts"
        if original.exists():
            backup = work / "artifacts-before-recovery"
            if not backup.exists():
                shutil.copytree(original, backup)
        original.mkdir(exist_ok=True)
        for path in output.iterdir():
            shutil.copy2(path, original / path.name)
        shutil.copy2(meeting / "status.json", work / "status-before-recovery.json")
        service._write_status(
            meeting, "blocked_on_ai", phase="blocked_on_ai", progress=0.9,
            message="Transcript recovered locally; queued for the normal AI report",
            segments_total=len(all_segments), segments_done=len(all_segments),
            result={"segments": len(all_segments), "tracks": len(tracks), "cloud_asr_submissions": 0},
        )
    print(f"recovered_segments={len(all_segments)} tracks={len(tracks)} cloud_submissions=0 published={args.publish}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
