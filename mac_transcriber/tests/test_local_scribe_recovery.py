import json

import pytest

from mac_transcriber import asr, scribe
from mac_transcriber.scripts import recover_scribe_locally as recovery


def setup_tracks(tmp_path):
    source = tmp_path / "input"
    participants = source / "participants"
    participants.mkdir(parents=True)
    for index in (1, 2):
        (participants / f"{index:02d}.m4a").write_bytes(f"audio-{index}".encode())
    (source / "metadata.json").write_text(json.dumps({"title": "Recovery"}))
    cache = tmp_path / "models"
    cache.mkdir()
    for name in ("v3_e2e_rnnt.ckpt", "v3_e2e_rnnt_tokenizer.model"):
        (cache / name).write_bytes(b"model")
    return cache


def test_recovery_reuses_cloud_cache_and_checkpoints_local_tracks(tmp_path, monkeypatch):
    model_cache = setup_tracks(tmp_path)
    requests = []
    local_calls = []

    def cached(**kwargs):
        assert kwargs["cache_only"] is True
        track = kwargs["tracks"][0]
        requests.append(track.path.name)
        if track.path.name == "02.m4a":
            raise scribe.ScribeCacheMiss("uncertain paid submission")
        return [asr.Segment("Speaker", track.path.name, 0, 2, text="Cached speech")], [track]

    def local(segments, **kwargs):
        local_calls.append(kwargs["model_name"])
        segments[0].text = "Local speech"
        return segments

    monkeypatch.setattr(scribe, "transcribe_tracks", cached)
    monkeypatch.setattr(asr, "build_segments", lambda tracks: [asr.Segment(tracks[0].speaker, tracks[0].path.name, 3, 5)])
    monkeypatch.setattr(asr, "transcribe_segments", local)
    args = [str(tmp_path), "--model-cache", str(model_cache)]
    assert recovery.main(args) == 0
    assert recovery.main(args) == 0
    assert requests == ["01.m4a", "02.m4a"]
    assert local_calls == ["v3_e2e_rnnt"]
    transcript = json.loads((tmp_path / "local_recovery/artifacts/transcript.json").read_text())
    assert [item["text"] for item in transcript] == ["Cached speech", "Local speech"]
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "status.json").exists()


def test_failure_keeps_completed_checkpoint_without_publishing(tmp_path, monkeypatch):
    cache = setup_tracks(tmp_path)

    def cached(**kwargs):
        track = kwargs["tracks"][0]
        if track.path.name == "02.m4a":
            raise RuntimeError("recovery interrupted")
        return [asr.Segment("Speaker", track.path.name, 0, 1, text="Kept")], [track]

    monkeypatch.setattr(scribe, "transcribe_tracks", cached)
    with pytest.raises(RuntimeError, match="interrupted"):
        recovery.main([str(tmp_path), "--model-cache", str(cache)])
    assert (tmp_path / "local_recovery/track-01.json").exists()
    assert not (tmp_path / "local_recovery/artifacts").exists()
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("status", ["completed", "blocked_on_ai"])
def test_published_recovery_does_not_overwrite_report_or_reset_status(tmp_path, status):
    (tmp_path / "status.json").write_text(json.dumps({"status": status}))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "asr_metadata.json").write_text(json.dumps({"model_name": "local_recovery"}))
    report = artifacts / "report.md"
    report.write_text("Existing verified report")
    assert recovery.main([str(tmp_path), "--publish"]) == 0
    assert json.loads((tmp_path / "status.json").read_text())["status"] == status
    assert report.read_text() == "Existing verified report"
