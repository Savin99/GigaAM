"""Тесты пер-трековой диаризации (общий микрофон в переговорке).

Диаризацию мокаем: `build_diarized_segments`/`build_segments` подменяются, реальный
pyannote и загрузка аудио не вызываются. Проверяем сквозную нумерацию говорящих,
сохранение Zoom-имени для чистых дорожек и маршрутизацию по флагу.
"""

from pathlib import Path

from mac_transcriber import asr


def _seg(speaker: str, track: str, start: float, end: float) -> asr.Segment:
    return asr.Segment(speaker=speaker, track=track, start=start, end=end, wav=None)


# --- флаг и опции -----------------------------------------------------------


def test_diarize_tracks_enabled_default_off(monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_DIARIZE_TRACKS", raising=False)
    assert asr.diarize_tracks_enabled() is False


def test_diarize_tracks_enabled_parses_truthy(monkeypatch):
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS", value)
        assert asr.diarize_tracks_enabled() is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS", value)
        assert asr.diarize_tracks_enabled() is False


def test_per_track_options_never_force_num_speakers(monkeypatch):
    # Главное: ОДНУ дорожку нельзя резать по числу участников встречи.
    monkeypatch.delenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MAX_SPEAKERS", raising=False)
    assert asr.per_track_diarization_options() == {}
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MAX_SPEAKERS", "4")
    assert asr.per_track_diarization_options() == {"max_speakers": 4}


def test_stable_speaker_names_offset():
    turns = [
        asr.DiarizedTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
        asr.DiarizedTurn(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]
    assert asr.stable_speaker_names(turns) == {
        "SPEAKER_00": "Speaker 1",
        "SPEAKER_01": "Speaker 2",
    }
    assert asr.stable_speaker_names(turns, offset=2) == {
        "SPEAKER_00": "Speaker 3",
        "SPEAKER_01": "Speaker 4",
    }


# --- speaker_tracks_from_segments_multi -------------------------------------


def test_speaker_tracks_from_segments_multi_pairs():
    tracks = [
        asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Ilya"),
        asr.TrackSpec(path=Path("/in/02.m4a"), speaker="Aziz"),
    ]
    segments = [
        _seg("Speaker 1", "01.m4a", 0.0, 1.0),
        _seg("Speaker 2", "01.m4a", 1.0, 2.0),
        _seg("Aziz", "02.m4a", 2.0, 3.0),
        _seg("Speaker 1", "01.m4a", 3.0, 4.0),  # повтор не дублируется
    ]
    assert asr.speaker_tracks_from_segments_multi(tracks, segments) == [
        asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Speaker 1"),
        asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Speaker 2"),
        asr.TrackSpec(path=Path("/in/02.m4a"), speaker="Aziz"),
    ]


# --- build_per_track_segments -----------------------------------------------


def test_per_track_splits_room_keeps_clean(monkeypatch):
    """01.m4a — комната (2 голоса) -> Speaker 1/2; 02.m4a — чистый Aziz -> имя."""
    in_room = asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Ilya")
    clean = asr.TrackSpec(path=Path("/in/02.m4a"), speaker="Aziz")

    def fake_diarized(path, *, device, metadata, options=None, name_offset=0):
        if path.name == "01.m4a":
            return [
                _seg(f"Speaker {name_offset + 1}", "01.m4a", 0.0, 1.0),
                _seg(f"Speaker {name_offset + 2}", "01.m4a", 2.0, 3.0),
                _seg(f"Speaker {name_offset + 1}", "01.m4a", 4.0, 5.0),
            ]
        return []  # чистая дорожка: диаризация не нашла второго голоса

    def fake_build_segments(tracks, progress_callback=None):
        track = tracks[0]
        return [_seg(track.speaker, track.path.name, 10.0, 11.0)]

    monkeypatch.setattr(asr, "build_diarized_segments", fake_diarized)
    monkeypatch.setattr(asr, "build_segments", fake_build_segments)

    segments = asr.build_per_track_segments([in_room, clean], device="cpu", metadata={})

    speakers = {segment.speaker for segment in segments}
    assert speakers == {"Speaker 1", "Speaker 2", "Aziz"}
    # отсортировано по времени
    assert [round(s.start, 1) for s in segments] == [0.0, 2.0, 4.0, 10.0]


def test_per_track_global_speaker_numbering(monkeypatch):
    """Две дорожки-комнаты: нумерация не сбрасывается (Speaker 1..4)."""
    room_a = asr.TrackSpec(path=Path("/in/01.m4a"), speaker="RoomA")
    room_b = asr.TrackSpec(path=Path("/in/02.m4a"), speaker="RoomB")

    def fake_diarized(path, *, device, metadata, options=None, name_offset=0):
        return [
            _seg(f"Speaker {name_offset + 1}", path.name, 0.0, 1.0),
            _seg(f"Speaker {name_offset + 2}", path.name, 1.0, 2.0),
        ]

    monkeypatch.setattr(asr, "build_diarized_segments", fake_diarized)
    monkeypatch.setattr(
        asr, "build_segments", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )

    segments = asr.build_per_track_segments([room_a, room_b], device="cpu", metadata={})
    assert {s.speaker for s in segments} == {
        "Speaker 1",
        "Speaker 2",
        "Speaker 3",
        "Speaker 4",
    }


def test_per_track_falls_back_when_diarization_unavailable(monkeypatch):
    track = asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Ilya")

    def raise_unavailable(path, *, device, metadata, options=None, name_offset=0):
        raise asr.DiarizationUnavailable("no token")

    def fake_build_segments(tracks, progress_callback=None):
        return [_seg(tracks[0].speaker, tracks[0].path.name, 0.0, 1.0)]

    monkeypatch.setattr(asr, "build_diarized_segments", raise_unavailable)
    monkeypatch.setattr(asr, "build_segments", fake_build_segments)

    segments = asr.build_per_track_segments([track], device="cpu", metadata={})
    assert [s.speaker for s in segments] == ["Ilya"]


# --- маршрутизация build_input_segments по флагу ----------------------------


def _zoom_meeting(tmp_path: Path) -> tuple[Path, dict]:
    participants_dir = tmp_path / "participants"
    participants_dir.mkdir()
    (tmp_path / "audio.m4a").write_bytes(b"placeholder")
    (participants_dir / "01.m4a").write_bytes(b"room")
    (participants_dir / "02.m4a").write_bytes(b"aziz")
    metadata = {
        "zoom_participant_tracks": [
            {"speaker_name": "Ilya"},
            {"speaker_name": "Aziz"},
        ]
    }
    return participants_dir, metadata


def test_routing_uses_per_track_when_flag_on(monkeypatch, tmp_path):
    participants_dir, metadata = _zoom_meeting(tmp_path)
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS", "1")
    monkeypatch.setattr(asr, "diarization_enabled", lambda: True)

    produced = [
        _seg("Speaker 1", "01.m4a", 0.0, 1.0),
        _seg("Speaker 2", "01.m4a", 2.0, 3.0),
        _seg("Aziz", "02.m4a", 4.0, 5.0),
    ]
    seen = {}

    def fake_per_track(tracks, *, device, metadata, progress_callback=None):
        seen["tracks"] = tracks
        return produced

    def fail_build_segments(*a, **k):
        raise AssertionError("build_segments must not run when flag is on")

    monkeypatch.setattr(asr, "build_per_track_segments", fake_per_track)
    monkeypatch.setattr(asr, "build_segments", fail_build_segments)

    segments, tracks = asr.build_input_segments(
        input_dir=tmp_path, metadata=metadata, device="cpu"
    )

    assert segments is produced
    # список говорящих собран из сегментов: комната -> Speaker 1/2, Aziz -> имя
    assert [(t.path, t.speaker) for t in tracks] == [
        (participants_dir / "01.m4a", "Speaker 1"),
        (participants_dir / "01.m4a", "Speaker 2"),
        (participants_dir / "02.m4a", "Aziz"),
    ]
    assert seen["tracks"] == [
        asr.TrackSpec(path=participants_dir / "01.m4a", speaker="Ilya"),
        asr.TrackSpec(path=participants_dir / "02.m4a", speaker="Aziz"),
    ]


def test_routing_skips_per_track_when_flag_off(monkeypatch, tmp_path):
    participants_dir, metadata = _zoom_meeting(tmp_path)
    monkeypatch.delenv("MAC_TRANSCRIBER_DIARIZE_TRACKS", raising=False)

    fallback = [_seg("Ilya", "01.m4a", 0.0, 1.0)]

    def fake_build_segments(tracks, progress_callback=None):
        return fallback

    def fail_per_track(*a, **k):
        raise AssertionError("per-track diarization must not run when flag is off")

    monkeypatch.setattr(asr, "build_segments", fake_build_segments)
    monkeypatch.setattr(asr, "build_per_track_segments", fail_per_track)

    segments, tracks = asr.build_input_segments(
        input_dir=tmp_path, metadata=metadata, device="cpu"
    )

    assert segments is fallback
    assert tracks == [
        asr.TrackSpec(path=participants_dir / "01.m4a", speaker="Ilya"),
        asr.TrackSpec(path=participants_dir / "02.m4a", speaker="Aziz"),
    ]
