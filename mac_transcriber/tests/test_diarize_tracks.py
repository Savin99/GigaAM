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
    """01.m4a — комната (2 голоса): доминирующий -> Zoom-имя Ilya, второй -> Speaker N.

    02.m4a — чистый Aziz -> имя сохраняется.
    """
    # короткие тестовые сегменты -> понижаем порог заметного говорящего
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", "0.5")
    in_room = asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Ilya")
    clean = asr.TrackSpec(path=Path("/in/02.m4a"), speaker="Aziz")

    def fake_diarized(path, *, device, metadata, options=None, name_offset=0):
        if path.name == "01.m4a":
            return [
                _seg(
                    f"Speaker {name_offset + 1}", "01.m4a", 0.0, 1.0
                ),  # 2с -> доминирует
                _seg(f"Speaker {name_offset + 2}", "01.m4a", 2.0, 3.0),  # 1с
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
    # доминирующий голос комнаты унаследовал Zoom-имя дорожки, второй пронумерован
    assert speakers == {"Ilya", "Speaker 1", "Aziz"}
    # имя Ilya достаётся доминирующему по talk-time кластеру (2с против 1с)
    ilya_starts = sorted(s.start for s in segments if s.speaker == "Ilya")
    assert ilya_starts == [0.0, 4.0]
    # отсортировано по времени
    assert [round(s.start, 1) for s in segments] == [0.0, 2.0, 4.0, 10.0]


def test_per_track_global_speaker_numbering(monkeypatch):
    """Две дорожки-комнаты без реальных Zoom-имён: нумерация не сбрасывается (Speaker 1..4)."""
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", "0.5")
    # служебные имена дорожек (нет реального участника) -> наследовать нечего, нумеруем всех
    room_a = asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Zoom participant 1")
    room_b = asr.TrackSpec(path=Path("/in/02.m4a"), speaker="Zoom participant 2")

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


def test_is_placeholder_speaker():
    # служебные метки дорожки -> наследовать нечего
    assert asr._is_placeholder_speaker("Speaker") is True
    assert asr._is_placeholder_speaker("Speaker 2") is True
    assert asr._is_placeholder_speaker("Zoom participant 1") is True
    assert asr._is_placeholder_speaker(None) is True  # отсутствие имени тоже служебное
    # реальные Zoom-имена -> наследуются доминирующему говорящему
    assert asr._is_placeholder_speaker("Ilya") is False
    assert asr._is_placeholder_speaker("Вячеслав") is False


def test_per_track_dominant_inherits_zoom_name(monkeypatch):
    """Регресс по сегодняшнему созвону: дорожка Ilya с двумя голосами не должна терять имя.

    Раньше при ≥2 заметных голосах на дорожке оба становились Speaker N и Zoom-имя
    терялось. Теперь доминирующий по talk-time кластер наследует имя дорожки.
    """
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", "20")
    track = asr.TrackSpec(path=Path("/in/02.m4a"), speaker="Ilya")

    def fake_diarized(path, *, device, metadata, options=None, name_offset=0):
        return [
            _seg("Speaker 1", "02.m4a", 0.0, 160.0),  # 160с -> доминирует, это Ilya
            _seg(
                "Speaker 2", "02.m4a", 160.0, 220.0
            ),  # 60с -> второй человек в комнате
        ]

    monkeypatch.setattr(asr, "build_diarized_segments", fake_diarized)
    monkeypatch.setattr(
        asr, "build_segments", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )

    segments = asr.build_per_track_segments([track], device="cpu", metadata={})
    by_speaker = {s.speaker: s.start for s in segments}
    assert set(by_speaker) == {"Ilya", "Speaker 1"}
    assert by_speaker["Ilya"] == 0.0  # доминирующий кластер
    assert by_speaker["Speaker 1"] == 160.0


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


def test_per_track_ignores_tiny_noise_cluster(monkeypatch):
    """Регресс: 1 большой голос + крошечный шум -> имя из Zoom, НЕ дробим на Speaker N.

    Именно отсутствие порога ломало all-remote встречи: 2-секундный шумовой кластер
    считался вторым спикером и дробил чистую дорожку.
    """
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", "20")
    track = asr.TrackSpec(path=Path("/in/01.m4a"), speaker="Вячеслав")

    def fake_diarized(path, *, device, metadata, options=None, name_offset=0):
        return [
            _seg("Speaker 1", "01.m4a", 0.0, 60.0),  # 60с — реальный голос
            _seg("Speaker 1", "01.m4a", 60.0, 100.0),  # ещё 40с
            _seg("Speaker 2", "01.m4a", 100.0, 102.0),  # 2с — шум, ниже порога
        ]

    def fake_build_segments(tracks, progress_callback=None):
        return [_seg(tracks[0].speaker, tracks[0].path.name, 0.0, 1.0)]

    monkeypatch.setattr(asr, "build_diarized_segments", fake_diarized)
    monkeypatch.setattr(asr, "build_segments", fake_build_segments)

    segments = asr.build_per_track_segments([track], device="cpu", metadata={})
    assert {s.speaker for s in segments} == {"Вячеслав"}


def test_track_min_speaker_seconds_parsing(monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", raising=False)
    assert asr.track_min_speaker_seconds() == 20.0
    monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", "30")
    assert asr.track_min_speaker_seconds() == 30.0
    for bad in ("bad", "-5", "0"):
        monkeypatch.setenv("MAC_TRANSCRIBER_DIARIZE_TRACKS_MIN_SPEAKER_S", bad)
        assert asr.track_min_speaker_seconds() == 20.0


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
