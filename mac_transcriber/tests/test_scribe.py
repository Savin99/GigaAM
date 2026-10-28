import json

import httpx
import pytest

from mac_transcriber import asr, scribe


def word(text, start, end, speaker="speaker_0"):
    return dict(type="word", text=text, start=start, end=end, speaker_id=speaker)


def mock_api(monkeypatch, handler):
    client = httpx.Client
    def mocked_client(**kwargs):
        kwargs.pop("transport", None)
        return client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(
        scribe.httpx,
        "Client",
        mocked_client,
    )
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-test-key")


def test_upload_cache_and_word_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "audio.m4a").write_bytes(b"test audio")
    (source / "metadata.json").write_text(
        json.dumps({"title": "Meeting", "terms_hint": "GigaAM; Scribe"})
    )
    calls = []

    def handler(request):
        calls.append(request)
        body = request.read()
        assert request.url == scribe.ENDPOINT
        assert request.headers["xi-api-key"] == "secret-test-key"
        for value in (
            b"scribe_v2",
            b"diarize",
            b"true",
            b"no_verbatim",
            b"false",
            b"GigaAM",
        ):
            assert value in body
        assert body.count(b'name="keyterms"') >= 2
        return httpx.Response(
            200,
            json={
                "text": "Привет, мир! Да.",
                "words": [
                    word("Привет,", 0, 0.5),
                    {"type": "spacing", "text": " "},
                    word("мир!", 0.6, 1),
                    word("Да.", 2, 3, "speaker_1"),
                ],
            },
        )

    mock_api(monkeypatch, handler)

    def forbidden(*args, **kwargs):
        pytest.fail("Scribe must bypass local models")

    monkeypatch.setattr(asr, "build_input_segments", forbidden)
    monkeypatch.setattr(asr, "transcribe_segments", forbidden)
    statuses = []
    for _ in range(2):
        result = asr.transcribe_meeting(
            meeting_dir=tmp_path,
            model_name="scribe_v2",
            cache_dir="unused",
            device="cpu",
            batch_size=4,
            include_report=False,
            status_callback=lambda state, **payload: statuses.append((state, payload)),
        )
    assert len(calls) == 1
    assert result["segments"] == 2
    artifacts = tmp_path / "artifacts"
    transcript = json.loads((artifacts / "transcript.json").read_text())
    assert [s["text"] for s in transcript] == ["Привет, мир!", "Да."]
    assert [s["speaker"] for s in transcript] == ["Speaker 1", "Speaker 2"]
    assert transcript[1]["start"] == 2
    assert "scribe_v2" in (artifacts / "transcript.md").read_text()
    assert (
        json.loads((artifacts / "summary.json").read_text())["model_name"]
        == "scribe_v2"
    )
    assert statuses[-1][0] == "completed"


def test_participant_names_and_global_speakers(tmp_path, monkeypatch):
    replies = iter(
        [
            {"words": [word("Первый", 0, 3), word("Второй", 4, 5, "speaker_1")]},
            {"words": [word("Третий", 1, 2)]},
            {"words": [word("Четвертый", 3, 4)]},
        ]
    )
    monkeypatch.setattr(scribe, "request_transcript", lambda *a, **k: next(replies))
    tracks = [
        asr.TrackSpec(tmp_path / f"{i}.m4a", name)
        for i, name in enumerate(["Ilya", "Aziz", "Zoom participant 3"])
    ]
    segments, mapped = scribe.transcribe_tracks(
        tracks=tracks,
        metadata={
            "zoom_participant_tracks": [
                {"microphone_mode": "room"},
                {"microphone_mode": "single"},
                {"microphone_mode": "unknown"},
            ]
        },
        cache_dir=tmp_path,
    )
    assert [s.speaker for s in segments] == [
        "Speaker 1",
        "Aziz",
        "Speaker 3",
        "Speaker 2",
    ]
    assert len(mapped) == 4


@pytest.mark.parametrize("status", [401, 403, 422, 500])
def test_errors_safe_and_no_cache(tmp_path, monkeypatch, status):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    mock_api(
        monkeypatch,
        lambda req: httpx.Response(status, text="secret-test-key private transcript"),
    )
    with pytest.raises(scribe.ScribeError) as exc:
        scribe.request_transcript(path, terms=[], cache_dir=tmp_path / "cache")
    assert str(status) in str(exc.value)
    assert "secret-test-key" not in str(exc.value)
    assert not list((tmp_path / "cache").glob("*.json")) or all(
        p.name.endswith(".request.json") for p in (tmp_path / "cache").glob("*.json")
    )


def test_disconnect_journal_blocks_another_paid_upload(tmp_path, monkeypatch):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadError("private transcript secret-test-key", request=request)

    mock_api(monkeypatch, handler)
    cache = tmp_path / "cache"
    with pytest.raises(scribe.ScribeError, match="ReadError") as exc:
        scribe.request_transcript(path, terms=[], cache_dir=cache)
    assert "private transcript" not in str(exc.value)
    journal = json.loads(next(cache.glob("*.request.json")).read_text())
    assert journal["state"] == "unknown"
    assert journal["transport_error"] == "ReadError"
    assert "secret-test-key" not in json.dumps(journal)
    with pytest.raises(scribe.ScribeError, match="requires reconciliation"):
        scribe.request_transcript(path, terms=[], cache_dir=cache)
    assert len(calls) == 1


def test_cache_only_never_opens_cloud_client(tmp_path, monkeypatch):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")

    def forbidden(**kwargs):
        pytest.fail("Offline recovery must never open a paid ASR connection")

    monkeypatch.setattr(scribe.httpx, "Client", forbidden)
    with pytest.raises(scribe.ScribeCacheMiss):
        scribe.request_transcript(path, terms=[], cache_dir=tmp_path / "cache", cache_only=True)
    assert not (tmp_path / "cache").exists()


def test_cache_only_reuses_validated_response_without_api_key(tmp_path, monkeypatch):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    payload = {"words": [word("Готово", 0, 1)]}
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    cache = tmp_path / "cache"
    assert scribe.request_transcript(path, terms=[], cache_dir=cache) == payload
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.setattr(scribe.httpx, "Client", lambda **kwargs: pytest.fail("Cache hit attempted a cloud connection"))
    # Use matching terms: the cache key covers the provider's exact request.
    assert scribe.request_transcript(path, terms=[], cache_dir=cache, cache_only=True) == payload


def test_headers_retained_when_response_body_disconnects(tmp_path, monkeypatch):
    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            raise httpx.ReadTimeout("private body")
            yield b""

    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    mock_api(monkeypatch, lambda request: httpx.Response(
        200, headers={"x-trace-id": "provider-trace"}, stream=BrokenBody()
    ))
    with pytest.raises(scribe.ScribeError, match="ReadTimeout"):
        scribe.request_transcript(path, terms=[], cache_dir=tmp_path / "cache")
    journal = json.loads(next((tmp_path / "cache").glob("*.request.json")).read_text())
    assert journal["request_id"] == "provider-trace"
    assert journal["http_status"] == 200


def test_keepalive_is_scoped_to_scribe(tmp_path, monkeypatch):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-test-key")
    original = httpx.HTTPTransport
    options = []

    def transport(**kwargs):
        options.append(kwargs["socket_options"])
        return original(**kwargs)

    mock_api(monkeypatch, lambda request: httpx.Response(200, json={"words": []}))
    monkeypatch.setattr(scribe.httpx, "HTTPTransport", transport)
    scribe.request_transcript(path, terms=[], cache_dir=tmp_path / "cache")
    assert options == [scribe._socket_options()]
    assert (scribe.socket.SOL_SOCKET, scribe.socket.SO_KEEPALIVE, 1) in options[0]


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    with pytest.raises(scribe.ScribeError, match="missing"):
        scribe.request_transcript(path, terms=[], cache_dir=tmp_path / "cache")


def test_rate_limit_retry(tmp_path, monkeypatch):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"audio")
    replies = iter(
        [
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json={"text": "", "words": []}),
        ]
    )
    mock_api(monkeypatch, lambda req: next(replies))
    monkeypatch.setattr(scribe.time, "sleep", lambda delay: None)
    assert (
        scribe.request_transcript(path, terms=[], cache_dir=tmp_path / "cache")["words"]
        == []
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "Lost", "words": []},
        {"words": [word("bad", -1, 2)]},
        {"words": [word("bad", 3, 2)]},
    ],
)
def test_invalid_response(payload):
    with pytest.raises(scribe.ScribeError):
        scribe.validate_response(payload)


def test_keyterms_validation():
    assert scribe.keyterms(" Scribe ; Scribe\nGigaAM, <bad>, " + "x" * 50) == [
        "Scribe",
        "GigaAM",
    ]


def test_report_contract_preserved(tmp_path, monkeypatch):
    segment = asr.Segment("Speaker 1", "audio.m4a", 0, 1, text="Текст")
    monkeypatch.setattr(
        scribe, "transcribe_tracks", lambda **kwargs: ([segment], kwargs["tracks"])
    )
    calls = []
    monkeypatch.setattr(asr, "write_artifacts", lambda **kwargs: calls.append(kwargs))
    asr.transcribe_meeting(
        meeting_dir=tmp_path,
        model_name="scribe_v2",
        cache_dir="unused",
        device="cpu",
        batch_size=4,
    )
    assert calls[0]["include_report"] is True
    assert calls[0]["model_name"] == "scribe_v2"
    assert calls[0]["segments"] == [segment]


def test_four_room_voices_keep_short_turn_and_single_track_name(tmp_path, monkeypatch):
    replies = iter(
        [
            {
                "words": [
                    word("Длинный доклад", 0, 60),
                    word("Да", 61, 61.2, "speaker_1"),
                    word("Согласен", 62, 62.5, "speaker_2"),
                    word("Нет", 63, 63.2, "speaker_3"),
                ]
            },
            {"words": [word("Ответ удалённого участника", 65, 67)]},
        ]
    )
    monkeypatch.setattr(scribe, "request_transcript", lambda *a, **k: next(replies))
    tracks = [
        asr.TrackSpec(tmp_path / "room.m4a", "Room owner"),
        asr.TrackSpec(tmp_path / "remote.m4a", "Aziz"),
    ]
    segments, mapped = scribe.transcribe_tracks(
        tracks=tracks,
        metadata={
            "zoom_participant_tracks": [
                {"microphone_mode": "room"},
                {"microphone_mode": "single"},
                {"microphone_mode": "unknown"},
            ]
        },
        cache_dir=tmp_path,
    )
    assert [s.speaker for s in segments] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 3",
        "Speaker 4",
        "Aziz",
    ]
    assert segments[1].text == "Да"
    assert len(mapped) == 5
    asr.write_artifacts(
        output_dir=tmp_path / "artifacts",
        meeting_id="test",
        title="Test",
        source_filename="room.m4a",
        model_name="scribe_v2",
        tracks=mapped,
        segments=segments,
        elapsed_seconds=1,
        include_report=False,
    )
    transcript = json.loads((tmp_path / "artifacts/transcript.json").read_text())
    assert [s["speaker"] for s in transcript] == [s.speaker for s in segments]
    assert "Room owner" not in (tmp_path / "artifacts/transcript.md").read_text()


def test_explicit_personal_microphone_ignores_spurious_speaker(tmp_path, monkeypatch):
    calls = []

    def response(*args, **kwargs):
        calls.append(kwargs)
        return {
            "words": [word("Основная речь", 0, 30), word("Да", 31, 31.1, "speaker_1")]
        }

    monkeypatch.setattr(scribe, "request_transcript", response)
    tracks = [asr.TrackSpec(tmp_path / "01.m4a", "Александр")]
    segments, _ = scribe.transcribe_tracks(
        tracks=tracks, metadata={"microphone_mode": "single"}, cache_dir=tmp_path
    )
    assert {s.speaker for s in segments} == {"Александр"}
    assert calls[0]["single_speaker"] is True
    assert "LangGraph" in calls[0]["terms"]


def test_unknown_mic_never_claims_zoom_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MAC_TRANSCRIBER_MICROPHONE_MODES_FILE", str(tmp_path / "missing.json")
    )
    monkeypatch.setattr(
        scribe,
        "request_transcript",
        lambda *a, **k: {"words": [word("Привет всем", 0, 1)]},
    )
    segments, _ = scribe.transcribe_tracks(
        tracks=[asr.TrackSpec(tmp_path / "01.m4a", "Егор")],
        metadata={},
        cache_dir=tmp_path,
    )
    assert segments[0].speaker == "Speaker 1"


def test_mic_override_priority(tmp_path, monkeypatch):
    config = tmp_path / "modes.json"
    config.write_text('{"Егор":"single"}')
    monkeypatch.setenv("MAC_TRANSCRIBER_MICROPHONE_MODES_FILE", str(config))
    track = asr.TrackSpec(tmp_path / "01.m4a", "Егор")
    assert scribe.microphone_mode(track, 0, {}) == "auto"
    assert (
        scribe.microphone_mode(
            track, 0, {"zoom_participant_tracks": [{"microphone_mode": "room"}]}
        )
        == "room"
    )


def sustained(sid, start=0, count=30):
    return [word('слово', start+i*0.6, start+i*0.6+0.5, sid) for i in range(count)]


def test_auto_same_connection_changes_each_meeting(tmp_path, monkeypatch):
    tracks = [asr.TrackSpec(tmp_path/'01.m4a', 'Егор')]
    for words, mode, labels in [
        (sustained('a'), 'single', {'Егор'}),
        (sustained('a')+sustained('b', 30), 'room', {'Егор / Speaker 1', 'Егор / Speaker 2'}),
        ([word('Да', 0, 0.3)], 'unknown', {'Speaker 1'}),
        (sustained('a')+[word('Это', 25, 25.1, 'b')], 'single', {'Егор', 'Speaker 1'}),
        (sustained('a')+sustained('b', 30, 5), 'unknown', {'Speaker 1', 'Speaker 2'}),
    ]:
        monkeypatch.setattr(scribe, 'request_transcript', lambda *a, **k: {'words': words})
        diagnostics = []
        segments, _ = scribe.transcribe_tracks(tracks=tracks, metadata={}, cache_dir=tmp_path, diagnostics=diagnostics)
        assert {s.speaker for s in segments} == labels
        assert diagnostics[0]['mode'] == mode
        assert sum(len(s.text.split()) for s in segments) == len(words)


def test_unknown_speaker_id_cannot_claim_name():
    assert scribe.infer_microphone(sustained('unknown'))['mode'] == 'unknown'
