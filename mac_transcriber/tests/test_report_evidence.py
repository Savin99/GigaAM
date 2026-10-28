from types import SimpleNamespace
import pytest
from mac_transcriber.report_evidence import validate_payload
from mac_transcriber import reporting


def item(quote="Сделаю MR сегодня-завтра.", certainty="confirmed"):
    return {
        "title": "MR",
        "text": "Подготовить MR",
        "certainty": certainty,
        "citations": ["S0002"],
        "evidence": [{"segment_id": "S0002", "quote": quote}],
    }


def test_evidence_accepts_matching_source_and_retains_proposal():
    records = [SimpleNamespace(segment_id="S0002", text="Сделаю MR сегодня-завтра.")]
    payload, audit = validate_payload({"action_items": [item()]}, records)
    assert len(payload["action_items"]) == 1 and audit["status"] == "passed"
    payload, _ = validate_payload(
        {"action_items": [item(certainty="proposed")]}, records
    )
    assert payload["action_items"] == []
    assert "не подтверждено" in payload["open_questions"][0]["text"]


def test_wrong_neighbor_citation_is_rejected():
    records = [
        SimpleNamespace(
            segment_id="S0002",
            text="Ты мне не говорил, что надо делать версионирование.",
        )
    ]
    with pytest.raises(ValueError, match="does not match"):
        validate_payload({"action_items": [item()]}, records)


def test_tentative_only_cannot_be_confirmed():
    quote = "Могу попробовать подготовить документ."
    with pytest.raises(ValueError, match="proposal-only"):
        validate_payload(
            {"action_items": [item(quote)]},
            [SimpleNamespace(segment_id="S0002", text=quote)],
        )


def test_missing_evidence_is_rejected():
    bad = item()
    del bad["evidence"]
    with pytest.raises(ValueError, match="quotes required"):
        validate_payload({"decisions": [bad]}, [])


def test_all_protocol_references_survive_rendering():
    report = reporting.build_local_report(
        meeting_id="test",
        title="test",
        source_filename="test.m4a",
        model_name="scribe_v2",
        segments=[
            {"speaker": "A", "start": 0, "end": 3, "text": "Мы готовы к мержу?"},
            {
                "speaker": "B",
                "start": 4,
                "end": 7,
                "text": "Да, сегодня можно мержить.",
            },
        ],
    )
    ref = reporting._protocol_ref(report, ["S0001", "S0002", "S0001"])
    assert ref == "S0001 · 00:00; S0002 · 00:04"
