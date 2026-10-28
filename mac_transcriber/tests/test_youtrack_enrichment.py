import json

from mac_transcriber import youtrack_enrichment as yt


def _agent_input():
    return {
        "segments": [
            {
                "segment_id": "S0001",
                "text": "В CH 594 нужно отдавать дополнительные поля документов.",
            },
            {
                "segment_id": "S0002",
                "text": "Это ручка оборудования для фронта.",
            },
        ]
    }


class FakeClient:
    def __init__(self):
        self.config = yt.YouTrackMcpConfig(
            endpoint="https://youtrack.example/mcp",
            token="not-used",
            project="CH",
        )
        self.queries = []

    def search_issues(self, query, *, limit):
        self.queries.append((query, limit))
        return [
            {
                "id": "CH-594",
                "summary": "Отдавать доп поля для фронта по документам",
            }
        ]

    def get_issue(self, issue_id):
        assert issue_id == "CH-594"
        return {
            "id": "CH-594",
            "url": "https://youtrack.example/issue/CH-594",
            "summary": "[Оборудование] Отдавать доп поля для фронта по документам",
            "description": "Ручка листа документов должна отдавать доп поля.",
            "parentIssue": {"id": "CH-592", "summary": "[Epic] Оборудование"},
            "updatedAt": "2026-07-27 11:42:49",
            "customFields": {
                "Stage": "Develop",
                "State": "In Progress",
                "Assignee": "George",
            },
        }


def test_explicit_issue_intents_normalize_space_and_deduplicate():
    payload = _agent_input()
    payload["segments"].append(
        {"segment_id": "S0003", "text": "Вернёмся к ch-594 после теста."}
    )

    intents = yt.explicit_issue_intents(payload)

    assert intents == [
        {
            "intent_id": "EXPLICIT-CH-594",
            "kind": "explicit_issue",
            "summary": "Явное упоминание CH-594",
            "queries": [],
            "explicit_issue_ids": ["CH-594"],
            "segment_ids": ["S0001", "S0003"],
        }
    ]


def test_normalize_model_intents_keeps_real_segments_and_plain_queries():
    intents = yt.normalize_model_intents(
        {
            "intents": [
                {
                    "kind": "feature",
                    "summary": "Поля документов для оборудования CH-999",
                    "queries": [
                        "оборудование State: Done or документы",
                        "фронт дополнительные поля",
                    ],
                    "segment_ids": ["S0001", "S9999"],
                }
            ]
        },
        _agent_input(),
    )

    assert intents[0]["segment_ids"] == ["S0001"]
    assert intents[0]["explicit_issue_ids"] == []
    assert intents[0]["queries"] == [
        "оборудование State Done документы",
        "фронт дополнительные поля",
    ]


def test_build_candidate_snapshot_uses_read_only_bounded_search():
    client = FakeClient()

    snapshot = yt.build_candidate_snapshot(
        agent_input=_agent_input(),
        model_intents={
            "intents": [
                {
                    "kind": "feature",
                    "summary": "Дополнительные поля документов",
                    "queries": ["оборудование дополнительные поля документы"],
                    "segment_ids": ["S0001", "S0002"],
                }
            ]
        },
        client=client,
    )

    assert snapshot["status"] == "ok"
    assert snapshot["candidates"][0]["issue"]["id"] == "CH-594"
    assert snapshot["candidates"][0]["issue"]["stage"] == "Develop"
    assert snapshot["candidates"][0]["retrieved_by"] == [
        "explicit_id",
        "semantic_search",
    ]
    assert client.queries == [
        ("project: CH оборудование дополнительные поля документы", 5)
    ]


def test_finalize_context_confirms_only_real_candidate_with_two_signals():
    client = FakeClient()
    agent_input = _agent_input()
    snapshot = yt.build_candidate_snapshot(
        agent_input=agent_input,
        model_intents={
            "intents": [
                {
                    "kind": "feature",
                    "summary": "Дополнительные поля документов",
                    "queries": ["оборудование дополнительные поля документы"],
                    "segment_ids": ["S0001", "S0002"],
                }
            ]
        },
        client=client,
    )
    semantic_intent_id = next(
        item["intent_id"]
        for item in snapshot["intents"]
        if item["kind"] == "feature"
    )

    context = yt.finalize_context(
        candidate_snapshot=snapshot,
        verification={
            "matches": [
                {
                    "intent_id": semantic_intent_id,
                    "issue_id": "CH-594",
                    "decision": "confirmed",
                    "signals": ["same_object", "same_action"],
                    "conflicts": [],
                    "segment_ids": ["S0001", "S0002", "S9999"],
                    "rationale": "Совпадают объект и требуемое изменение.",
                },
                {
                    "intent_id": semantic_intent_id,
                    "issue_id": "CH-999",
                    "decision": "confirmed",
                    "signals": ["same_object", "same_action"],
                    "segment_ids": ["S0001"],
                },
            ]
        },
        agent_input=agent_input,
        generated_by="claude-opus-5 · two-pass verifier",
    )

    assert context["status"] == "partial"
    assert len(context["confirmed"]) == 1
    assert context["confirmed"][0]["issue"]["id"] == "CH-594"
    assert context["confirmed"][0]["segment_ids"] == ["S0001", "S0002"]
    assert any("non-candidate" in error for error in context["errors"])


def test_finalize_context_demotes_weak_or_conflicted_confirmation():
    agent_input = _agent_input()
    snapshot = {
        "status": "ok",
        "project": "CH",
        "fetched_at": "2026-07-27T12:00:00+00:00",
        "errors": [],
        "intents": [
            {
                "intent_id": "YT001",
                "kind": "feature",
                "summary": "Поля документов",
                "segment_ids": ["S0001"],
                "explicit_issue_ids": [],
            }
        ],
        "candidates": [
            {
                "issue": {
                    "id": "CH-594",
                    "summary": "Поля документов",
                    "url": "https://youtrack.example/issue/CH-594",
                },
                "intent_ids": ["YT001"],
                "retrieved_by": ["semantic_search"],
            }
        ],
    }

    context = yt.finalize_context(
        candidate_snapshot=snapshot,
        verification={
            "matches": [
                {
                    "intent_id": "YT001",
                    "issue_id": "CH-594",
                    "decision": "confirmed",
                    "signals": ["same_object"],
                    "conflicts": ["Действие в задаче отличается."],
                    "segment_ids": ["S0001"],
                    "rationale": "Совпадает только объект.",
                }
            ]
        },
        agent_input=agent_input,
        generated_by="claude-opus-5 · two-pass verifier",
    )

    assert context["confirmed"] == []
    assert context["possible"][0]["issue"]["id"] == "CH-594"


def test_augment_report_payload_adds_deterministic_sections():
    context = {
        "confirmed": [
            {
                "issue": {
                    "id": "CH-594",
                    "summary": "Поля документов",
                    "url": "https://youtrack.example/issue/CH-594",
                    "stage": "Develop",
                    "state": "In Progress",
                    "assignee": "George",
                    "parent": {"id": "CH-592", "summary": "Оборудование"},
                },
                "segment_ids": ["S0001"],
                "rationale": "Совпали объект и действие.",
            }
        ],
        "possible": [],
    }

    result = yt.augment_report_payload(
        {"adaptive_sections": [{"kind": "topic", "title": "Основное"}]},
        context,
    )

    assert [section["kind"] for section in result["adaptive_sections"]] == [
        "topic",
        "youtrack_confirmed",
    ]
    tracker_item = result["adaptive_sections"][1]["items"][0]
    assert tracker_item["title"].startswith("CH-594")
    assert tracker_item["citations"] == ["S0001"]


def test_health_annotation_is_non_blocking_degraded_warning(tmp_path):
    health_path = tmp_path / "report_health.json"
    health_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "alerts": [],
                "checks": [],
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    yt.write_json(
        tmp_path / "youtrack_context.json",
        yt.empty_context(
            status="unavailable",
            generated_by="claude-opus-5 · two-pass verifier",
            error="MCP timeout.",
        ),
    )
    context = yt.read_json_object(tmp_path / "youtrack_context.json")

    health = yt.annotate_report_health(health_path, context)

    assert health["status"] == "degraded"
    assert health["checks"][-1]["name"] == "youtrack_enrichment"
    assert health["checks"][-1]["ok"] is False
    assert health["artifacts"]["youtrack_context_json"]["exists"] is True


def test_load_mcp_config_reads_local_codex_section_without_serializing(tmp_path, monkeypatch):
    monkeypatch.delenv("YOUTRACK_URL", raising=False)
    monkeypatch.delenv("YOUTRACK_TOKEN", raising=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp_servers.youtrack]
command = "python3"

[mcp_servers.youtrack.env]
YOUTRACK_URL = "https://youtrack.example"
YOUTRACK_TOKEN = "secret-for-test"
""".strip(),
        encoding="utf-8",
    )

    config = yt.load_mcp_config(config_path)

    assert config is not None
    assert config.endpoint == "https://youtrack.example/mcp"
    assert config.token == "secret-for-test"
