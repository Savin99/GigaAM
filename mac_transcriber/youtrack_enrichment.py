"""Read-only YouTrack enrichment for meeting reports.

The module deliberately separates three trust domains:

* Claude proposes search intents from cited transcript segments.
* Python retrieves real issue candidates through the YouTrack MCP endpoint.
* Claude compares candidates with the cited context, while Python validates that
  every selected issue and segment really existed in the retrieved snapshot.

Only ``search_issues`` and ``get_issue`` are allowed.  Tracker credentials never
enter prompts, artifacts, or logs.
"""

from __future__ import annotations

import copy
import json
import os
import re
import ssl
import tomllib
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ISSUE_ID_RE = re.compile(r"\bCH[\s-]?(\d{2,6})\b", re.IGNORECASE)
SAFE_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_-]+")
QUERY_OPERATORS = {"and", "or", "not"}
ALLOWED_KINDS = {
    "explicit_issue",
    "action_item",
    "decision",
    "problem",
    "feature",
}
ALLOWED_SIGNALS = {
    "explicit_issue_id",
    "same_object",
    "same_action",
    "same_component",
    "same_deliverable",
    "same_person",
    "same_parent_epic",
    "same_time_window",
}
MAX_INTENTS = 12
MAX_QUERIES_PER_INTENT = 3
MAX_CANDIDATES = 24
MAX_RESULTS_PER_QUERY = 5


class YouTrackEnrichmentError(RuntimeError):
    """Safe, credential-free enrichment failure."""


@dataclass(frozen=True)
class YouTrackMcpConfig:
    endpoint: str
    token: str
    project: str = "CH"
    timeout_seconds: float = 20.0
    verify_tls: bool = True


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def load_mcp_config(
    config_path: Path | None = None,
    *,
    project: str | None = None,
) -> YouTrackMcpConfig | None:
    """Load MCP endpoint credentials from env, then the local Codex config.

    The values are returned only in memory.  Callers must never serialize the
    resulting config object.
    """

    url = (os.environ.get("YOUTRACK_URL") or "").strip()
    token = (os.environ.get("YOUTRACK_TOKEN") or "").strip()

    if not (url and token):
        path = config_path or (Path.home() / ".codex" / "config.toml")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            server = ((data.get("mcp_servers") or {}).get("youtrack") or {})
            values = server.get("env") or {}
            url = url or str(values.get("YOUTRACK_URL") or "").strip()
            token = token or str(values.get("YOUTRACK_TOKEN") or "").strip()
        except (OSError, tomllib.TOMLDecodeError, TypeError):
            return None

    if not (url and token):
        return None

    timeout_raw = os.environ.get("MAC_TRANSCRIBER_YOUTRACK_TIMEOUT", "20")
    try:
        timeout = max(1.0, float(timeout_raw))
    except ValueError:
        timeout = 20.0
    resolved_project = (
        project or os.environ.get("MAC_TRANSCRIBER_YOUTRACK_PROJECT") or "CH"
    ).strip()
    return YouTrackMcpConfig(
        endpoint=f"{url.rstrip('/')}/mcp",
        token=token,
        project=resolved_project or "CH",
        timeout_seconds=timeout,
        verify_tls=_env_flag("MAC_TRANSCRIBER_YOUTRACK_VERIFY_TLS", True),
    )


class YouTrackMcpClient:
    """Minimal JSON-RPC client restricted to read-only YouTrack MCP tools."""

    _allowed_tools = {"search_issues", "get_issue"}

    def __init__(self, config: YouTrackMcpConfig):
        self.config = config
        self._request_id = 0

    @classmethod
    def from_local_config(
        cls, config_path: Path | None = None
    ) -> "YouTrackMcpClient":
        config = load_mcp_config(config_path)
        if config is None:
            raise YouTrackEnrichmentError(
                "YouTrack MCP is not configured in env or ~/.codex/config.toml."
            )
        return cls(config)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._allowed_tools:
            raise YouTrackEnrichmentError(
                f"YouTrack MCP tool is not allowed in report enrichment: {name}"
            )
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.token}",
            },
            method="POST",
        )
        context = None
        if not self.config.verify_tls:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed configured MCP URL
                request,
                timeout=self.config.timeout_seconds,
                context=context,
            ) as response:
                message = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise YouTrackEnrichmentError(
                f"YouTrack MCP returned HTTP {exc.code}."
            ) from exc
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise YouTrackEnrichmentError(
                f"YouTrack MCP request failed: {type(exc).__name__}."
            ) from exc

        if not isinstance(message, dict):
            raise YouTrackEnrichmentError("YouTrack MCP returned a non-object response.")
        if message.get("error"):
            error = message["error"]
            safe_message = (
                str(error.get("message") or "unknown MCP error")
                if isinstance(error, dict)
                else "unknown MCP error"
            )
            raise YouTrackEnrichmentError(
                f"YouTrack MCP tool {name} failed: {safe_message[:300]}"
            )
        result = message.get("result") or {}
        if not isinstance(result, dict):
            raise YouTrackEnrichmentError(
                f"YouTrack MCP tool {name} returned an invalid result."
            )
        if result.get("isError"):
            raise YouTrackEnrichmentError(
                f"YouTrack MCP tool {name} reported an error."
            )
        for item in result.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise YouTrackEnrichmentError(
            f"YouTrack MCP tool {name} returned no JSON object."
        )

    def search_issues(
        self, query: str, *, limit: int = MAX_RESULTS_PER_QUERY
    ) -> list[dict[str, Any]]:
        result = self._call_tool(
            "search_issues",
            {
                "query": query,
                "limit": min(max(1, limit), 20),
                "customFieldsToReturn": [
                    "Type",
                    "State",
                    "Stage",
                    "Assignee",
                    "Priority",
                ],
            },
        )
        issues = result.get("issuesPage") or []
        return [issue for issue in issues if isinstance(issue, dict)]

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        result = self._call_tool("get_issue", {"issueId": issue_id})
        if str(result.get("id") or "").upper() != issue_id.upper():
            raise YouTrackEnrichmentError(
                f"YouTrack MCP returned a mismatched issue for {issue_id}."
            )
        return result


def _known_segments(agent_input: dict[str, Any]) -> dict[str, str]:
    known: dict[str, str] = {}
    for segment in agent_input.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id") or "").strip()
        text = str(segment.get("text") or "").strip()
        if segment_id:
            known[segment_id] = text
    return known


def _issue_ids(text: str) -> list[str]:
    return [f"CH-{match}" for match in ISSUE_ID_RE.findall(text)]


def explicit_issue_intents(agent_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build deterministic intents for issue IDs literally present in transcript."""

    by_issue: dict[str, dict[str, Any]] = {}
    for segment_id, text in _known_segments(agent_input).items():
        for issue_id in _issue_ids(text):
            intent = by_issue.setdefault(
                issue_id,
                {
                    "intent_id": f"EXPLICIT-{issue_id}",
                    "kind": "explicit_issue",
                    "summary": f"Явное упоминание {issue_id}",
                    "queries": [],
                    "explicit_issue_ids": [issue_id],
                    "segment_ids": [],
                },
            )
            intent["segment_ids"].append(segment_id)
    return list(by_issue.values())


def _safe_query_text(value: object) -> str:
    tokens = [
        token
        for token in SAFE_TOKEN_RE.findall(str(value or ""))
        if token.lower() not in QUERY_OPERATORS
    ]
    return " ".join(tokens[:14]).strip()


def normalize_model_intents(
    raw: object, agent_input: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate model intents against real segment IDs and plain search terms."""

    known = _known_segments(agent_input)
    source = raw.get("intents") if isinstance(raw, dict) else None
    if not isinstance(source, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(source[:MAX_INTENTS], start=1):
        if not isinstance(item, dict):
            continue
        segment_ids = list(
            dict.fromkeys(
                str(value)
                for value in (item.get("segment_ids") or [])
                if str(value) in known
            )
        )
        summary = str(item.get("summary") or "").strip()[:600]
        if not segment_ids or not summary:
            continue
        kind = str(item.get("kind") or "feature").strip()
        if kind not in ALLOWED_KINDS:
            kind = "feature"
        raw_queries = item.get("queries") or item.get("search_terms") or []
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        queries = []
        for query in raw_queries[:MAX_QUERIES_PER_INTENT]:
            safe = _safe_query_text(query)
            if len(safe) >= 4 and safe not in queries:
                queries.append(safe)
        if not queries:
            fallback = _safe_query_text(summary)
            if fallback:
                queries.append(fallback)
        normalized.append(
            {
                "intent_id": f"YT{index:03d}",
                "kind": kind,
                "summary": summary,
                "queries": queries,
                # Model-proposed IDs are intentionally not trusted as explicit IDs.
                "explicit_issue_ids": [],
                "segment_ids": segment_ids,
            }
        )
    return normalized


def _merge_intents(
    explicit: list[dict[str, Any]], semantic: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    combined = [*explicit, *semantic]
    return combined[: MAX_INTENTS + len(explicit)]


def _normalize_issue(raw: dict[str, Any]) -> dict[str, Any]:
    fields = raw.get("customFields") or {}
    if not isinstance(fields, dict):
        fields = {}
    parent = raw.get("parentIssue")
    parent_out = None
    if isinstance(parent, dict) and parent.get("id"):
        parent_out = {
            "id": str(parent.get("id")),
            "summary": str(parent.get("summary") or ""),
        }
    return {
        "id": str(raw.get("id") or ""),
        "url": str(raw.get("url") or ""),
        "summary": str(raw.get("summary") or "")[:500],
        "description": str(raw.get("description") or "")[:5000],
        "state": str(fields.get("State") or ""),
        "stage": str(fields.get("Stage") or ""),
        "assignee": str(fields.get("Assignee") or ""),
        "priority": str(fields.get("Priority") or ""),
        "parent": parent_out,
        "resolved_at": raw.get("resolvedAt"),
        "created_at": raw.get("createdAt"),
        "updated_at": raw.get("updatedAt"),
    }


def build_candidate_snapshot(
    *,
    agent_input: dict[str, Any],
    model_intents: object,
    client: YouTrackMcpClient,
) -> dict[str, Any]:
    """Retrieve bounded candidates for transcript-grounded intents."""

    explicit = explicit_issue_intents(agent_input)
    semantic = normalize_model_intents(model_intents, agent_input)
    intents = _merge_intents(explicit, semantic)
    candidate_links: dict[str, set[str]] = defaultdict(set)
    retrieval: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    query_log: list[dict[str, Any]] = []

    for intent in intents:
        intent_id = intent["intent_id"]
        for issue_id in intent.get("explicit_issue_ids") or []:
            candidate_links[issue_id].add(intent_id)
            retrieval[issue_id].add("explicit_id")

        for search_text in (intent.get("queries") or [])[:MAX_QUERIES_PER_INTENT]:
            if len(candidate_links) >= MAX_CANDIDATES:
                break
            query = f"project: {client.config.project} {search_text}".strip()
            try:
                hits = client.search_issues(query, limit=MAX_RESULTS_PER_QUERY)
            except YouTrackEnrichmentError as exc:
                errors.append(f"{intent_id}: {exc}")
                query_log.append(
                    {"intent_id": intent_id, "query": query, "status": "error"}
                )
                continue
            query_log.append(
                {
                    "intent_id": intent_id,
                    "query": query,
                    "status": "ok",
                    "result_count": len(hits),
                }
            )
            for issue in hits:
                issue_id = str(issue.get("id") or "").upper()
                if not re.fullmatch(r"CH-\d+", issue_id):
                    continue
                candidate_links[issue_id].add(intent_id)
                retrieval[issue_id].add("semantic_search")
                if len(candidate_links) >= MAX_CANDIDATES:
                    break

    candidates: list[dict[str, Any]] = []
    for issue_id in list(candidate_links)[:MAX_CANDIDATES]:
        try:
            issue = _normalize_issue(client.get_issue(issue_id))
        except YouTrackEnrichmentError as exc:
            errors.append(f"{issue_id}: {exc}")
            continue
        candidates.append(
            {
                "issue": issue,
                "intent_ids": sorted(candidate_links[issue_id]),
                "retrieved_by": sorted(retrieval[issue_id]),
            }
        )

    return {
        "schema_version": 1,
        "source": "youtrack_mcp",
        "project": client.config.project,
        "fetched_at": datetime.now(UTC).isoformat(),
        "status": "partial" if errors else "ok",
        "intents": intents,
        "queries": query_log,
        "candidates": candidates,
        "errors": errors[:20],
    }


def _safe_list(value: object, *, limit: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:limit]:
        text = str(item or "").strip()[:max_len]
        if text and text not in result:
            result.append(text)
    return result


def finalize_context(
    *,
    candidate_snapshot: dict[str, Any],
    verification: object,
    agent_input: dict[str, Any],
    generated_by: str,
    extra_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Validate Claude's comparison and build the durable report context."""

    known_segments = set(_known_segments(agent_input))
    intents = {
        str(item.get("intent_id")): item
        for item in candidate_snapshot.get("intents") or []
        if isinstance(item, dict) and item.get("intent_id")
    }
    candidates = {
        str(item.get("issue", {}).get("id")): item
        for item in candidate_snapshot.get("candidates") or []
        if isinstance(item, dict)
        and isinstance(item.get("issue"), dict)
        and item["issue"].get("id")
    }
    raw_matches = (
        verification.get("matches")
        if isinstance(verification, dict)
        and isinstance(verification.get("matches"), list)
        else []
    )

    matches: list[dict[str, Any]] = []
    errors = [
        str(error)[:500] for error in (candidate_snapshot.get("errors") or [])
    ]
    errors.extend(str(error)[:500] for error in (extra_errors or []))

    for raw in raw_matches:
        if not isinstance(raw, dict):
            continue
        intent_id = str(raw.get("intent_id") or "")
        issue_id = str(raw.get("issue_id") or "").upper()
        intent = intents.get(intent_id)
        candidate = candidates.get(issue_id)
        if (
            intent is None
            or candidate is None
            or intent_id not in (candidate.get("intent_ids") or [])
        ):
            errors.append(
                f"Verifier selected a non-candidate pair: {intent_id}/{issue_id}."
            )
            continue

        allowed_segments = set(intent.get("segment_ids") or []) & known_segments
        segment_ids = [
            value
            for value in _safe_list(
                raw.get("segment_ids"), limit=12, max_len=32
            )
            if value in allowed_segments
        ]
        if not segment_ids:
            errors.append(
                f"Verifier provided no valid transcript evidence: {intent_id}/{issue_id}."
            )
            continue
        signals = [
            value
            for value in _safe_list(raw.get("signals"), limit=8, max_len=64)
            if value in ALLOWED_SIGNALS
        ]
        conflicts = _safe_list(raw.get("conflicts"), limit=8, max_len=500)
        requested = str(raw.get("decision") or "possible").lower()
        explicit = (
            "explicit_issue_id" in signals
            and issue_id in (intent.get("explicit_issue_ids") or [])
        )
        enough_evidence = bool(segment_ids) and (explicit or len(set(signals)) >= 2)
        decision = "confirmed" if requested == "confirmed" and enough_evidence else "possible"
        if conflicts:
            decision = "possible"
        if requested == "rejected":
            continue
        matches.append(
            {
                "intent_id": intent_id,
                "issue": copy.deepcopy(candidate["issue"]),
                "decision": decision,
                "signals": signals,
                "conflicts": conflicts,
                "segment_ids": segment_ids,
                "rationale": str(raw.get("rationale") or "").strip()[:1000],
            }
        )

    # Ambiguous intents cannot have multiple confirmed semantic matches.  An exact
    # literal issue ID wins; otherwise all competing matches remain "possible".
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        if match["decision"] == "confirmed":
            by_intent[match["intent_id"]].append(match)
    for intent_id, confirmed in by_intent.items():
        if len(confirmed) <= 1:
            continue
        exact = [
            match
            for match in confirmed
            if "explicit_issue_id" in match["signals"]
            and match["issue"]["id"]
            in (intents[intent_id].get("explicit_issue_ids") or [])
        ]
        winner = exact[0] if len(exact) == 1 else None
        for match in confirmed:
            if match is not winner:
                match["decision"] = "possible"

    confirmed = [match for match in matches if match["decision"] == "confirmed"]
    possible = [match for match in matches if match["decision"] == "possible"]
    linked_intents = {match["intent_id"] for match in matches}
    unmatched = [
        {
            "intent_id": intent_id,
            "kind": str(intent.get("kind") or ""),
            "summary": str(intent.get("summary") or ""),
            "segment_ids": list(intent.get("segment_ids") or []),
        }
        for intent_id, intent in intents.items()
        if intent_id not in linked_intents
    ]
    snapshot_status = str(candidate_snapshot.get("status") or "ok")
    status = "partial" if errors or snapshot_status != "ok" else "ok"
    return {
        "schema_version": 1,
        "source": "youtrack_mcp",
        "project": str(candidate_snapshot.get("project") or "CH"),
        "status": status,
        "fetched_at": candidate_snapshot.get("fetched_at"),
        "generated_by": generated_by,
        "confirmed": confirmed,
        "possible": possible,
        "unmatched": unmatched,
        "candidate_count": len(candidates),
        "candidate_ids": sorted(candidates),
        "intent_count": len(intents),
        "search_queries": list(candidate_snapshot.get("queries") or []),
        "errors": list(dict.fromkeys(errors))[:20],
    }


def empty_context(
    *,
    status: str,
    generated_by: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "youtrack_mcp",
        "project": os.environ.get("MAC_TRANSCRIBER_YOUTRACK_PROJECT", "CH"),
        "status": status,
        "fetched_at": datetime.now(UTC).isoformat(),
        "generated_by": generated_by,
        "confirmed": [],
        "possible": [],
        "unmatched": [],
        "candidate_count": 0,
        "candidate_ids": [],
        "intent_count": 0,
        "search_queries": [],
        "errors": [error[:500]] if error else [],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def inject_context(agent_input_path: Path, context: dict[str, Any]) -> None:
    payload = read_json_object(agent_input_path)
    if payload is None:
        raise YouTrackEnrichmentError(
            f"Agent input is not valid JSON: {agent_input_path.name}"
        )
    payload["youtrack_context"] = context
    write_json(agent_input_path, payload)


def _issue_item(match: dict[str, Any], *, possible: bool) -> dict[str, Any]:
    issue = match["issue"]
    state_parts = []
    if issue.get("stage"):
        state_parts.append(f"этап: {issue['stage']}")
    if issue.get("state"):
        state_parts.append(f"состояние: {issue['state']}")
    if issue.get("assignee"):
        state_parts.append(f"исполнитель: {issue['assignee']}")
    parent = issue.get("parent")
    if isinstance(parent, dict) and parent.get("id"):
        state_parts.append(
            f"родитель: {parent['id']} {str(parent.get('summary') or '').strip()}"
        )
    prefix = "Возможное совпадение, требует проверки." if possible else "Связь подтверждена."
    rationale = str(match.get("rationale") or "").strip()
    details = "; ".join(state_parts)
    text_parts = [prefix]
    if details:
        text_parts.append(details + ".")
    if rationale:
        text_parts.append(rationale)
    if issue.get("url"):
        text_parts.append(f"YouTrack: {issue['url']}")
    return {
        "title": f"{issue['id']} · {issue.get('summary') or 'Без названия'}",
        "text": " ".join(text_parts),
        "citations": list(match.get("segment_ids") or []),
    }


def _dedupe_report_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_issue: dict[str, dict[str, Any]] = {}
    for match in matches:
        issue = match.get("issue")
        if not isinstance(issue, dict) or not issue.get("id"):
            continue
        issue_id = str(issue["id"])
        if issue_id not in by_issue:
            by_issue[issue_id] = copy.deepcopy(match)
            continue
        current = by_issue[issue_id]
        current["segment_ids"] = list(
            dict.fromkeys(
                [
                    *(current.get("segment_ids") or []),
                    *(match.get("segment_ids") or []),
                ]
            )
        )
        current["signals"] = list(
            dict.fromkeys(
                [*(current.get("signals") or []), *(match.get("signals") or [])]
            )
        )
    return list(by_issue.values())


def augment_report_payload(
    payload: dict[str, Any], context: dict[str, Any] | None
) -> dict[str, Any]:
    """Append deterministic tracker sections to the model report payload."""

    result = copy.deepcopy(payload)
    sections = [
        section
        for section in (result.get("adaptive_sections") or [])
        if not (
            isinstance(section, dict)
            and str(section.get("kind") or "").startswith("youtrack_")
        )
    ]
    if not context:
        result["adaptive_sections"] = sections
        return result

    confirmed = _dedupe_report_matches(
        [
            match
            for match in context.get("confirmed") or []
            if isinstance(match, dict)
        ]
    )
    confirmed_ids = {
        str(match.get("issue", {}).get("id") or "") for match in confirmed
    }
    possible = _dedupe_report_matches(
        [
            match
            for match in context.get("possible") or []
            if isinstance(match, dict)
            and isinstance(match.get("issue"), dict)
            and str(match.get("issue", {}).get("id") or "") not in confirmed_ids
        ]
    )
    for kind, title, summary, matches, is_possible in (
        (
            "youtrack_confirmed",
            "Подтверждённые связи с YouTrack",
            "Задачи получены через read-only MCP и сверены с конкретными фрагментами встречи.",
            confirmed,
            False,
        ),
        (
            "youtrack_possible",
            "Возможные связи с YouTrack",
            "Совпадения недостаточно надёжны для утверждения и требуют ручной проверки.",
            possible,
            True,
        ),
    ):
        if not matches:
            continue
        items = [_issue_item(match, possible=is_possible) for match in matches]
        citations = list(
            dict.fromkeys(
                segment_id
                for item in items
                for segment_id in item.get("citations") or []
            )
        )
        sections.append(
            {
                "kind": kind,
                "title": title,
                "purpose": "Связать содержание встречи с текущим состоянием задач.",
                "summary": summary,
                "items": items,
                "citations": citations,
                "accent": "blue",
            }
        )
    result["adaptive_sections"] = sections
    return result


def publish_context(work_path: Path, output_dir: Path) -> Path | None:
    context = read_json_object(work_path)
    if context is None:
        return None
    target = output_dir / "youtrack_context.json"
    write_json(target, context)
    return target


def annotate_report_health(
    health_path: Path, context: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Add a non-blocking enrichment check to an existing report health file."""

    health = read_json_object(health_path)
    if health is None or context is None:
        return health
    status = str(context.get("status") or "unavailable")
    ok = status in {"ok", "disabled"}
    if status == "ok":
        details = (
            "YouTrack enrichment completed: "
            f"{len(context.get('confirmed') or [])} confirmed, "
            f"{len(context.get('possible') or [])} possible."
        )
    elif status == "disabled":
        details = "YouTrack enrichment was explicitly disabled."
    else:
        errors = context.get("errors") or []
        reason = str(errors[0])[:240] if errors else f"status={status}"
        details = f"YouTrack enrichment is {status}: {reason}"

    checks = [
        check
        for check in (health.get("checks") or [])
        if not (
            isinstance(check, dict)
            and check.get("name") == "youtrack_enrichment"
        )
    ]
    checks.append({"name": "youtrack_enrichment", "ok": ok, "details": details})
    health["checks"] = checks
    alerts = [str(alert) for alert in (health.get("alerts") or [])]
    if not ok and details not in alerts:
        alerts.append(details)
    health["alerts"] = alerts
    if not ok and health.get("status") != "failed":
        health["status"] = "degraded"
    target = health_path.parent / "youtrack_context.json"
    artifacts = health.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        health["artifacts"] = artifacts
    artifacts["youtrack_context_json"] = {
        "exists": target.exists(),
        "size_bytes": target.stat().st_size if target.exists() else 0,
    }
    health["youtrack_enrichment"] = {
        "status": status,
        "confirmed": len(context.get("confirmed") or []),
        "possible": len(context.get("possible") or []),
        "unmatched": len(context.get("unmatched") or []),
        "fetched_at": context.get("fetched_at"),
    }
    write_json(health_path, health)
    return health
