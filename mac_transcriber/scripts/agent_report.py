"""Генерация отчёта встречи суб-агентом вместо LLM-API.

Конвейер `reporting.py` оставляем как есть, заменяя ровно один шаг — обращение
к внешнему LLM. Суб-агент (Claude) играет роль AI-генератора:

  prepare  : transcript.json -> базовый MeetingReport (use_ai=False) ->
             agent_input.json с каноническими segment_id (S0001, ...), которые
             агент обязан цитировать.
  finalize : ai_payload.json (ответ агента по ai_report_schema) ->
             _filter_payload_citations -> _merge_ai_payload -> валидация ->
             write_report_artifacts_from_report (md/json/html/typ/pdf/coverage/
             health/slack).

Coverage достраивается автоматически из секций (build_coverage), а неизвестные
segment_id вырезаются фильтром — поэтому шаг устойчив к огрехам агента.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mac_transcriber import reporting  # noqa: E402
from mac_transcriber import youtrack_enrichment as yt  # noqa: E402

MODEL_NAME = "v3_e2e_rnnt"
DEFAULT_CLAUDE_MODEL = "opus"


def _generated_by(model: str | None = None) -> str:
    resolved = (
        model
        if model is not None
        else os.environ.get("MAC_TRANSCRIBER_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    )
    resolved = (resolved or DEFAULT_CLAUDE_MODEL).strip()
    return f"{resolved} · mac_transcriber subagent"


def _clock(seconds: float) -> str:
    """Секунды -> H:MM:SS (или MM:SS для коротких записей)."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _load_segments(meeting_dir: Path) -> list[dict[str, Any]]:
    path = meeting_dir / "artifacts" / "transcript.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"transcript.json is not a list: {path}")
    return data


def _read_status(meeting_dir: Path) -> dict[str, Any]:
    for candidate in (
        meeting_dir / "status.json",
        meeting_dir / "artifacts" / "status.json",
    ):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def _meeting_meta(meeting_dir: Path) -> dict[str, Any]:
    status = _read_status(meeting_dir)
    meta = status.get("metadata", {}) if isinstance(status, dict) else {}
    meeting_id = str(meta.get("meeting_id") or meeting_dir.name)
    source = str(meta.get("source_filename") or "audio.m4a")
    title_seed = Path(source).stem or meeting_dir.name
    participants = [str(p) for p in (meta.get("participants") or [])]
    return {
        "meeting_id": meeting_id,
        "source_filename": source,
        "title_seed": title_seed,
        "participants": participants,
    }


def _build_base(meeting_dir: Path) -> reporting.MeetingReport:
    meta = _meeting_meta(meeting_dir)
    segments = _load_segments(meeting_dir)
    model_name = MODEL_NAME
    model_meta = meeting_dir / "artifacts" / "asr_metadata.json"
    if model_meta.exists():
        model_name = json.loads(model_meta.read_text()).get("model_name") or MODEL_NAME
    else:
        transcript_md = meeting_dir / "artifacts" / "transcript.md"
        if transcript_md.exists():
            for line in transcript_md.read_text().splitlines()[:20]:
                if line.startswith("- Model: "):
                    model_name = line.removeprefix("- Model: ").strip()
                    break
    return reporting.build_report(
        meeting_id=meta["meeting_id"],
        title=meta["title_seed"],
        source_filename=meta["source_filename"],
        model_name=model_name,
        segments=segments,
        use_ai=False,
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    meeting_dir = Path(args.meeting_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    meta = _meeting_meta(meeting_dir)
    base = _build_base(meeting_dir)

    segments_out = [
        {
            "segment_id": r.segment_id,
            "start": round(r.start, 2),
            "end": round(r.end, 2),
            "clock": f"{_clock(r.start)}-{_clock(r.end)}",
            "speaker": r.speaker,
            "text": r.text,
        }
        for r in base.transcript
    ]

    payload = {
        "evidence_contract_version": 1,
        "meeting_id": meta["meeting_id"],
        "title_seed": meta["title_seed"],
        "source_filename": meta["source_filename"],
        "participants": meta["participants"],
        "duration": base.duration,
        "segment_count": base.segment_count,
        "profile_hint": {
            "kind": base.profile.kind,
            "label": base.profile.label,
            "confidence": base.profile.confidence,
            "rationale": base.profile.rationale,
        },
        "segments": segments_out,
    }
    out = work_dir / "agent_input.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"prepared meeting={meta['meeting_id']} segments={base.segment_count} "
        f"profile={base.profile.kind} -> {out}"
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    meeting_dir = Path(args.meeting_dir)
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)

    payload_path = work_dir / "ai_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ai_payload.json is not an object: {payload_path}")

    base = _build_base(meeting_dir)
    known_ids = {r.segment_id for r in base.transcript}

    from mac_transcriber.report_evidence import validate_payload
    agent_input = json.loads((work_dir / "agent_input.json").read_text(encoding="utf-8")) if (work_dir / "agent_input.json").exists() else {}
    evidence_audit = None
    if agent_input.get("evidence_contract_version") == 1:
        payload, evidence_audit = validate_payload(payload, base.transcript)

    youtrack_path = work_dir / "youtrack_context.json"
    youtrack_context = yt.read_json_object(youtrack_path)
    payload = yt.augment_report_payload(payload, youtrack_context)
    title_override = str(payload.get("title") or "").strip()
    filtered = reporting._filter_payload_citations(payload, known_ids)
    report = reporting._merge_ai_payload(
        base,
        filtered,
        report_model=_generated_by(getattr(args, "report_model", None)),
        recover_base_items=False,
    )
    if title_override:
        report = dataclasses.replace(report, title=title_override)

    reporting.validate_report_citations(report)

    if evidence_audit is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report_evidence.json").write_text(json.dumps(evidence_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    yt.publish_context(youtrack_path, out_dir)
    artifacts = reporting.write_report_artifacts_from_report(
        output_dir=out_dir,
        report=report,
        requested_ai=True,
        make_pdf=args.pdf,
    )
    health = yt.annotate_report_health(artifacts.health_path, youtrack_context)
    if evidence_audit is not None and isinstance(health, dict):
        health.setdefault("checks", []).append({
            "name": "report_source_quotes", "ok": True,
            "details": "Exact evidence quotes verified for formal decisions/tasks; semantic entailment remains an author check.",
        })
        artifacts.health_path.write_text(json.dumps(health, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if isinstance(health, dict):
        slack_summary = reporting.render_slack_summary(
            protocol=reporting.build_protocol_data(report),
            health=health,
            output_dir=out_dir,
            pdf_path=artifacts.pdf_path,
        )
        artifacts.slack_summary_path.write_text(slack_summary, encoding="utf-8")
    final_status = (
        str(health.get("status") or artifacts.status)
        if isinstance(health, dict)
        else artifacts.status
    )
    final_alerts = (
        list(health.get("alerts") or [])
        if isinstance(health, dict)
        else artifacts.alerts
    )

    summary = {
        "meeting_id": report.meeting_id,
        "title": report.title,
        "status": final_status,
        "generated_by": artifacts.generated_by,
        "alerts": final_alerts,
        "sections": len(report.adaptive_sections),
        "decisions": len(report.decisions),
        "action_items": len(report.action_items),
        "open_questions": len(report.open_questions),
        "risks": len(report.risks),
        "timeline": len(report.timeline),
        "youtrack_confirmed": len(
            (youtrack_context or {}).get("confirmed") or []
        ),
        "youtrack_possible": len((youtrack_context or {}).get("possible") or []),
        "report_md": str(out_dir / "report.md"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="transcript.json -> agent_input.json")
    p_prep.add_argument("--meeting-dir", required=True, dest="meeting_dir")
    p_prep.add_argument("--work-dir", required=True, dest="work_dir")
    p_prep.set_defaults(func=cmd_prepare)

    p_fin = sub.add_parser("finalize", help="ai_payload.json -> rendered artifacts")
    p_fin.add_argument("--meeting-dir", required=True, dest="meeting_dir")
    p_fin.add_argument("--work-dir", required=True, dest="work_dir")
    p_fin.add_argument("--out-dir", required=True, dest="out_dir")
    p_fin.add_argument("--report-model", default=None, dest="report_model")
    p_fin.add_argument("--pdf", action="store_true", default=False)
    p_fin.set_defaults(func=cmd_finalize)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
