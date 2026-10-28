"""Validate extractive evidence before publishing headless-generated commitments.

This is a provenance gate, not a proof of semantic entailment. The author must
also check that quotes establish the claim, owner, deadline and certainty.
"""

from copy import deepcopy
import re


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold().replace("ё", "е")


def validate_payload(payload: dict, transcript) -> tuple[dict, dict]:
    result = deepcopy(payload)
    records = {r.segment_id: r.text for r in transcript}
    audit = {
        "version": 1,
        "status": "passed",
        "checks": [],
        "limitation": "Exact source quotes verified; semantic entailment is not mechanically proven.",
    }
    for section in ("decisions", "action_items"):
        kept = []
        for index, item in enumerate(result.get(section) or []):
            label = f"{section}[{index}]"
            certainty = item.get("certainty")
            if certainty not in {"confirmed", "proposed"}:
                raise ValueError(f"{label}: certainty must be confirmed or proposed")
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"{label}: source evidence quotes required")
            citations = item.get("citations") or []
            quoted_ids = []
            quotes = []
            for entry in evidence:
                sid, quote = entry.get("segment_id"), entry.get("quote")
                if (
                    sid not in records
                    or sid not in citations
                    or not isinstance(quote, str)
                    or not quote.strip()
                ):
                    raise ValueError(
                        f"{label}: evidence must refer to a cited source segment"
                    )
                if _normalize(quote) not in _normalize(records[sid]):
                    raise ValueError(f"{label}: evidence quote does not match {sid}")
                quoted_ids.append(sid)
                quotes.append(quote)
            if set(citations) != set(quoted_ids):
                raise ValueError(
                    f"{label}: every citation requires an exact evidence quote"
                )
            if len(re.findall(r"\w+", " ".join(quotes))) < 3:
                raise ValueError(
                    f"{label}: include context, not only a short acknowledgement"
                )
            tentative = all(
                re.search(
                    r"могу попробовать|возможно.{0,12}(?:стоит|мог)|может быть.{0,12}стоит",
                    _normalize(q),
                )
                for q in quotes
            )
            if certainty == "confirmed" and tentative:
                raise ValueError(
                    f"{label}: proposal-only evidence cannot establish a commitment"
                )
            audit["checks"].append(
                {
                    "section": section,
                    "index": index,
                    "certainty": certainty,
                    "citations": citations,
                    "evidence": evidence,
                }
            )
            if certainty == "proposed":
                result.setdefault("open_questions", []).append(
                    {
                        "title": item.get("title", "Предложение"),
                        "text": "Предложено, но не подтверждено как договорённость: "
                        + item.get("text", ""),
                        "citations": citations,
                    }
                )
            else:
                kept.append(item)
        result[section] = kept
    return result, audit
