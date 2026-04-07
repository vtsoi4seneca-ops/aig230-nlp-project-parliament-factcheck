#!/usr/bin/env python3
"""
Create a cleaned deterministic gold dataset from the existing broad factcheck master.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


DEFAULT_INPUT = Path("data/factcheck_master.csv")
DEFAULT_OUTPUT_DIR = Path("data")
MASTER_FILENAME = "gold_factcheck_master.csv"
TRAINING_FILENAME = "gold_training_pairs.json"
STATS_FILENAME = "gold_factcheck_stats.csv"

APPENDED_COLUMNS = [
    "source_context_local",
    "dataset_tier",
    "label_source",
    "gold_resolution_method",
    "gold_resolution_score",
    "source_session_id",
    "source_bill_debated_id",
]

TRAINING_PROMPT_INSTRUCTIONS = (
    "Provide your answer in this exact format:\n"
    "VERDICT: [TRUE/FALSE]\n"
    "CONFIDENCE: [0-100]\n"
    "EXPLANATION: [Short evidence-grounded explanation]\n"
    "CORRECTION: [If false, give the correct fact; otherwise N/A]"
)

SESSION_PARLIAMENT_RE = re.compile(
    r"(\d+)(?:st|nd|rd|th)\s+session\s+of\s+the\s+(\d+)(?:st|nd|rd|th)\s+parliament",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned deterministic gold dataset exports.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input broad deterministic master CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for gold outputs.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="openparliament")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or "")
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")


def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def trim_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def infer_local_context(source_context: str, claim_text: str, max_chars: int = 1200) -> str:
    source_context = re.sub(r"\s+", " ", (source_context or "").strip())
    claim_text = re.sub(r"\s+", " ", (claim_text or "").strip())
    if not source_context:
        return claim_text
    if not claim_text:
        return trim_text(source_context, max_chars)

    idx = source_context.find(claim_text)
    if idx >= 0:
        window = source_context[max(0, idx - 420) : min(len(source_context), idx + len(claim_text) + 420)]
        return trim_text(window, max_chars)

    sentences = re.split(r"(?<=[.!?])\s+", source_context)
    for pos, sentence in enumerate(sentences):
        if normalize_text(claim_text) in normalize_text(sentence):
            local_sentences = sentences[max(0, pos - 1) : min(len(sentences), pos + 2)]
            return trim_text(" ".join(local_sentences), max_chars)

    idx = normalize_text(source_context).find(normalize_text(claim_text))
    if idx >= 0:
        window = source_context[max(0, idx - 420) : min(len(source_context), idx + len(claim_text) + 420)]
        return trim_text(window, max_chars)

    return trim_text(source_context, max_chars)


def load_csv_rows(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def connect_db(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        cursor_factory=RealDictCursor,
    )


def fetch_statement_metadata(conn, statement_ids: Sequence[int]) -> Dict[int, dict]:
    if not statement_ids:
        return {}
    query = """
        SELECT
            s.id AS statement_id,
            s.bill_debated_id,
            d.session_id AS source_session_id
        FROM hansards_statement s
        JOIN hansards_document d ON d.id = s.document_id
        WHERE s.id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(query, (list(statement_ids),))
        rows = cur.fetchall()
    return {int(row["statement_id"]): dict(row) for row in rows}


def fetch_bill_metadata(conn, bill_ids: Sequence[int]) -> Dict[int, dict]:
    if not bill_ids:
        return {}
    query = """
        SELECT
            id,
            number,
            session_id,
            introduced,
            status_date,
            latest_debate_date
        FROM bills_bill
        WHERE id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(query, (list(bill_ids),))
        rows = cur.fetchall()
    return {int(row["id"]): dict(row) for row in rows}


def extract_explicit_session_parliament(claim_text: str) -> Optional[Tuple[str, str]]:
    match = SESSION_PARLIAMENT_RE.search(claim_text or "")
    if not match:
        return None
    session_number, parliament_number = match.group(1), match.group(2)
    return parliament_number, session_number


def as_int(value: str) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def score_candidate(row: dict, statement_meta: Optional[dict], bill_meta: Optional[dict]) -> int:
    score = 0
    source_date = parse_date(row.get("source_date", ""))
    vote_date = parse_date(row.get("vote_date", ""))
    introduced = bill_meta.get("introduced") if bill_meta else None
    latest_debate_date = bill_meta.get("latest_debate_date") if bill_meta else None

    candidate_bill_id = as_int(row.get("bill_id"))
    if statement_meta:
        if statement_meta.get("bill_debated_id") and candidate_bill_id == statement_meta.get("bill_debated_id"):
            score += 1000
        if statement_meta.get("source_session_id") and statement_meta.get("source_session_id") == row.get("session_id"):
            score += 275

    explicit_session = extract_explicit_session_parliament(row.get("claim_text", ""))
    if explicit_session:
        explicit_parliament, explicit_session_number = explicit_session
        if (
            explicit_parliament == row.get("parliament_number")
            and explicit_session_number == row.get("session_number")
        ):
            score += 450
        else:
            score -= 450

    if source_date and vote_date:
        delta = (source_date - vote_date).days
        if delta >= 0:
            score += max(0, 180 - min(delta, 180))
        else:
            score -= 240 + min(abs(delta), 365)

    if source_date and introduced:
        delta = (source_date - introduced).days
        if delta >= 0:
            score += max(0, 120 - min(delta, 120))
        else:
            score -= 160 + min(abs(delta), 365)

    if source_date and latest_debate_date:
        delta = (source_date - latest_debate_date).days
        if delta >= 0:
            score += max(0, 40 - min(delta, 40))

    if row.get("claim_type") == "party_vote_direction" and row.get("vote_date"):
        score += 50
    if row.get("claim_type") in {"introduced_date", "status_snapshot", "session_parliament", "sponsor"}:
        score += 30

    return score


def choose_group_row(rows: Sequence[dict], statement_meta_map: Dict[int, dict], bill_meta_map: Dict[int, dict]) -> Tuple[Optional[dict], str, int]:
    if len(rows) == 1:
        row = dict(rows[0])
        row["gold_resolution_score"] = "0"
        return row, "single_candidate", 0

    ranked: List[Tuple[int, dict]] = []
    for row in rows:
        statement_meta = statement_meta_map.get(as_int(row.get("source_statement_id")))
        bill_meta = bill_meta_map.get(as_int(row.get("bill_id")))
        score = score_candidate(row, statement_meta, bill_meta)
        ranked.append((score, row))

    ranked.sort(
        key=lambda item: (
            item[0],
            1 if item[1].get("claim_type") == "party_vote_direction" else 0,
            item[1].get("vote_date") or "",
            item[1].get("session_id") or "",
            item[1].get("bill_id") or "",
        ),
        reverse=True,
    )

    best_score, best_row = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else None
    if second_score is not None and best_score == second_score:
        return None, "score_tie", best_score
    if best_score < 0:
        return None, "negative_best_score", best_score

    row = dict(best_row)
    row["gold_resolution_score"] = str(best_score)
    return row, "scored_resolution", best_score


def enrich_row(row: dict, statement_meta: Optional[dict], resolution_method: str) -> dict:
    enriched = dict(row)
    enriched["source_context_local"] = infer_local_context(row.get("source_context", ""), row.get("claim_text", ""))
    enriched["dataset_tier"] = "gold"
    enriched["label_source"] = "deterministic_db_rule"
    enriched["gold_resolution_method"] = resolution_method
    enriched["source_session_id"] = statement_meta.get("source_session_id", "") if statement_meta else ""
    enriched["source_bill_debated_id"] = str(statement_meta.get("bill_debated_id", "") or "") if statement_meta else ""
    return enriched


def build_gold_rows(rows: Sequence[dict], statement_meta_map: Dict[int, dict], bill_meta_map: Dict[int, dict]) -> Tuple[List[dict], Counter]:
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    stats: Counter = Counter()
    for row in rows:
        key = (row.get("source_statement_id", ""), row.get("claim_text", ""), row.get("claim_type", ""))
        grouped[key].append(row)

    selected: List[dict] = []
    for group_rows in grouped.values():
        chosen, resolution_method, score = choose_group_row(group_rows, statement_meta_map, bill_meta_map)
        stats[f"resolution.{resolution_method}"] += 1
        if chosen is None:
            stats["skipped.unresolved_ambiguous_group"] += 1
            continue

        statement_meta = statement_meta_map.get(as_int(chosen.get("source_statement_id")))
        selected.append(enrich_row(chosen, statement_meta, resolution_method))
        if score:
            stats["resolution.scored_group_rows"] += len(group_rows)

    return selected, stats


def build_gold_prompt(row: dict) -> str:
    provenance_summary = (
        f"Claim type: {row.get('claim_type', 'unknown')}\n"
        f"Bill: {row.get('bill_number') or 'N/A'}\n"
        f"Bill session: {row.get('session_id') or 'N/A'}\n"
        f"Source statement id: {row.get('source_statement_id') or 'N/A'}\n"
        f"Evidence ids: {row.get('evidence_primary_ids') or 'N/A'}"
    )
    return (
        "Analyze this claim from Canadian Parliament and provide a fact-check:\n\n"
        f"Claim: \"{row.get('claim_text', '')}\"\n"
        f"Speaker: {row.get('source_speaker') or 'Unknown'}\n"
        f"Date: {row.get('source_date') or 'Unknown'}\n"
        f"Context: {row.get('source_context_local') or row.get('source_context') or 'N/A'}\n"
        f"Evidence: {row.get('evidence_text') or 'N/A'}\n"
        f"Provenance:\n{provenance_summary}\n\n"
        f"{TRAINING_PROMPT_INSTRUCTIONS}"
    )


def build_gold_correction(row: dict) -> str:
    if row.get("label") == "True":
        return "N/A"
    return trim_text(row.get("evidence_text", "") or row.get("explanation", ""), 240)


def build_gold_completion(row: dict) -> str:
    return "\n".join(
        [
            f"VERDICT: {str(row.get('label', '')).upper()}",
            "CONFIDENCE: 100",
            f"EXPLANATION: {trim_text(row.get('explanation', ''), 300)}",
            f"CORRECTION: {build_gold_correction(row)}",
        ]
    )


def build_training_pairs(rows: Sequence[dict]) -> List[dict]:
    payload: List[dict] = []
    for row in rows:
        payload.append(
            {
                "prompt": build_gold_prompt(row),
                "completion": build_gold_completion(row),
                "metadata": {
                    "claim_id": row.get("claim_id"),
                    "dataset_tier": "gold",
                    "label_source": "deterministic_db_rule",
                    "label": row.get("label"),
                    "claim_family": row.get("claim_family"),
                    "claim_type": row.get("claim_type"),
                    "bill_id": row.get("bill_id"),
                    "bill_number": row.get("bill_number"),
                    "source_statement_id": row.get("source_statement_id"),
                },
            }
        )
    return payload


def build_stats_rows(rows: Sequence[dict], stats: Counter) -> List[dict]:
    output = [
        {"category": "overall", "name": "total_rows", "value": str(len(rows))},
        {"category": "overall", "name": "true_rows", "value": str(sum(row.get("label") == "True" for row in rows))},
        {"category": "overall", "name": "false_rows", "value": str(sum(row.get("label") == "False" for row in rows))},
    ]

    for category_name, values in [
        ("claim_family", Counter(row.get("claim_family", "") for row in rows)),
        ("claim_type", Counter(row.get("claim_type", "") for row in rows)),
        ("rule_id", Counter(row.get("rule_id", "") for row in rows)),
        ("gold_resolution_method", Counter(row.get("gold_resolution_method", "") for row in rows)),
    ]:
        for name, value in sorted(values.items()):
            output.append({"category": category_name, "name": name, "value": str(value)})

    for key, value in sorted(stats.items()):
        category, _, metric = key.partition(".")
        output.append({"category": category, "name": metric or key, "value": str(value)})

    return output


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log(f"Loading deterministic rows from {input_path}")
    rows, input_columns = load_csv_rows(input_path)
    log(f"Loaded {len(rows)} rows")

    statement_ids = sorted({as_int(row.get("source_statement_id")) for row in rows if as_int(row.get("source_statement_id"))})
    bill_ids = sorted({as_int(row.get("bill_id")) for row in rows if as_int(row.get("bill_id"))})

    log("Loading statement and bill metadata from PostgreSQL")
    with connect_db(args) as conn:
        statement_meta_map = fetch_statement_metadata(conn, statement_ids)
        bill_meta_map = fetch_bill_metadata(conn, bill_ids)

    gold_rows, stats = build_gold_rows(rows, statement_meta_map, bill_meta_map)
    gold_rows.sort(key=lambda row: (row.get("source_date", ""), row.get("claim_id", "")))

    output_dir.mkdir(parents=True, exist_ok=True)
    master_columns = list(input_columns)
    for column in APPENDED_COLUMNS:
        if column not in master_columns:
            master_columns.append(column)

    log(f"Writing {len(gold_rows)} gold rows")
    write_csv(output_dir / MASTER_FILENAME, gold_rows, master_columns)
    write_json(output_dir / TRAINING_FILENAME, build_training_pairs(gold_rows))
    write_csv(output_dir / STATS_FILENAME, build_stats_rows(gold_rows, stats), ["category", "name", "value"])

    log(f"Gold export complete: {MASTER_FILENAME}, {TRAINING_FILENAME}, {STATS_FILENAME}")


if __name__ == "__main__":
    main()
