import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


PARSER_VERSION = "vote_only_v1"
RULE_ID = "hansard_party_vote_sentence_v1"
CLAIM_FAMILY = "vote_fact"
CLAIM_TYPE = "party_vote_direction"
MASTER_COLUMNS = [
    "claim_id",
    "claim_text",
    "source_context",
    "source_sentence",
    "label",
    "explanation",
    "evidence_text",
    "claim_origin",
    "source_table",
    "source_statement_id",
    "source_document_id",
    "source_speaker",
    "source_party",
    "source_date",
    "claim_family",
    "claim_type",
    "claim_subtype",
    "match_status",
    "bill_id",
    "bill_number",
    "bill_title",
    "session_id",
    "parliament_number",
    "session_number",
    "votequestion_id",
    "vote_date",
    "party_id",
    "party_name",
    "politician_id",
    "politician_name",
    "evidence_source_tables",
    "evidence_primary_ids",
    "evidence_json",
    "rule_id",
    "confidence",
    "contradiction_field",
    "parser_version",
    "generated_at",
    "notes",
    "source_context_local",
    "votequestion_description",
    "vote_result",
    "yea_total",
    "nay_total",
    "claim_vote_direction",
    "evidence_vote_direction",
    "verdict_relation",
]
DERIVED_OUTPUTS = {
    "factcheck_claim_label.csv": ["claim_text", "label"],
    "factcheck_claim_context_label.csv": ["claim_text", "source_context", "label"],
    "factcheck_claim_evidence_label.csv": ["claim_text", "evidence_text", "label"],
    "factcheck_claim_evidence_explanation.csv": [
        "claim_text",
        "evidence_text",
        "label",
        "explanation",
    ],
}
NORMALIZE_SPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
VOTE_PHRASE_RE = re.compile(r"\bvoted\s+(for|against)\b", re.IGNORECASE)
BILL_NUMBER_RE = re.compile(r"\bBill\s+([A-Z]-\d+[A-Z0-9-]*)\b", re.IGNORECASE)
SPEAKER_PARTY_RE = re.compile(r"\((?:[^()]*,\s*)?([A-Za-z.]+)\)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ELECTORATE_LANGUAGE_RE = re.compile(
    r"\b("
    r"voters?|election|mandate|people voted|Quebeckers voted|"
    r"Canadians voted|voted for us|voted for them|gave us our mandate"
    r")\b",
    re.IGNORECASE,
)
UNCERTAINTY_LANGUAGE_RE = re.compile(
    r"\b(i think|maybe|might(?:\s+even)?\s+have\s+voted|cannot remember|can't remember|perhaps)\b",
    re.IGNORECASE,
)
VOTE_OBJECT_CUES = ("motion", "amendment", "report", "section")
LOCAL_PARTY_VOTE_GAP = 80
LOCAL_CONTEXT_MAX_CHARS = 1200
TRAINING_PROMPT_INSTRUCTIONS = """Provide your assessment in this exact format:
VERDICT: [TRUE/FALSE]
CONFIDENCE: [0-100]
EXPLANATION: [Short explanation grounded in the evidence]
CORRECTION: [If false, provide the corrected information; otherwise N/A]"""


@dataclass(frozen=True)
class VoteFact:
    partyvote_id: int
    votequestion_id: int
    bill_id: int
    bill_number: str
    bill_title: str
    session_id: Optional[int]
    parliament_number: Optional[int]
    session_number: Optional[int]
    vote_date: str
    votequestion_description: str
    vote_result: str
    yea_total: Optional[int]
    nay_total: Optional[int]
    party_id: int
    party_name: str
    vote: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a natural-first fact-check dataset.")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--database", default=os.getenv("PGDATABASE", "openparliament"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "postgres"))
    parser.add_argument("--password-env", default="PGPASSWORD")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--synthetic-false-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--setup-sql", default="sql/factcheck_work_setup.sql")
    parser.add_argument("--skip-setup-sql", action="store_true")
    parser.add_argument("--statement-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def connect_db(args: argparse.Namespace):
    password = os.getenv(args.password_env)
    if not password:
        raise RuntimeError(f"Environment variable {args.password_env} is not set.")
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=password,
    )


def run_setup_sql(conn, setup_sql_path: Path) -> List[str]:
    if not setup_sql_path.exists():
        return []
    sql_text = setup_sql_path.read_text(encoding="utf-8")
    if not sql_text.strip():
        return []
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()
    return ["factcheck_work", "factcheck_work.bill_vote_fact_view", "factcheck_work.hansard_claim_candidates"]


def log_progress(message: str) -> None:
    print(message, flush=True)


def normalize_spaces(text: str) -> str:
    return NORMALIZE_SPACE_RE.sub(" ", text).strip()


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    return normalize_spaces(text)


def normalize_claim_key(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    if normalized.endswith("."):
        normalized = normalized[:-1].rstrip()
    return normalized


def trim_text(text: str, limit: int) -> str:
    text = normalize_spaces(text)
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 3].rsplit(" ", 1)[0].rstrip(",;: ")
    return (trimmed or text[: limit - 3]) + "..."


def parse_source_party(who_en: Optional[str]) -> str:
    if not who_en:
        return ""
    match = SPEAKER_PARTY_RE.search(who_en.strip())
    return match.group(1).strip() if match else ""


def canonical_party_aliases(party_rows: Sequence[dict]) -> List[Tuple[re.Pattern, int, str]]:
    by_short = {row["short_name_en"].lower(): row for row in party_rows if row["short_name_en"]}
    by_name = {row["name_en"].lower(): row for row in party_rows if row["name_en"]}
    alias_specs = [
        ("conservative", ["conservative party of canada", "conservative", "conservatives", "cpc"]),
        ("liberal", ["liberal party of canada", "liberal", "liberals", "lpc", "lib."]),
        ("ndp", ["new democratic party", "new democrat", "new democrats", "ndp"]),
        ("bloc", ["bloc québécois", "bloc quebecois", "bloc", "bq"]),
        ("green", ["green party of canada", "green party", "greens", "green"]),
    ]
    patterns: List[Tuple[re.Pattern, int, str]] = []
    for short_name, aliases in alias_specs:
        row = by_short.get(short_name) or next(
            (candidate for key, candidate in by_name.items() if short_name in key),
            None,
        )
        if not row:
            continue
        escaped = [re.escape(alias) for alias in aliases]
        pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
        patterns.append((pattern, row["id"], row["name_en"]))
    return patterns


def extract_bill_number(sentence: str) -> Optional[str]:
    match = BILL_NUMBER_RE.search(sentence)
    return match.group(1).upper() if match else None


def detect_vote_direction(sentence: str) -> Optional[str]:
    directions = {match.group(1).lower() for match in VOTE_PHRASE_RE.finditer(sentence)}
    if len(directions) != 1:
        return None
    return next(iter(directions))


def vote_code_to_direction(vote: str) -> str:
    return "for" if vote == "Y" else "against"


def extract_party_vote_claim(
    sentence: str,
    party_patterns: Sequence[Tuple[re.Pattern, int, str]],
) -> Tuple[Optional[Tuple[int, str]], Optional[str], str]:
    direction = detect_vote_direction(sentence)
    if not direction:
        return None, None, "direction_ambiguous"
    if ELECTORATE_LANGUAGE_RE.search(sentence):
        return None, None, "electorate_language"
    if UNCERTAINTY_LANGUAGE_RE.search(sentence):
        return None, None, "uncertain_claim_language"

    vote_match = next(VOTE_PHRASE_RE.finditer(sentence), None)
    if not vote_match:
        return None, None, "direction_ambiguous"

    all_parties = set()
    found = set()
    for pattern, party_id, party_name in party_patterns:
        for party_match in pattern.finditer(sentence):
            all_parties.add((party_id, party_name))
            if party_match.end() > vote_match.start():
                continue
            if vote_match.start() - party_match.end() > LOCAL_PARTY_VOTE_GAP:
                continue
            gap = sentence[party_match.end() : vote_match.start()]
            if any(marker in gap for marker in (".", "!", "?", "\n")):
                continue
            if re.search(r"\bthey\b|\bto vote\b", gap, re.IGNORECASE):
                continue
            found.add((party_id, party_name))

    if len(all_parties) != 1:
        return None, None, "party_ambiguous_or_missing"
    if len(found) != 1:
        return None, None, "party_ambiguous_or_missing"
    return next(iter(found)), direction, ""


def extract_sentence_candidates(text: str) -> List[str]:
    matches = list(VOTE_PHRASE_RE.finditer(text))
    if not matches:
        return []
    sentences: List[str] = []
    seen = set()
    for match in matches:
        start = max(
            text.rfind(".", 0, match.start()),
            text.rfind("?", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        end_positions = [
            pos
            for pos in (
                text.find(".", match.end()),
                text.find("?", match.end()),
                text.find("!", match.end()),
                text.find("\n", match.end()),
            )
            if pos != -1
        ]
        end = min(end_positions) if end_positions else len(text)
        sentence = text[start + 1 : end + 1].strip()
        sentence = normalize_spaces(sentence)
        if sentence and sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    return sentences


def fetch_party_rows(conn) -> List[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, name_en, short_name_en
            FROM public.core_party
            ORDER BY id
            """
        )
        return list(cur.fetchall())


def fetch_vote_facts(conn) -> Dict[Tuple[int, int], List[VoteFact]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                pv.id AS partyvote_id,
                pv.vote,
                pv.party_id,
                cp.name_en AS party_name,
                vq.id AS votequestion_id,
                vq.date AS vote_date,
                COALESCE(vq.description_en, '') AS votequestion_description,
                COALESCE(vq.result, '') AS vote_result,
                vq.yea_total,
                vq.nay_total,
                vq.bill_id,
                b.number AS bill_number,
                COALESCE(NULLIF(b.short_title_en, ''), NULLIF(b.name_en, ''), b.number) AS bill_title,
                b.session_id,
                cs.parliamentnum AS parliament_number,
                cs.sessnum AS session_number
            FROM public.bills_partyvote pv
            JOIN public.bills_votequestion vq ON vq.id = pv.votequestion_id
            JOIN public.bills_bill b ON b.id = vq.bill_id
            LEFT JOIN public.core_session cs ON cs.id = b.session_id
            JOIN public.core_party cp ON cp.id = pv.party_id
            WHERE pv.vote IN ('Y', 'N')
              AND vq.bill_id IS NOT NULL
              AND vq.date IS NOT NULL
              AND b.number IS NOT NULL
            ORDER BY vq.bill_id, pv.party_id, vq.date, vq.id
            """
        )
        facts: Dict[Tuple[int, int], List[VoteFact]] = defaultdict(list)
        for row in cur.fetchall():
            fact = VoteFact(
                partyvote_id=row["partyvote_id"],
                votequestion_id=row["votequestion_id"],
                bill_id=row["bill_id"],
                bill_number=row["bill_number"],
                bill_title=row["bill_title"] or row["bill_number"],
                session_id=row["session_id"],
                parliament_number=row["parliament_number"],
                session_number=row["session_number"],
                vote_date=row["vote_date"].isoformat(),
                votequestion_description=clean_text(row["votequestion_description"]),
                vote_result=row["vote_result"] or "",
                yea_total=row["yea_total"],
                nay_total=row["nay_total"],
                party_id=row["party_id"],
                party_name=row["party_name"],
                vote=row["vote"],
            )
            facts[(fact.bill_id, fact.party_id)].append(fact)
        return facts


def fetch_hansard_candidates(conn) -> List[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                s.id AS statement_id,
                s.document_id,
                d.date AS source_date,
                s.who_en,
                regexp_replace(s.content_en, '<[^>]+>', ' ', 'g') AS content_text,
                s.bill_debated_id AS bill_id,
                b.number AS bill_number,
                COALESCE(NULLIF(b.short_title_en, ''), NULLIF(b.name_en, ''), b.number) AS bill_title,
                b.session_id,
                cs.parliamentnum AS parliament_number,
                cs.sessnum AS session_number
            FROM public.hansards_statement s
            JOIN public.hansards_document d ON d.id = s.document_id
            JOIN public.bills_bill b ON b.id = s.bill_debated_id
            LEFT JOIN public.core_session cs ON cs.id = b.session_id
            WHERE s.procedural = false
              AND s.bill_debated_id IS NOT NULL
              AND s.content_en IS NOT NULL
              AND (
                    s.content_en ILIKE '%voted for%'
                 OR s.content_en ILIKE '%voted against%'
              )
            ORDER BY d.date, s.id
            """
        )
        return list(cur.fetchall())


def resolve_vote_fact(
    facts_by_key: Dict[Tuple[int, int], List[VoteFact]],
    bill_id: int,
    party_id: int,
    source_date: Optional[str],
) -> Tuple[Optional[VoteFact], str]:
    facts = facts_by_key.get((bill_id, party_id), [])
    if not facts:
        return None, "no_vote_fact"
    if len(facts) == 1:
        return facts[0], ""
    if source_date:
        exact = [fact for fact in facts if fact.vote_date == source_date]
        if len(exact) == 1:
            return exact[0], ""
        if len(exact) > 1:
            return None, "ambiguous_vote_fact"
    return None, "multiple_vote_events"


def is_vote_object_grounded(sentence: str, fact: VoteFact) -> Tuple[bool, str]:
    description = normalize_spaces(fact.votequestion_description).lower()
    for cue in VOTE_OBJECT_CUES:
        if re.search(rf"\b{cue}\b", sentence, re.IGNORECASE) and not re.search(rf"\b{cue}\b", description):
            return False, cue
    return True, ""


def short_votequestion_description(fact: VoteFact) -> str:
    description = clean_text(fact.votequestion_description)
    fallback = f"Vote on Bill {fact.bill_number}"
    return trim_text(description or fallback, 220)


def build_local_context(source_context: str, claim_text: str) -> str:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(source_context) if part.strip()]
    for index, part in enumerate(parts):
        if claim_text in part:
            local = " ".join(parts[max(0, index - 1) : min(len(parts), index + 2)])
            return trim_text(local, LOCAL_CONTEXT_MAX_CHARS)

    position = source_context.find(claim_text)
    if position == -1:
        return trim_text(claim_text, LOCAL_CONTEXT_MAX_CHARS)
    start = max(0, position - 400)
    end = min(len(source_context), position + len(claim_text) + 400)
    return trim_text(source_context[start:end], LOCAL_CONTEXT_MAX_CHARS)


def build_evidence_text(fact: VoteFact) -> str:
    direction = vote_code_to_direction(fact.vote)
    question = short_votequestion_description(fact)
    return (
        f'House vote question: "{question}" The {fact.party_name} voted {direction} on {fact.vote_date}.'
    )


def build_explanation(label: str, fact: VoteFact) -> str:
    direction = vote_code_to_direction(fact.vote)
    if label == "True":
        return f"True. DB vote record shows the {fact.party_name} voted {direction} on that House vote question on {fact.vote_date}."
    opposite = "against" if fact.vote == "Y" else "for"
    return (
        f"False. DB vote record shows the {fact.party_name} voted {direction}, not {opposite}, on that House vote question on {fact.vote_date}."
    )


def build_correction(label: str, fact: VoteFact) -> str:
    if label == "True":
        return "N/A"
    return (
        f'The {fact.party_name} voted {vote_code_to_direction(fact.vote)} on the House vote question '
        f'"{short_votequestion_description(fact)}" on {fact.vote_date}.'
    )


def build_row(
    *,
    claim_id: str,
    claim_text: str,
    source_context: str,
    source_sentence: str,
    label: str,
    claim_origin: str,
    source_statement_id: Optional[int],
    source_document_id: Optional[int],
    source_speaker: str,
    source_party: str,
    source_date: Optional[str],
    claim_vote_direction: str,
    fact: VoteFact,
    contradiction_field: str = "",
    notes: str = "",
) -> dict:
    evidence_vote_direction = vote_code_to_direction(fact.vote)
    evidence_source_tables = json.dumps(
        ["public.bills_partyvote", "public.bills_votequestion", "public.bills_bill", "public.core_party"],
        ensure_ascii=True,
    )
    evidence_primary_ids = json.dumps(
        {
            "partyvote_id": fact.partyvote_id,
            "votequestion_id": fact.votequestion_id,
            "bill_id": fact.bill_id,
            "party_id": fact.party_id,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    evidence_json = json.dumps(
        {
            "bill_id": fact.bill_id,
            "bill_number": fact.bill_number,
            "nay_total": fact.nay_total,
            "party_id": fact.party_id,
            "party_name": fact.party_name,
            "vote_result": fact.vote_result,
            "votequestion_description": fact.votequestion_description,
            "vote": fact.vote,
            "vote_date": fact.vote_date,
            "votequestion_id": fact.votequestion_id,
            "partyvote_id": fact.partyvote_id,
            "yea_total": fact.yea_total,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "source_context": source_context,
        "source_sentence": source_sentence,
        "label": label,
        "explanation": build_explanation(label, fact),
        "evidence_text": build_evidence_text(fact),
        "claim_origin": claim_origin,
        "source_table": "public.hansards_statement" if claim_origin != "synthetic_fallback" else "synthetic_from_public.hansards_statement",
        "source_statement_id": source_statement_id or "",
        "source_document_id": source_document_id or "",
        "source_speaker": source_speaker,
        "source_party": source_party,
        "source_date": source_date or "",
        "claim_family": CLAIM_FAMILY,
        "claim_type": CLAIM_TYPE,
        "claim_subtype": "explicit_party_vote_direction",
        "match_status": "matched" if label == "True" else "contradicted",
        "bill_id": fact.bill_id,
        "bill_number": fact.bill_number,
        "bill_title": fact.bill_title,
        "session_id": fact.session_id or "",
        "parliament_number": fact.parliament_number or "",
        "session_number": fact.session_number or "",
        "votequestion_id": fact.votequestion_id,
        "vote_date": fact.vote_date,
        "party_id": fact.party_id,
        "party_name": fact.party_name,
        "politician_id": "",
        "politician_name": "",
        "evidence_source_tables": evidence_source_tables,
        "evidence_primary_ids": evidence_primary_ids,
        "evidence_json": evidence_json,
        "rule_id": RULE_ID,
        "confidence": "1.0",
        "contradiction_field": contradiction_field,
        "parser_version": PARSER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "source_context_local": build_local_context(source_context, source_sentence),
        "votequestion_description": fact.votequestion_description,
        "vote_result": fact.vote_result,
        "yea_total": fact.yea_total if fact.yea_total is not None else "",
        "nay_total": fact.nay_total if fact.nay_total is not None else "",
        "claim_vote_direction": claim_vote_direction,
        "evidence_vote_direction": evidence_vote_direction,
        "verdict_relation": "SUPPORTED" if label == "True" else "CONTRADICTED",
    }


def build_natural_rows(
    statements: Sequence[dict],
    facts_by_key: Dict[Tuple[int, int], List[VoteFact]],
    party_patterns: Sequence[Tuple[re.Pattern, int, str]],
    stats: Counter,
    progress_every: int = 0,
) -> List[dict]:
    rows: List[dict] = []
    counter = 1
    total = len(statements)
    for index, statement in enumerate(statements, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            log_progress(f"Processing statement {index}/{total}")
        source_context = clean_text(statement["content_text"])
        if not source_context:
            stats["skipped.empty_source_context"] += 1
            continue
        source_date = statement["source_date"].isoformat() if statement["source_date"] else ""
        source_party = parse_source_party(statement["who_en"])
        statement_rows = 0
        for sentence in extract_sentence_candidates(source_context):
            stats["candidate.sentences"] += 1
            explicit_bill = extract_bill_number(sentence)
            if explicit_bill and explicit_bill != (statement["bill_number"] or "").upper():
                stats["skipped.bill_number_mismatch"] += 1
                continue
            party, claim_vote_direction, skip_reason = extract_party_vote_claim(sentence, party_patterns)
            if not party:
                stats[f"skipped.{skip_reason}"] += 1
                continue
            fact, skip_reason = resolve_vote_fact(
                facts_by_key,
                statement["bill_id"],
                party[0],
                source_date,
            )
            if not fact:
                stats[f"skipped.{skip_reason}"] += 1
                continue
            grounded, cue = is_vote_object_grounded(sentence, fact)
            if not grounded:
                stats[f"skipped.vote_object_mismatch_{cue}"] += 1
                continue
            label = "True" if vote_code_to_direction(fact.vote) == claim_vote_direction else "False"
            row = build_row(
                claim_id=f"claim-{counter}",
                claim_text=sentence,
                source_context=source_context,
                source_sentence=sentence,
                label=label,
                claim_origin="hansard_natural",
                source_statement_id=statement["statement_id"],
                source_document_id=statement["document_id"],
                source_speaker=statement["who_en"] or "",
                source_party=source_party,
                source_date=source_date,
                claim_vote_direction=claim_vote_direction,
                fact=fact,
                contradiction_field="vote_direction" if label == "False" else "",
            )
            rows.append(row)
            counter += 1
            statement_rows += 1
        if not statement_rows:
            stats["unresolved.statements"] += 1
    return rows


def build_synthetic_false_rows(true_rows: Sequence[dict], stats: Counter, ratio: float, seed: int) -> List[dict]:
    natural_false = sum(1 for row in true_rows if row["label"] == "False")
    natural_true = sum(1 for row in true_rows if row["label"] == "True")
    target_false = min(natural_true, int(natural_true * ratio))
    needed = max(0, target_false - natural_false)
    if needed <= 0:
        return []

    candidates = [row for row in true_rows if row["label"] == "True" and row["claim_origin"] == "hansard_natural"]
    random.Random(seed).shuffle(candidates)
    synthetic_rows: List[dict] = []
    for candidate in candidates:
        if len(synthetic_rows) >= needed:
            break
        claim_text = candidate["claim_text"]
        if re.search(r"\bvoted for\b", claim_text, flags=re.IGNORECASE):
            mutated = re.sub(r"\bvoted for\b", "voted against", claim_text, count=1, flags=re.IGNORECASE)
        elif re.search(r"\bvoted against\b", claim_text, flags=re.IGNORECASE):
            mutated = re.sub(r"\bvoted against\b", "voted for", claim_text, count=1, flags=re.IGNORECASE)
        else:
            stats["skipped.synthetic_no_flip_phrase"] += 1
            continue
        if normalize_claim_key(mutated) == normalize_claim_key(claim_text):
            stats["skipped.synthetic_identity"] += 1
            continue
        synthetic = dict(candidate)
        synthetic["claim_id"] = f"claim-synth-{len(synthetic_rows) + 1}"
        synthetic["claim_text"] = mutated
        synthetic["label"] = "False"
        synthetic["claim_origin"] = "synthetic_fallback"
        synthetic["match_status"] = "contradicted"
        synthetic["explanation"] = candidate["explanation"].replace("True.", "False.", 1)
        synthetic["contradiction_field"] = "vote_direction"
        synthetic["claim_vote_direction"] = "against" if candidate["claim_vote_direction"] == "for" else "for"
        synthetic["verdict_relation"] = "CONTRADICTED"
        synthetic["notes"] = "synthetic_corruption_from_natural_sentence"
        synthetic["generated_at"] = datetime.now(timezone.utc).isoformat()
        synthetic["source_table"] = "synthetic_from_public.hansards_statement"
        synthetic_rows.append(synthetic)
    stats["generated.synthetic_false_rows"] += len(synthetic_rows)
    return synthetic_rows


def deduplicate_rows(rows: Sequence[dict], stats: Counter) -> List[dict]:
    kept: List[dict] = []
    labels_by_claim: Dict[str, str] = {}
    seen_pairs = set()
    for row in rows:
        claim_key = normalize_claim_key(row["claim_text"])
        pair_key = (claim_key, normalize_claim_key(row["evidence_text"]))
        existing_label = labels_by_claim.get(claim_key)
        if existing_label and existing_label != row["label"]:
            stats["skipped.conflicting_claim_label"] += 1
            continue
        if pair_key in seen_pairs:
            stats["skipped.duplicate_claim_evidence"] += 1
            continue
        labels_by_claim[claim_key] = row["label"]
        seen_pairs.add(pair_key)
        kept.append(row)
    return kept


def validate_rows(rows: Sequence[dict]) -> None:
    seen_claim_labels: Dict[str, str] = {}
    seen_pairs = set()
    for row in rows:
        if not row["claim_text"]:
            raise RuntimeError("Empty claim_text detected.")
        if row["label"] not in {"True", "False"}:
            raise RuntimeError(f"Invalid label: {row['label']}")
        for required in ("explanation", "evidence_text", "evidence_source_tables", "evidence_primary_ids"):
            if not row[required]:
                raise RuntimeError(f"Missing required field {required} in row {row['claim_id']}")
        for required in ("source_context_local", "claim_vote_direction", "evidence_vote_direction", "verdict_relation"):
            if not row[required]:
                raise RuntimeError(f"Missing required field {required} in row {row['claim_id']}")
        if row["label"] == "False" and not row["contradiction_field"]:
            raise RuntimeError(f"False row missing contradiction_field: {row['claim_id']}")
        if row["claim_origin"] == "hansard_natural" and row["claim_text"] != row["source_sentence"]:
            raise RuntimeError(f"Natural row claim_text mismatch: {row['claim_id']}")
        if row["label"] == "True" and row["verdict_relation"] != "SUPPORTED":
            raise RuntimeError(f"True row has wrong verdict_relation: {row['claim_id']}")
        if row["label"] == "False" and row["verdict_relation"] != "CONTRADICTED":
            raise RuntimeError(f"False row has wrong verdict_relation: {row['claim_id']}")
        claim_key = normalize_claim_key(row["claim_text"])
        existing = seen_claim_labels.get(claim_key)
        if existing and existing != row["label"]:
            raise RuntimeError(f"Conflicting labels survived for claim {row['claim_text']}")
        seen_claim_labels[claim_key] = row["label"]
        pair_key = (claim_key, normalize_claim_key(row["evidence_text"]))
        if pair_key in seen_pairs:
            raise RuntimeError(f"Duplicate claim/evidence pair survived: {row['claim_id']}")
        seen_pairs.add(pair_key)


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_training_prompt(row: dict) -> str:
    return f"""Analyze this claim from Canadian Parliament and provide a fact-check:

Claim: "{row['claim_text']}"
Speaker: {row['source_speaker'] or 'Unknown'}
Date: {row['source_date'] or 'Unknown'}
Bill: {row['bill_number']}
Named party: {row['party_name']}
Vote date: {row['vote_date']}

Context: {row['source_context_local']}

Evidence: {row['evidence_text']}

{TRAINING_PROMPT_INSTRUCTIONS}"""


def build_row_correction(row: dict) -> str:
    if row["label"] == "True":
        return "N/A"
    question = trim_text(row["votequestion_description"] or f'Vote on Bill {row["bill_number"]}', 220)
    return (
        f'The {row["party_name"]} voted {row["evidence_vote_direction"]} on the House vote question '
        f'"{question}" '
        f'on {row["vote_date"]}.'
    )


def build_training_completion(row: dict) -> str:
    verdict = row["label"].upper()
    return "\n".join(
        [
            f"VERDICT: {verdict}",
            "CONFIDENCE: 100",
            f"EXPLANATION: {row['explanation']}",
            f"CORRECTION: {build_row_correction(row)}",
        ]
    )


def build_training_pairs(rows: Sequence[dict]) -> List[dict]:
    pairs: List[dict] = []
    for row in rows:
        pairs.append(
            {
                "prompt": build_training_prompt(row),
                "completion": build_training_completion(row),
                "metadata": {
                    "claim_id": row["claim_id"],
                    "label": row["label"],
                    "verdict_relation": row["verdict_relation"],
                    "claim_type": row["claim_type"],
                    "bill_number": row["bill_number"],
                    "votequestion_id": row["votequestion_id"],
                    "party_name": row["party_name"],
                },
            }
        )
    return pairs


def build_stats_rows(rows: Sequence[dict], stats: Counter) -> List[dict]:
    output: List[dict] = []
    output.append({"category": "overall", "name": "total_rows", "value": str(len(rows))})
    output.append({"category": "overall", "name": "true_rows", "value": str(sum(row["label"] == "True" for row in rows))})
    output.append({"category": "overall", "name": "false_rows", "value": str(sum(row["label"] == "False" for row in rows))})

    claim_type_counts = Counter(row["claim_type"] for row in rows)
    claim_origin_counts = Counter(row["claim_origin"] for row in rows)
    rule_counts = Counter(row["rule_id"] for row in rows)

    for name, value in sorted(claim_type_counts.items()):
        output.append({"category": "claim_type", "name": name, "value": str(value)})
    for name, value in sorted(claim_origin_counts.items()):
        output.append({"category": "claim_origin", "name": name, "value": str(value)})
    for name, value in sorted(rule_counts.items()):
        output.append({"category": "rule_id", "name": name, "value": str(value)})
    for name, value in sorted(stats.items()):
        category, _, metric = name.partition(".")
        output.append({"category": category, "name": metric or name, "value": str(value)})
    return output


def export_outputs(output_dir: Path, rows: Sequence[dict], stats: Counter) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "factcheck_master.csv", rows, MASTER_COLUMNS)
    for filename, columns in DERIVED_OUTPUTS.items():
        write_csv(output_dir / filename, rows, columns)
    write_csv(output_dir / "factcheck_stats.csv", build_stats_rows(rows, stats), ["category", "name", "value"])
    write_json(output_dir / "training_pairs.json", build_training_pairs(rows))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    setup_sql_path = Path(args.setup_sql)
    stats: Counter = Counter()

    log_progress(f"Connecting to PostgreSQL database '{args.database}' on {args.host}:{args.port}")
    with connect_db(args) as conn:
        if args.skip_setup_sql:
            created_objects = []
            log_progress("Skipping helper SQL setup.")
        else:
            log_progress(f"Running helper SQL setup from {setup_sql_path}")
            created_objects = run_setup_sql(conn, setup_sql_path)
            log_progress("Helper SQL setup complete.")
        log_progress("Loading party aliases.")
        party_patterns = canonical_party_aliases(fetch_party_rows(conn))
        log_progress("Loading vote facts.")
        facts_by_key = fetch_vote_facts(conn)
        log_progress("Loading Hansard candidate statements.")
        statements = fetch_hansard_candidates(conn)

    if args.statement_limit is not None:
        statements = statements[: args.statement_limit]
        log_progress(f"Applying statement limit: {len(statements)}")

    stats["candidate.statements"] = len(statements)
    log_progress(f"Building rows from {len(statements)} candidate statements.")
    rows = build_natural_rows(
        statements,
        facts_by_key,
        party_patterns,
        stats,
        progress_every=args.progress_every,
    )
    log_progress(f"Built {len(rows)} natural rows.")
    synthetic_rows = build_synthetic_false_rows(rows, stats, args.synthetic_false_ratio, args.seed)
    combined = rows + synthetic_rows
    log_progress(f"Built {len(synthetic_rows)} synthetic fallback rows.")
    deduped = deduplicate_rows(combined, stats)
    log_progress(f"Kept {len(deduped)} rows after deduplication.")
    validate_rows(deduped)
    log_progress(f"Writing CSV outputs to {output_dir}")
    export_outputs(output_dir, deduped, stats)

    print(f"Created helper objects: {', '.join(created_objects) if created_objects else 'none'}")
    print(f"Natural rows kept: {sum(row['claim_origin'] == 'hansard_natural' for row in deduped)}")
    print(f"Synthetic fallback rows kept: {sum(row['claim_origin'] == 'synthetic_fallback' for row in deduped)}")
    print(f"Final rows: {len(deduped)}")
    print(f"True rows: {sum(row['label'] == 'True' for row in deduped)}")
    print(f"False rows: {sum(row['label'] == 'False' for row in deduped)}")


if __name__ == "__main__":
    main()
