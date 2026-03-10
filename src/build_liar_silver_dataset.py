#!/usr/bin/env python3
"""
Build a separate LIAR-style silver dataset using broader natural claims and OpenAI labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_INPUTS = [Path("data/processed_claims.json"), Path("data/extracted_claims.json")]
DEFAULT_EXISTING_SILVER = Path("data/liar_silver_master.csv")
MASTER_FILENAME = "liar_silver_master.csv"
TRAINING_FILENAME = "liar_silver_training_pairs.json"
STATS_FILENAME = "liar_silver_stats.csv"
ALLOWED_LABELS = {
    "true",
    "mostly-true",
    "half-true",
    "barely-true",
    "false",
    "pants-fire",
}
OFFICIAL_DOMAIN_SUFFIXES = (
    ".gc.ca",
    "canada.ca",
    "parl.ca",
    "ourcommons.ca",
    "sen.parl.gc.ca",
    "elections.ca",
    "pbo-dpb.ca",
)

FACT_PATTERNS = [
    re.compile(r"\b(statistics? show|data indicates?|according to|research finds?)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(percent|per cent|billion|million|thousand)\b", re.IGNORECASE),
    re.compile(r"\b(unemployment|inflation|gdp|economy|growth|deficit|debt)\b", re.IGNORECASE),
    re.compile(r"\b(bill|legislation|act|law)\b", re.IGNORECASE),
    re.compile(r"\b(voted|vote|supported|opposed|introduced|passed|defeated)\b", re.IGNORECASE),
    re.compile(r"\b(because of|caused by|thanks to|as a result of|record|highest|lowest|the only|the first|one of|among)\b", re.IGNORECASE),
]
LOW_SIGNAL_PATTERNS = [
    re.compile(r"^\s*(what|why|how)\b", re.IGNORECASE),
    re.compile(r"\b(i think|i believe|we believe|i feel|i hope)\b", re.IGNORECASE),
    re.compile(r"\?$"),
]
MIDDLE_LABEL_CUE_PATTERNS = [
    re.compile(r"\b(about|around|almost|nearly|roughly)\b", re.IGNORECASE),
    re.compile(r"\b(more than|less than|over|under)\b", re.IGNORECASE),
    re.compile(r"\b(record|highest|lowest|always|never|all|none|every)\b", re.IGNORECASE),
    re.compile(r"\b(one of|among|the only|the first)\b", re.IGNORECASE),
    re.compile(r"\b(because of|caused by|thanks to|as a result of)\b", re.IGNORECASE),
]
BINARY_CLEAN_PATTERNS = [
    re.compile(r"\bvoted?\s+(for|against)\s+bill\s+[cs]-\d+\b", re.IGNORECASE),
    re.compile(r"\bbill\s+[cs]-\d+\s+(was|is)\s+(introduced|passed|defeated)\b", re.IGNORECASE),
]
LABEL_PRIORITY_VALUES = {"middle_first", "all_labels", "middle_only"}
MIDDLE_LABELS = {"mostly-true", "half-true", "barely-true", "pants-fire"}
FALLBACK_LABELS = {"true", "false"}

TRAINING_PROMPT_INSTRUCTIONS = (
    "Provide your answer in this exact format:\n"
    "VERDICT: [TRUE/MOSTLY-TRUE/HALF-TRUE/BARELY-TRUE/FALSE/PANTS-FIRE]\n"
    "CONFIDENCE: [0-100]\n"
    "EXPLANATION: [Short evidence-grounded explanation]\n"
    "CORRECTION: [Short correction or N/A]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LIAR-style silver dataset with OpenAI labels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-claims", type=int, default=900, help="Maximum number of candidate claims to prepare.")
    parser.add_argument("--min-confidence", type=int, default=60)
    parser.add_argument("--statement-limit", type=int, default=20000, help="DB statement limit if augmentation is needed.")
    parser.add_argument("--start-date", default="2024-01-01", help="Earliest debate date for DB augmentation.")
    parser.add_argument("--target-new-rows", type=int, default=150, help="Target number of newly accepted silver rows.")
    parser.add_argument("--max-api-calls", type=int, default=360, help="Maximum number of OpenAI labeling calls.")
    parser.add_argument(
        "--label-priority",
        choices=sorted(LABEL_PRIORITY_VALUES),
        default="middle_first",
        help="How to prioritize middle labels versus true/false.",
    )
    parser.add_argument(
        "--exclude-existing-silver",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip claims already present in the current silver master.",
    )
    parser.add_argument("--existing-silver-path", default=str(DEFAULT_EXISTING_SILVER))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="openparliament")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or "")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "gpt-4.1-mini")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def load_dotenv_fallback(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")


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
            return trim_text(" ".join(sentences[max(0, pos - 1) : min(len(sentences), pos + 2)]), max_chars)
    return trim_text(source_context, max_chars)


def classify_claim_type(text: str) -> str:
    text_lower = (text or "").lower()
    if any(word in text_lower for word in ["economy", "gdp", "inflation", "unemployment", "deficit"]):
        return "economic"
    if any(word in text_lower for word in ["climate", "environment", "carbon", "emission"]):
        return "environmental"
    if any(word in text_lower for word in ["health", "hospital", "medicare", "vaccine"]):
        return "health"
    if any(word in text_lower for word in ["bill", "legislation", "act", "law"]):
        return "legislative"
    if any(word in text_lower for word in ["vote", "voted", "supported", "opposed"]):
        return "vote"
    return "general"


def claim_is_usable(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 30 or len(text) > 420:
        return False
    if any(pattern.search(text) for pattern in LOW_SIGNAL_PATTERNS):
        return False
    return any(pattern.search(text) for pattern in FACT_PATTERNS)


def middle_label_signal_score(text: str) -> int:
    text = text or ""
    score = 0
    for pattern in MIDDLE_LABEL_CUE_PATTERNS:
        if pattern.search(text):
            score += 3
    for pattern in BINARY_CLEAN_PATTERNS:
        if pattern.search(text):
            score -= 5
    if re.search(r"\b\d+(?:\.\d+)?\s*(percent|per cent|million|billion|thousand)\b", text, re.IGNORECASE):
        score += 2
    if re.search(r"\b(canadians|families|workers|businesses|people)\b", text, re.IGNORECASE):
        score += 1
    return score


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]


def extract_candidate_claims(text: str, speaker: str, context: dict) -> List[dict]:
    clean_text = re.sub(r"<[^>]+>", "", text or "").strip()
    claims: List[dict] = []
    for sentence in split_sentences(clean_text):
        if not claim_is_usable(sentence):
            continue
        claims.append(
            {
                "claim_text": sentence,
                "full_context": clean_text,
                "speaker": speaker,
                "context": context,
                "claim_type": classify_claim_type(sentence),
            }
        )
    return claims


def connect_db(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        cursor_factory=RealDictCursor,
    )


def fetch_recent_statements(conn, start_date: str, limit: int) -> List[dict]:
    query = """
        SELECT
            s.id,
            s.document_id,
            d.date AS debate_date,
            d.session_id,
            s.who_en AS speaker_name,
            s.h1_en AS topic_h1,
            s.h2_en AS topic_h2,
            s.h3_en AS topic_h3,
            s.content_en AS content,
            s.procedural
        FROM hansards_statement s
        JOIN hansards_document d ON d.id = s.document_id
        WHERE d.date >= %s
          AND s.content_en IS NOT NULL
          AND s.content_en != ''
          AND COALESCE(s.procedural, FALSE) = FALSE
          AND s.who_en IS NOT NULL
          AND s.who_en != ''
        ORDER BY d.date DESC, s.sequence
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (start_date, limit))
        return [dict(row) for row in cur.fetchall()]


def load_seed_claims() -> List[dict]:
    claims: List[dict] = []
    for path in DEFAULT_INPUTS:
        if path.exists():
            claims.extend(json.loads(path.read_text(encoding="utf-8")))
    return claims


def load_existing_silver_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def existing_claim_keys(rows: Sequence[dict]) -> Set[str]:
    return {normalize_text(row.get("claim_text", "")) for row in rows if normalize_text(row.get("claim_text", ""))}


def dedupe_candidates(candidates: Sequence[dict], excluded_claims: Optional[Set[str]] = None) -> List[dict]:
    seen = set()
    output: List[dict] = []
    excluded_claims = excluded_claims or set()
    for candidate in candidates:
        key = normalize_text(candidate.get("claim_text", ""))
        if not key or key in seen or key in excluded_claims:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def select_diverse_candidates(candidates: Sequence[dict], max_claims: int) -> List[dict]:
    buckets: Dict[str, List[dict]] = {}
    for candidate in candidates:
        claim_type = candidate.get("claim_type") or "general"
        buckets.setdefault(claim_type, []).append(candidate)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda item: (
                item.get("candidate_score", 0),
                len(item.get("claim_text", "")),
            ),
            reverse=True,
        )

    selected: List[dict] = []
    claim_types = sorted(buckets)
    while len(selected) < max_claims and claim_types:
        next_round: List[str] = []
        for claim_type in claim_types:
            bucket = buckets[claim_type]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= max_claims:
                    break
            if bucket:
                next_round.append(claim_type)
        claim_types = next_round
    return selected


def score_candidate(candidate: dict) -> int:
    text = candidate.get("claim_text", "")
    claim_type = candidate.get("claim_type", "general")
    score = middle_label_signal_score(text)
    if claim_type in {"economic", "general", "environmental", "health"}:
        score += 2
    if claim_type == "vote":
        score -= 2
    if claim_type == "legislative":
        score += 1
    if re.search(r"\bthis week|today|yesterday|currently|now\b", text, re.IGNORECASE):
        score += 1
    return score


def build_candidate_pool(args: argparse.Namespace, excluded_claims: Optional[Set[str]] = None) -> List[dict]:
    candidates = [claim for claim in load_seed_claims() if claim_is_usable(claim.get("claim_text", ""))]
    for candidate in candidates:
        candidate["candidate_score"] = score_candidate(candidate)
    candidates = dedupe_candidates(candidates, excluded_claims)
    if len(candidates) >= args.max_claims:
        return select_diverse_candidates(candidates, args.max_claims)

    log(f"Seed claim files only provide {len(candidates)} usable claims. Augmenting from DB.")
    with connect_db(args) as conn:
        statements = fetch_recent_statements(conn, args.start_date, args.statement_limit)

    for statement in statements:
        context = {
            "date": str(statement.get("debate_date")),
            "session": statement.get("session_id"),
            "topic_h1": statement.get("topic_h1") or "",
            "topic_h2": statement.get("topic_h2") or "",
            "topic_h3": statement.get("topic_h3") or "",
            "document_id": statement.get("document_id"),
            "statement_id": statement.get("id"),
        }
        candidates.extend(
            extract_candidate_claims(
                statement.get("content", ""),
                statement.get("speaker_name", ""),
                context,
            )
        )
        if len(candidates) >= args.max_claims * 4:
            break

    for candidate in candidates:
        candidate["candidate_score"] = score_candidate(candidate)
    candidates = dedupe_candidates(candidates, excluded_claims)
    return select_diverse_candidates(candidates, args.max_claims)


def extract_output_text(payload: dict) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    parts: List[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text_value = content.get("text")
            if text_value:
                parts.append(text_value)
    return "\n".join(parts).strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def official_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    host = host.split("@")[-1]
    return any(host == suffix or host.endswith(f".{suffix}") or host.endswith(suffix) for suffix in OFFICIAL_DOMAIN_SUFFIXES)


def parse_model_json(text: str) -> Optional[dict]:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def build_openai_prompt(candidate: dict) -> str:
    context = candidate.get("context", {})
    source_summary = (
        f"Speaker: {candidate.get('speaker') or 'Unknown'}\n"
        f"Date: {context.get('date') or 'Unknown'}\n"
        f"Session: {context.get('session') or 'Unknown'}\n"
        f"Topic: {context.get('topic_h2') or context.get('topic_h1') or 'Unknown'}"
    )
    return f"""
You are building a SILVER fact-check dataset for Canadian parliamentary claims.
Use the Hansard context below and, if needed, official or primary Canadian public sources.
Do not treat this as deterministic truth. If evidence is too weak or too ambiguous, reject the row.

Return JSON only with this schema:
{{
  "accept": true,
  "label": "true|mostly-true|half-true|barely-true|false|pants-fire",
  "confidence": 0,
  "not_binary_reason": "why this is not simply true or false",
  "rationale": "short rationale grounded in evidence",
  "evidence_summary": "short evidence summary",
  "used_sources": [
    {{"title": "source title", "url": "https://...", "source_type": "official|parliamentary|primary", "why_relevant": "short note"}}
  ]
}}

Reject rows that are mainly rhetorical, too ambiguous, or unsupported. In that case return:
{{"accept": false, "reason": "brief reason"}}

Label guidance:
- true: fully supported in all material respects
- mostly-true: directionally correct, but a minor number, scope, or context detail is off
- half-true: meaningful mix of supported and contradicted material elements
- barely-true: a small fragment is true, but the overall claim is materially misleading
- false: core claim is contradicted
- pants-fire: false and egregiously misleading or wildly exaggerated

Prefer non-binary labels when the claim is directionally right but numerically off, missing important context, overstated, compressed, or only partly supported.
Do not default to mostly-true. Use mostly-true only when the error is genuinely minor.
If the claim contains both important support and important contradiction, prefer half-true.
If only a narrow fragment is correct but the overall impression is misleading, prefer barely-true.
For sweeping causal, absolute, or comparative claims, prefer half-true or barely-true over mostly-true when evidence is mixed.
If you choose mostly-true, half-true, barely-true, or pants-fire, `not_binary_reason` is required.

Claim:
{candidate.get("claim_text", "")}

Hansard local context:
{candidate.get("source_context_local", "")}

Hansard full context:
{candidate.get("full_context", "")}

Source metadata:
{source_summary}
""".strip()


def call_openai_label(api_key: str, model: str, candidate: dict) -> Tuple[Optional[dict], Optional[str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": build_openai_prompt(candidate),
        "tools": [
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate", "country": "CA", "city": "Ottawa"},
                "search_context_size": "medium",
            }
        ],
        "max_output_tokens": 900,
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
    parsed = parse_model_json(extract_output_text(body))
    return parsed, body.get("model")


def build_evidence_text(candidate: dict, label_payload: dict) -> str:
    base = label_payload.get("evidence_summary") or ""
    local = candidate.get("source_context_local") or ""
    if base and local:
        return trim_text(f"Hansard context: {local} Official evidence: {base}", 900)
    return trim_text(base or local, 900)


def build_provenance(candidate: dict, label_payload: dict) -> List[dict]:
    provenance = [
        {
            "title": "Hansard statement",
            "url": "",
            "source_type": "parliamentary",
            "why_relevant": f"Statement {candidate.get('context', {}).get('statement_id', '')} provides the natural claim context.",
        }
    ]
    for item in label_payload.get("used_sources", []) or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if url and not official_url(url):
            continue
        provenance.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "source_type": item.get("source_type", "official"),
                "why_relevant": item.get("why_relevant", ""),
            }
        )
    return provenance


def normalize_confidence(value) -> Optional[int]:
    try:
        numeric = float(value)
        if 0 <= numeric <= 1:
            numeric *= 100
        return max(0, min(100, int(round(numeric))))
    except (TypeError, ValueError):
        return None


def candidate_claim_id(candidate: dict) -> str:
    base = f"{candidate.get('claim_text','')}|{candidate.get('context', {}).get('statement_id','')}|{candidate.get('context', {}).get('date','')}"
    return "silver_" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def label_candidate(api_key: str, args: argparse.Namespace, candidate: dict) -> Optional[dict]:
    payload, model_name = call_openai_label(api_key, args.model, candidate)
    if not payload or not payload.get("accept"):
        return None

    label = str(payload.get("label", "")).strip().lower()
    confidence = normalize_confidence(payload.get("confidence"))
    not_binary_reason = trim_text(payload.get("not_binary_reason", ""), 220)
    rationale = trim_text(payload.get("rationale", ""), 320)
    evidence_summary = trim_text(payload.get("evidence_summary", ""), 420)
    provenance = build_provenance(candidate, payload)
    has_official_source = any(item.get("url") for item in provenance)

    if label not in ALLOWED_LABELS:
        return None
    if confidence is not None and confidence < args.min_confidence:
        return None
    if not rationale or not evidence_summary:
        return None
    if not has_official_source:
        return None
    if label in MIDDLE_LABELS and not not_binary_reason:
        return None

    context = candidate.get("context", {})
    row = {
        "claim_id": candidate_claim_id(candidate),
        "claim_text": candidate.get("claim_text", ""),
        "source_context": candidate.get("full_context", ""),
        "source_context_local": candidate.get("source_context_local", ""),
        "source_speaker": candidate.get("speaker", ""),
        "source_date": context.get("date", ""),
        "source_session": context.get("session", ""),
        "source_document_id": context.get("document_id", ""),
        "source_statement_id": context.get("statement_id", ""),
        "source_topic_h1": context.get("topic_h1", ""),
        "source_topic_h2": context.get("topic_h2", ""),
        "source_topic_h3": context.get("topic_h3", ""),
        "claim_type": candidate.get("claim_type", "general"),
        "label": label,
        "dataset_tier": "silver",
        "label_source": "openai_llm",
        "label_rationale": rationale,
        "not_binary_reason": not_binary_reason,
        "label_confidence": "" if confidence is None else str(confidence),
        "evidence_text": build_evidence_text(candidate, payload),
        "evidence_provenance_json": json.dumps(provenance, ensure_ascii=False),
        "model_name": model_name or args.model,
        "prompt_version": "liar_silver_v2_middle_first",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Silver / LLM-labeled. Non-deterministic.",
    }
    return row


def build_training_prompt(row: dict) -> str:
    return (
        "Analyze this political claim from Canadian Parliament and provide a fact-check.\n\n"
        f"Claim: \"{row.get('claim_text', '')}\"\n"
        f"Speaker: {row.get('source_speaker') or 'Unknown'}\n"
        f"Date: {row.get('source_date') or 'Unknown'}\n"
        f"Claim type: {row.get('claim_type') or 'general'}\n"
        f"Local context: {row.get('source_context_local') or row.get('source_context') or 'N/A'}\n"
        f"Evidence: {row.get('evidence_text') or 'N/A'}\n\n"
        f"{TRAINING_PROMPT_INSTRUCTIONS}"
    )


def build_training_completion(row: dict) -> str:
    correction = "N/A"
    if row.get("label") in {"false", "pants-fire", "barely-true", "half-true"}:
        correction = trim_text(row.get("evidence_text", ""), 220)
    verdict = str(row.get("label", "")).upper()
    return "\n".join(
        [
            f"VERDICT: {verdict}",
            f"CONFIDENCE: {row.get('label_confidence') or 'N/A'}",
            f"EXPLANATION: {trim_text(row.get('label_rationale', ''), 260)}",
            f"CORRECTION: {correction}",
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
                    "claim_id": row.get("claim_id"),
                    "dataset_tier": "silver",
                    "label_source": "openai_llm",
                    "label": row.get("label"),
                    "not_binary_reason": row.get("not_binary_reason"),
                    "claim_type": row.get("claim_type"),
                    "source_statement_id": row.get("source_statement_id"),
                    "model_name": row.get("model_name"),
                },
            }
        )
    return pairs


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_stats_rows(rows: Sequence[dict], stats: Counter) -> List[dict]:
    output = [
        {"category": "overall", "name": "total_rows", "value": str(len(rows))},
        {"category": "overall", "name": "distinct_claim_types", "value": str(len({row.get('claim_type') for row in rows}))},
    ]
    for category_name, counter in [
        ("label", Counter(row.get("label", "") for row in rows)),
        ("claim_type", Counter(row.get("claim_type", "") for row in rows)),
        ("model_name", Counter(row.get("model_name", "") for row in rows)),
    ]:
        for name, value in sorted(counter.items()):
            output.append({"category": category_name, "name": name, "value": str(value)})
    for key, value in sorted(stats.items()):
        category, _, metric = key.partition(".")
        output.append({"category": category, "name": metric or key, "value": str(value)})
    return output


def main() -> None:
    load_dotenv_fallback()
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the silver dataset builder.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_silver_path = Path(args.existing_silver_path)
    existing_rows = load_existing_silver_rows(existing_silver_path) if args.exclude_existing_silver else []
    excluded_claims = existing_claim_keys(existing_rows) if args.exclude_existing_silver else set()

    log("Building silver candidate pool")
    candidates = build_candidate_pool(args, excluded_claims=excluded_claims)
    for candidate in candidates:
        candidate["source_context_local"] = infer_local_context(candidate.get("full_context", ""), candidate.get("claim_text", ""))
    log(f"Prepared {len(candidates)} candidate claims")

    middle_rows: List[dict] = []
    fallback_rows: List[dict] = []
    stats: Counter = Counter()
    calls_made = 0
    for index, candidate in enumerate(candidates, start=1):
        if calls_made >= args.max_api_calls:
            stats["stopped.max_api_calls"] += 1
            break
        try:
            row = label_candidate(api_key, args, candidate)
            calls_made += 1
        except Exception as exc:
            stats["skipped.api_error"] += 1
            log(f"Skipping candidate {index} due to API error: {exc}")
            continue

        if row is None:
            stats["skipped.filtered_or_rejected"] += 1
            continue

        if row["label"] in MIDDLE_LABELS:
            middle_rows.append(row)
        elif row["label"] in FALLBACK_LABELS:
            fallback_rows.append(row)
        else:
            stats["skipped.unknown_label_bucket"] += 1
            continue

        new_rows_kept = len(middle_rows) + len(fallback_rows)
        if index % args.progress_every == 0 or index == len(candidates):
            log(
                f"Labeled {index}/{len(candidates)} candidates, kept {new_rows_kept} new rows "
                f"({len(middle_rows)} middle, {len(fallback_rows)} fallback)"
            )
        if args.label_priority == "middle_only" and len(middle_rows) >= args.target_new_rows:
            stats["stopped.target_new_rows"] += 1
            break
        if args.label_priority in {"middle_first", "all_labels"} and new_rows_kept >= args.target_new_rows:
            stats["stopped.target_new_rows"] += 1
            break

    if args.label_priority == "middle_only":
        new_rows = middle_rows[: args.target_new_rows]
    elif args.label_priority == "middle_first":
        needed = max(0, args.target_new_rows - len(middle_rows))
        new_rows = middle_rows + fallback_rows[:needed]
    else:
        new_rows = (middle_rows + fallback_rows)[: args.target_new_rows]

    rows = existing_rows + new_rows
    stats["overall.existing_rows"] = len(existing_rows)
    stats["overall.new_rows"] = len(new_rows)
    stats["overall.middle_rows"] = sum(row.get("label") in MIDDLE_LABELS for row in new_rows)
    stats["overall.fallback_rows"] = sum(row.get("label") in FALLBACK_LABELS for row in new_rows)
    stats["overall.api_calls"] = calls_made

    write_csv(
        output_dir / MASTER_FILENAME,
        rows,
        [
            "claim_id",
            "claim_text",
            "source_context",
            "source_context_local",
            "source_speaker",
            "source_date",
            "source_session",
            "source_document_id",
            "source_statement_id",
            "source_topic_h1",
            "source_topic_h2",
            "source_topic_h3",
            "claim_type",
            "label",
            "dataset_tier",
            "label_source",
            "label_rationale",
            "not_binary_reason",
            "label_confidence",
            "evidence_text",
            "evidence_provenance_json",
            "model_name",
            "prompt_version",
            "generated_at",
            "notes",
        ],
    )
    write_json(output_dir / TRAINING_FILENAME, build_training_pairs(rows))
    write_csv(output_dir / STATS_FILENAME, build_stats_rows(rows, stats), ["category", "name", "value"])
    log(f"Silver export complete: {MASTER_FILENAME}, {TRAINING_FILENAME}, {STATS_FILENAME}")


if __name__ == "__main__":
    main()
