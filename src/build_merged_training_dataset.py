#!/usr/bin/env python3
"""
Merge the silver training dataset with a curated balanced subset of the gold dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATA_DIR = Path("data")
GOLD_MASTER = DEFAULT_DATA_DIR / "gold_factcheck_master.csv"
GOLD_TRAINING = DEFAULT_DATA_DIR / "gold_training_pairs.json"
SILVER_MASTER = DEFAULT_DATA_DIR / "liar_silver_master.csv"
SILVER_TRAINING = DEFAULT_DATA_DIR / "liar_silver_training_pairs.json"
OUTPUT_JSON = DEFAULT_DATA_DIR / "merged_gold_silver_training_pairs.json"
OUTPUT_STATS = DEFAULT_DATA_DIR / "merged_gold_silver_stats.csv"

TARGET_TRUE_TOTAL = 500
TARGET_FALSE_TOTAL = 500
MAJOR_TYPE_WEIGHTS = {
    "status_snapshot": 0.4,
    "introduced_date": 0.4,
    "party_vote_direction": 0.2,
}
RARE_TYPES = ("session_parliament", "sponsor")
LOW_SIGNAL_PATTERNS = [
    re.compile(r"^(That is why we are moving the following motion:)", re.IGNORECASE),
    re.compile(r"^(I rise(?: tonight)? to speak)", re.IGNORECASE),
    re.compile(r"^(Mr\. Speaker, I rise)", re.IGNORECASE),
    re.compile(r"^(The Speaker then responded:)", re.IGNORECASE),
]
PROMPT_LABEL_RE = re.compile(r"^VERDICT:\s*([A-Z-]+)", re.MULTILINE)
CLAIM_PROMPT_RE = re.compile(r'Claim:\s*"(.+?)"\n', re.DOTALL)


@dataclass
class TrainingItem:
    pair: dict
    master: dict
    label: str
    source_dataset: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge gold and silver training datasets.")
    parser.add_argument("--gold-master", default=str(GOLD_MASTER))
    parser.add_argument("--gold-training", default=str(GOLD_TRAINING))
    parser.add_argument("--silver-master", default=str(SILVER_MASTER))
    parser.add_argument("--silver-training", default=str(SILVER_TRAINING))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-stats", default=str(OUTPUT_STATS))
    parser.add_argument("--target-true-total", type=int, default=TARGET_TRUE_TOTAL)
    parser.add_argument("--target-false-total", type=int, default=TARGET_FALSE_TOTAL)
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def parse_training_label(completion: str) -> str:
    match = PROMPT_LABEL_RE.search(completion or "")
    return match.group(1).lower() if match else ""


def parse_claim_text_from_prompt(prompt: str) -> str:
    match = CLAIM_PROMPT_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_master_map(rows: Sequence[dict]) -> Dict[str, dict]:
    return {row["claim_id"]: row for row in rows}


def build_training_items(training_pairs: Sequence[dict], master_map: Dict[str, dict], source_dataset: str) -> List[TrainingItem]:
    items: List[TrainingItem] = []
    for pair in training_pairs:
        metadata = pair.get("metadata", {})
        claim_id = metadata.get("claim_id")
        master = master_map.get(claim_id)
        if not master:
            continue
        items.append(
            TrainingItem(
                pair=pair,
                master=master,
                label=parse_training_label(pair.get("completion", "")),
                source_dataset=source_dataset,
            )
        )
    return items


def is_good_gold_candidate(item: TrainingItem) -> bool:
    text = item.master.get("claim_text", "")
    if len(text) < 45 or len(text) > 320:
        return False
    if any(pattern.search(text) for pattern in LOW_SIGNAL_PATTERNS):
        return False
    return True


def gold_candidate_score(item: TrainingItem) -> Tuple[int, int, int, int]:
    text = item.master.get("claim_text", "")
    score = 0
    text_len = len(text)
    if 60 <= text_len <= 220:
        score += 4
    elif 45 <= text_len <= 320:
        score += 2
    else:
        score -= 4

    if re.search(r"\b[C|S]-\d+\b", text, re.IGNORECASE):
        score += 2
    if re.search(r"\b\d{4}\b", text):
        score += 1
    if re.search(r"\b(voted|introduced|committee|senate|house|royal assent|third reading|second reading|report stage)\b", text, re.IGNORECASE):
        score += 1

    prompt_len = len(item.pair.get("prompt", ""))
    if prompt_len <= 2200:
        score += 1
    elif prompt_len > 3000:
        score -= 1

    source_date = item.master.get("source_date", "")
    year = 0
    if source_date and len(source_date) >= 4 and source_date[:4].isdigit():
        year = int(source_date[:4])
        if year >= 2020:
            score += 2
        elif year >= 2010:
            score += 1

    claim_type = item.master.get("claim_type", "")
    if claim_type == "party_vote_direction":
        score -= 1

    return score, year, -prompt_len, -text_len


def allocate_major_type_targets(target: int, capacities: Dict[str, int]) -> Dict[str, int]:
    quotas: Dict[str, int] = {claim_type: 0 for claim_type in MAJOR_TYPE_WEIGHTS}
    remaining = target

    for claim_type, weight in MAJOR_TYPE_WEIGHTS.items():
        proposed = int(target * weight)
        take = min(capacities.get(claim_type, 0), proposed)
        quotas[claim_type] = take
        remaining -= take

    while remaining > 0:
        candidates = [claim_type for claim_type in MAJOR_TYPE_WEIGHTS if quotas[claim_type] < capacities.get(claim_type, 0)]
        if not candidates:
            break
        candidates.sort(key=lambda claim_type: (MAJOR_TYPE_WEIGHTS[claim_type], capacities.get(claim_type, 0) - quotas[claim_type]), reverse=True)
        quotas[candidates[0]] += 1
        remaining -= 1

    return quotas


def select_gold_subset(
    gold_items: Sequence[TrainingItem],
    excluded_claims: set[str],
    target_true_total: int,
    target_false_total: int,
    silver_label_counts: Counter,
) -> List[TrainingItem]:
    target_true_from_gold = max(0, target_true_total - silver_label_counts.get("true", 0))
    target_false_from_gold = max(0, target_false_total - silver_label_counts.get("false", 0))

    filtered = [
        item
        for item in gold_items
        if item.label in {"true", "false"}
        and is_good_gold_candidate(item)
        and normalize_text(item.master.get("claim_text", "")) not in excluded_claims
    ]

    by_bucket: Dict[Tuple[str, str], List[TrainingItem]] = {}
    for label in {"true", "false"}:
        for claim_type in set(MAJOR_TYPE_WEIGHTS) | set(RARE_TYPES):
            bucket = [item for item in filtered if item.label == label and item.master.get("claim_type") == claim_type]
            bucket.sort(key=gold_candidate_score, reverse=True)
            by_bucket[(label, claim_type)] = bucket

    selected: List[TrainingItem] = []
    targets = {"true": target_true_from_gold, "false": target_false_from_gold}
    for label, label_target in targets.items():
        label_selected: List[TrainingItem] = []

        for claim_type in RARE_TYPES:
            bucket = by_bucket[(label, claim_type)]
            label_selected.extend(bucket)

        remaining = max(0, label_target - len(label_selected))
        capacities = {claim_type: len(by_bucket[(label, claim_type)]) for claim_type in MAJOR_TYPE_WEIGHTS}
        quotas = allocate_major_type_targets(remaining, capacities)
        for claim_type, quota in quotas.items():
            label_selected.extend(by_bucket[(label, claim_type)][:quota])

        if len(label_selected) < label_target:
            leftovers: List[TrainingItem] = []
            already = {item.master["claim_id"] for item in label_selected}
            for claim_type in MAJOR_TYPE_WEIGHTS:
                leftovers.extend(
                    item
                    for item in by_bucket[(label, claim_type)]
                    if item.master["claim_id"] not in already
                )
            leftovers.sort(key=gold_candidate_score, reverse=True)
            label_selected.extend(leftovers[: label_target - len(label_selected)])

        selected.extend(label_selected[:label_target])

    return selected


def merged_pair(item: TrainingItem) -> dict:
    pair = json.loads(json.dumps(item.pair))
    metadata = dict(pair.get("metadata", {}))
    metadata["source_dataset"] = item.source_dataset
    metadata["merged_label"] = item.label
    metadata["claim_text"] = item.master.get("claim_text") or parse_claim_text_from_prompt(pair.get("prompt", ""))
    if "label" in metadata and isinstance(metadata["label"], str):
        metadata["label"] = metadata["label"].lower()
    pair["metadata"] = metadata
    return pair


def build_stats_rows(merged_items: Sequence[TrainingItem], selected_gold: Sequence[TrainingItem], silver_items: Sequence[TrainingItem]) -> List[dict]:
    output = [
        {"category": "overall", "name": "total_rows", "value": str(len(merged_items))},
        {"category": "overall", "name": "silver_rows", "value": str(len(silver_items))},
        {"category": "overall", "name": "selected_gold_rows", "value": str(len(selected_gold))},
    ]

    for category_name, counter in [
        ("label", Counter(item.label for item in merged_items)),
        ("source_dataset", Counter(item.source_dataset for item in merged_items)),
        ("claim_type", Counter(item.master.get("claim_type", "") for item in merged_items)),
        ("gold_claim_type", Counter(item.master.get("claim_type", "") for item in selected_gold)),
        ("silver_claim_type", Counter(item.master.get("claim_type", "") for item in silver_items)),
    ]:
        for name, value in sorted(counter.items()):
            output.append({"category": category_name, "name": name, "value": str(value)})

    return output


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    args = parse_args()
    gold_master_rows = load_csv_rows(Path(args.gold_master))
    silver_master_rows = load_csv_rows(Path(args.silver_master))
    gold_pairs = load_json(Path(args.gold_training))
    silver_pairs = load_json(Path(args.silver_training))

    gold_items = build_training_items(gold_pairs, build_master_map(gold_master_rows), "gold")
    silver_items = build_training_items(silver_pairs, build_master_map(silver_master_rows), "silver")

    silver_label_counts = Counter(item.label for item in silver_items)
    excluded_claims = {normalize_text(item.master.get("claim_text", "")) for item in silver_items}

    log("Selecting balanced gold subset for merged training dataset")
    selected_gold = select_gold_subset(
        gold_items=gold_items,
        excluded_claims=excluded_claims,
        target_true_total=args.target_true_total,
        target_false_total=args.target_false_total,
        silver_label_counts=silver_label_counts,
    )

    merged_items = list(silver_items) + selected_gold
    merged_payload = [merged_pair(item) for item in merged_items]

    log(f"Writing merged training dataset with {len(merged_payload)} rows")
    write_json(Path(args.output_json), merged_payload)
    write_csv(
        Path(args.output_stats),
        build_stats_rows(merged_items, selected_gold, silver_items),
        ["category", "name", "value"],
    )

    log("Merge complete")


if __name__ == "__main__":
    main()
