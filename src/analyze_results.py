"""
Compute aggregate statistics and significance tests for the psychoactive framing study.

Reads the raw JSONL outputs, derives Wilson confidence intervals, counts missing answers,
and runs Fisher exact tests comparing each condition to the control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import fisher_exact

DEFAULT_RAW_DIR = Path("results/raw")
DEFAULT_AGG_DIR = Path("results/aggregates")
Z_95 = 1.959963984540054


@dataclass
class ConditionStats:
    condition: str
    correct: int
    total: int
    missing: int
    accuracy: float
    ci_low: float
    ci_high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark outputs with confidence intervals.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Path to a specific raw JSONL file. Defaults to the newest file in results/raw.",
    )
    parser.add_argument(
        "--control-condition",
        type=str,
        default="control",
        help="Condition name to treat as the control group for significance tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AGG_DIR,
        help="Directory to store summary CSVs.",
    )
    return parser.parse_args()


def find_latest_jsonl(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("arc_responses_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No JSONL files found in {raw_dir}")
    return candidates[-1]


def load_records(jsonl_path: Path) -> pd.DataFrame:
    data: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def wilson_interval(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denom = 1 + (Z_95**2) / n
    centre = p_hat + (Z_95**2) / (2 * n)
    adj = Z_95 * ((p_hat * (1 - p_hat) + (Z_95**2) / (4 * n)) / n) ** 0.5
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)


def summarize_conditions(df: pd.DataFrame) -> Dict[str, ConditionStats]:
    stats: Dict[str, ConditionStats] = {}
    for condition, group in df.groupby("condition"):
        total = len(group)
        correct = int(group["is_correct"].sum())
        missing = int(group["prediction"].isna().sum())
        ci_low, ci_high = wilson_interval(correct, total)
        stats[condition] = ConditionStats(
            condition=condition,
            correct=correct,
            total=total,
            missing=missing,
            accuracy=correct / total if total else 0.0,
            ci_low=ci_low,
            ci_high=ci_high,
        )
    return stats


def pairwise_tests(
    stats: Dict[str, ConditionStats], control_condition: str
) -> List[Dict[str, float]]:
    if control_condition not in stats:
        raise ValueError(f"Control condition '{control_condition}' not found in stats.")
    control = stats[control_condition]
    results: List[Dict[str, float]] = []
    for condition, cond_stats in stats.items():
        if condition == control_condition:
            continue
        table = [
            [cond_stats.correct, cond_stats.total - cond_stats.correct],
            [control.correct, control.total - control.correct],
        ]
        _, p_value = fisher_exact(table, alternative="two-sided")
        results.append(
            {
                "condition": condition,
                "control_condition": control_condition,
                "delta_accuracy": cond_stats.accuracy - control.accuracy,
                "p_value": p_value,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    jsonl_path = args.jsonl or find_latest_jsonl(DEFAULT_RAW_DIR)
    df = load_records(jsonl_path)
    stats = summarize_conditions(df)

    timestamp = jsonl_path.stem.split("_")[-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats_records = [
        {
            "condition": s.condition,
            "correct": s.correct,
            "total": s.total,
            "accuracy": s.accuracy,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "missing_predictions": s.missing,
        }
        for s in stats.values()
    ]
    stats_df = pd.DataFrame(stats_records).sort_values("condition")
    stats_path = args.output_dir / f"stats_{timestamp}.csv"
    stats_df.to_csv(stats_path, index=False)

    signif_records = pairwise_tests(stats, args.control_condition)
    signif_df = pd.DataFrame(signif_records).sort_values("condition")
    signif_path = args.output_dir / f"significance_{timestamp}.csv"
    signif_df.to_csv(signif_path, index=False)

    print(f"Wrote per-condition stats to {stats_path}")
    print(f"Wrote pairwise significance tests to {signif_path}")


if __name__ == "__main__":
    main()
