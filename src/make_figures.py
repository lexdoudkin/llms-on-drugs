"""
Generate simple SVG bar charts for benchmark metrics without relying on Matplotlib.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_DIR = Path("results/aggregates")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_summary() -> pd.DataFrame:
    summaries = sorted(SUMMARY_DIR.glob("summary_*.csv"))
    if not summaries:
        raise FileNotFoundError("No summary CSVs found in results/aggregates.")
    df = pd.read_csv(summaries[-1])
    order = ["control", "lsd", "cocaine", "alcohol", "cannabis"]
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    return df.sort_values("condition")


def svg_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    *,
    title: str,
    y_label: str,
    ymax: float | None = None,
) -> None:
    width, height = 640, 400
    margin = 60
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    labels = list(df["condition"])
    values = list(df[value_col])

    if ymax is None:
        ymax = max(values) if values else 1.0
        if math.isclose(ymax, 0):
            ymax = 1.0
        else:
            ymax = round((ymax * 1.1) + 1e-9, 2)

    bar_width = chart_width / (len(values) * 1.3)
    spacing = bar_width * 0.3
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="{margin/2}" text-anchor="middle" '
        f'font-size="18" font-family="Helvetica">{title}</text>',
        f'<text x="{margin/2}" y="{margin}" text-anchor="middle" transform="rotate(-90 {margin/2},{margin})" '
        f'font-size="14" font-family="Helvetica">{y_label}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" '
        f'stroke="#333" stroke-width="2"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" '
        f'stroke="#333" stroke-width="2"/>',
    ]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin + idx * (bar_width + spacing) + spacing
        bar_height = 0 if ymax == 0 else (value / ymax) * chart_height
        y = height - margin - bar_height
        color = "#2a9d8f"
        svg_parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'fill="{color}" opacity="0.8"/>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width/2:.2f}" y="{height - margin + 20}" text-anchor="middle" '
            f'font-size="12" font-family="Helvetica">{label}</text>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width/2:.2f}" y="{y - 5:.2f}" text-anchor="middle" '
            f'font-size="12" font-family="Helvetica">{value:.2f}</text>'
        )

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = height - margin - frac * chart_height
        value = frac * ymax
        svg_parts.append(
            f'<line x1="{margin - 5}" y1="{y:.2f}" x2="{width - margin}" y2="{y:.2f}" '
            f'stroke="#999" stroke-width="0.5" stroke-dasharray="4,4"/>'
        )
        svg_parts.append(
            f'<text x="{margin - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" '
            f'font-family="Helvetica">{value:.2f}</text>'
        )

    svg_parts.append("</svg>")
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    df = load_latest_summary()
    svg_bar_chart(
        df,
        "accuracy",
        FIG_DIR / "accuracy_by_condition.svg",
        title="ARC Accuracy by Framing Condition",
        y_label="Accuracy",
        ymax=1.0,
    )
    svg_bar_chart(
        df,
        "avg_latency",
        FIG_DIR / "latency_by_condition.svg",
        title="Average Response Latency by Condition",
        y_label="Seconds",
        ymax=None,
    )


if __name__ == "__main__":
    main()
