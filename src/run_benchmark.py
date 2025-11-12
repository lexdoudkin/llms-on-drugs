"""
Run the psychoactive framing benchmark on GPT-5-mini using ARC-Challenge.

The script loads a slice of ARC-Challenge, prefixes each prompt with a
drug framing (plus a sober control), queries the OpenAI Responses API,
and writes both raw generations and aggregate metrics to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a meticulous science and reasoning tutor. "
    "You must decide which answer choice (A-E) is correct. "
    "Think step-by-step if needed, but always present the final answer "
    "on a separate line starting with 'Answer: <LETTER>'."
)

CONDITIONS = {
    "control": {
        "label": "Control",
        "prefix": "You are sober, calm, and on-task. Answer with maximum precision.",
    },
    "lsd": {
        "label": "LSD",
        "prefix": (
            "You are on LSD. Colours pulse around each idea and associations feel expansive, "
            "yet you must still answer the question."
        ),
    },
    "cocaine": {
        "label": "Cocaine",
        "prefix": (
            "You are on cocaine. Thoughts race, confidence surges, and you feel unstoppable. "
            "Answer decisively."
        ),
    },
    "alcohol": {
        "label": "Alcohol",
        "prefix": "You are on alcohol. Loose, conversational, but try to stay coherent as you answer.",
    },
    "cannabis": {
        "label": "Cannabis",
        "prefix": (
            "You are on cannabis. Ideas drift and feel novel, yet you remain introspective. "
            "Ground yourself and answer."
        ),
    },
}

PROMPT_TEMPLATE = """{prefix}

You will receive one ARC multiple-choice question. Provide reasoning and finish with the letter.

Question: {question}
Choices:
{choices}
"""

CHOICE_OPTION_TEMPLATE = "{label}. {text}"

CHOICE_REGEX = re.compile(r"\b([A-E])(?=[).:\s])", re.IGNORECASE)


@dataclass
class BenchmarkConfig:
    dataset: str = "ai2_arc"
    subset: str = "ARC-Challenge"
    split: str = "validation"
    num_samples: int = 40
    temperature: float = 0.2
    top_p: float = 0.95
    max_output_tokens: int = 300
    seed: int = 13
    retries: int = 4
    pause: float = 2.5
    output_dir: Path = Path("results")
    model: str = "gpt-5-mini"


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run psychoactive framing benchmark on GPT-5-mini.")
    parser.add_argument("--num-samples", type=int, default=40, help="Number of ARC items to evaluate.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to sample from.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for sampling and API calls.")
    parser.add_argument(
        "--max-output-tokens", type=int, default=300, help="Maximum tokens to allow in each completion."
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for outputs.")
    args = parser.parse_args()
    return BenchmarkConfig(
        num_samples=args.num_samples,
        split=args.split,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
    )


def load_arc_subset(cfg: BenchmarkConfig) -> Dataset:
    dataset = load_dataset(cfg.dataset, cfg.subset, split=cfg.split)
    dataset = dataset.shuffle(seed=cfg.seed)
    if cfg.num_samples:
        dataset = dataset.select(range(min(cfg.num_samples, len(dataset))))
    return dataset


def format_choices(question: Dict) -> Tuple[str, Dict[str, str]]:
    labels = question["choices"]["label"]
    texts = question["choices"]["text"]
    formatted_lines = []
    label_to_text = {}
    for label, text in zip(labels, texts):
        formatted_lines.append(CHOICE_OPTION_TEMPLATE.format(label=label, text=text))
        label_to_text[label] = text
    return "\n".join(formatted_lines), label_to_text


def build_prompt(prefix: str, sample: Dict) -> Tuple[str, Dict[str, str]]:
    question = sample["question"]
    choices_block, label_map = format_choices(sample)
    prompt = PROMPT_TEMPLATE.format(prefix=prefix, question=question, choices=choices_block)
    return prompt, label_map


def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    match = CHOICE_REGEX.search(text)
    if match:
        return match.group(1).upper()
    # Look for 'Answer: X'
    lowered = text.lower()
    for label in ["a", "b", "c", "d", "e"]:
        if f"answer: {label}" in lowered:
            return label.upper()
    return None


def response_to_text(response: Any) -> str:
    chunks: List[str] = []
    for item in response.output or []:
        if item.type == "message":
            for content in item.content or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(content.text)
        elif item.type == "output_text":
            chunks.append(item.text)
    return "\n".join(chunks).strip()


def call_model_with_retry(
    client: OpenAI, user_prompt: str, cfg: BenchmarkConfig, *, seed_offset: int = 0
) -> Any:
    delay = cfg.pause
    last_error: Optional[Exception] = None
    for attempt in range(cfg.retries):
        try:
            return client.responses.create(
                model=cfg.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=cfg.max_output_tokens,
            )
        except Exception as err:  # noqa: BLE001 - we need to retry on any transport/API error
            last_error = err
            sleep_for = delay * (2 ** attempt)
            time.sleep(sleep_for)
    raise RuntimeError(f"Model call failed after {cfg.retries} retries: {last_error}") from last_error


def ensure_dirs(base: Path) -> Dict[str, Path]:
    raw_dir = base / "raw"
    aggregates_dir = base / "aggregates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "aggregates": aggregates_dir}


def run_benchmark(cfg: BenchmarkConfig) -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Please populate .env before running.")
    client = OpenAI()

    dataset = load_arc_subset(cfg)
    output_dirs = ensure_dirs(cfg.output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = output_dirs["raw"] / f"arc_responses_{timestamp}.jsonl"

    records: List[Dict] = []
    jsonl_file = raw_path.open("w", encoding="utf-8")

    try:
        for condition_key, condition in CONDITIONS.items():
            for idx, sample in enumerate(
                tqdm(dataset, desc=f"{condition['label']} ({condition_key})", leave=False)
            ):
                prompt, label_map = build_prompt(condition["prefix"], sample)
                start = time.time()
                response = call_model_with_retry(client, prompt, cfg, seed_offset=idx)
                latency = time.time() - start
                text = response_to_text(response)
                choice = extract_choice_letter(text)
                gold = sample["answerKey"].strip().upper()
                correct = choice == gold
                usage = response.usage or {}

                record = {
                    "timestamp": timestamp,
                    "condition": condition_key,
                    "condition_label": condition["label"],
                    "question_id": sample["id"],
                    "question": sample["question"],
                    "choices": label_map,
                    "gold": gold,
                    "prediction": choice,
                    "is_correct": bool(correct),
                    "prompt": prompt,
                    "response_text": text,
                    "latency_sec": latency,
                    "model": cfg.model,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_output_tokens": cfg.max_output_tokens,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                records.append(record)
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
    finally:
        jsonl_file.close()

    df = pd.DataFrame(records)
    csv_path = output_dirs["raw"] / f"arc_responses_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("condition")
        .agg(
            accuracy=("is_correct", "mean"),
            avg_latency=("latency_sec", "mean"),
            avg_response_chars=("response_text", lambda x: x.str.len().mean()),
        )
        .reset_index()
    )
    summary_path = output_dirs["aggregates"] / f"summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote raw responses to {raw_path}")
    print(f"Wrote CSV export to {csv_path}")
    print(f"Wrote summary metrics to {summary_path}")


if __name__ == "__main__":
    config = parse_args()
    run_benchmark(config)
