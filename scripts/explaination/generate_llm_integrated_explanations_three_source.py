#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate plain-language explanations from structured detection payloads.

Part A receives three model-only evidence sources:
1. offline Event-level detection,
2. streaming time-localised detection,
3. ConvAE pre-window early-warning detection.

The deterministic fusion script has already assigned anomaly / normal /
review_required.  The LLM must reproduce and explain that status; it must not
change it.

Part B remains streaming-only for additional off-log review candidates.

The LLM does not modify detection results. It only explains the supplied
structured evidence for non-technical readers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


LOGGED_SYSTEM_INSTRUCTIONS = """
You write short, plain-English monitoring summaries for wind-turbine operators
and non-technical readers.

For each logged analysis window you receive evidence from three complementary
detection approaches:

1. Offline detection:
   Reviews the complete analysis window and gives an overall Event-level
   assessment.

2. Streaming detection:
   Looks through the timeline and highlights specific periods of unusual
   behaviour in or near the logged window.

3. Early-warning detection:
   Reviews the period before the logged window and reports whether the selected
   Convolutional Autoencoder found confirmed pre-window warning episodes.

A deterministic fusion layer has already combined these evidence sources into
one final status: anomaly, normal, or review_required.

Write a clear and practical explanation based only on the supplied evidence.

Strict rules:
- Never use the terms "V2", "V5", "Offline V2", "Streaming V5", "CAE", or
  internal version names.
- Call the three approaches exactly "offline detection", "streaming detection",
  and "early-warning detection".
- The "Final assessment" section MUST reproduce the supplied deterministic final
  status. Render:
    anomaly -> "Anomaly"
    normal -> "Normal"
    review_required -> "Review required"
  Never invent or change the category.
- "Normal" means the three-source framework found low abnormal model evidence;
  it does NOT prove that the turbine is physically healthy or fault-free.
- Use simple everyday English. Avoid internal JSON field names, detector codes,
  model architecture terms, and unnecessary decimal values.
- Do not list raw feature names such as names containing underscores.
- Human-readable signal descriptions from early-warning evidence may be
  summarised, but they are supporting reconstruction-error evidence, not proven
  failed components or root causes.
- Explain the three approaches separately because they answer different
  temporal questions: whole-window pattern, time-localised behaviour, and
  pre-window early-warning behaviour.
- Explain whether the three evidence sources agree, partly agree, disagree, or
  remain uncertain.
- Do not invent a fault type, failed component, root cause, maintenance action,
  maintenance fact, or diagnosis.
- Do not claim that any approach is correct.
- Do not change or override any supplied detection result.
- A vote fraction describes how consistently repeated model checks voted for
  unusual behaviour. It is not the probability of a real physical fault.
- No streaming alert does not prove normal operation.
- No early-warning episode does not prove that no degradation existed.
- Detection evidence indicates unusual behaviour, not a confirmed diagnosis.
- Recommend human review when the deterministic status is "review_required" or
  when practical verification is useful.
- Keep the explanation between 110 and 180 words.
- Use short paragraphs and complete sentences.

Use exactly these headings:
Final assessment
Offline detection
Streaming detection
Early-warning detection
Overall interpretation
Recommended action
""".strip()


OFFLOG_SYSTEM_INSTRUCTIONS = """
You write short, plain-English monitoring notices for wind-turbine operators
and non-technical readers.

You receive evidence for an additional streaming-detection interval outside
its source Event's logged analysis window and configured time buffer.

Strict rules:
- Never use the terms "V2" or "V5".
- Call the method "streaming detection".
- Describe the interval as a review candidate, not a confirmed fault.
- Use simple everyday English and avoid internal detector codes or raw field
  names.
- Briefly explain when the interval occurred, how long it lasted, and what
  general behaviour caused it to be prioritised.
- Explain its position relative to its own source Event when that information
  is available.
- Do not invent a failed component, fault type, root cause, or maintenance fact.
- State that maintenance records and confirmed labels were not used to create
  the candidate.
- Recommend a practical manual check of trends, status history, and relevant
  maintenance records.
- Keep the explanation between 60 and 100 words.
- Use short paragraphs and complete sentences.

Use exactly these headings:
Review candidate
Why it was highlighted
What it means
Recommended action
""".strip()



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per non-empty line."""
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}."
                )

            records.append(payload)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def extract_record_id(payload: dict[str, Any]) -> str:
    """Return a stable identifier for saving and restart support."""
    task = str(payload.get("task", "")).strip()

    if task == "logged_analysis_window_model_only_explanation":
        window = payload.get("analysis_window", {})
        return f"event_{window.get('event_id', 'unknown')}"

    if task == "additional_unlogged_v5_review_candidate":
        candidate = payload.get("candidate", {})
        return str(candidate.get("candidate_id", "unknown_candidate"))

    return "unknown_record"


def select_instructions(payload: dict[str, Any]) -> str:
    task = str(payload.get("task", "")).strip()

    if task == "logged_analysis_window_model_only_explanation":
        return LOGGED_SYSTEM_INSTRUCTIONS

    if task == "additional_unlogged_v5_review_candidate":
        return OFFLOG_SYSTEM_INSTRUCTIONS

    raise ValueError(f"Unsupported payload task: {task!r}")


def make_public_facing_payload(value: Any) -> Any:
    """Rename internal labels and remove version names before the API call."""
    key_map = {
        "offline_v2": "offline_detection",
        "streaming_v5": "streaming_detection",
        "early_warning_cae": "early_warning_detection",
        "v2_reason": "offline_detection_reason",
        "v5_reason": "streaming_detection_reason",
        "cae_reason": "early_warning_detection_reason",
    }

    if isinstance(value, dict):
        return {
            key_map.get(str(key), str(key)): make_public_facing_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [make_public_facing_payload(item) for item in value]
    if isinstance(value, str):
        replacements = [
            ("Offline V2", "offline detection"),
            ("Streaming V5", "streaming detection"),
            ("offline V2", "offline detection"),
            ("streaming V5", "streaming detection"),
            ("Convolutional Autoencoder", "early-warning detection"),
            ("ConvAE", "early-warning detection"),
            ("CAE", "early-warning detection"),
            ("V2", "offline detection"),
            ("V5", "streaming detection"),
        ]
        result = value
        for old, new in replacements:
            result = result.replace(old, new)
        return result
    return value


def clean_public_explanation(text: str) -> str:
    """Final safeguard so the public report contains no version labels."""
    replacements = [
        ("Offline V2", "Offline detection"),
        ("Streaming V5", "Streaming detection"),
        ("offline V2", "offline detection"),
        ("streaming V5", "streaming detection"),
        ("Convolutional Autoencoder", "early-warning detection"),
        ("ConvAE", "early-warning detection"),
        ("CAE", "early-warning detection"),
        ("V2", "offline detection"),
        ("V5", "streaming detection"),
    ]
    cleaned = text
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return cleaned.strip()


def call_llm(
    client: OpenAI,
    payload: dict[str, Any],
    model: str,
    max_retries: int,
) -> dict[str, Any]:
    """Call the Responses API with retry handling."""
    record_id = extract_record_id(payload)
    instructions = select_instructions(payload)

    public_payload = make_public_facing_payload(payload)
    user_input = (
        "Write the requested public-facing explanation from this evidence. "
        "Do not repeat internal JSON field names.\n\n"
        + json.dumps(public_payload, ensure_ascii=False, indent=2)
    )

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
            )

            explanation = clean_public_explanation(response.output_text)

            if not explanation:
                raise RuntimeError("The API returned an empty explanation.")

            usage = getattr(response, "usage", None)

            return {
                "record_id": record_id,
                "task": payload.get("task"),
                "status": "success",
                "model": model,
                "response_id": response.id,
                "explanation": explanation,
                "usage": (
                    usage.model_dump()
                    if usage is not None
                    and hasattr(usage, "model_dump")
                    else None
                ),
                "source_payload": payload,
            }

        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            wait_seconds = min(30, 2 ** attempt)

            print(
                f"[RETRY] {record_id}: attempt {attempt} failed: {exc}"
            )
            print(f"[WAIT] {wait_seconds} seconds")
            time.sleep(wait_seconds)

    return {
        "record_id": record_id,
        "task": payload.get("task"),
        "status": "failed",
        "model": model,
        "error": str(last_error),
        "source_payload": payload,
    }


def load_completed_ids(path: Path) -> set[str]:
    """
    Read successful IDs from an existing output file.

    This allows the script to resume after interruption without paying again
    for already completed records.
    """
    if not path.exists():
        return set()

    completed: set[str] = set()

    for record in read_jsonl(path):
        if record.get("status") == "success":
            completed.add(str(record.get("record_id", "")))

    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def create_markdown_report(
    result_files: list[Path],
    output_path: Path,
) -> None:
    """Create one readable Markdown report from successful responses."""
    records: list[dict[str, Any]] = []

    for path in result_files:
        if path.exists():
            records.extend(read_jsonl(path))

    successful = [
        record
        for record in records
        if record.get("status") == "success"
    ]

    logged = [
        record
        for record in successful
        if record.get("task")
        == "logged_analysis_window_model_only_explanation"
    ]

    offlog = [
        record
        for record in successful
        if record.get("task")
        == "additional_unlogged_v5_review_candidate"
    ]

    lines: list[str] = [
        "# Plain-Language Detection Explanations",
        "",
        "For logged analysis windows, the explanations below were generated from "
        "structured offline, streaming and early-warning model evidence. Part B remains "
        "streaming-only. Maintenance labels, recorded fault descriptions and true "
        "outcomes were not supplied to the LLM.",
        "",
        "## Part A — Logged analysis windows",
        "",
    ]

    for record in logged:
        lines.extend(
            [
                f"### {record['record_id']}",
                "",
                record["explanation"],
                "",
            ]
        )

    lines.extend(
        [
            "## Part B — Additional streaming-detection review candidates",
            "",
        ]
    )

    if not offlog:
        lines.append("No additional candidate explanation was generated.")
    else:
        for record in offlog:
            lines.extend(
                [
                    f"### {record['record_id']}",
                    "",
                    record["explanation"],
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def process_file(
    client: OpenAI,
    input_path: Path,
    output_path: Path,
    model: str,
    max_retries: int,
    delay_seconds: float,
    limit: int | None,
) -> None:
    payloads = read_jsonl(input_path)
    completed_ids = load_completed_ids(output_path)

    pending = [
        payload
        for payload in payloads
        if extract_record_id(payload) not in completed_ids
    ]

    if limit is not None:
        pending = pending[:limit]

    print(f"[INPUT] {input_path}")
    print(f"[TOTAL] {len(payloads)}")
    print(f"[COMPLETED] {len(completed_ids)}")
    print(f"[PENDING THIS RUN] {len(pending)}")

    for index, payload in enumerate(pending, start=1):
        record_id = extract_record_id(payload)

        print(
            f"[CALL {index}/{len(pending)}] {record_id}"
        )

        result = call_llm(
            client=client,
            payload=payload,
            model=model,
            max_retries=max_retries,
        )

        append_jsonl(output_path, result)

        if result["status"] == "success":
            print(f"[SUCCESS] {record_id}")
        else:
            print(
                f"[FAILED] {record_id}: {result.get('error', '')}"
            )

        if delay_seconds > 0 and index < len(pending):
            time.sleep(delay_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create plain-language three-source Part-A and streaming-only Part-B explanations."
    )

    parser.add_argument(
        "--logged-payloads",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--offlog-payloads",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model ID.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Delay between API calls.",
    )

    parser.add_argument(
        "--limit-logged",
        type=int,
        default=None,
        help="Optional test limit for logged Event payloads.",
    )

    parser.add_argument(
        "--limit-offlog",
        type=int,
        default=None,
        help="Optional test limit for additional candidates.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. In PowerShell run:\n"
            '$env:OPENAI_API_KEY="your_api_key_here"'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    logged_output = (
        args.output_dir / "logged_event_llm_explanations.jsonl"
    )

    offlog_output = (
        args.output_dir / "offlog_candidate_llm_explanations.jsonl"
    )

    process_file(
        client=client,
        input_path=args.logged_payloads,
        output_path=logged_output,
        model=args.model,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        limit=args.limit_logged,
    )

    process_file(
        client=client,
        input_path=args.offlog_payloads,
        output_path=offlog_output,
        model=args.model,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        limit=args.limit_offlog,
    )

    create_markdown_report(
        result_files=[logged_output, offlog_output],
        output_path=(
            args.output_dir
            / "llm_integrated_explanation_report.md"
        ),
    )

    print("[DONE] LLM explanations generated.")
    print(f"[OUTPUT] {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())