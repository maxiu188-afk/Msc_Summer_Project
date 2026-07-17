"""Compare Claude Sonnet 5 low/high effort on the six blind BioASQ review cases."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.evaluation.claude_judge import (
    CLAUDE_JUDGE_SYSTEM_PROMPT,
    build_semantic_quality_prompt,
    extract_label,
)


CASE_RE = re.compile(r"^## Review (?P<case_id>R\d+)\n(?P<body>.*?)(?=^## Review |^## Overall notes)", re.MULTILINE | re.DOTALL)
QUESTION_RE = re.compile(r"Question:\n\n> (?P<value>.*?)\n\nIdeal reference:", re.DOTALL)
REFERENCE_RE = re.compile(r"Ideal reference:\n\n> (?P<value>.*?)\n\nCandidate answer:", re.DOTALL)
CANDIDATE_RE = re.compile(r"Candidate answer:\n\n> (?P<value>.*?)\n\nReview:", re.DOTALL)


def clean_quote(value: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in value.splitlines()).strip()


def load_cases(path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    content = path.read_text(encoding="utf-8")
    for match in CASE_RE.finditer(content):
        body = match.group("body")
        fields = [QUESTION_RE.search(body), REFERENCE_RE.search(body), CANDIDATE_RE.search(body)]
        if any(field is None for field in fields):
            raise ValueError(f"Could not parse {match.group('case_id')} from {path}.")
        question, reference, candidate = (clean_quote(field.group("value")) for field in fields)
        cases.append(
            {
                "case_id": match.group("case_id"),
                "question": question,
                "reference": reference,
                "candidate": candidate,
            }
        )
    if len(cases) != 6:
        raise ValueError(f"Expected 6 manual-review cases, found {len(cases)}.")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review_file",
        type=Path,
        default=PROJECT_ROOT.parent / "server_results/manual_review/bioasq_low_quality_blind_review.md",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=PROJECT_ROOT.parent / "server_results/manual_review/claude_sonnet5_effort_smoke.csv",
    )
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--env_file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output_csv}; pass --overwrite.")
    if args.max_tokens < 32:
        raise ValueError("max_tokens must be at least 32.")

    load_dotenv(args.env_file)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(f"ANTHROPIC_API_KEY was not found in {args.env_file}.")
    from anthropic import Anthropic

    cases = load_cases(args.review_file)
    client = Anthropic()
    rows: list[dict[str, object]] = []
    for effort in ("low", "high"):
        for case in cases:
            prompt = build_semantic_quality_prompt(
                question=case["question"],
                references=[case["reference"]],
                candidate=case["candidate"],
            )
            started = time.monotonic()
            response = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                system=CLAUDE_JUDGE_SYSTEM_PROMPT,
                output_config={"effort": effort},
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - started
            rows.append(
                {
                    "case_id": case["case_id"],
                    "effort": effort,
                    "label": extract_label(response),
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "elapsed_seconds": f"{elapsed:.2f}",
                }
            )
            print(f"{effort} {case['case_id']}: {rows[-1]['label']} ({elapsed:.1f}s)", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} judgments to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
