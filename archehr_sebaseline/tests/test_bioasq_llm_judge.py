from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.evaluation.bioasq_llm_judge import (
    parse_judge_response,
    run_llm_judge_validation,
    select_stratified_examples,
    spearman,
)
from archehr_sebaseline.data_io import write_jsonl


class BioASQLLMJudgeTests(unittest.TestCase):
    def test_select_stratified_examples_is_fixed_and_balanced(self) -> None:
        rows = [
            {"example_id": f"ex-{index:02d}", "quality_score": index / 100, "quality_rank_within_type": index / 29}
            for index in range(30)
        ]
        first = select_stratified_examples(rows, sample_size=9, seed=17)
        second = select_stratified_examples(rows, sample_size=9, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(
            {stratum: sum(row["stratum"] == stratum for row in first) for stratum in {row["stratum"] for row in first}},
            {"bottom_third": 3, "middle_third": 3, "top_third": 3},
        )

    def test_parse_judge_response_accepts_fenced_json_and_normalizes(self) -> None:
        parsed = parse_judge_response(
            '```json\n{"correctness":4,"completeness":3,"evidence_support":2,"relevance":2,"rationale":"sound"}\n```'
        )
        self.assertAlmostEqual(parsed["judge_score"], 11 / 14)
        self.assertEqual(parsed["rationale"], "sound")

    def test_parse_judge_response_rejects_out_of_range_score(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_response(
                '{"correctness":5,"completeness":3,"evidence_support":2,"relevance":2}'
            )

    def test_spearman_handles_ties(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [10, 20, 20, 40]) or 0, 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]) or 0, -1.0)

    def test_validation_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            eval_dir = run_dir / "bioasq_eval"
            output_dir = run_dir / "judge"
            eval_dir.mkdir(parents=True)
            examples = []
            generations = []
            quality_examples = []
            quality_generations = []
            se_examples = []
            for index in range(9):
                example_id = f"ex-{index}"
                examples.append(
                    {
                        "id": example_id,
                        "question": f"Question {index}?",
                        "ideal_answers": [f"Reference {index}"],
                        "evidence_sentences": [f"Evidence {index}"],
                    }
                )
                generations.append({"example_id": example_id, "sample_id": 0, "raw_answer": f"Answer {index}"})
                quality_examples.append({"example_id": example_id, "quality_score": index / 10, "quality_rank_within_type": index / 8})
                quality_generations.append({"example_id": example_id, "sample_id": 0, "quality_score": index / 10})
                se_examples.append({"example_id": example_id, "normalized_discrete_semantic_entropy": index / 8})
            write_jsonl(examples, run_dir / "examples.jsonl")
            write_jsonl(generations, run_dir / "cleaned_generations.jsonl")
            for filename, rows in (
                ("bioasq_quality_examples.csv", quality_examples),
                ("bioasq_quality_generations.csv", quality_generations),
                ("bioasq_se_eval_examples.csv", se_examples),
            ):
                with (eval_dir / filename).open("w", encoding="utf-8", newline="") as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            response = json.dumps(
                {"correctness": 3, "completeness": 3, "evidence_support": 3, "relevance": 2, "rationale": "ok"}
            )
            result = run_llm_judge_validation(
                run_dir,
                output_dir,
                judge=lambda prompt: response,
                judge_model_name="fake-judge",
                sample_size=9,
            )
            self.assertEqual(result["summary"]["valid_judgments"], 9)
            self.assertTrue((output_dir / "judge_raw_responses.jsonl").exists())
            self.assertTrue((output_dir / "judge_validation_report.md").exists())


if __name__ == "__main__":
    unittest.main()
