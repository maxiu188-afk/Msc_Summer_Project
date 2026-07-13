from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.dataset_adapters import (
    load_archehr_common_examples,
    load_bioasq_common_examples,
    load_pubmedqa_common_examples,
    archehr_records_to_common,
    bioasq_records_to_common,
    pubmedqa_records_to_common,
)
from archehr_sebaseline.prompting import build_prompt, build_prompt_records


class DatasetAdapterTests(unittest.TestCase):
    def test_pubmedqa_record_to_common_schema(self) -> None:
        examples = pubmedqa_records_to_common(
            {
                "123": {
                    "QUESTION": "Does treatment improve survival?",
                    "CONTEXTS": ["Sentence one.", "Sentence two."],
                    "LONG_ANSWER": "Treatment improved survival in this study.",
                    "final_decision": "yes",
                }
            },
            split="pqal",
        )
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["dataset"], "pubmedqa")
        self.assertEqual(example["id"], "123")
        self.assertEqual(example["split"], "pqal")
        self.assertEqual(example["question"], "Does treatment improve survival?")
        self.assertEqual(example["evidence_sentences"], ["Sentence one.", "Sentence two."])
        self.assertEqual(example["label"], "yes")
        self.assertEqual(example["options"], ["yes", "no", "maybe"])

    def test_load_pubmedqa_common_examples_limit(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_fixtures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ori_pqal_test.json"
        path.write_text(
            json.dumps(
                {
                    "2": {"QUESTION": "Q2?", "CONTEXTS": ["C2"], "LONG_ANSWER": "A2"},
                    "1": {"QUESTION": "Q1?", "CONTEXTS": ["C1"], "LONG_ANSWER": "A1"},
                }
            ),
            encoding="utf-8",
        )
        examples = load_pubmedqa_common_examples(path, limit=1)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], "1")

    def test_bioasq_record_to_common_schema(self) -> None:
        examples = bioasq_records_to_common(
            [
                {
                    "id": "bio-1",
                    "body": "Describe RankMHC",
                    "type": "summary",
                    "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/1"],
                    "snippets": [
                        {
                            "text": "RankMHC is a learning-to-rank predictor.",
                            "document": "http://www.ncbi.nlm.nih.gov/pubmed/1",
                        }
                    ],
                    "ideal_answer": ["RankMHC is a learning-to-rank predictor."],
                }
            ],
            split="train13b",
        )
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["dataset"], "bioasq")
        self.assertEqual(example["id"], "bio-1")
        self.assertEqual(example["split"], "train13b")
        self.assertEqual(example["bioasq_type"], "summary")
        self.assertEqual(example["evidence_sentences"], ["RankMHC is a learning-to-rank predictor."])
        self.assertEqual(example["evidence_sentence_ids"], ["S1"])
        self.assertEqual(example["ideal_answers"], ["RankMHC is a learning-to-rank predictor."])
        self.assertEqual(example["documents"], ["http://www.ncbi.nlm.nih.gov/pubmed/1"])

    def test_bioasq_yesno_label_and_exact_answers(self) -> None:
        examples = bioasq_records_to_common(
            [
                {
                    "id": "bio-2",
                    "body": "Is Papilin secreted?",
                    "type": "yesno",
                    "snippets": [{"text": "Papilin is a secreted protein."}],
                    "exact_answer": "yes",
                    "ideal_answer": ["Yes, papilin is secreted."],
                }
            ]
        )
        self.assertEqual(examples[0]["label"], "yes")
        self.assertEqual(examples[0]["options"], ["yes", "no"])
        self.assertEqual(examples[0]["exact_answers"], ["yes"])

    def test_load_bioasq_common_examples_filters_type(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_fixtures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "bioasq_test.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "id": "summary-1",
                            "body": "Summarize X.",
                            "type": "summary",
                            "snippets": [{"text": "X is described."}],
                            "ideal_answer": ["X is described."],
                        },
                        {
                            "id": "factoid-1",
                            "body": "Name X.",
                            "type": "factoid",
                            "snippets": [{"text": "X is ABC."}],
                            "exact_answer": ["ABC"],
                            "ideal_answer": ["X is ABC."],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        examples = load_bioasq_common_examples(path, question_type="summary")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], "summary-1")

    def test_bioasq_prompt_uses_biomedical_snippets(self) -> None:
        example = bioasq_records_to_common(
            [
                {
                    "id": "bio-3",
                    "body": "List EGFR ligands.",
                    "type": "list",
                    "snippets": [{"text": "EGF and epiregulin bind EGFR."}],
                    "ideal_answer": ["EGF and epiregulin bind EGFR."],
                    "exact_answer": [["EGF"], ["epiregulin"]],
                }
            ]
        )[0]
        prompt = build_prompt(example)
        self.assertIn("BioASQ biomedical question", prompt)
        self.assertIn("Question type: list", prompt)
        self.assertIn("[S1] EGF and epiregulin bind EGFR.", prompt)
        self.assertIn("comma-separated list", prompt)
        self.assertEqual(build_prompt_records([example])[0]["prompt_version"], "bioasq_grounded_v1")

    def test_archehr_records_to_common_preserves_sentence_ids(self) -> None:
        examples = archehr_records_to_common(
            [
                {
                    "id": "case-1",
                    "patient_question": "Why was the medication changed?",
                    "clinician_question": "Why was therapy switched?",
                    "evidence": [
                        {"sentence_id": "S1", "text": "The patient had nausea."},
                        {"sentence_id": "S2", "text": "The medication was changed."},
                    ],
                    "gold_answer": "Therapy was switched because of nausea.",
                    "gold_relevant_sentence_ids": ["S1", "S2"],
                }
            ]
        )
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["dataset"], "archehr_qa")
        self.assertEqual(example["id"], "case-1")
        self.assertEqual(example["evidence_sentence_ids"], ["S1", "S2"])
        self.assertEqual(example["gold_relevant_sentence_ids"], ["S1", "S2"])

    def test_load_archehr_common_examples_jsonl(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_fixtures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "archehr_test.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "case-2",
                            "question": "What happened?",
                            "sentences": ["Sentence one.", "Sentence two."],
                        }
                    )
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        examples = load_archehr_common_examples(path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], "case-2")
        self.assertEqual(examples[0]["evidence_sentence_ids"], ["S1", "S2"])

    def test_load_archehr_common_examples_xml(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_fixtures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "archehr_test.xml"
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<annotations>
  <case id="case-xml">
    <clinical_specialty>medicine</clinical_specialty>
    <patient_question>Why was the test ordered?</patient_question>
    <clinician_question>Why was imaging recommended?</clinician_question>
    <note_excerpt_sentences>
      <sentence id="S1" paragraph_id="p1" start_char_index="0" length="10">The patient had pain.</sentence>
      <sentence id="S2" paragraph_id="p1" start_char_index="11" length="10">Imaging was recommended.</sentence>
    </note_excerpt_sentences>
  </case>
</annotations>
""",
            encoding="utf-8",
        )
        examples = load_archehr_common_examples(path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], "case-xml")
        self.assertEqual(examples[0]["question"], "Why was the test ordered?")
        self.assertEqual(examples[0]["clinician_question"], "Why was imaging recommended?")
        self.assertEqual(examples[0]["evidence_sentence_ids"], ["S1", "S2"])

    def test_load_archehr_common_examples_xml_nested_questions(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_fixtures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "archehr_nested_test.xml"
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<annotations>
  <case id="1">
    <questions>
      <question role="patient">Why did I need antibiotics?</question>
      <question role="clinician">Why were antibiotics prescribed?</question>
    </questions>
    <note_excerpt_sentences>
      <sentence id="S1">The patient had a suspected infection.</sentence>
    </note_excerpt_sentences>
  </case>
</annotations>
""",
            encoding="utf-8",
        )
        examples = load_archehr_common_examples(path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], "1")
        self.assertEqual(examples[0]["question"], "Why did I need antibiotics?")
        self.assertEqual(examples[0]["clinician_question"], "Why were antibiotics prescribed?")


if __name__ == "__main__":
    unittest.main()
