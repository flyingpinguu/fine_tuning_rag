"""Small checks for metric behavior; no models, downloads, or API calls."""

import copy
import unittest

from evaluate import FALLBACK, contains_term, evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = [
            {"id": "supported", "question": "May I do that?", "expected_sources": ["policy"],
             "required_terms": ["not"], "forbidden_terms": [], "should_abstain": False},
            {"id": "unsupported", "question": "Unknown question?", "expected_sources": [],
             "required_terms": [], "forbidden_terms": [], "should_abstain": True},
        ]
        hit = {"source_id": "policy", "score": .9}
        generated = {"generated_tokens": 8, "generation_seconds": .1, "answer_latency_seconds": .11}
        fallback = {"answer": FALLBACK, "generated_tokens": 0, "generation_seconds": 0, "answer_latency_seconds": .01}
        self.trace = {"configuration": {"max_new_tokens": 80}, "cases": [
            {"id": "supported", "question": "May I do that?", "all_retrieved": [hit], "evidence": [hit], "status": "generated",
             "baseline": {**generated, "answer": "Nothing is wrong. [policy]"},
             "lora": {**generated, "answer": "Do not do that. [missing]"}},
            {"id": "unsupported", "question": "Unknown question?", "all_retrieved": [{"source_id": "policy", "score": .1}],
             "evidence": [], "status": "no-evidence", "baseline": fallback, "lora": fallback},
        ]}

    def test_phrase_boundaries(self):
        self.assertFalse(contains_term("Nothing is wrong", "not"))
        self.assertTrue(contains_term("Do NOT do that", "not"))

    def test_missing_citation_source_is_rejected(self):
        rows = evaluate(self.trace, self.labels)["cases"]
        self.assertTrue(rows[0]["baseline"]["citation_valid"])
        self.assertFalse(rows[0]["lora"]["citation_valid"])
        self.assertEqual(rows[0]["lora"]["guard_status"], "rejected")

    def test_term_coverage_does_not_infer_meaning(self):
        rows = evaluate(self.trace, self.labels)["cases"]
        self.assertEqual(rows[0]["baseline"]["term_coverage"], 0)
        self.assertEqual(rows[0]["lora"]["term_coverage"], 1)
        self.assertIsNone(rows[1]["lora"]["term_coverage"])

    def test_fallback_does_not_become_model_generation(self):
        result = evaluate(self.trace, self.labels)
        self.assertFalse(result["cases"][1]["baseline"]["generated"])
        self.assertTrue(result["cases"][1]["baseline"]["abstained"])
        self.assertEqual(result["summary"]["systems"]["baseline"]["mean_generated_tokens"], 8)

    def test_duplicate_trace_cases_fail(self):
        trace = copy.deepcopy(self.trace)
        trace["cases"].append(trace["cases"][0])
        with self.assertRaises(ValueError):
            evaluate(trace, self.labels)

    def test_modified_answer_cannot_reuse_judgment(self):
        reviews = {"cases": [{"id": label["id"], "baseline": {"answer_sha256": "stale"}, "lora": {"answer_sha256": "stale"}} for label in self.labels]}
        with self.assertRaisesRegex(ValueError, "Judgment does not match"):
            evaluate(self.trace, self.labels, reviews)


if __name__ == "__main__":
    unittest.main()
