"""Offline tests of check_model.py's verdict: a model matches its own fingerprint, another model's numbers do not, and
a prompt that reached the model changed never matches. Run with: python -m unittest discover -s tests"""
import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_model as cm  # noqa: E402

FINGERPRINTS = sorted(name for name in os.listdir(os.path.join(ROOT, "fingerprints")) if name.endswith(".json"))


def load(name):
    with open(os.path.join(ROOT, "fingerprints", name), encoding="utf-8") as handle:
        return json.load(handle)


def answers_from(fingerprint, offset=0):
    """What the API would return if it served exactly the fingerprinted weights."""
    return [(copy.deepcopy(row["top"]), row["prompt_tokens"] + offset) for row in fingerprint["rows"]]


class Verdict(unittest.TestCase):
    def test_every_fingerprint_matches_its_own_numbers(self):
        for name in FINGERPRINTS:
            fingerprint = load(name)
            verdict = cm.compare(fingerprint, answers_from(fingerprint))
            self.assertEqual(verdict["result"], "matches", name)
            self.assertEqual(verdict["typical_gap"], 0.0, name)

    def test_another_models_numbers_do_not_match(self):
        # Two fingerprints of the same questions from different models: each model's answers fail the other's file.
        for first in FINGERPRINTS:
            for second in FINGERPRINTS:
                if first == second:
                    continue
                a, b = load(first), load(second)
                if [r["question"] for r in a["rows"]] != [r["question"] for r in b["rows"]]:
                    continue
                answers = [(row["top"], ra["prompt_tokens"]) for row, ra in zip(b["rows"], a["rows"])]
                self.assertEqual(cm.compare(a, answers)["result"], "DOES NOT MATCH", (first, second))

    def test_a_small_constant_template_difference_is_allowed_and_a_longer_prompt_is_not(self):
        fingerprint = load(FINGERPRINTS[0])
        self.assertEqual(cm.compare(fingerprint, answers_from(fingerprint, offset=1))["result"], "matches")
        self.assertEqual(cm.compare(fingerprint, answers_from(fingerprint, offset=cm.PROMPT_SLACK + 1))["result"],
                         "DOES NOT MATCH")
        # Lengths that differ from question to question mean the prompts were not the ones sent.
        uneven = answers_from(fingerprint)
        uneven[0] = (uneven[0][0], uneven[0][1] + 1)
        self.assertFalse(cm.compare(fingerprint, uneven)["prompt_as_sent"])

    def test_numbers_moved_past_the_limit_do_not_match(self):
        fingerprint = load(FINGERPRINTS[0])
        limit = fingerprint["limits"]["typical_gap"]
        moved = [([[token, value - 2 * limit] for token, value in top], length) for top, length in answers_from(fingerprint)]
        self.assertEqual(cm.compare(fingerprint, moved)["result"], "DOES NOT MATCH")

    def test_answers_with_nothing_in_common_or_missing_never_match(self):
        fingerprint = load(FINGERPRINTS[0])
        strangers = [([["zz%d" % i, -1.0]], length) for i, (_, length) in enumerate(answers_from(fingerprint))]
        self.assertEqual(cm.compare(fingerprint, strangers)["result"], "DOES NOT MATCH")
        self.assertEqual(cm.compare(fingerprint, answers_from(fingerprint)[:-1])["result"], "DOES NOT MATCH")


if __name__ == "__main__":
    unittest.main()
