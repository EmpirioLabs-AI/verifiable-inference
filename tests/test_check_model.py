"""Offline tests of check_model.py: a model matches its own fingerprint and another's numbers do not, a prompt that
reached the model changed never matches, near-certain questions alone can never say "matches", framing the API leaves
out is skipped by position, and an API without a list of alternatives is read from sampled answers.
Run with: python -m unittest discover -s tests"""
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


def score(fingerprint, answers):
    return cm.score(fingerprint["rows"], answers, fingerprint.get("limits") or {})


def near_certain(n=50, tail=0.0):
    """A fingerprint whose every question is near-certain (top token above 99%), and answers that move only the tail."""
    rows = [{"question": "q%d" % i, "prompt_tokens": 30,
             "top": [["A%d" % i, -0.001]] + [["t%d_%d" % (i, j), -8.0 - j] for j in range(19)]} for i in range(n)]
    answers = [([["A%d" % i, -0.001]] + [["t%d_%d" % (i, j), -8.0 - j - tail] for j in range(19)], 30) for i in range(n)]
    return {"rows": rows, "limits": {"typical_gap": 0.15, "first_token": 0.8}, "system": "s"}, answers


class Verdict(unittest.TestCase):
    def test_every_fingerprint_matches_its_own_numbers(self):
        for name in FINGERPRINTS:
            fingerprint = load(name)
            verdict = score(fingerprint, answers_from(fingerprint))
            self.assertEqual(verdict["result"], "matches", name)
            self.assertEqual(verdict["typical_gap"], 0.0, name)

    def test_another_models_numbers_do_not_match(self):
        for first in FINGERPRINTS:
            for second in FINGERPRINTS:
                if first == second:
                    continue
                a, b = load(first), load(second)
                if [r["question"] for r in a["rows"]] != [r["question"] for r in b["rows"]]:
                    continue
                answers = [(row["top"], ra["prompt_tokens"]) for row, ra in zip(b["rows"], a["rows"])]
                self.assertEqual(score(a, answers)["result"], "DOES NOT MATCH", (first, second))

    def test_a_small_constant_template_difference_is_allowed_and_a_longer_prompt_is_not(self):
        fingerprint = load(FINGERPRINTS[0])
        self.assertEqual(score(fingerprint, answers_from(fingerprint, offset=1))["result"], "matches")
        self.assertEqual(score(fingerprint, answers_from(fingerprint, offset=cm.PROMPT_SLACK + 1))["result"], "DOES NOT MATCH")
        uneven = answers_from(fingerprint)
        uneven[0] = (uneven[0][0], uneven[0][1] + 1)
        self.assertFalse(score(fingerprint, uneven)["prompt_as_sent"])

    def test_numbers_moved_past_the_limit_do_not_match(self):
        fingerprint = load(FINGERPRINTS[0])
        limit = fingerprint["limits"]["typical_gap"]
        moved = [([[token, value - 2 * limit] for token, value in top], length) for top, length in answers_from(fingerprint)]
        self.assertEqual(score(fingerprint, moved)["result"], "DOES NOT MATCH")

    def test_missing_or_empty_answers_never_match(self):
        fingerprint = load(FINGERPRINTS[0])
        self.assertEqual(score(fingerprint, answers_from(fingerprint)[:-1])["result"], "DOES NOT MATCH")
        empty = [([], length) for _, length in answers_from(fingerprint)]
        self.assertEqual(score(fingerprint, empty)["result"], "DOES NOT MATCH")

    def test_near_certain_questions_alone_cannot_say_matches(self):
        fingerprint, answers = near_certain(tail=3.0)
        # Every first token agrees and the weighted gap is tiny, yet the tail moved by 3: without a tail limit the
        # check cannot tell, and with one it does not match.
        self.assertEqual(score(fingerprint, answers)["result"], "CANNOT TELL")
        fingerprint["limits"]["tail_gap"] = 0.5
        self.assertEqual(score(fingerprint, answers)["result"], "DOES NOT MATCH")
        same, answers = near_certain(tail=0.0)
        same["limits"]["tail_gap"] = 0.5
        self.assertEqual(score(same, answers)["result"], "matches")


class FakeAPI:
    """The live API as check_receipt.post returns it: (request bytes, response bytes)."""

    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def __call__(self, path, body, key, timeout=120):
        self.sent.append(body)
        data = self.reply(body)
        return json.dumps(body).encode(), json.dumps(data).encode()


def answer(entries, prompt_tokens=30):
    return {"choices": [{"index": 0, "logprobs": {"content": entries}}], "usage": {"prompt_tokens": prompt_tokens}}


def entry(token, logprob, alternatives=None):
    return {"token": token, "logprob": logprob, "top_logprobs": alternatives if alternatives is not None else
            [{"token": token, "logprob": logprob}] + [{"token": "x%d" % i, "logprob": -9.0 - i} for i in range(19)]}


class Asking(unittest.TestCase):
    def setUp(self):
        self.real = cm.check_receipt.post

    def tearDown(self):
        cm.check_receipt.post = self.real

    def test_framing_the_api_leaves_out_is_skipped_by_asking_for_more_tokens(self):
        api = FakeAPI(lambda body: answer([] if body["max_tokens"] == 1 else [entry("Name", -0.001)]))
        cm.check_receipt.post = api
        asker = cm.Asker("m", {"system": "s"}, "key", None)
        top, length = asker.top("q")
        self.assertEqual(top[0], ["Name", -0.001])
        self.assertEqual([b["max_tokens"] for b in api.sent], [1, cm.ROOM])

    def test_an_answer_without_log_probabilities_is_an_error_not_a_verdict(self):
        cm.check_receipt.post = FakeAPI(lambda body: {"choices": [{"index": 0}], "usage": {}})
        with self.assertRaises(ValueError):
            cm.Asker("m", {"system": "s"}, "key", None).top("q")

    def test_an_api_without_a_list_of_alternatives_is_read_from_samples(self):
        draws = iter([("Apple", -0.1), ("Apple", -0.1), ("Apple", -0.1), ("Pear", -2.3), ("Apple", -0.1)])

        def reply(body):
            token, value = next(draws)
            return answer([entry(token, value, alternatives=[{"token": "glued", "logprob": value}])])
        api = FakeAPI(reply)
        cm.check_receipt.post = api
        rows = [{"question": "q", "prompt_tokens": 30, "top": [["Apple", -0.1], ["Pear", -2.3]]}]
        verdict, receipts, sampled = cm.run_check("m", {"system": "s", "rows": rows, "limits": {}}, "key", None)
        self.assertEqual(sampled, 1)
        # One request for the list (one alternative came back), then four samples, the first at temperature 0.
        self.assertEqual([b["temperature"] for b in api.sent], [1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(verdict["same_first"], 1)

    def test_every_answer_receipt_is_checked_when_keys_are_given(self):
        cm.check_receipt.post = FakeAPI(lambda body: answer([entry("A", -0.1)]))
        asker = cm.Asker("m", {"system": "s"}, "key", [])
        asker.top("q")
        self.assertEqual(asker.receipts, [(False, "the response carries no receipt")])

    def test_the_request_is_the_documented_one_and_a_file_can_add_settings(self):
        body = cm.request_body("m", {"system": "s", "request": {"reasoning_effort": "low"}}, "q")
        self.assertEqual(body["messages"], [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
        self.assertEqual((body["max_tokens"], body["logprobs"], body["top_logprobs"], body["reasoning_effort"]),
                         (1, True, cm.TOP, "low"))


if __name__ == "__main__":
    unittest.main()
