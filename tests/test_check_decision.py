"""Offline tests of check_decision.py, with the API replaced by a table: the model's own probabilities match, a
little noise still matches, another model's probabilities, probabilities moved past the limit, other answers or a
missing answer never do, an API that does not answer gives no verdict, the request sent is the fingerprint's own,
a caller can send the requests through its own post, check_model.py runs this check for a decision fingerprint,
and the check needs only the standard library.
Run with: python -m unittest discover -s tests"""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_decision as cd  # noqa: E402
import check_model  # noqa: E402

FINGERPRINTS = os.path.join(ROOT, "fingerprints")
REAL = os.path.join(FINGERPRINTS, "aplomb-1.json")
# The answers another public checkpoint (Qwen/Qwen3.5-4B, the base aplomb-1 was trained from) gives to the same
# requests through aplomb-1's published reference code: measured once, for the control that another model does not
# match.
OTHER = os.path.join(ROOT, "tests", "testdata", "aplomb-1-base.answers.json")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


FILES = [os.path.join(FINGERPRINTS, name) for name in sorted(os.listdir(FINGERPRINTS))
         if name.endswith(".json") and load(os.path.join(FINGERPRINTS, name)).get("kind") == cd.KIND]


def key_of(state):
    return json.dumps(state, sort_keys=True)


def run(change, path=REAL, check=None):
    """check_one() with every request answered by `change(the fingerprint's own answers for it)`; (exit, output)."""
    by_state = dict((key_of(r["state"]), r["answers"]) for r in load(path)["requests"])

    def ask(model, body, key, post=None):
        return change(json.loads(json.dumps(by_state[key_of(body["state"])])))
    out = io.StringIO()
    with mock.patch.object(cd, "ask", ask), contextlib.redirect_stdout(out):
        code = (check or cd.check_one)("aplomb-1", path, "not-a-key")
    return code, out.getvalue()


def shift(answers, delta):
    """Every probability moved by `delta` (clamped to [0, 1]): what a slightly different computation looks like."""
    def move(node):
        if isinstance(node, dict):
            return {k: (min(1.0, max(0.0, v + delta)) if k in ("noul", "abstain", "probability_true") and isinstance(v, float)
                        else {kk: min(1.0, max(0.0, vv + delta)) for kk, vv in v.items()} if k == "probabilities"
                        else move(v)) for k, v in node.items()}
        return node
    return move(answers)


class Fingerprint(unittest.TestCase):
    def test_the_decision_fingerprints_are_aplomb_1(self):
        self.assertEqual([os.path.basename(f)[:-len(".json")] for f in FILES], ["aplomb-1"])

    def test_it_names_the_public_checkpoint_and_asks_the_requests_make_writes(self):
        data = load(REAL)
        self.assertEqual((data["kind"], data["model"]), (cd.KIND, "aplomb-1"))
        self.assertEqual(data["weights"], "empiriolabsai/aplomb-1")
        self.assertEqual(len(data["revision"]), 40)
        self.assertTrue(data["weight_files_sha256"] and all(len(v) == 64 for v in data["weight_files_sha256"].values()))
        self.assertEqual([r["state"] for r in data["requests"]], [r["state"] for r in cd.REQUESTS])
        self.assertEqual([r["questions"] for r in data["requests"]], [r["questions"] for r in cd.REQUESTS])
        self.assertTrue(all(set(r["answers"]) == set(q["questions"]) for r, q in zip(data["requests"], cd.REQUESTS)))
        self.assertGreaterEqual(sum(len(r["answers"]) for r in data["requests"]), 40)
        self.assertTrue(0.8 <= data["limits"]["same_answer"] <= 1 and 0 < data["limits"]["typical_gap"] <= 0.05)


class Verdict(unittest.TestCase):
    def test_its_own_numbers_match(self):
        for path in FILES:
            code, out = run(lambda answers: answers, path)
            self.assertEqual(code, 0, out)
            self.assertIn(": matches", out)

    def test_the_same_numbers_with_a_little_noise_match(self):
        limit = load(REAL)["limits"]["typical_gap"]
        code, out = run(lambda answers: shift(answers, limit / 4))
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_another_models_answers_do_not_match(self):
        other = load(OTHER)["answers"]
        by_state = dict((key_of(body["state"]), a) for body, a in zip(cd.REQUESTS, other))
        out = io.StringIO()
        with mock.patch.object(cd, "ask", lambda model, body, key, post=None: by_state[key_of(body["state"])]), \
                contextlib.redirect_stdout(out):
            code = cd.check_one("aplomb-1", REAL, "not-a-key")
        self.assertEqual(code, 1, out.getvalue())
        self.assertIn("DOES NOT MATCH", out.getvalue())

    def test_every_probability_moved_past_the_limit_does_not_match(self):
        limit = load(REAL)["limits"]["typical_gap"]
        code, out = run(lambda answers: shift(answers, 3 * limit))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_another_answer_on_many_questions_does_not_match(self):
        def flip(answers):
            for a in answers.values():
                if a.get("type") == "noul":
                    a["noul"] = round(1.0 - a["noul"], 8)
            return answers
        code, out = run(flip)
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_a_missing_answer_counts_against_the_model(self):
        code, out = run(lambda answers: {})
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_check_model_runs_this_check_for_a_decision_fingerprint_and_fails_no_receipt(self):
        code, out = run(lambda answers: answers, check=lambda model, path, key: check_model.check_one(model, path, key, []))
        self.assertEqual(code, 0, out)
        self.assertIn("aplomb-1 against the fingerprint of empiriolabsai/aplomb-1: matches", out)
        self.assertIn("  receipts: not on this endpoint", out)


class Asking(unittest.TestCase):
    def test_an_api_that_does_not_answer_gives_no_verdict(self):
        def fail(model, body, key, post=None):
            raise cd.Unanswered("HTTP 503: try again")
        out = io.StringIO()
        with mock.patch.object(cd, "ask", fail), contextlib.redirect_stdout(out):
            code = cd.check_one("aplomb-1", REAL, "not-a-key")
        self.assertEqual(code, 1)
        self.assertIn("could not be checked", out.getvalue())
        self.assertNotIn("MATCH", out.getvalue())

    def test_the_request_sent_is_the_model_the_state_and_the_questions(self):
        sent = []

        def post(path, body, key, timeout=300):
            sent.append((path, body))
            return json.dumps(body).encode(), json.dumps({"answers": {"q": {"type": "noul", "noul": 0.9}}}).encode()
        body = load(REAL)["requests"][0]
        with mock.patch.object(cd.check_receipt, "post", post):
            answers = cd.ask("aplomb-1", body, "not-a-key")
        self.assertEqual(answers, {"q": {"type": "noul", "noul": 0.9}})
        self.assertEqual(sent, [("/decisions", {"model": "aplomb-1", "state": body["state"],
                                                "questions": body["questions"]})])

    def test_a_caller_can_send_through_its_own_post(self):
        fingerprint = load(REAL)
        by_state = dict((key_of(r["state"]), r["answers"]) for r in fingerprint["requests"])
        sent = []

        def post(body, key):
            sent.append(body)
            return json.dumps(body).encode(), json.dumps({"answers": by_state[key_of(body["state"])]}).encode()

        def unused(*args, **kwargs):
            raise AssertionError("the default post was used")
        with mock.patch.object(cd.check_receipt, "post", unused):
            v, live = cd.run_check("aplomb-1", fingerprint, "not-a-key", post=post)
        self.assertEqual(v["result"], "matches")
        self.assertEqual(sent, [{"model": "aplomb-1", "state": r["state"], "questions": r["questions"]}
                                for r in fingerprint["requests"]])
        # An answer the caller's post returns that is not an answer is still no verdict.
        with self.assertRaises(cd.Unanswered):
            cd.ask("aplomb-1", fingerprint["requests"][0], "not-a-key", post=lambda body, key: (b"{}", b"<html>"))

    def test_an_error_of_the_callers_own_post_reaches_the_caller(self):
        class Stop(Exception):
            pass

        def post(body, key):
            raise Stop("no time left")

        def unused(*args, **kwargs):
            raise AssertionError("the default post was used")
        with mock.patch.object(cd.check_receipt, "post", unused), self.assertRaises(Stop):
            cd.run_check("aplomb-1", load(REAL), "not-a-key", post=post)

    def test_the_command_line_check_writes_the_answers(self):
        by_state = dict((key_of(r["state"]), r["answers"]) for r in load(REAL)["requests"])
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "answers.json")
            with mock.patch.object(cd, "ask", lambda model, body, key, post=None: by_state[key_of(body["state"])]), \
                    mock.patch.dict(os.environ, {"EMPIRIOLABS_API_KEY": "k"}), contextlib.redirect_stdout(io.StringIO()):
                code = cd.main(["check", "--model", "aplomb-1", "--out", target])
            written = load(target)
        self.assertEqual(code, 0)
        self.assertEqual((written["result"], len(written["answers"])), ("matches", len(by_state)))


class StandardLibrary(unittest.TestCase):
    def test_the_check_needs_only_the_standard_library(self):
        names_of = getattr(sys, "stdlib_module_names", None)
        if names_of is None:
            self.skipTest("Python 3.10 or later lists the standard library")
        with open(os.path.join(ROOT, "check_decision.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
        self.assertTrue(names)
        self.assertLessEqual(names, set(names_of) | {"check_receipt"}, names - set(names_of))


if __name__ == "__main__":
    unittest.main()
