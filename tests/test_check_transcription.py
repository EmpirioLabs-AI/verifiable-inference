"""Offline tests of check_transcription.py, with the API replaced by a table: the model's own numbers match, a little
noise still matches, another real model's numbers, another transcript or a missing field never do, the check reads
nothing but the numbers it compares, a job answer is followed to its end, an API that does not answer gives no
verdict, a changed clip is never sent, check_model.py runs this check for a transcription fingerprint, and the check
needs only the standard library. Run with: python -m unittest discover -s tests"""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_model  # noqa: E402
import check_transcription as ct  # noqa: E402

FINGERPRINTS = os.path.join(ROOT, "fingerprints")
REAL = os.path.join(FINGERPRINTS, "whisper-large-v3-turbo.json")
# The answers another public checkpoint (openai/whisper-large-v3) gives for the same clips with the same settings,
# every language: measured once, for the control that another model does not match.
OTHER = os.path.join(ROOT, "tests", "testdata", "whisper-large-v3.answers.json")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


FILES = [os.path.join(FINGERPRINTS, name) for name in sorted(os.listdir(FINGERPRINTS))
         if name.endswith(".json") and load(os.path.join(FINGERPRINTS, name)).get("kind") == ct.KIND]


def api_answer(row, language_shift=0.0, logprob_shift=0.0, tokens=None):
    """A transcription answer as the API gives it, built from a fingerprint row (or a measured answer)."""
    return {"text": row.get("text", ""), "language": row["language"],
            "language_probability": math.exp(row["languages"][0][1]),
            "segments": [{"id": 1, "tokens": row["tokens"] if tokens is None else tokens,
                          "avg_logprob": row["avg_logprob"] + logprob_shift, "no_speech_prob": row.get("no_speech_prob", 0.0)}],
            "info": {"all_language_probs": [[code, math.exp(lp + language_shift)] for code, lp in row["languages"]]}}


def run(answer, path=REAL, check=None):
    """check_one() with every clip answered by `answer(the fingerprint's row for it)`; (exit status, output)."""
    rows = dict((row["sha256"], row) for row in load(path)["clips"])

    def ask(model, clip, suffix, key, settings):
        return answer(rows[hashlib.sha256(clip).hexdigest()])
    out = io.StringIO()
    with mock.patch.object(ct, "ask", ask), contextlib.redirect_stdout(out):
        code = (check or ct.check_one)("whisper-large-v3-turbo", path, "not-a-key")
    return code, out.getvalue()


class Fingerprint(unittest.TestCase):
    def test_the_transcription_fingerprints_are_whisper_large_v3_turbo(self):
        self.assertEqual([os.path.basename(f)[:-len(".json")] for f in FILES], ["whisper-large-v3-turbo"])

    def test_it_names_the_public_checkpoint_and_the_settings_the_check_sends(self):
        data = load(REAL)
        self.assertEqual((data["kind"], data["model"]), (ct.KIND, "whisper-large-v3-turbo"))
        self.assertEqual(data["weights"], "openai/whisper-large-v3-turbo")
        self.assertEqual(len(data["revision"]), 40)
        self.assertTrue(data["weight_files_sha256"] and all(len(v) == 64 for v in data["weight_files_sha256"].values()))
        self.assertEqual(data["request"], ct.REQUEST)
        self.assertGreaterEqual(len(data["clips"]), 10)
        self.assertTrue(all(len(clip["languages"]) == ct.TOP_LANGUAGES for clip in data["clips"]))


class Verdict(unittest.TestCase):
    def test_its_own_numbers_match(self):
        for path in FILES:
            code, out = run(api_answer, path)
            self.assertEqual(code, 0, out)
            self.assertIn(": matches", out)

    def test_the_same_numbers_with_a_little_noise_match(self):
        limits = load(REAL)["limits"]
        code, out = run(lambda row: api_answer(row, language_shift=limits["language_gap"] / 4,
                                               logprob_shift=limits["logprob_gap"] / 4))
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_another_models_answers_do_not_match(self):
        other = dict((row["sha256"], row) for row in load(OTHER)["clips"])
        code, out = run(lambda row: api_answer(other[row["sha256"]]))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_another_transcript_on_every_clip_does_not_match(self):
        code, out = run(lambda row: api_answer(row, tokens=row["tokens"][:-1] + [row["tokens"][-1] + 1]))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_an_answer_without_language_probabilities_does_not_match(self):
        def bare(row):
            answer = api_answer(row)
            del answer["info"]
            return answer
        code, out = run(bare)
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_the_check_reads_only_the_language_probabilities_tokens_and_average_log_probability(self):
        # Fields beside those numbers (timings, the options in `info`, the text) may be added, changed or removed:
        # the verdict stays the same. Every language's probability may also come at the top level, without `info`.
        def extra(row):
            answer = api_answer(row)
            answer["timings"] = {"total_seconds": 1.0}
            answer["info"].update({"transcription_options": {"beam_size": 1}, "vad_options": {"threshold": 0.5},
                                   "duration": 6.0})
            return answer

        def stripped(row):
            answer = api_answer(row)
            answer["all_language_probs"] = answer.pop("info")["all_language_probs"]
            del answer["text"], answer["language_probability"]
            answer["segments"][0] = {"tokens": answer["segments"][0]["tokens"],
                                     "avg_logprob": answer["segments"][0]["avg_logprob"]}
            return answer
        for answer in (extra, stripped):
            code, out = run(answer)
            self.assertEqual(code, 0, (answer.__name__, out))
            self.assertIn(": matches", out)

    def test_check_model_runs_this_check_for_a_transcription_fingerprint_and_fails_no_receipt(self):
        code, out = run(api_answer, check=lambda model, path, key: check_model.check_one(model, path, key, []))
        self.assertEqual(code, 0, out)
        self.assertIn("whisper-large-v3-turbo against the fingerprint of openai/whisper-large-v3-turbo: matches", out)
        self.assertIn("  receipts: not on this endpoint", out)


class Asking(unittest.TestCase):
    def test_a_job_answer_is_followed_to_its_end(self):
        calls = []

        def call(method, path, key, body=None):
            calls.append((method, path))
            if method == "POST":
                return 202, {"job_id": "job_1", "status": "processing", "poll_url": "/v1/jobs/job_1"}
            if len(calls) < 3:
                return 200, {"job_id": "job_1", "status": "processing"}
            return 200, {"job_id": "job_1", "status": "completed", "result": {"text": "done"}}
        with mock.patch.object(ct, "_call", call), mock.patch.object(ct.time, "sleep", lambda seconds: None):
            self.assertEqual(ct.ask("m", b"fLaC", ".flac", "k"), {"text": "done"})
        self.assertEqual(calls, [("POST", "/v1/audio/transcriptions"), ("GET", "/v1/jobs/job_1"), ("GET", "/v1/jobs/job_1")])

    def test_a_job_is_polled_on_the_apis_own_origin_whatever_its_poll_url_says(self):
        calls = []

        def call(method, path, key, body=None):
            calls.append((method, path))
            if method == "POST":
                return 202, {"job_id": "job/2", "status": "processing", "poll_url": "@elsewhere.example/v1/jobs/job_2"}
            return 200, {"job_id": "job/2", "status": "completed", "result": {"text": "done"}}
        with mock.patch.object(ct, "_call", call), mock.patch.object(ct.time, "sleep", lambda seconds: None):
            self.assertEqual(ct.ask("m", b"fLaC", ".flac", "k"), {"text": "done"})
        self.assertEqual(calls, [("POST", "/v1/audio/transcriptions"), ("GET", "/v1/jobs/job%2F2")])

    def test_the_request_is_sent_to_the_public_api(self):
        self.assertEqual(ct.ORIGIN, "https://api.empiriolabs.ai")

    def test_an_api_that_does_not_answer_gives_no_verdict(self):
        def fail(model, clip, suffix, key, settings):
            raise ct.Unanswered("HTTP 503: try again")
        out = io.StringIO()
        with mock.patch.object(ct, "ask", fail), contextlib.redirect_stdout(out):
            code = ct.check_one("whisper-large-v3-turbo", REAL, "not-a-key")
        self.assertEqual(code, 1)
        self.assertIn("could not be checked", out.getvalue())
        self.assertNotIn("MATCH", out.getvalue())

    def test_a_clip_that_was_changed_is_refused_and_no_clip_is_sent(self):
        data = load(REAL)
        with tempfile.TemporaryDirectory() as folder:
            for clip in data["clips"]:
                target = os.path.join(folder, clip["file"])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(FINGERPRINTS, clip["file"]), target)
            first = os.path.join(folder, data["clips"][0]["file"])
            with open(first, "ab") as handle:
                handle.write(b"\0")
            copy = os.path.join(folder, "copy.json")
            with open(copy, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            sent = []
            out = io.StringIO()
            with mock.patch.object(ct, "ask", lambda *a: sent.append(a)), contextlib.redirect_stdout(out):
                code = ct.check_one("whisper-large-v3-turbo", copy, "not-a-key")
        self.assertEqual((code, sent), (1, []))
        self.assertIn("not the clip", out.getvalue())
        self.assertNotIn("MATCH", out.getvalue())

    def test_the_command_line_check_writes_the_numbers_it_read(self):
        rows = dict((row["sha256"], row) for row in load(REAL)["clips"])
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "answers.json")
            with mock.patch.object(ct, "ask", lambda model, clip, suffix, key, settings: api_answer(
                    rows[hashlib.sha256(clip).hexdigest()])), mock.patch.dict(os.environ, {"EMPIRIOLABS_API_KEY": "k"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = ct.main(["check", "--model", "whisper-large-v3-turbo", "--out", target])
            written = load(target)
        self.assertEqual(code, 0)
        self.assertEqual((written["result"], len(written["per_clip"]), len(written["answers"])),
                         ("matches", len(rows), len(rows)))


class StandardLibrary(unittest.TestCase):
    def test_the_check_needs_only_the_standard_library(self):
        names_of = getattr(sys, "stdlib_module_names", None)
        if names_of is None:
            self.skipTest("Python 3.10 or later lists the standard library")
        with open(os.path.join(ROOT, "check_transcription.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
        self.assertTrue(names)
        self.assertLessEqual(names, set(names_of) | {"check_receipt"}, names - set(names_of))


if __name__ == "__main__":
    unittest.main()
