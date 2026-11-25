"""Offline tests of check_ocr.py, with the API replaced by a table: the model's own reading matches, the same reading
with boxes a few units off still matches, another reading (other text, other blocks, other boxes) does not, the edit
distance is Levenshtein's, the model's grounding text becomes the API's blocks, a job answer is followed to its end on
the API's own origin, a changed page is never sent, an API that does not answer gives no verdict, check_model.py runs
this check for an OCR fingerprint and checks every receipt, and the check needs only the standard library.
Run with: python -m unittest discover -s tests"""
import ast
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_model  # noqa: E402
import check_ocr as co  # noqa: E402
from test_check_receipt import KEYS, receipt_for  # noqa: E402  (the test signer, never a key of the API)

FINGERPRINTS = os.path.join(ROOT, "fingerprints")
MODEL = "ocr-test-model"
# Grounding text as the model card shows it: each block a placeholder line with its label and box, then its text.
OUTPUTS = [
    "![title](70,52,700,95)\n# Return of the Whole Number of Persons\n\n"
    "![text](70,160,930,250)\nThe first census of the United States counted the inhabitants of each district.\n\n"
    "![table](70,280,930,770)\n<table><tr><th>State</th><th>Total</th></tr><tr><td>Connecticut</td>"
    "<td>237,946</td></tr><tr><td>Delaware</td><td>59,094</td></tr></table>\n\n"
    "![page_number](480,930,520,960)\n3",
    "![title](350,70,660,120)\n# The Bill of Rights\n\n"
    "![list](80,250,930,860)\n1. Congress shall make no law respecting an establishment of religion.\n"
    "2. A well regulated Militia, being necessary to the security of a free State.\n\n"
    "![text](70,870,400,900)\nRatified December 15, 1791.",
    "![header](64,30,470,60)\nTHE AUTOBIOGRAPHY OF BENJAMIN FRANKLIN\n\n"
    "![section](66,110,380,160)\n## Names of Virtues\n\n"
    "![list](80,250,930,500)\n- Temperance. Eat not to Dullness. Drink not to Elevation.\n"
    "![list+](80,500,930,640)\n- Silence. Speak not but what may benefit others or yourself.\n\n"
    "![image](700,700,930,900)\nA small engraved portrait.\n\n"
    "![page_number](480,940,520,975)\n103",
    "![title](70,80,400,130)\n# The Planets\n\n"
    "![formula](440,260,580,300)\n$$T^2 = a^3$$\n\n"
    "![table](70,430,930,850)\n<table><tr><th>Planet</th><th>Period (years)</th></tr><tr><td>Mercury</td>"
    "<td>0.241</td></tr><tr><td>Venus</td><td>0.615</td></tr></table>\n\n"
    "![caption](70,880,520,910)\nTable 1. Mean distances and periods of the planets.",
]
SIZES = [(1024, 992), (1024, 768), (1024, 928), (1024, 864)]


def api_page(output, width, height, shift=0, text=None):
    """One page of the API's answer for the model's grounding text, as the endpoint gives it: each block's box in
    pixels of the page (rounded), its type, its content. `shift` moves every box; `text` rewrites each content."""
    blocks = []
    for block in co.parse_blocks(output):
        x1, y1, x2, y2 = (min(1000, max(0, v + shift)) for v in block["box"])
        blocks.append({"type": block["type"], "top_left_x": round(x1 * width / 1000), "top_left_y": round(y1 * height / 1000),
                       "bottom_right_x": round(x2 * width / 1000), "bottom_right_y": round(y2 * height / 1000),
                       "content": text(block["content"]) if text else block["content"], "confidence_scores": None})
    return {"index": 0, "markdown": "\n\n".join(b["content"] for b in blocks), "images": [], "tables": [],
            "hyperlinks": [], "header": None, "footer": None,
            "dimensions": {"dpi": 72, "height": height, "width": width}, "confidence_scores": None, "blocks": blocks}


class Fixture:
    """A fingerprint of four pages in a temporary folder, each page some bytes and the model's text for it."""

    def __init__(self, folder):
        self.folder = folder
        os.makedirs(os.path.join(folder, MODEL))
        rows = []
        for number, (output, (width, height)) in enumerate(zip(OUTPUTS, SIZES), 1):
            data = b"\x89PNG\r\n\x1a\n page %d" % number
            name = "page-%d.png" % number
            with open(os.path.join(folder, MODEL, name), "wb") as handle:
                handle.write(data)
            rows.append({"file": "%s/%s" % (MODEL, name), "sha256": hashlib.sha256(data).hexdigest(),
                         "source": "https://example.org/page-%d" % number, "license": "Public domain",
                         "processing": "rendered", "width": width, "height": height, "output": output})
        self.fingerprint = {"kind": co.KIND, "model": MODEL, "weights": "example/ocr-model", "revision": "0" * 40,
                            "weight_files_sha256": {"model.safetensors": "1" * 64},
                            "made_with": "plain transformers code on cpu", "made_at": "2026-11-24T00:00:00Z",
                            "request": dict(co.REQUEST),
                            "limits": {"page_cer": 0.02, "typical_cer": 0.01, "box_gap": 4.0, "close": 0.75},
                            "pages": rows}
        self.path = os.path.join(folder, MODEL + ".json")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.fingerprint, handle)
        self.by_hash = {row["sha256"]: row for row in rows}

    def answer(self, page_bytes, **changes):
        row = self.by_hash[hashlib.sha256(page_bytes).hexdigest()]
        return api_page(row["output"], row["width"], row["height"], **changes)


def run(fixture, answer, check=None, signed=True):
    """check_one() with every page answered by `answer(page bytes)`, each answer signed unless `signed` is false;
    (exit status, output)."""

    def ask(model, page_bytes, key, request=None, exchanges=None, send=None):
        given = answer(page_bytes)
        if exchanges is not None:
            sent = json.dumps({"model": model, "page": hashlib.sha256(page_bytes).hexdigest(), **(request or {})}).encode()
            received = json.dumps({"pages": [given], "model": model}).encode()
            if signed:
                receipt = json.dumps(receipt_for(sent, received, model=model), separators=(",", ":")).encode()
                received = received[:-1] + b',"receipt":' + receipt + b"}"
            exchanges.append((sent, received))
        return given
    out = io.StringIO()
    with mock.patch.object(co, "ask", ask), contextlib.redirect_stdout(out):
        code = (check or co.check_one)(MODEL, fixture.path, "not-a-key")
    return code, out.getvalue()


class EditDistance(unittest.TestCase):
    @staticmethod
    def table(a, b):
        previous = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            current = [i]
            for j, y in enumerate(b, 1):
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
            previous = current
        return previous[-1]

    def test_it_is_levenshteins_distance(self):
        rng = random.Random(7)
        alphabet = "ab cdéé中,."
        for _ in range(400):
            a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            self.assertEqual(co.edit_distance(a, b), self.table(a, b), (a, b))
        self.assertEqual(co.edit_distance("", ""), 0)
        self.assertEqual(co.edit_distance("kitten", "sitting"), 3)

    def test_a_long_page_takes_well_under_a_second(self):
        import time

        text = " ".join("word%d" % (i % 97) for i in range(900))  # about 6,000 characters
        changed = text.replace("word13", "w0rd13")
        started = time.time()
        self.assertEqual(co.edit_distance(text, changed), text.count("word13"))
        self.assertLess(time.time() - started, 2.0)

    def test_the_error_rate_counts_against_the_reference_after_whitespace_is_normalized(self):
        self.assertEqual(co.character_error_rate(co.normalize("a  b\n\nc"), co.normalize(" a b c ")), 0.0)
        self.assertAlmostEqual(co.character_error_rate("abcd", "abed"), 0.25)
        self.assertEqual(co.character_error_rate("", ""), 0.0)


class Blocks(unittest.TestCase):
    def test_the_grounding_text_becomes_typed_boxed_blocks(self):
        blocks = co.parse_blocks(OUTPUTS[2])
        self.assertEqual([b["type"] for b in blocks], ["header", "title", "list", "image", "footer"])
        self.assertEqual(blocks[2]["box"], (80, 250, 930, 640), "a continuation joins its block, boxes united")
        self.assertIn("Silence.", blocks[2]["content"])
        self.assertEqual(blocks[3]["content"], "A small engraved portrait.")

    def test_a_page_number_is_a_header_at_the_top_and_a_footer_below(self):
        self.assertEqual(co.parse_blocks("![page_number](480,20,520,40)\n7")[0]["type"], "header")
        self.assertEqual(co.parse_blocks("![page_number](480,940,520,975)\n7")[0]["type"], "footer")

    def test_text_before_any_placeholder_and_empty_blocks(self):
        blocks = co.parse_blocks("stray words\n![text](1,2,3,4)\n\n![image](5,6,7,8)\n![text](9,9,9,9)\n")
        # The text before any placeholder covers the page; a text block with no text is dropped, an image is kept.
        self.assertEqual([(b["type"], b["box"]) for b in blocks], [("text", (0, 0, 1000, 1000)), ("image", (5, 6, 7, 8))])

    def test_the_answers_pixel_boxes_come_back_to_the_models_scale(self):
        page = api_page(OUTPUTS[0], 1024, 992)
        live = co.answer_blocks(page)
        for ref, got in zip(co.parse_blocks(OUTPUTS[0]), live):
            self.assertEqual(ref["type"], got["type"])
            self.assertTrue(all(abs(a - b) < 1.0 for a, b in zip(ref["box"], got["box"])))
        self.assertIsNone(co.answer_blocks({"dimensions": {"width": 1, "height": 1}}))
        self.assertIsNone(co.answer_blocks({"blocks": []}))


class Verdict(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.fixture = Fixture(self.folder)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_its_own_reading_matches(self):
        code, out = run(self.fixture, self.fixture.answer)
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_boxes_a_few_units_off_still_match(self):
        code, out = run(self.fixture, lambda page: self.fixture.answer(page, shift=2))
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_boxes_far_off_do_not_match(self):
        code, out = run(self.fixture, lambda page: self.fixture.answer(page, shift=12))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_a_typical_box_gap_limit_of_its_own_is_kept_when_the_file_states_one(self):
        # Boxes two units off are within the per-page box limit (4) but their typical gap is not within 1.
        self.fixture.fingerprint["limits"]["typical_box_gap"] = 1.0
        with open(self.fixture.path, "w", encoding="utf-8") as handle:
            json.dump(self.fixture.fingerprint, handle)
        code, out = run(self.fixture, lambda page: self.fixture.answer(page, shift=2))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)
        self.assertIn("(limit 1.00, of 1000)", out)
        code, out = run(self.fixture, self.fixture.answer)
        self.assertEqual(code, 0, out)

    def test_another_reading_of_the_text_does_not_match(self):
        code, out = run(self.fixture, lambda page: self.fixture.answer(page, text=lambda t: t.replace("e", "c")))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_other_blocks_do_not_match(self):
        def merged(page):
            answer = self.fixture.answer(page)
            answer["blocks"] = answer["blocks"][:1]
            return answer
        code, out = run(self.fixture, merged)
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_an_answer_without_blocks_does_not_match(self):
        def bare(page):
            answer = self.fixture.answer(page)
            answer["blocks"] = None
            return answer
        code, out = run(self.fixture, bare)
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_check_model_runs_this_check_for_an_ocr_fingerprint_and_checks_every_receipt(self):
        def check(model, path, key):
            return check_model.check_one(model, path, key, KEYS)
        code, out = run(self.fixture, self.fixture.answer, check=check)
        self.assertEqual(code, 0, out)
        self.assertIn("%s against the fingerprint of example/ocr-model: matches" % MODEL, out)
        self.assertIn("  receipts: every answer signed by EmpirioLabs for %s" % MODEL, out)
        code, out = run(self.fixture, self.fixture.answer, check=check, signed=False)
        self.assertEqual(code, 1, out)
        self.assertIn("  receipts: 0 of 4 answers carry a valid receipt naming %s" % MODEL, out)

    def test_a_page_that_was_changed_is_refused_and_no_page_is_sent(self):
        with open(os.path.join(self.folder, self.fixture.fingerprint["pages"][1]["file"]), "ab") as handle:
            handle.write(b"\0")
        sent = []
        out = io.StringIO()
        with mock.patch.object(co, "ask", lambda *a, **k: sent.append(a)), contextlib.redirect_stdout(out):
            code = co.check_one(MODEL, self.fixture.path, "not-a-key")
        self.assertEqual((code, sent), (1, []))
        self.assertIn("not the page", out.getvalue())
        self.assertNotIn("MATCH", out.getvalue())

    def test_an_api_that_does_not_answer_gives_no_verdict(self):
        def fail(*args, **kwargs):
            raise co.Unanswered("HTTP 503: try again")
        out = io.StringIO()
        with mock.patch.object(co, "ask", fail), contextlib.redirect_stdout(out):
            code = co.check_one(MODEL, self.fixture.path, "not-a-key")
        self.assertEqual(code, 1)
        self.assertIn("could not be checked", out.getvalue())
        self.assertNotIn("MATCH", out.getvalue())

    def test_the_command_line_check_writes_the_numbers_it_read(self):
        target = os.path.join(self.folder, "answers.json")
        with mock.patch.object(co, "ask", lambda model, page, key, request=None, exchanges=None, send=None:
                               self.fixture.answer(page)), \
                mock.patch.dict(os.environ, {"EMPIRIOLABS_API_KEY": "k"}), contextlib.redirect_stdout(io.StringIO()):
            code = co.main(["check", "--model", MODEL, "--fingerprint", self.fixture.path, "--out", target,
                            "--keys", "none"])
        with open(target, encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertEqual(code, 0)
        self.assertEqual((written["result"], len(written["per_page"]), len(written["answers"])), ("matches", 4, 4))


class Asking(unittest.TestCase):
    def test_a_job_answer_is_followed_to_its_end_on_the_apis_own_origin(self):
        calls = []

        def send(method, path, key, body=None):
            calls.append((method, path))
            sent = json.dumps(body).encode() if body else None
            if method == "POST":
                self.assertEqual(body["document"]["type"], "image_url")
                self.assertTrue(body["document"]["image_url"].startswith("data:image/png;base64,"))
                self.assertIs(body["include_blocks"], True)
                return 202, {"job_id": "job/1", "status": "processing", "poll_url": "@elsewhere.example/x"}, sent, b"{}"
            if len(calls) < 3:
                return 200, {"status": "processing"}, sent, b"{}"  # a poll answer need not repeat the job's id
            done = json.dumps({"job_id": "job/1", "status": "completed", "result": {"pages": [{"index": 0}]}}).encode()
            return 200, json.loads(done), sent, done
        exchanges = []
        with mock.patch.object(co.time, "sleep", lambda seconds: None):
            page = co.ask("m", b"\x89PNG", "k", exchanges=exchanges, send=send)
        self.assertEqual(page, {"index": 0})
        self.assertEqual(calls, [("POST", "/v1/ocr"), ("GET", "/v1/jobs/job%2F1"), ("GET", "/v1/jobs/job%2F1")])
        self.assertEqual(len(exchanges), 1)
        self.assertIn(b'"document"', exchanges[0][0])
        self.assertIn(b'"completed"', exchanges[0][1])

    def test_an_answer_without_pages_is_no_answer(self):
        with self.assertRaises(co.Unanswered):
            co.ask("m", b"x", "k", send=lambda *a, **k: (200, {"pages": []}, b"{}", b"{}"))

    def test_the_request_is_sent_to_the_public_api(self):
        self.assertEqual(co.ORIGIN, "https://api.empiriolabs.ai")


class StandardLibrary(unittest.TestCase):
    def test_the_check_needs_only_the_standard_library(self):
        names_of = getattr(sys, "stdlib_module_names", None)
        if names_of is None:
            self.skipTest("Python 3.10 or later lists the standard library")
        with open(os.path.join(ROOT, "check_ocr.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
        self.assertTrue(names)
        self.assertLessEqual(names, set(names_of) | {"check_receipt"}, names - set(names_of))


if __name__ == "__main__":
    unittest.main()
