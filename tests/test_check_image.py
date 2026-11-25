"""Offline tests of check_image.py: the standard-library PNG reader decodes every row filter and the layouts an image
API returns, the thumbnail averages its blocks, the model's own pictures match, a little noise still matches, another
model's pictures or another seed's never do, an API that does not answer gives no verdict, a job is polled on the
API's own origin only, every answer's receipt and file are checked, check_model.py runs this check for an image
fingerprint, and the check needs only the standard library.
Run with: python -m unittest discover -s tests"""
import ast
import base64
import contextlib
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_image as ci  # noqa: E402
import check_model  # noqa: E402
from test_check_receipt import KEYS, receipt_for  # noqa: E402  (the test signer, never a key of the API)

FINGERPRINTS = os.path.join(ROOT, "fingerprints")


# ------------------------------------------------------------------------------------------------- a PNG writer
def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def _filtered(line, previous, method, step):
    out = bytearray(len(line))
    for i, value in enumerate(line):
        left = line[i - step] if i >= step else 0
        up = previous[i]
        upper_left = previous[i - step] if i >= step else 0
        predictor = (0, left, up, (left + up) >> 1, _paeth(left, up, upper_left))[method]
        out[i] = (value - predictor) & 255
    return bytes(out)


def png(width, height, pixel, color=2, depth=8, methods=(0,), palette=None):
    """A PNG of `pixel(x, y)` (a tuple of channel values in the color type's order), each row filtered with the next
    method of `methods` in turn."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    step = channels * depth // 8
    raw, previous = bytearray(), bytes(width * step)
    for y in range(height):
        line = bytearray()
        for x in range(width):
            for value in pixel(x, y):
                line += struct.pack(">H", value) if depth == 16 else bytes([value])
        method = methods[y % len(methods)]
        raw += bytes([method]) + _filtered(line, previous, method, step)
        previous = bytes(line)
    head = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
    body = _chunk(b"IHDR", head) + (_chunk(b"PLTE", palette) if palette else b"")
    return ci.PNG_SIGNATURE + body + _chunk(b"IDAT", zlib.compress(bytes(raw))) + _chunk(b"IEND", b"")


def picture_of(thumbnail, scale=4, noise=0, seed=0):
    """A picture whose thumbnail is `thumbnail` (plus up to `noise` of uniform noise per pixel)."""
    rng = random.Random(seed)
    size = ci.THUMBNAIL

    def pixel(x, y):
        block = 3 * ((y // scale) * size + x // scale)
        return tuple(max(0, min(255, thumbnail[block + c] + rng.randint(-noise, noise))) for c in range(3))
    return png(size * scale, size * scale, pixel)


def fingerprint(count=8, seed=1):
    rng = random.Random(seed)
    probes = [{"prompt": "a test picture %d" % i, "seed": 100 + i,
               "thumbnail": ci.encode_thumbnail([rng.randrange(256) for _ in range(3 * ci.THUMBNAIL ** 2)])}
              for i in range(count)]
    return {"kind": "image", "model": "test-image-model", "weights": "org/test-image-model", "revision": "a" * 40,
            "weight_files_sha256": {"transformer/model.safetensors": "b" * 64},
            "made_with": "the pipeline published with the weights, on cuda", "made_at": "2026-11-24T00:00:00Z",
            "request": {"width": 128, "height": 128, "format": "png", "response_format": "b64_json"},
            "limits": {"image_distance": 0.1, "typical_distance": 0.05, "close": 0.75}, "probes": probes}


class Png(unittest.TestCase):
    def test_every_row_filter_decodes_to_the_pixels_written(self):
        pixel = lambda x, y: ((x * 37 + y) % 256, (y * 11) % 256, (x * y) % 256)  # noqa: E731
        for methods in ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)):
            width, height, rows = ci.decode_png(png(9, 7, pixel, methods=methods))
            self.assertEqual((width, height), (9, 7))
            for y in range(7):
                self.assertEqual(list(rows[y]), [v for x in range(9) for v in pixel(x, y)], (methods, y))

    def test_alpha_grey_palette_and_sixteen_bit_layouts_decode_to_red_green_blue(self):
        rgba = ci.decode_png(png(3, 2, lambda x, y: (10 * x, 20 * y, 30, 99), color=6, methods=(4,)))[2]
        self.assertEqual(list(rgba[1]), [0, 20, 30, 10, 20, 30, 20, 20, 30])
        grey = ci.decode_png(png(2, 1, lambda x, y: (40 + x,), color=0, methods=(1,)))[2]
        self.assertEqual(list(grey[0]), [40, 40, 40, 41, 41, 41])
        grey_alpha = ci.decode_png(png(2, 1, lambda x, y: (70 + x, 5), color=4))[2]
        self.assertEqual(list(grey_alpha[0]), [70, 70, 70, 71, 71, 71])
        palette = bytes([1, 2, 3, 200, 201, 202])
        indexed = ci.decode_png(png(2, 1, lambda x, y: (x,), color=3, palette=palette))[2]
        self.assertEqual(list(indexed[0]), [1, 2, 3, 200, 201, 202])
        deep = ci.decode_png(png(2, 1, lambda x, y: (0x1234, 0xABCD, 0x00FF), depth=16, methods=(2,)))[2]
        self.assertEqual(list(deep[0]), [0x12, 0xAB, 0x00] * 2)

    def test_what_is_not_a_png_or_not_a_layout_it_reads_is_refused(self):
        with self.assertRaises(ci.NotAnImage):
            ci.decode_png(b"RIFF....WEBPVP8 ")
        interlaced = bytearray(png(2, 2, lambda x, y: (1, 2, 3)))
        interlaced[8 + 8 + 12] = 1  # IHDR's interlace byte
        with self.assertRaises(ci.NotAnImage):
            ci.decode_png(bytes(interlaced))


class Thumbnail(unittest.TestCase):
    def test_each_block_is_the_rounded_average_of_its_pixels(self):
        data = png(64, 64, lambda x, y: (x % 2 * 255, 100, 7 if y < 32 else 9))
        values = ci.thumbnail(data)
        self.assertEqual(len(values), 3 * 32 * 32)
        self.assertEqual(values[:3], [128, 100, 7])
        self.assertEqual(values[-3:], [128, 100, 9])

    def test_a_thumbnail_survives_its_text_form_and_distance_reads_layout_not_brightness(self):
        values = [0, 255, 17] * (32 * 32)
        self.assertEqual(ci.decode_thumbnail(ci.encode_thumbnail(values)), values)
        self.assertAlmostEqual(ci.distance(values, values), 0, places=9)
        lighter = [min(255, v + 20) for v in values]
        self.assertLess(ci.distance(values, lighter), 0.01)
        self.assertAlmostEqual(ci.distance(values, [255 - v for v in values]), 2, places=9)
        self.assertEqual(ci.distance([0] * 6, [255] * 6), 1)


def run_with(fp, picture_for, keys=None):
    """check_one() with each probe answered by `picture_for(probe)`; (exit, output)."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, fp["model"] + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fp, handle)

        def ask(model, probe, request, key, exchanges=None, call=None):
            picture = picture_for(probe)
            if exchanges is not None:
                sent = json.dumps({"model": model, "prompt": probe["prompt"], "seed": probe["seed"], **request}).encode()
                body = json.dumps({"data": [{"b64_json": base64.b64encode(picture).decode(),
                                             "sha256": __import__("hashlib").sha256(picture).hexdigest()}]}).encode()
                receipt = json.dumps(receipt_for(sent, body, model=model), separators=(",", ":")).encode()
                exchanges.append((sent, body[:-1] + b',"receipt":' + receipt + b"}", picture))
            return picture
        out = io.StringIO()
        with mock.patch.object(ci, "ask", ask), contextlib.redirect_stdout(out):
            code = ci.check_one(fp["model"], path, "key", keys=keys)
        return code, out.getvalue()


class Verdict(unittest.TestCase):
    def test_the_models_own_pictures_match_even_with_a_little_noise(self):
        fp = fingerprint()
        own = lambda probe: picture_of(ci.decode_thumbnail(probe["thumbnail"]), noise=6, seed=probe["seed"])  # noqa: E731
        code, out = run_with(fp, own)
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_another_models_pictures_do_not_match(self):
        fp, other = fingerprint(), fingerprint(seed=2)
        by_seed = {p["seed"]: p for p in other["probes"]}
        code, out = run_with(fp, lambda probe: picture_of(ci.decode_thumbnail(by_seed[probe["seed"]]["thumbnail"])))
        self.assertEqual(code, 1)
        self.assertIn("DOES NOT MATCH", out)

    def test_two_far_pictures_in_eight_still_match_but_three_do_not(self):
        fp, other = fingerprint(), fingerprint(seed=3)

        def answers(wrong):
            def picture_for(probe):
                index = probe["seed"] - 100
                source = other if index < wrong else fp
                return picture_of(ci.decode_thumbnail(source["probes"][index]["thumbnail"]))
            return picture_for
        self.assertEqual(run_with(fp, answers(2))[0], 0)
        self.assertEqual(run_with(fp, answers(3))[0], 1)

    def test_an_api_that_does_not_answer_gives_no_verdict(self):
        def refuse(probe):
            raise ci.Unanswered("HTTP 503: busy")
        code, out = run_with(fingerprint(), refuse)
        self.assertEqual(code, 1)
        self.assertIn("could not be checked", out)
        self.assertNotIn("matches", out)


class Receipts(unittest.TestCase):
    def test_every_answer_signed_for_its_picture_is_reported_signed(self):
        fp = fingerprint()
        own = lambda probe: picture_of(ci.decode_thumbnail(probe["thumbnail"]))  # noqa: E731
        code, out = run_with(fp, own, keys=KEYS)
        self.assertEqual(code, 0, out)
        self.assertIn("every answer signed by EmpirioLabs for test-image-model", out)

    def test_a_picture_that_is_not_the_file_its_answer_lists_is_not_signed(self):
        fp = fingerprint()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "test-image-model.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fp, handle)

            def ask(model, probe, request, key, exchanges=None, call=None):
                picture = picture_of(ci.decode_thumbnail(probe["thumbnail"]))
                sent = b'{"model":"test-image-model"}'
                body = json.dumps({"data": [{"b64_json": "", "sha256": "0" * 64}]}).encode()
                receipt = json.dumps(receipt_for(sent, body, model=model), separators=(",", ":")).encode()
                exchanges.append((sent, body[:-1] + b',"receipt":' + receipt + b"}", picture))
                return picture
            out = io.StringIO()
            with mock.patch.object(ci, "ask", ask), contextlib.redirect_stdout(out):
                code = ci.check_one("test-image-model", path, "key", keys=KEYS)
        self.assertEqual(code, 1)
        self.assertIn("is not a file this answer lists", out.getvalue())


class Ask(unittest.TestCase):
    def test_a_job_is_polled_on_the_apis_own_origin_until_it_holds_the_picture(self):
        picture = png(32, 32, lambda x, y: (1, 2, 3))
        done = {"status": "completed", "result": {"data": [{"b64_json": base64.b64encode(picture).decode()}]}}
        calls = []

        def send(method, path, key, body=None):
            calls.append((method, path))
            if method == "POST":
                answer = {"status": "queued", "job_id": "j 1", "poll_url": "https://elsewhere.example/steal"}
                return 202, answer, json.dumps(body).encode(), json.dumps(answer).encode()
            return 200, done, None, json.dumps(done).encode()
        exchanges = []
        with mock.patch.object(ci, "_send", send), mock.patch.object(ci.time, "sleep", lambda s: None):
            got = ci.ask("m", {"prompt": "p", "seed": 5}, {"format": "png"}, "key", exchanges)
        self.assertEqual(got, picture)
        self.assertEqual(calls, [("POST", "/v1/images/generations"), ("GET", "/v1/jobs/j%201")])
        sent = json.loads(exchanges[0][0])
        self.assertEqual((sent["prompt"], sent["seed"], sent["format"]), ("p", 5, "png"))
        self.assertEqual(json.loads(exchanges[0][1])["status"], "completed")


class Wiring(unittest.TestCase):
    def test_check_model_runs_this_check_for_an_image_fingerprint(self):
        self.assertIs(check_model.OTHER_KINDS["image"], ci)
        fp = fingerprint()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "test-image-model.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fp, handle)
            with mock.patch.object(ci, "check_one", return_value=0) as one:
                self.assertEqual(check_model.check_one("test-image-model", path, "key", None), 0)
            one.assert_called_once()

    def test_the_check_needs_only_the_standard_library(self):
        if getattr(sys, "stdlib_module_names", None) is None:
            self.skipTest("Python 3.10 or later lists the standard library")
        with open(os.path.join(ROOT, "check_image.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                top.add((node.module or "").split(".")[0])
        allowed = set(getattr(sys, "stdlib_module_names", ())) | {"check_receipt"}
        self.assertLessEqual(top, allowed, top - allowed)

    def test_each_published_image_fingerprint_holds_whole_thumbnails(self):
        for name in sorted(os.listdir(FINGERPRINTS)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(FINGERPRINTS, name), encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("kind") != ci.KIND:
                continue
            for probe in data["probes"]:
                self.assertEqual(len(ci.decode_thumbnail(probe["thumbnail"])), 3 * ci.THUMBNAIL ** 2, name)


if __name__ == "__main__":
    unittest.main()
