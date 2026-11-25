"""Offline tests of check_video.py: frames are read from ffmpeg's PPM stream one at a time, the frames kept are the
shares of the clip the file states, a clip's own frames match even with a little noise, another model's or another
length's never do, the check refuses to send anything without ffmpeg or with a starting image that is not the file the
fingerprint records, every probe is sent before any job is polled and each job is polled on the API's own origin, each
answer's receipt is checked against the POST that made its job, check_model.py runs this check for a video
fingerprint, the check needs only the standard library, and a real ffmpeg (when one is installed) decodes an MP4 to the
frames it was made from. Run with: python -m unittest discover -s tests"""
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_image  # noqa: E402
import check_model  # noqa: E402
import check_video as cv  # noqa: E402
from test_check_receipt import KEYS, receipt_for  # noqa: E402  (the test signer, never a key of the API)

FINGERPRINTS = os.path.join(ROOT, "fingerprints")
SIZE = 3 * check_image.THUMBNAIL ** 2


def ppm(width, height, pixel):
    """One binary PPM frame of `pixel(x, y)` -> (red, green, blue)."""
    body = bytes(value for y in range(height) for x in range(width) for value in pixel(x, y))
    return b"P6\n%d %d\n255\n" % (width, height) + body


def frame_of(thumbnail, scale=2, noise=0, seed=0):
    """(width, height, bytes) of a frame whose thumbnail is `thumbnail`, plus up to `noise` of noise per pixel."""
    rng = random.Random(seed)
    size = check_image.THUMBNAIL
    width = height = size * scale
    out = bytearray()
    for y in range(height):
        for x in range(width):
            block = 3 * ((y // scale) * size + x // scale)
            out += bytes(max(0, min(255, thumbnail[block + c] + rng.randint(-noise, noise))) for c in range(3))
    return width, height, bytes(out)


def fingerprint(count=8, seed=1, frames=25, image=None):
    rng = random.Random(seed)
    indices = cv.frame_indices(frames)
    probes = []
    for i in range(count):
        probe = {"prompt": "a test clip %d" % i, "seed": 100 + i, "frame_count": frames, "width": 64, "height": 64,
                 "frames": indices,
                 "thumbnails": [check_image.encode_thumbnail([rng.randrange(256) for _ in range(SIZE)])
                                for _ in indices]}
        if image:
            probe["image"] = image
        probes.append(probe)
    return {"kind": "video", "model": "test-video-model", "weights": "org/test-video-model", "revision": "a" * 40,
            "weight_files_sha256": {"transformer/model.safetensors": "b" * 64},
            "made_with": "the pipeline published with the weights, on cuda", "made_at": "2026-11-25T00:00:00Z",
            "request": {"resolution": "480p", "duration": 1, "steps": 10, "cfg_scale": 1},
            "limits": {"clip_distance": 0.1, "typical_distance": 0.05, "close": 0.75}, "probes": probes}


class FakeClips:
    """Clips as tokens, and a stand-in for ffmpeg that turns a token into frames: b"<seed>:<variant>:<frames>"."""

    def __init__(self, fps, noise=0):
        self.by_variant = {name: {p["seed"]: p for p in fp["probes"]} for name, fp in fps.items()}
        self.noise = noise

    def clip(self, probe, variant="own", frames=None):
        return ("%d:%s:%d" % (probe["seed"], variant, frames or probe["frame_count"])).encode()

    def frames(self, data, wanted=(), program=None):
        seed, variant, count = data.decode().split(":")
        probe = self.by_variant[variant][int(seed)]
        thumbs = dict(zip(probe["frames"], probe["thumbnails"]))
        kept = {}
        for index in wanted:
            if index < int(count):
                thumb = check_image.decode_thumbnail(thumbs.get(index, probe["thumbnails"][0]))
                kept[index] = frame_of(thumb, noise=self.noise, seed=index)
        return int(count), kept


@contextlib.contextmanager
def decoder(fake):
    with mock.patch.object(cv, "frames", fake.frames), mock.patch.object(cv, "require_ffmpeg", lambda: "ffmpeg"):
        yield


def run_with(fp, clip_for, fake, keys=None, folder=None):
    """check_one() with each probe's clip from `clip_for(probe)`; (exit, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(folder or tmp, fp["model"] + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fp, handle)

        def ask_all(model, fingerprint, key, folder, call=None):
            out = []
            for probe in fingerprint["probes"]:
                clip = clip_for(probe)
                sent = json.dumps({"model": model, "prompt": probe["prompt"], "seed": probe["seed"]}).encode()
                body = json.dumps({"job_id": "j", "status": "completed",
                                   "result": {"data": [{"url": "https://media.example/c.mp4",
                                                        "sha256": hashlib.sha256(clip).hexdigest()}]}}).encode()
                receipt = json.dumps(receipt_for(sent, body, model=model), separators=(",", ":")).encode()
                out.append((sent, body[:-1] + b',"receipt":' + receipt + b"}", clip))
            return out
        out = io.StringIO()
        with decoder(fake), mock.patch.object(cv, "ask_all", ask_all), contextlib.redirect_stdout(out):
            code = cv.check_one(fp["model"], path, "key", keys=keys)
        return code, out.getvalue()


class Frames(unittest.TestCase):
    def test_a_ppm_stream_is_read_one_frame_at_a_time_and_only_wanted_frames_are_kept(self):
        stream = io.BytesIO(ppm(3, 2, lambda x, y: (x, y, 7)) + ppm(3, 2, lambda x, y: (9, 9, 9))
                            + ppm(3, 2, lambda x, y: (1, 2, 3)))
        count, kept = cv.read_frames(stream, {0, 2})
        self.assertEqual(count, 3)
        self.assertEqual(sorted(kept), [0, 2])
        self.assertEqual(kept[0][:2], (3, 2))
        self.assertEqual(kept[0][2][:6], bytes([0, 0, 7, 1, 0, 7]))
        self.assertEqual(kept[2][2], bytes([1, 2, 3] * 6))

    def test_a_header_comment_is_skipped_and_a_cut_frame_or_another_format_is_refused(self):
        count, kept = cv.read_frames(io.BytesIO(b"P6 # made by a test\n1 1\n255\n\x01\x02\x03"), {0})
        self.assertEqual((count, kept[0][2]), (1, b"\x01\x02\x03"))
        with self.assertRaises(cv.NotAVideo):
            cv.read_frames(io.BytesIO(ppm(4, 4, lambda x, y: (0, 0, 0))[:-5]), {0})
        with self.assertRaises(cv.NotAVideo):
            cv.read_frames(io.BytesIO(b"P5\n1 1\n255\n\x00"), {0})
        with self.assertRaises(cv.NotAVideo):
            cv.read_frames(io.BytesIO(b"P6\n1 1\n65535\n\x00\x00\x00\x00\x00\x00"), {0})

    def test_a_frames_thumbnail_is_the_image_kinds_block_average(self):
        width, height, raw = frame_of([5, 6, 7] * (check_image.THUMBNAIL ** 2), scale=3)
        self.assertEqual(cv.frame_thumbnail(width, height, raw), [5, 6, 7] * (check_image.THUMBNAIL ** 2))

    def test_the_frames_kept_are_shares_of_the_clip_never_its_very_first(self):
        self.assertEqual(cv.frame_indices(25), [6, 12, 18, 24])
        self.assertEqual(cv.frame_indices(49), [12, 24, 36, 48])
        self.assertEqual(cv.frame_indices(309), [77, 154, 231, 308])
        self.assertEqual(cv.frame_indices(1), [0])
        self.assertEqual(cv.frame_indices(25, (0, 1)), [0, 24])


class Verdict(unittest.TestCase):
    def test_a_models_own_clips_match_even_with_a_little_noise(self):
        fp = fingerprint()
        fake = FakeClips({"own": fp}, noise=6)
        code, out = run_with(fp, lambda probe: fake.clip(probe), fake)
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)
        self.assertIn("every clip the length its fingerprint states", out)

    def test_another_models_clips_do_not_match(self):
        fp, other = fingerprint(), fingerprint(seed=2)
        fake = FakeClips({"own": fp, "other": other})
        code, out = run_with(fp, lambda probe: fake.clip(probe, "other"), fake)
        self.assertEqual(code, 1)
        self.assertIn("DOES NOT MATCH", out)

    def test_two_far_clips_in_eight_still_match_but_three_do_not(self):
        fp, other = fingerprint(), fingerprint(seed=3)
        fake = FakeClips({"own": fp, "other": other})
        for wrong, expected in ((2, 0), (3, 1)):
            code, out = run_with(fp, lambda probe: fake.clip(probe, "other" if probe["seed"] - 100 < wrong else "own"),
                                 fake)
            self.assertEqual(code, expected, out)

    def test_a_clip_of_another_length_never_counts_as_close(self):
        fp = fingerprint()
        fake = FakeClips({"own": fp})
        code, out = run_with(fp, lambda probe: fake.clip(probe, frames=33 if probe["seed"] < 103 else None), fake)
        self.assertEqual(code, 1)
        self.assertIn("3 clips of another length", out)

    def test_every_answer_signed_for_its_clip_is_reported_signed(self):
        fp = fingerprint()
        fake = FakeClips({"own": fp})
        code, out = run_with(fp, lambda probe: fake.clip(probe), fake, keys=KEYS)
        self.assertEqual(code, 0, out)
        self.assertIn("every answer signed by EmpirioLabs for test-video-model", out)


class Refusals(unittest.TestCase):
    def test_without_ffmpeg_nothing_is_sent_and_the_check_says_why(self):
        fp = fingerprint()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "test-video-model.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fp, handle)
            sent = []
            out = io.StringIO()
            with mock.patch.object(cv.shutil, "which", lambda name: None), \
                    mock.patch.dict(os.environ, {cv.FFMPEG_ENV: ""}), \
                    mock.patch.object(cv, "_send", lambda *a, **k: sent.append(a)), contextlib.redirect_stdout(out):
                code = cv.check_one("test-video-model", path, "key", keys=None)
        self.assertEqual(code, 1)
        self.assertEqual(sent, [])
        self.assertIn("could not be checked: ffmpeg is not installed here", out.getvalue())
        self.assertIn("Nothing was sent.", out.getvalue())

    def test_a_named_ffmpeg_that_is_not_a_file_is_no_ffmpeg(self):
        with mock.patch.dict(os.environ, {cv.FFMPEG_ENV: os.path.join(ROOT, "no-such-ffmpeg")}):
            self.assertIsNone(cv.ffmpeg_program())

    def test_a_starting_image_that_is_not_the_recorded_file_is_never_sent(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "test-video-model"))
            with open(os.path.join(folder, "test-video-model", "start.png"), "wb") as handle:
                handle.write(check_image.PNG_SIGNATURE + b"not the recorded bytes")
            image = {"file": "test-video-model/start.png", "sha256": "0" * 64, "source": "https://example.org/a",
                     "license": "Public domain"}
            fp = fingerprint(image=image)
            sent = []

            def call(method, path, key, body=None):
                sent.append(path)
                return 202, {"job_id": "x"}, b"{}", b"{}"
            with mock.patch.object(cv, "require_ffmpeg", lambda: "ffmpeg"):
                with self.assertRaises(ValueError) as caught:
                    cv.run_check("test-video-model", fp, "key", folder=folder, call=call)
        self.assertIn("is not the file this fingerprint records", str(caught.exception))
        self.assertEqual(sent, [])


class Asking(unittest.TestCase):
    def test_every_probe_is_sent_before_any_job_is_polled_and_jobs_are_polled_on_the_apis_own_origin(self):
        fp = fingerprint(count=3)
        calls, reads = [], {}

        def call(method, path, key, body=None):
            calls.append((method, path))
            if method == "POST":
                job = "job %d" % body["seed"]
                answer = {"job_id": job, "status": "processing", "poll_url": "https://elsewhere.example/steal"}
                reads[job] = 0
                return 202, answer, json.dumps(body).encode(), json.dumps(answer).encode()
            job = path.rsplit("/", 1)[1].replace("%20", " ")
            reads[job] += 1
            if reads[job] < 2:
                answer = {"job_id": job, "status": "processing"}
            else:
                answer = {"job_id": job, "status": "completed",
                          "result": {"data": [{"b64_json": base64.b64encode(job.encode()).decode()}]}}
            return 200, answer, None, json.dumps(answer).encode()
        with mock.patch.object(cv.time, "sleep", lambda s: None):
            clips = cv.ask_all("m", fp, "key", FINGERPRINTS, call=call)
        self.assertEqual(calls[:3], [("POST", cv.ENDPOINT)] * 3)
        self.assertTrue(all(path.startswith("/v1/jobs/job%20") for method, path in calls[3:]), calls)
        self.assertEqual([clip for _, _, clip in clips], [b"job 100", b"job 101", b"job 102"])
        for (sent, received, _clip), probe in zip(clips, fp["probes"]):
            self.assertEqual(json.loads(sent)["seed"], probe["seed"])  # each answer kept with the POST of its job
            self.assertEqual(json.loads(received)["status"], "completed")

    def test_a_starting_image_goes_as_a_data_url_beside_the_fingerprints_settings(self):
        body = cv.request_body("m", {"prompt": "p", "seed": 7}, {"mode": "i2v", "duration": 2}, b"\x89PNG", "a/b.png")
        self.assertEqual((body["prompt"], body["seed"], body["mode"]), ("p", 7, "i2v"))
        self.assertEqual(body["image"], "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode())

    def test_a_failed_job_gives_no_verdict(self):
        state = {"poll": "/v1/jobs/x", "started": cv.time.time(), "done": False}
        with self.assertRaises(cv.Unanswered):
            cv.poll(state, "key", lambda *a: (200, {"status": "failed", "error": {"message": "busy"}}, None, b"{}"))

    def test_a_refused_request_is_sent_again_after_a_429_and_only_then(self):
        answers = [cv.urllib.error.HTTPError("u", 429, "slow down", {}, io.BytesIO(b"")), FakeResponse(b'{"ok": 1}')]
        sleeps = []

        def urlopen(request, timeout):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        with mock.patch.object(cv.urllib.request, "urlopen", urlopen), mock.patch.object(cv.time, "sleep", sleeps.append):
            status, data, _sent, _received = cv._send("POST", "/v1/videos/generations", "key", {"a": 1})
        self.assertEqual((status, data, sleeps), (200, {"ok": 1}, [5]))


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Wiring(unittest.TestCase):
    def test_check_model_runs_this_check_for_a_video_fingerprint(self):
        self.assertIs(check_model.OTHER_KINDS["video"], cv)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "test-video-model.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fingerprint(), handle)
            with mock.patch.object(cv, "check_one", return_value=0) as one:
                self.assertEqual(check_model.check_one("test-video-model", path, "key", None), 0)
            one.assert_called_once()

    def test_the_check_needs_only_the_standard_library(self):
        with open(os.path.join(ROOT, "check_video.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                top.add((node.module or "").split(".")[0])
        allowed = set(getattr(sys, "stdlib_module_names", ())) | {"check_receipt", "check_image"}
        self.assertLessEqual(top, allowed, top - allowed)

    def test_each_published_video_fingerprint_holds_whole_thumbnails(self):
        for name in sorted(os.listdir(FINGERPRINTS)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(FINGERPRINTS, name), encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("kind") != cv.KIND:
                continue
            for probe in data["probes"]:
                self.assertEqual(len(probe["frames"]), len(probe["thumbnails"]), name)
                for thumbnail in probe["thumbnails"]:
                    self.assertEqual(len(check_image.decode_thumbnail(thumbnail)), SIZE, name)


@unittest.skipUnless(cv.ffmpeg_program(), "ffmpeg is not installed")
class RealDecoder(unittest.TestCase):
    """A real ffmpeg: frames written to an MP4 (ffmpeg's own MPEG-4 encoder, near lossless) come back as frames whose
    thumbnails are the ones written, and a file that is not a clip is refused."""

    def test_an_mp4_decodes_to_the_frames_it_was_made_from(self):
        rng = random.Random(5)
        thumbs = [[rng.randrange(40, 216) for _ in range(SIZE)] for _ in range(9)]
        frames_written = [frame_of(t, scale=4) for t in thumbs]
        width, height = frames_written[0][:2]
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "clip.mp4")
            encoder = subprocess.run(
                [cv.ffmpeg_program(), "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "%dx%d" % (width, height), "-r", "24", "-i", "pipe:0", "-c:v", "mpeg4", "-q:v", "1",
                 "-pix_fmt", "yuv420p", "-f", "mp4", target],
                input=b"".join(raw for _, _, raw in frames_written), capture_output=True)
            self.assertEqual(encoder.returncode, 0, encoder.stderr[-400:])
            with open(target, "rb") as handle:
                data = handle.read()
        made = cv.signature(data)
        self.assertEqual(made["frame_count"], 9)
        self.assertEqual(made["frames"], [2, 4, 6, 8])
        self.assertEqual((made["width"], made["height"]), (width, height))
        for index, live in zip(made["frames"], made["thumbnails"]):
            self.assertLess(check_image.distance(thumbs[index], live), 0.02, index)
            self.assertGreater(check_image.distance(thumbs[(index + 1) % 9], live), 0.5, index)

    def test_bytes_that_are_not_a_clip_are_refused(self):
        with self.assertRaises(cv.NotAVideo):
            cv.signature(b"\x00\x00\x00\x18ftypisomnot really a clip at all")


if __name__ == "__main__":
    unittest.main()
