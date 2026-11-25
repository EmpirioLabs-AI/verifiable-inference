"""Offline tests of check_audio.py, with the API replaced by a table: WAV files of every common layout decode (integer
PCM through the wave module, floating point through the RIFF walk), the transform is the discrete Fourier transform,
the signature puts a tone in its band whatever the sample rate, a clip's own signature matches, the same clip with a
little noise still matches, another seed's clip does not, a shorter clip pays for its missing cells, a job answer is
followed to its end and speech comes back as bytes with its header receipt, check_model.py runs this check for an
audio fingerprint and checks every receipt, a 10 s clip is read in seconds, and the check needs only the standard
library. With torch installed, the seeded draws of `make` are checked too.
Run with: python -m unittest discover -s tests"""
import array
import ast
import cmath
import contextlib
import hashlib
import io
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
import unittest
import wave
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_audio as ca  # noqa: E402
import check_model  # noqa: E402
from test_check_receipt import KEYS, receipt_for  # noqa: E402  (the test signer, never a key of the API)

MODEL = "audio-test-model"


def song(prompt, seed, seconds=2.0, rate=48000, noise=0.0, gain=1.0):
    """A stand-in model's clip: a few tones and a beat chosen by the prompt and the seed, stereo floats. `gain` and
    `noise` stand for the small differences between two computers' runs of the same weights."""
    rng = random.Random("%s/%d" % (prompt, seed))
    tones = [(rng.uniform(80, 6000), rng.uniform(0.05, 0.25)) for _ in range(4)]
    beat = rng.uniform(1.5, 4.0)
    jitter = random.Random(seed + 1)
    left, right = [], []
    for i in range(int(seconds * rate)):
        t = i / rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * beat * t) ** 2
        value = gain * envelope * sum(a * math.sin(2 * math.pi * f * t) for f, a in tones)
        if noise:
            value += noise * (jitter.random() * 2 - 1)
        left.append(value)
        right.append(value * 0.8)
    return rate, [left, right]


def wav(rate, channels):
    return ca.encode_wav(rate, channels)


class Decoding(unittest.TestCase):
    def frames(self, width, values, channels, rate=16000):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as clip:
            clip.setnchannels(channels)
            clip.setsampwidth(width)
            clip.setframerate(rate)
            if width == 1:
                clip.writeframes(bytes(v + 128 for v in values))
            elif width == 3:
                clip.writeframes(b"".join(v.to_bytes(3, "little", signed=True) for v in values))
            else:
                clip.writeframes(array.array({2: "h", 4: "i"}[width], values).tobytes())
        return buffer.getvalue()

    def test_integer_pcm_of_every_width_and_channel_count(self):
        for width, full in ((1, 127), (2, 32767), (3, 8388607), (4, 2147483647)):
            data = self.frames(width, [full, -full, full // 2, 0], 2)
            rate, mono = ca.decode_wav(data)
            self.assertEqual(rate, 16000)
            self.assertEqual(len(mono), 2, width)
            self.assertAlmostEqual(mono[0], 0.0, places=6)
            self.assertAlmostEqual(mono[1], (full // 2) / (full + 1) / 2, places=5)
        rate, mono = ca.decode_wav(self.frames(2, [100, 200, 300], 3))
        self.assertAlmostEqual(mono[0], 200 / 32768, places=9)

    def float_wav(self, samples, channels, bits=32, extensible=False):
        body = array.array("f" if bits == 32 else "d", samples).tobytes()
        if extensible:
            fmt = struct.pack("<HHIIHHHHIH14s", 0xFFFE, channels, 24000, 24000 * channels * bits // 8,
                              channels * bits // 8, bits, 22, bits, 0, 3, b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71")
        else:
            fmt = struct.pack("<HHIIHH", 3, channels, 24000, 24000 * channels * bits // 8, channels * bits // 8, bits)
        chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body
        return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks

    def test_floating_point_wav_is_read_by_the_riff_walk(self):
        for bits in (32, 64):
            for extensible in (False, True):
                rate, mono = ca.decode_wav(self.float_wav([0.5, -0.5, 0.25, 0.75], 2, bits, extensible))
                self.assertEqual(rate, 24000)
                self.assertEqual(mono, [0.0, 0.5])

    def test_anything_else_is_refused(self):
        for data in (b"", b"ID3\x04 an mp3", b"RIFF\x00\x00\x00\x00WAVEjunk", b"fLaC\x00\x00"):
            with self.assertRaises(ca.NotAudio):
                ca.decode_wav(data)

    def test_the_16_bit_writer_rounds_and_clips(self):
        rate, mono = ca.decode_wav(wav(8000, [[0.5, 2.0, -2.0, 0.0]]))
        self.assertEqual(rate, 8000)
        self.assertEqual([round(v * 32768) for v in mono], [16384, 32767, -32768, 0])


class Signature(unittest.TestCase):
    def test_the_transform_is_the_discrete_fourier_transform(self):
        rng = random.Random(3)
        for n in (1, 2, 8, 64):
            values = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
            direct = [sum(v * cmath.exp(-2j * math.pi * k * t / n) for t, v in enumerate(values)) for k in range(n)]
            for a, b in zip(ca._fft(values), direct):
                self.assertAlmostEqual(abs(a - b), 0.0, places=9)

    def tone(self, frequency, rate, seconds=1.0, amplitude=0.5):
        return [amplitude * math.sin(2 * math.pi * frequency * i / rate) for i in range(int(seconds * rate))]

    def test_a_tone_is_loudest_in_its_band_whatever_the_sample_rate(self):
        edges = [ca.LOW_HZ * (ca.HIGH_HZ / ca.LOW_HZ) ** (b / ca.BANDS) for b in range(ca.BANDS + 1)]
        for band in (9, 16, 25):
            frequency = math.sqrt(edges[band] * edges[band + 1])  # the band's middle
            rows = {}
            for rate in (16000, 24000, 44100, 48000):
                rows[rate] = ca.signature(rate, self.tone(frequency, rate))
                self.assertEqual(len(rows[rate]), 4, rate)
                middle = rows[rate][1]
                self.assertEqual(max(range(ca.BANDS), key=middle.__getitem__), band, (frequency, rate))
            self.assertLess(ca.distance(rows[44100], rows[48000]), 2.0, frequency)

    def test_silence_is_the_floor_and_every_quarter_second_is_a_cell(self):
        rows = ca.signature(48000, [0.0] * 48000 * 3)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(v == ca.FLOOR_DB for row in rows for v in row))
        hiss = random.Random(1)
        quiet = ca.signature(48000, [0.0002 * (hiss.random() * 2 - 1) for _ in range(48000)])
        self.assertTrue(all(v == ca.FLOOR_DB for row in quiet for v in row), "hiss 80 dB down stays at the floor")
        self.assertEqual(len(ca.signature(48000, [0.0] * 1000)), 1)

    def test_the_kept_signature_is_the_signature_to_half_a_db(self):
        rate, channels = song("calm piano", 7, seconds=1.0)
        exact = ca.signature(*ca.decode_wav(wav(rate, channels)))
        kept = ca.decode_signature(ca.encode_signature(exact))
        self.assertEqual(len(kept), len(exact))
        self.assertLessEqual(max(abs(a - b) for x, y in zip(exact, kept) for a, b in zip(x, y)), 0.25 + 1e-9)
        with self.assertRaises(ValueError):
            ca.decode_signature(ca.encode_signature(exact)[:-4])

    def test_distance(self):
        rate, channels = song("calm piano", 7, seconds=2.0)
        mine = ca.clip_signature(wav(rate, channels))
        self.assertEqual(ca.distance(mine, mine), 0.0)
        near = ca.clip_signature(wav(*song("calm piano", 7, seconds=2.0, noise=0.0002, gain=0.97)))
        self.assertLess(ca.distance(mine, near), 1.0)
        other = ca.clip_signature(wav(*song("calm piano", 8, seconds=2.0)))
        self.assertGreater(ca.distance(mine, other), 4.0)
        short = mine[:len(mine) // 2]
        missing = sum(v - ca.FLOOR_DB for row in mine[len(mine) // 2:] for v in row) / (len(mine) * ca.BANDS)
        self.assertAlmostEqual(ca.distance(mine, short), missing, msg="each missing cell counts as silence")
        self.assertGreater(ca.distance(mine, short), 3.0)

    def test_a_ten_second_clip_is_read_in_seconds(self):
        rate, channels = song("timing", 1, seconds=10.0)
        data = wav(rate, channels)
        started = time.time()
        rows = ca.clip_signature(data)
        seconds = time.time() - started
        self.assertEqual(len(rows), 40)
        self.assertLess(seconds, 15.0, "10 s of 48 kHz stereo took %.1f s" % seconds)


class Fixture:
    """A fingerprint of three probes made by the stand-in model, in a temporary folder."""

    def __init__(self, folder, endpoint="/v1/audio/generations"):
        speech = endpoint == "/v1/audio/speech"
        probes = [{"input": "Hello there.", "voice": "emma", "seed": 1}, {"input": "A second line.", "voice": "emma",
                                                                          "seed": 2}] if speech else \
            [{"prompt": "calm piano", "seed": 11, "audio_duration": 10}, {"prompt": "drum loop", "seed": 12,
                                                                          "audio_duration": 10},
             {"prompt": "folk song", "lyrics": "[Verse]\nRow, row, row your boat", "seed": 13, "audio_duration": 10}]
        rows = []
        for probe in probes:
            rate, channels = self.clip(probe)
            rows.append({**probe, "seconds": len(channels[0]) / rate,
                         "signature": ca.encode_signature(ca.signature(*ca.decode_wav(wav(rate, channels))))})
        self.fingerprint = {"kind": ca.KIND, "model": MODEL, "weights": "example/audio-model", "revision": "0" * 40,
                            "weight_files_sha256": {"model.safetensors": "1" * 64},
                            "made_with": "the pipeline published with the weights, on cpu",
                            "made_at": "2026-11-24T00:00:00Z", "endpoint": endpoint,
                            "request": {"output_format": "wav"} if speech else {"format": "wav"},
                            "limits": {"clip_distance": 3.0, "typical_distance": 2.0, "close": 0.66}, "probes": rows}
        self.path = os.path.join(folder, MODEL + ".json")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.fingerprint, handle)

    @staticmethod
    def clip(probe, seed_shift=0, noise=0.0, gain=1.0):
        return song(probe.get("prompt") or probe.get("input"), probe["seed"] + seed_shift, seconds=1.5, noise=noise,
                    gain=gain)

    def answer(self, probe, **changes):
        return wav(*self.clip(probe, **changes))


def run(fixture, answer, check=None, signed=True):
    """check_one() with every probe answered by `answer(probe)` (WAV bytes), each answer a signed job result listing
    the file unless `signed` is false; (exit status, output)."""

    def ask(model, probe, request, endpoint, key, exchanges=None, send=None, download=None):
        clip = answer(probe)
        if exchanges is not None:
            sent = json.dumps({"model": model, **request, **probe}).encode()
            received = json.dumps({"job_id": "j", "status": "completed", "result": {
                "data": [{"url": "https://media.example/clip.wav", "sha256": hashlib.sha256(clip).hexdigest()}]}}).encode()
            if signed:
                receipt = json.dumps(receipt_for(sent, received, model=model), separators=(",", ":")).encode()
                received = received[:-1] + b',"receipt":' + receipt + b"}"
            exchanges.append((sent, received, None, clip, True))
        return clip
    out = io.StringIO()
    with mock.patch.object(ca, "ask", ask), contextlib.redirect_stdout(out):
        code = (check or ca.check_one)(MODEL, fixture.path, "not-a-key")
    return code, out.getvalue()


class Verdict(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.fixture = Fixture(self.folder)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.folder, ignore_errors=True)

    def test_its_own_clips_match(self):
        code, out = run(self.fixture, self.fixture.answer)
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_the_same_clips_a_little_off_match(self):
        code, out = run(self.fixture, lambda probe: self.fixture.answer(probe, noise=0.0002, gain=0.97))
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_another_seeds_clips_do_not_match(self):
        code, out = run(self.fixture, lambda probe: self.fixture.answer(probe, seed_shift=100))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_an_answer_that_is_not_wav_gives_no_verdict(self):
        code, out = run(self.fixture, lambda probe: b"ID3\x04 an mp3")
        self.assertEqual(code, 1, out)
        self.assertIn("could not be checked", out)
        self.assertNotIn("MATCH", out)

    def test_check_model_runs_this_check_for_an_audio_fingerprint_and_checks_every_receipt(self):
        def check(model, path, key):
            return check_model.check_one(model, path, key, KEYS)
        code, out = run(self.fixture, self.fixture.answer, check=check)
        self.assertEqual(code, 0, out)
        self.assertIn("%s against the fingerprint of example/audio-model: matches" % MODEL, out)
        self.assertIn("  receipts: every answer signed by EmpirioLabs for %s" % MODEL, out)
        code, out = run(self.fixture, self.fixture.answer, check=check, signed=False)
        self.assertEqual(code, 1, out)
        self.assertIn("  receipts: 0 of 3 answers carry a valid receipt naming %s" % MODEL, out)

    def test_a_clip_that_is_not_the_listed_file_is_not_signed(self):
        def ask(model, probe, request, endpoint, key, exchanges=None, send=None, download=None):
            clip = self.fixture.answer(probe)
            sent = json.dumps(probe).encode()
            received = json.dumps({"result": {"data": [{"sha256": "0" * 64}]}}).encode()
            receipt = json.dumps(receipt_for(sent, received, model=model), separators=(",", ":")).encode()
            exchanges.append((sent, received[:-1] + b',"receipt":' + receipt + b"}", None, clip, True))
            return clip
        out = io.StringIO()
        with mock.patch.object(ca, "ask", ask), contextlib.redirect_stdout(out):
            code = check_model.check_one(MODEL, self.fixture.path, "not-a-key", KEYS)
        self.assertEqual(code, 1, out.getvalue())
        self.assertIn("is not a file this answer lists", out.getvalue())

    def test_the_command_line_check_writes_the_signatures_it_read(self):
        target = os.path.join(self.folder, "answers.json")
        with mock.patch.object(ca, "ask", lambda model, probe, request, endpoint, key, exchanges=None, send=None,
                               download=None: self.fixture.answer(probe)), \
                mock.patch.dict(os.environ, {"EMPIRIOLABS_API_KEY": "k"}), contextlib.redirect_stdout(io.StringIO()):
            code = ca.main(["check", "--model", MODEL, "--fingerprint", self.fixture.path, "--out", target,
                            "--keys", "none"])
        with open(target, encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertEqual(code, 0)
        self.assertEqual((written["result"], len(written["signatures"])), ("matches", 3))

    def with_references(self, *shifts):
        """The fixture with each probe's one signature replaced by `references`: the clip of seed + shift for each
        shift."""
        for row in self.fixture.fingerprint["probes"]:
            items = []
            for shift in shifts:
                rate, channels = self.fixture.clip(row, seed_shift=shift)
                items.append({"seconds": len(channels[0]) / rate,
                              "signature": ca.encode_signature(ca.signature(*ca.decode_wav(wav(rate, channels))))})
            del row["signature"], row["seconds"]
            row["references"] = items
        with open(self.fixture.path, "w", encoding="utf-8") as handle:
            json.dump(self.fixture.fingerprint, handle)

    def test_a_clip_matches_the_closest_of_a_probes_references(self):
        self.with_references(50, 0, 60)
        code, out = run(self.fixture, self.fixture.answer)
        self.assertEqual(code, 0, out)
        self.assertIn("typical distance: 0.00 dB", out)
        for shift in (50, 60):
            code, out = run(self.fixture, lambda probe: self.fixture.answer(probe, seed_shift=shift))
            self.assertEqual(code, 0, out)

    def test_a_clip_far_from_every_reference_does_not_match(self):
        self.with_references(50, 0, 60)
        code, out = run(self.fixture, lambda probe: self.fixture.answer(probe, seed_shift=100))
        self.assertEqual(code, 1, out)
        self.assertIn("DOES NOT MATCH", out)

    def test_a_probe_of_one_reference_reads_as_one_signature(self):
        rows = self.fixture.fingerprint["probes"]
        single = [ca.decode_signature(row["signature"]) for row in rows]
        self.with_references(0)
        self.assertEqual([ca.references(row) for row in rows], [[s] for s in single])
        live = [ca.clip_signature(self.fixture.answer(row, seed_shift=100)) for row in rows]
        before = {**self.fixture.fingerprint, "probes": [{"signature": ca.encode_signature(s)} for s in single]}
        self.assertEqual(ca.verdict(self.fixture.fingerprint, live)["distances"], ca.verdict(before, live)["distances"])

    def test_an_endpoint_the_check_does_not_ask_is_refused(self):
        self.fixture.fingerprint["endpoint"] = "/v1/audio/transcriptions"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ca.check_one(MODEL, self.fixture.path, "k", fingerprint=self.fixture.fingerprint)
        self.assertEqual(code, 1)
        self.assertIn("could not be checked", out.getvalue())


class Asking(unittest.TestCase):
    def test_music_is_a_job_followed_to_its_end_on_the_apis_own_origin(self):
        calls, clip = [], wav(*song("x", 1, seconds=0.5))

        def send(method, path, key, body=None, accept="application/json"):
            calls.append((method, path, accept))
            sent = json.dumps(body).encode() if body else None
            if method == "POST":
                self.assertEqual((body["format"], body["seed"], body["lyrics"]), ("wav", 5, "la"))
                self.assertNotIn("voice", body)
                return 200, {"Content-Type": "application/json"}, json.dumps(
                    {"job_id": "job/1", "status": "processing", "poll_url": "@elsewhere.example/x"}).encode(), sent
            if len(calls) < 3:
                return 200, {}, b'{"status": "processing"}', sent
            return 200, {}, json.dumps({"status": "completed", "result": {"data": [
                {"url": "https://media.example/c.wav", "sha256": hashlib.sha256(clip).hexdigest()}]}}).encode(), sent
        exchanges = []
        with mock.patch.object(ca.time, "sleep", lambda seconds: None):
            got = ca.ask("m", {"prompt": "x", "seed": 5, "lyrics": "la", "voice": "ignored"}, {"format": "wav"},
                         "/v1/audio/generations", "k", exchanges, send=send, download=lambda url: clip)
        self.assertEqual(got, clip)
        self.assertEqual([c[:2] for c in calls], [("POST", "/v1/audio/generations"), ("GET", "/v1/jobs/job%2F1"),
                                                  ("GET", "/v1/jobs/job%2F1")])
        sent, received, header, kept, listed = exchanges[0]
        self.assertIn(b'"seed": 5', sent)
        self.assertIn(b'"completed"', received)
        self.assertEqual((header, kept, listed), (None, clip, True))

    def test_speech_comes_back_as_bytes_with_its_header_receipt(self):
        clip = wav(*song("hello", 1, seconds=0.5))

        def send(method, path, key, body=None, accept="application/json"):
            self.assertEqual((method, path, accept), ("POST", "/v1/audio/speech", "audio/wav"))
            self.assertEqual((body["input"], body["voice"], body["output_format"]), ("Hello.", "emma", "wav"))
            return 200, {"Content-Type": "audio/wav", "X-Receipt": '{"version": 1}'}, clip, json.dumps(body).encode()
        exchanges = []
        got = ca.ask("m", {"input": "Hello.", "voice": "emma", "seed": 3}, {"output_format": "wav"},
                     "/v1/audio/speech", "k", exchanges, send=send)
        self.assertEqual(got, clip)
        self.assertEqual(exchanges[0][2:], ('{"version": 1}', clip, False))

    def test_a_job_that_failed_or_holds_no_clip_is_no_answer(self):
        def failed(method, path, key, body=None, accept=None):
            if method == "POST":
                return 202, {}, b'{"job_id": "j", "status": "processing"}', b"{}"
            return 200, {}, b'{"status": "failed", "error": {"message": "no"}}', None
        with mock.patch.object(ca.time, "sleep", lambda seconds: None), self.assertRaises(ca.Unanswered):
            ca.ask("m", {"prompt": "x", "seed": 1}, {}, "/v1/audio/generations", "k", send=failed)
        with self.assertRaises(ca.Unanswered):
            ca.ask("m", {"prompt": "x", "seed": 1}, {}, "/v1/audio/generations", "k",
                   send=lambda *a, **k: (200, {}, b'{"data": []}', b"{}"))

    def test_the_request_is_sent_to_the_public_api(self):
        self.assertEqual(ca.ORIGIN, "https://api.empiriolabs.ai")


class Make(unittest.TestCase):
    """The parts of `make` that draw noise, on tiny tensors (no model)."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        self.torch = torch

    def test_the_seeds_latents_are_float32_numbers_from_its_cpu_generator(self):
        torch = self.torch
        latents = ca.seed_latents(42, 250, 64)
        expected = torch.randn((1, 250, 64), generator=torch.Generator(device="cpu").manual_seed(42),
                               dtype=torch.float32)
        self.assertTrue(torch.equal(latents, expected))
        self.assertEqual(latents.dtype, torch.float32)

    def test_cpu_noise_gives_the_modules_seeded_draws_in_order_and_puts_torch_back(self):
        import types

        torch = self.torch
        module = types.ModuleType("stand_in_library")
        exec("import torch\n"
             "def run(x):\n"
             "    a = torch.randn([1, 3, 5], device='cpu')\n"
             "    b = torch.randn_like(x)\n"
             "    return a, b\n", module.__dict__)
        sys.modules["stand_in_library"] = module
        try:
            with ca.cpu_noise(9, ["stand_in_library"]) as seeded:
                a, b = module.run(torch.zeros(2, 4, dtype=torch.float16))
            generator = torch.Generator(device="cpu").manual_seed(9)
            self.assertTrue(torch.equal(a, torch.randn((1, 3, 5), generator=generator, dtype=torch.float32)))
            self.assertTrue(torch.equal(b, torch.randn((2, 4), generator=generator, dtype=torch.float32).half()))
            self.assertEqual([d[2] for d in seeded.draws], ["float32", "float16"])
            self.assertIs(module.torch, torch)
        finally:
            del sys.modules["stand_in_library"]


class StandardLibrary(unittest.TestCase):
    def test_the_check_needs_only_the_standard_library(self):
        names_of = getattr(sys, "stdlib_module_names", None)
        if names_of is None:
            self.skipTest("Python 3.10 or later lists the standard library")
        with open(os.path.join(ROOT, "check_audio.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
        self.assertTrue(names)
        self.assertLessEqual(names, set(names_of) | {"check_receipt"}, names - set(names_of))


if __name__ == "__main__":
    unittest.main()
