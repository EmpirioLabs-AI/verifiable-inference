"""Offline tests of the published files: every fingerprint names public weights exactly and holds only what its kind's
check needs, the clips beside a transcription fingerprint are the files it records, the key list is consistent, and
the README lists every fingerprint. Run with: python -m unittest discover -s tests"""
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERPRINTS = os.path.join(ROOT, "fingerprints")
KEYS = os.path.join(ROOT, "keys", "receipt-keys.json")
# Every kind: the fields a file may hold, and the ones it must.
COMMON = {"kind", "model", "weights", "revision", "weight_files_sha256", "made_with", "made_at", "limits"}
FIELDS = {
    "chat-first-token": COMMON | {"system", "rows", "request"},
    "transcription": COMMON | {"request", "clips"},
    "decision": COMMON | {"requests"},
    "image": COMMON | {"request", "settings", "adapters", "probes"},
    "ocr": COMMON | {"request", "pages"},
    "audio": COMMON | {"endpoint", "request", "probes"},
    "video": COMMON | {"request", "settings", "components", "probes"},
    "mesh": COMMON | {"request", "components", "probes"},
}
REQUIRED = {
    "chat-first-token": FIELDS["chat-first-token"] - {"request"},
    "transcription": FIELDS["transcription"],
    "decision": FIELDS["decision"],
    "image": COMMON | {"request", "probes"},
    "ocr": COMMON | {"request", "pages"},
    "audio": COMMON | {"endpoint", "request", "probes"},
    "video": COMMON | {"request", "probes"},
    "mesh": COMMON | {"request", "probes"},
}
LIMITS = {
    "chat-first-token": {"typical_gap", "first_token"},
    "transcription": {"same_transcript", "language_gap", "logprob_gap"},
    "decision": {"same_answer", "typical_gap"},
    "image": {"image_distance", "typical_distance", "close"},
    "ocr": {"page_cer", "typical_cer", "box_gap", "typical_box_gap", "close"},
    "audio": {"clip_distance", "typical_distance", "close"},
    "video": {"clip_distance", "typical_distance", "close"},
    "mesh": {"shape_distance", "typical_distance", "box_gap", "close"},
}
PAGE_FIELDS = {"file", "sha256", "source", "license", "processing", "width", "height", "output"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
AUDIO_PROBE_FIELDS = {"prompt", "lyrics", "input", "voice", "voice_model", "voice_s1", "voice_s2", "voice_s3",
                      "voice_s4", "seed", "audio_duration", "num_inference_steps", "guidance_scale", "shift",
                      "instrumental", "speed", "sample_rate", "model_quality", "volume", "temperature", "top_k",
                      "top_p", "repetition_penalty", "seconds", "signature", "references"}
# A reference clip of a probe that holds several.
AUDIO_REFERENCE_FIELDS = {"seconds", "signature"}
PROBE_FIELDS = {"prompt", "seed", "width", "height", "thumbnail"}
VIDEO_PROBE_FIELDS = {"prompt", "seed", "image", "frame_count", "width", "height", "frames", "thumbnails"}
MESH_PROBE_FIELDS = {"image", "seed", "box", "shape", "colors", "faces"}
PICTURE_FIELDS = {"file", "sha256", "source", "license", "processing"}
COMPONENT_FIELDS = {"role", "weights", "revision", "file", "sha256", "scale"}
PICTURE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")
CLIP_FIELDS = {"file", "sha256", "seconds", "source", "license", "cut", "processing", "language", "languages", "tokens",
               "text", "avg_logprob", "no_speech_prob"}
# A weight file of a public checkpoint: safetensors, or a speech or music model's PyTorch and ONNX parts.
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".onnx")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
MADE_AT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def files(kind=None):
    names = sorted(name for name in os.listdir(FINGERPRINTS) if name.endswith(".json"))
    for name in names:
        with open(os.path.join(FINGERPRINTS, name), encoding="utf-8") as handle:
            data = json.load(handle)
        if kind is None or data.get("kind") == kind:
            yield name, data


class Fingerprints(unittest.TestCase):
    def test_every_file_is_of_a_kind_a_check_reads(self):
        found = list(files())
        self.assertTrue(found)
        for name, data in found:
            self.assertIn(data.get("kind"), FIELDS, name)

    def test_each_file_names_public_weights_exactly_and_nothing_else(self):
        for name, data in files():
            kind = data["kind"]
            self.assertLessEqual(set(data), FIELDS[kind], name)
            self.assertLessEqual(REQUIRED[kind], set(data), name)
            self.assertEqual(name, data["model"] + ".json")
            self.assertRegex(data["model"], MODEL_ID)
            self.assertRegex(data["weights"], REPOSITORY)
            self.assertRegex(data["revision"], HEX40)
            self.assertRegex(data["made_at"], MADE_AT)
            self.assertIsInstance(data["made_with"], str)
            self.assertTrue(data["weight_files_sha256"], name)
            for file_name, digest in data["weight_files_sha256"].items():
                self.assertTrue(file_name.endswith(WEIGHT_SUFFIXES), (name, file_name))
                self.assertRegex(digest, HEX64)

    def test_each_limit_is_stated_and_not_loose(self):
        for name, data in files():
            limits = data["limits"]
            self.assertLessEqual(LIMITS[data["kind"]], set(limits), name)
            if data["kind"] == "chat-first-token":
                self.assertGreaterEqual(limits["first_token"], 0.8, name)
                self.assertLessEqual(limits["typical_gap"], 0.25, name)
            elif data["kind"] == "transcription":
                self.assertTrue(0.8 <= limits["same_transcript"] <= 1, name)
                self.assertTrue(0 < limits["language_gap"] <= 0.2, name)
                self.assertTrue(0 < limits["logprob_gap"] <= 0.05, name)
            elif data["kind"] == "image":
                self.assertTrue(0 < limits["typical_distance"] <= 0.15 and limits["typical_distance"] <= limits["image_distance"] <= 0.25, name)
                self.assertTrue(0.75 <= limits["close"] <= 1, name)
            elif data["kind"] == "ocr":
                self.assertTrue(0 < limits["typical_cer"] <= limits["page_cer"] <= 0.1, name)
                self.assertTrue(0 < limits["typical_box_gap"] <= limits["box_gap"] <= 15, name)
                self.assertTrue(0.75 <= limits["close"] <= 1, name)
            elif data["kind"] == "audio":
                self.assertTrue(0 < limits["typical_distance"] <= limits["clip_distance"] <= 10, name)
                self.assertTrue(0.75 <= limits["close"] <= 1, name)
            elif data["kind"] == "video":
                self.assertTrue(0 < limits["typical_distance"] <= limits["clip_distance"] <= 0.3, name)
                self.assertTrue(0.75 <= limits["close"] <= 1, name)
            elif data["kind"] == "mesh":
                self.assertTrue(0 < limits["typical_distance"] <= limits["shape_distance"] <= 0.3, name)
                self.assertTrue(0 < limits["box_gap"] <= 0.1, name)
                self.assertTrue(0 < limits.get("color_distance", 0.1) <= 0.3, name)
                self.assertTrue(0.75 <= limits["close"] <= 1, name)
            else:
                self.assertTrue(0.8 <= limits["same_answer"] <= 1, name)
                self.assertTrue(0 < limits["typical_gap"] <= 0.05, name)

    def test_each_chat_file_asks_the_same_questions_and_holds_twenty_tokens_each(self):
        questions = None
        for name, data in files("chat-first-token"):
            rows = data["rows"]
            self.assertEqual(len(rows), 50, name)
            asked = [row["question"] for row in rows]
            self.assertEqual(len(set(asked)), 50, name)
            questions = questions or asked
            self.assertEqual(asked, questions, name)
            for row in rows:
                self.assertEqual(set(row), {"question", "prompt_tokens", "top"}, name)
                self.assertIsInstance(row["prompt_tokens"], int)
                self.assertEqual(len(row["top"]), 20, (name, row["question"]))
                self.assertTrue(all(isinstance(token, str) and value <= 0 for token, value in row["top"]),
                                (name, row["question"]))

    def test_each_transcription_file_holds_public_domain_clips_it_records_exactly(self):
        for name, data in files("transcription"):
            folder = data["model"]
            self.assertIsInstance(data["request"], dict, name)
            self.assertTrue(data["request"], name)
            self.assertGreaterEqual(len(data["clips"]), 10, name)
            total = 0
            for clip in data["clips"]:
                self.assertEqual(set(clip), CLIP_FIELDS, (name, clip.get("file")))
                directory, base = clip["file"].split("/")
                self.assertEqual(directory, folder, clip["file"])
                with open(os.path.join(FINGERPRINTS, folder, base), "rb") as handle:
                    raw = handle.read()
                total += len(raw)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), clip["sha256"], clip["file"])
                self.assertEqual(raw[:4], b"fLaC", clip["file"])
                self.assertLessEqual(clip["seconds"], 10, clip["file"])
                self.assertIn("public domain", clip["license"].lower(), clip["file"])
                sources = clip["source"] if isinstance(clip["source"], list) else [clip["source"]]
                self.assertTrue(all(s.startswith("https://") for s in sources), clip["file"])
                self.assertEqual(len(clip["languages"]), 10, clip["file"])
                self.assertTrue(all(isinstance(code, str) and value <= 0 for code, value in clip["languages"]),
                                clip["file"])
                self.assertEqual(clip["language"], clip["languages"][0][0], clip["file"])
                self.assertTrue(clip["tokens"] and all(isinstance(t, int) for t in clip["tokens"]), clip["file"])
                self.assertIsInstance(clip["text"], str)
                self.assertLessEqual(clip["avg_logprob"], 0, clip["file"])
            self.assertLessEqual(total, 1500000, name)
            # The folder holds the recorded clips and nothing else.
            self.assertEqual(sorted(os.listdir(os.path.join(FINGERPRINTS, folder))),
                             sorted(clip["file"].split("/")[1] for clip in data["clips"]), name)

    def test_each_decision_file_holds_requests_with_an_answer_for_every_question(self):
        for name, data in files("decision"):
            self.assertTrue(data["requests"], name)
            for request in data["requests"]:
                self.assertEqual(set(request), {"state", "questions", "answers"}, name)
                self.assertTrue(request["questions"], name)
                self.assertEqual(set(request["answers"]), set(request["questions"]), name)
            self.assertGreaterEqual(sum(len(r["answers"]) for r in data["requests"]), 40, name)

    def test_each_image_file_holds_eight_or_more_probes_each_with_a_whole_thumbnail(self):
        for name, data in files("image"):
            self.assertGreaterEqual(len(data["probes"]), 8, name)
            self.assertEqual(len({probe["seed"] for probe in data["probes"]}), len(data["probes"]), name)
            for probe in data["probes"]:
                self.assertLessEqual(set(probe), PROBE_FIELDS, name)
                self.assertTrue(isinstance(probe["prompt"], str) and probe["prompt"], name)
                self.assertIsInstance(probe["seed"], int)
                raw = base64.b64decode(probe["thumbnail"])
                self.assertEqual(len(raw), 3 * 32 * 32, (name, probe["seed"]))
            self.assertEqual(data["request"].get("format"), "png", name)
            for adapter in data.get("adapters") or []:
                self.assertEqual(set(adapter), {"weights", "revision", "file", "sha256", "scale"}, name)
                self.assertRegex(adapter["weights"], REPOSITORY)
                self.assertRegex(adapter["revision"], HEX40)
                self.assertTrue(adapter["file"].endswith(".safetensors"), name)
                self.assertRegex(adapter["sha256"], HEX64)

    def test_each_ocr_file_holds_public_domain_pages_it_records_exactly(self):
        for name, data in files("ocr"):
            folder = data["model"]
            self.assertGreaterEqual(len(data["pages"]), 6, name)
            total = 0
            for page in data["pages"]:
                self.assertEqual(set(page), PAGE_FIELDS, (name, page.get("file")))
                directory, base = page["file"].split("/")
                self.assertEqual(directory, folder, page["file"])
                with open(os.path.join(FINGERPRINTS, folder, base), "rb") as handle:
                    raw = handle.read()
                total += len(raw)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), page["sha256"], page["file"])
                self.assertEqual(raw[:8], PNG_SIGNATURE, page["file"])
                self.assertLessEqual(max(page["width"], page["height"]), 2048, page["file"])
                license_text = page["license"].lower()
                self.assertTrue("public domain" in license_text or "cc0" in license_text, page["file"])
                self.assertTrue(page["source"].startswith("https://"), page["file"])
                self.assertTrue(isinstance(page["output"], str) and page["output"].strip(), page["file"])
            self.assertLessEqual(total, 1500000, name)
            # The folder holds the recorded pages and nothing else.
            self.assertEqual(sorted(os.listdir(os.path.join(FINGERPRINTS, folder))),
                             sorted(page["file"].split("/")[1] for page in data["pages"]), name)

    def test_each_audio_file_holds_probes_each_with_a_whole_signature(self):
        for name, data in files("audio"):
            self.assertIn(data["endpoint"], ("/v1/audio/generations", "/v1/audio/speech"), name)
            self.assertIn("wav", (data["request"].get("format"), data["request"].get("output_format")), name)
            self.assertGreaterEqual(len(data["probes"]), 6, name)
            self.assertEqual(len({probe["seed"] for probe in data["probes"]}), len(data["probes"]), name)
            for probe in data["probes"]:
                self.assertLessEqual(set(probe), AUDIO_PROBE_FIELDS, name)
                self.assertTrue(isinstance(probe.get("prompt") or probe.get("input"), str), name)
                self.assertIsInstance(probe["seed"], int)
                # One signature, or several references, never both.
                self.assertEqual(("signature" in probe) + ("references" in probe), 1, (name, probe["seed"]))
                references = probe.get("references") or [{"seconds": probe.get("seconds"),
                                                          "signature": probe["signature"]}]
                self.assertTrue(isinstance(references, list) and references, (name, probe["seed"]))
                for reference in references:
                    self.assertLessEqual(set(reference), AUDIO_REFERENCE_FIELDS, (name, probe["seed"]))
                    raw = base64.b64decode(reference["signature"])
                    self.assertTrue(raw and len(raw) % 30 == 0, (name, probe["seed"]))
                    # One row of 30 bands for every started quarter second of the clip.
                    self.assertEqual(len(raw) // 30, max(1, math.ceil(reference["seconds"] * 4 - 1e-9)),
                                     (name, probe["seed"]))

    def assert_pictures_recorded_exactly(self, name, data):
        """A video or mesh file's pictures sit in the folder of its model's name, each the file it records, public
        domain or CC0 with an https source, and the folder holds nothing else."""
        folder = data["model"]
        recorded = [probe["image"] for probe in data["probes"] if probe.get("image")]
        total = 0
        for picture in recorded:
            self.assertLessEqual(set(picture), PICTURE_FIELDS, (name, picture.get("file")))
            directory, base = picture["file"].split("/")
            self.assertEqual(directory, folder, picture["file"])
            with open(os.path.join(FINGERPRINTS, folder, base), "rb") as handle:
                raw = handle.read()
            total += len(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), picture["sha256"], picture["file"])
            self.assertTrue(raw.startswith(PICTURE_SIGNATURES), picture["file"])
            license_text = picture["license"].lower()
            self.assertTrue("public domain" in license_text or "cc0" in license_text, picture["file"])
            self.assertTrue(picture["source"].startswith("https://"), picture["file"])
        self.assertLessEqual(total, 3000000, name)
        if recorded:
            self.assertEqual(sorted(os.listdir(os.path.join(FINGERPRINTS, folder))),
                             sorted({picture["file"].split("/")[1] for picture in recorded}), name)

    def assert_components_named_exactly(self, name, data):
        for component in data.get("components") or []:
            self.assertLessEqual({"role", "weights", "revision"}, set(component), name)
            self.assertLessEqual(set(component), COMPONENT_FIELDS, name)
            self.assertRegex(component["weights"], REPOSITORY)
            self.assertRegex(component["revision"], HEX40)
            if "sha256" in component:
                self.assertRegex(component["sha256"], HEX64)

    def test_each_video_file_holds_probes_each_with_whole_thumbnails_of_the_frames_it_names(self):
        for name, data in files("video"):
            self.assertGreaterEqual(len(data["probes"]), 4, name)
            self.assertEqual(len({probe["seed"] for probe in data["probes"]}), len(data["probes"]), name)
            for probe in data["probes"]:
                self.assertLessEqual(set(probe), VIDEO_PROBE_FIELDS, name)
                self.assertTrue(isinstance(probe["prompt"], str) and probe["prompt"], name)
                self.assertIsInstance(probe["seed"], int)
                self.assertIsInstance(probe["frame_count"], int)
                self.assertEqual(probe["frames"], sorted(set(probe["frames"])), (name, probe["seed"]))
                self.assertTrue(all(0 <= index < probe["frame_count"] for index in probe["frames"]), name)
                self.assertEqual(len(probe["frames"]), len(probe["thumbnails"]), (name, probe["seed"]))
                for thumbnail in probe["thumbnails"]:
                    self.assertEqual(len(base64.b64decode(thumbnail)), 3 * 32 * 32, (name, probe["seed"]))
            self.assert_pictures_recorded_exactly(name, data)
            self.assert_components_named_exactly(name, data)

    def test_each_mesh_file_holds_pictures_with_seeds_and_a_whole_signature_each(self):
        for name, data in files("mesh"):
            self.assertGreaterEqual(len(data["probes"]), 6, name)
            for probe in data["probes"]:
                self.assertLessEqual(set(probe), MESH_PROBE_FIELDS, name)
                self.assertLessEqual({"image", "seed", "box", "shape"}, set(probe), name)
                self.assertIsInstance(probe["seed"], int)
                self.assertEqual(len(probe["box"]), 3, name)
                self.assertTrue(all(0 <= side <= 1 for side in probe["box"]) and max(probe["box"]) == 1, name)
                self.assertEqual(len(base64.b64decode(probe["shape"])), 16 ** 3, (name, probe["seed"]))
                if "colors" in probe:
                    self.assertEqual(len(base64.b64decode(probe["colors"])), 3 * 8 ** 3, (name, probe["seed"]))
            self.assert_pictures_recorded_exactly(name, data)
            self.assert_components_named_exactly(name, data)

    def test_nothing_but_the_clips_pages_or_pictures_of_a_fingerprint_sits_beside_the_fingerprints(self):
        folders = {data["model"] for _, data in files("transcription")} | {data["model"] for _, data in files("ocr")}
        folders |= {data["model"] for kind in ("video", "mesh") for _, data in files(kind)
                    if any(probe.get("image") for probe in data["probes"])}
        for entry in os.listdir(FINGERPRINTS):
            if not entry.endswith(".json"):
                self.assertIn(entry, folders, entry)


class Keys(unittest.TestCase):
    def test_every_listed_key_is_named_by_its_own_hash_and_one_is_current(self):
        with open(KEYS, encoding="utf-8") as handle:
            keys = json.load(handle)["keys"]
        self.assertTrue(keys)
        for key in keys:
            self.assertEqual(set(key), {"key_id", "algorithm", "public_key", "status"})
            self.assertEqual(key["algorithm"], "Ed25519")
            raw = base64.urlsafe_b64decode(key["public_key"] + "=" * (-len(key["public_key"]) % 4))
            self.assertEqual(len(raw), 32)
            self.assertEqual(hashlib.sha256(raw).hexdigest()[:16], key["key_id"])
            self.assertIn(key["status"], ("current", "retired"))
        self.assertEqual(sum(1 for key in keys if key["status"] == "current"), 1)
        self.assertEqual(len({key["key_id"] for key in keys}), len(keys))


class Readme(unittest.TestCase):
    def test_the_readme_lists_every_fingerprint(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "readme_table.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
