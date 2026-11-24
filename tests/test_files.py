"""Offline tests of the published files: every fingerprint names public weights exactly and holds only what a check
needs, the key list is consistent, and the README lists every fingerprint. Run with: python -m unittest discover -s tests"""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERPRINTS = os.path.join(ROOT, "fingerprints")
KEYS = os.path.join(ROOT, "keys", "receipt-keys.json")
FIELDS = {"kind", "model", "weights", "revision", "weight_files_sha256", "system", "made_with", "made_at", "limits", "rows",
          "request"}
REQUIRED = FIELDS - {"request"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Fingerprints(unittest.TestCase):
    def files(self):
        names = sorted(name for name in os.listdir(FINGERPRINTS) if name.endswith(".json"))
        self.assertTrue(names)
        for name in names:
            with open(os.path.join(FINGERPRINTS, name), encoding="utf-8") as handle:
                yield name, json.load(handle)

    def test_each_file_names_public_weights_exactly_and_nothing_else(self):
        for name, data in self.files():
            self.assertLessEqual(set(data), FIELDS, name)
            self.assertLessEqual(REQUIRED, set(data), name)
            self.assertEqual(data["kind"], "chat-first-token", name)
            self.assertEqual(name, data["model"] + ".json")
            self.assertRegex(data["model"], MODEL_ID)
            self.assertRegex(data["weights"], REPOSITORY)
            self.assertRegex(data["revision"], HEX40)
            self.assertTrue(data["weight_files_sha256"], name)
            for file_name, digest in data["weight_files_sha256"].items():
                self.assertTrue(file_name.endswith(".safetensors"), (name, file_name))
                self.assertRegex(digest, HEX64)

    def test_each_file_asks_the_same_questions_and_holds_twenty_tokens_each(self):
        questions = None
        for name, data in self.files():
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

    def test_each_limit_is_stated_and_not_loose(self):
        for name, data in self.files():
            self.assertGreaterEqual(data["limits"]["first_token"], 0.8, name)
            self.assertLessEqual(data["limits"]["typical_gap"], 0.25, name)


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
