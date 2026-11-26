"""Offline tests of check_receipt.py's statement check: EmpirioLabs' signed results of its own checks
(GET /v1/verification). What it accepts, that anything changed is refused, that a statement is never a receipt nor a
receipt a statement, and what `statement --file` prints. Run with: python -m unittest discover -s tests"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_receipt as rc  # noqa: E402
from test_check_receipt import KEYS, SIGNER, Signer, _b64url  # noqa: E402

FACTS = {
    "type": "verification-statement", "version": 1, "issuer": "api.empiriolabs.ai", "key_id": SIGNER.key_id,
    "issued_at": 1795720923,
    "models": [
        {"model": "aplomb-1", "kind": "decision", "weights": "empiriolabsai/aplomb-1",
         "revision": "1225d382f51a29d04c888cdadc8b94f21261c7b2", "fingerprint": "matches", "checked_on": "2026-11-26",
         "schedule": "daily", "answers_signed": True},
        {"model": "flux-2-klein-4b", "kind": "image", "weights": "black-forest-labs/FLUX.2-klein-4B",
         "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294", "fingerprint": "not checked",
         "checked_on": "2026-11-26", "schedule": "on request", "answers_signed": True},
    ],
    "privacy_test": {"models": ["aplomb-1", "qwen3-5-9b"], "since": "2026-11-25", "links_opened": 0,
                     "links_opened_is_minimum": False, "running": True, "checked_on": "2026-11-26"},
    "fingerprint_check": {"running": True, "checked_on": "2026-11-26"},
}


def signed(statement=None, signer=SIGNER):
    statement = copy.deepcopy(FACTS if statement is None else statement)
    statement.pop("signature", None)
    statement["key_id"] = signer.key_id
    statement["signature"] = _b64url(signer.sign(rc.canonical_json(statement)))
    return statement


class Statement(unittest.TestCase):
    def test_a_statement_signed_by_a_listed_key_verifies(self):
        ok, result = rc.check_statement(signed(), KEYS)
        self.assertTrue(ok, result)
        self.assertEqual(result["privacy_test"]["links_opened"], 0)

    def test_a_retired_key_still_verifies_and_an_unlisted_or_revoked_one_does_not(self):
        self.assertTrue(rc.check_statement(signed(), [SIGNER.listed("retired")])[0])
        self.assertFalse(rc.check_statement(signed(), [SIGNER.listed("revoked")])[0])
        other = Signer(b"\x09" * 32)
        ok, why = rc.check_statement(signed(signer=other), KEYS)
        self.assertFalse(ok)
        self.assertIn("not listed", why)

    def test_any_change_is_refused(self):
        changes = [
            lambda s: s.update(issued_at=s["issued_at"] + 1),
            lambda s: s["models"][1].update(fingerprint="matches"),
            lambda s: s["models"].pop(),
            lambda s: s["privacy_test"].update(links_opened=1),
            lambda s: s["privacy_test"].update(running=False),
            lambda s: s["fingerprint_check"].update(checked_on="2026-11-27"),
            lambda s: s.update(note="added"),
        ]
        for change in changes:
            statement = signed()
            change(statement)
            ok, why = rc.check_statement(statement, KEYS)
            self.assertFalse(ok)
            self.assertEqual(why, "the signature is not valid")

    def test_a_statement_is_never_a_receipt_and_a_receipt_never_a_statement(self):
        statement = signed()
        body = b'{"id":"x"}'
        response = body[:-1] + b',"receipt":' + json.dumps(statement, separators=(",", ":")).encode() + b"}"
        self.assertFalse(rc.check(body, response, KEYS)[0])
        receipt = {"version": 1, "issuer": "api.empiriolabs.ai", "key_id": SIGNER.key_id, "served_by": "empiriolabs-native",
                   "model": "qwen3-5-9b", "issued_at": 1795720923, "request_sha256": "0" * 64, "response_sha256": "1" * 64}
        receipt["signature"] = _b64url(SIGNER.sign(rc.canonical_json(receipt)))
        self.assertFalse(rc.check_statement(receipt, KEYS)[0])
        disguised = dict(receipt, type="verification-statement")
        disguised.pop("signature")
        disguised["signature"] = _b64url(SIGNER.sign(rc.canonical_json(disguised)))
        ok, why = rc.check_statement(disguised, KEYS)
        self.assertFalse(ok)
        self.assertEqual(why, "the statement does not name this issuer")

    def test_what_is_not_a_signed_version_1_statement_is_refused(self):
        for statement in (None, [], "x", dict(signed(), version=2), dict(signed(), type="other"),
                          dict(signed(), issuer="example.com")):
            self.assertFalse(rc.check_statement(statement, KEYS)[0])
        unsigned = signed()
        unsigned.pop("signature")
        self.assertEqual(rc.check_statement(unsigned, KEYS), (False, "the statement is not signed"))

    def test_the_command_checks_a_saved_statement_and_prints_its_results(self):
        with tempfile.TemporaryDirectory() as folder:
            keys, saved = os.path.join(folder, "keys.json"), os.path.join(folder, "statement.json")
            with open(keys, "w", encoding="utf-8") as handle:
                json.dump({"keys": KEYS}, handle)
            with open(saved, "w", encoding="utf-8") as handle:
                json.dump(signed(), handle)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = rc.main(["statement", "--file", saved, "--keys", keys])
            self.assertEqual(code, 0, out.getvalue())
            text = out.getvalue()
            self.assertIn("signed by EmpirioLabs", text)
            self.assertIn("privacy test: running, last 2026-11-26; test links opened since 2026-11-25: 0;", text)
            self.assertIn("flux-2-klein-4b", text)
            self.assertIn("on request", text)
            changed = signed()
            changed["privacy_test"]["links_opened"] = 3
            with open(saved, "w", encoding="utf-8") as handle:
                json.dump(changed, handle)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = rc.main(["statement", "--file", saved, "--keys", keys])
            self.assertEqual(code, 1)
            self.assertIn("NOT VERIFIED", out.getvalue())


class MaxAge(unittest.TestCase):
    def test_a_statement_issued_longer_ago_than_max_age_is_not_current_and_a_recent_one_passes(self):
        import time

        for hours, expected in ((30, 1), (0.1, 0)):
            statement = signed(dict(copy.deepcopy(FACTS), issued_at=int(time.time() - hours * 3600)))
            with tempfile.TemporaryDirectory() as folder:
                keys, saved = os.path.join(folder, "keys.json"), os.path.join(folder, "statement.json")
                with open(keys, "w", encoding="utf-8") as handle:
                    json.dump({"keys": KEYS}, handle)
                with open(saved, "w", encoding="utf-8") as handle:
                    json.dump(statement, handle)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = rc.main(["statement", "--file", saved, "--keys", keys, "--max-age", "26"])
            self.assertEqual(code, expected, out.getvalue())
            self.assertIn("hours ago", out.getvalue())
            self.assertEqual("NOT CURRENT" in out.getvalue(), bool(expected))

if __name__ == "__main__":
    unittest.main()
