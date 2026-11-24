"""Offline tests of check_receipt.py: Ed25519 against the standard's own vectors, and the receipt rules (what a
receipt covers, and that anything changed is refused). Run with: python -m unittest discover -s tests"""
import base64
import hashlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_receipt as rc  # noqa: E402

# RFC 8032, section 7.1: TEST 1, TEST 2, TEST 3 and TEST SHA(abc): (public key, message, signature).
RFC_8032 = [
    ("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    ("ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
     "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
     "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704"),
]


def _b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _compress(point):
    inverse = pow(point[2], rc._P - 2, rc._P)
    x, y = point[0] * inverse % rc._P, point[1] * inverse % rc._P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


class Signer:
    """RFC 8032 key generation and signing in plain integers, for the tests only (the seed is a test value)."""

    def __init__(self, seed):
        digest = hashlib.sha512(seed).digest()
        scalar = int.from_bytes(digest[:32], "little")
        scalar &= (1 << 254) - 8
        scalar |= 1 << 254
        self._scalar, self._prefix = scalar, digest[32:]
        self.public = _compress(rc._multiply(scalar, rc._BASE))
        self.key_id = hashlib.sha256(self.public).hexdigest()[:16]

    def sign(self, message):
        r = int.from_bytes(hashlib.sha512(self._prefix + message).digest(), "little") % rc._L
        encoded_r = _compress(rc._multiply(r, rc._BASE))
        h = int.from_bytes(hashlib.sha512(encoded_r + self.public + message).digest(), "little") % rc._L
        return encoded_r + ((r + h * self._scalar) % rc._L).to_bytes(32, "little")

    def listed(self, status="current"):
        return {"key_id": self.key_id, "algorithm": "Ed25519", "public_key": _b64url(self.public), "status": status}


SIGNER = Signer(b"\x07" * 32)
KEYS = [SIGNER.listed()]
REQUEST = json.dumps({"model": "qwen3-5-4b", "messages": [{"role": "user", "content": "Name one fruit."}]}).encode()


def receipt_for(request, covered, signer=SIGNER, **changes):
    receipt = {"version": 1, "issuer": "api.empiriolabs.ai", "key_id": signer.key_id, "served_by": "empiriolabs-native",
               "model": "qwen3-5-4b", "issued_at": 1795651200, "request_sha256": hashlib.sha256(request).hexdigest(),
               "response_sha256": hashlib.sha256(covered).hexdigest()}
    receipt.update(changes)
    receipt["signature"] = _b64url(signer.sign(rc.canonical_json(receipt)))
    return receipt


def whole(content="Apple", **changes):
    body = json.dumps({"id": "chatcmpl-1", "object": "chat.completion", "created": 1795651200, "model": "qwen3-5-4b",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 24, "completion_tokens": 1, "total_tokens": 25}},
                      separators=(",", ":")).encode()
    receipt = json.dumps(receipt_for(REQUEST, body, **changes), separators=(",", ":")).encode()
    return body[:-1] + b',"receipt":' + receipt + b"}"


def stream(pieces=("App", "le"), done=True, **changes):
    events = [b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"'
              + piece.encode() + b'"},"finish_reason":null}]}\n\n' for piece in pieces]
    events.append(b": keep-alive\n\n")
    events.append(b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
    covered = b"".join(events) + (b"data: [DONE]\n\n" if done else b"")
    receipt = json.dumps(receipt_for(REQUEST, covered, **changes), separators=(",", ":")).encode()
    return covered + b": receipt " + receipt + b"\n\n"


class Ed25519(unittest.TestCase):
    def test_the_standards_own_signatures_verify_and_any_change_does_not(self):
        for public, message, signature in RFC_8032:
            public, message, signature = bytes.fromhex(public), bytes.fromhex(message), bytes.fromhex(signature)
            self.assertTrue(rc.ed25519_verify(public, message, signature))
            self.assertFalse(rc.ed25519_verify(public, message + b"x", signature))
            for position in (0, 31, 32, 63):
                changed = bytearray(signature)
                changed[position] ^= 1
                self.assertFalse(rc.ed25519_verify(public, message, bytes(changed)), position)
            other = bytearray(public)
            other[0] ^= 1
            self.assertFalse(rc.ed25519_verify(bytes(other), message, signature))

    def test_wrong_lengths_and_a_second_form_of_one_signature_are_refused(self):
        public, message, signature = (bytes.fromhex(part) for part in RFC_8032[1])
        self.assertFalse(rc.ed25519_verify(public[:31], message, signature))
        self.assertFalse(rc.ed25519_verify(public, message, signature[:63]))
        scalar = int.from_bytes(signature[32:], "little") + rc._L
        self.assertFalse(rc.ed25519_verify(public, message, signature[:32] + scalar.to_bytes(32, "little")))

    def test_the_test_signer_agrees_with_the_standard(self):
        # RFC 8032 TEST 1's secret key signs the empty message into TEST 1's signature.
        signer = Signer(bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"))
        self.assertEqual(signer.public.hex(), RFC_8032[0][0])
        self.assertEqual(signer.sign(b"").hex(), RFC_8032[0][2])


class Receipts(unittest.TestCase):
    def test_a_whole_response_and_a_stream_each_verify(self):
        for response in (whole(), stream()):
            ok, receipt = rc.check(REQUEST, response, KEYS)
            self.assertTrue(ok, receipt)
            self.assertEqual((receipt["model"], receipt["key_id"]), ("qwen3-5-4b", SIGNER.key_id))

    def test_whitespace_around_a_whole_response_does_not_count(self):
        self.assertTrue(rc.check(REQUEST, b" \n \n" + whole() + b"\n", KEYS)[0])

    def test_a_changed_request_response_or_receipt_is_refused(self):
        for response in (whole(), stream()):
            self.assertEqual(rc.check(REQUEST + b" ", response, KEYS), (False, "the receipt is for another request"))
            self.assertEqual(rc.check(REQUEST, response.replace(b"App", b"Bpp"), KEYS),
                             (False, "the receipt is for another response"))
            self.assertEqual(rc.check(REQUEST, response.replace(b'"model":"qwen3-5-4b","issued_at"', b'"model":"qwen3-5-9b","issued_at"'), KEYS),
                             (False, "the signature is not valid"))
            self.assertEqual(rc.check(REQUEST, response.replace(b'"issued_at":1795651200', b'"issued_at":1795651201'), KEYS),
                             (False, "the signature is not valid"))

    def test_a_receipt_from_an_unlisted_key_or_another_issuer_is_refused(self):
        stranger = Signer(b"\x09" * 32)
        for make in (whole, stream):
            self.assertTrue(rc.check(REQUEST, make(signer=stranger), KEYS)[1].startswith("the receipt is signed with a key that is not listed"))
            # A stranger who writes our key's name on its own signature.
            self.assertEqual(rc.check(REQUEST, make(signer=stranger, key_id=SIGNER.key_id), KEYS), (False, "the signature is not valid"))
            self.assertEqual(rc.check(REQUEST, make(issuer="example.com"), KEYS), (False, "the receipt does not name this issuer"))
            self.assertEqual(rc.check(REQUEST, make(served_by="someone-else"), KEYS), (False, "the receipt does not name this issuer"))
            self.assertEqual(rc.check(REQUEST, make(version=2), KEYS), (False, "the receipt is not a version 1 receipt"))

    def test_a_listed_key_whose_name_is_not_its_own_hash_signs_nothing(self):
        stranger = Signer(b"\x0b" * 32)
        forged = dict(stranger.listed(), key_id=SIGNER.key_id)
        self.assertEqual(rc.check(REQUEST, whole(signer=stranger, key_id=SIGNER.key_id), [forged]),
                         (False, "the signature is not valid"))

    def test_a_response_without_a_receipt_says_so(self):
        plain = json.dumps({"id": "chatcmpl-1", "choices": []}).encode()
        self.assertEqual(rc.check(REQUEST, plain, KEYS), (False, "the response carries no receipt"))
        self.assertEqual(rc.check(REQUEST, b"data: {}\n\ndata: [DONE]\n\n", KEYS), (False, "the response carries no receipt"))
        self.assertEqual(rc.check(REQUEST, stream(done=False), KEYS), (False, "the response carries no receipt"))

    def test_answer_text_that_looks_like_a_receipt_changes_nothing(self):
        ok, receipt = rc.check(REQUEST, whole(content=',"receipt":{"version":1}'), KEYS)
        self.assertTrue(ok and receipt["model"] == "qwen3-5-4b")
        ok, receipt = rc.check(REQUEST, stream(pieces=("\\n: receipt {}", "x")), KEYS)
        self.assertTrue(ok and receipt["model"] == "qwen3-5-4b")

    def test_bytes_after_the_receipt_are_refused(self):
        self.assertFalse(rc.check(REQUEST, whole() + b'{"more":1}', KEYS)[0])
        self.assertFalse(rc.check(REQUEST, stream() + b"data: {}\n\n", KEYS)[0])

    def test_only_a_current_or_retired_key_signs(self):
        for status in ("revoked", "", None):
            self.assertTrue(rc.check(REQUEST, whole(), [SIGNER.listed(status)])[1].startswith(
                "the receipt is signed with a key that is not listed"))
        self.assertTrue(rc.check(REQUEST, whole(), [SIGNER.listed("retired")])[0])

    def test_both_shapes_of_a_key_list_are_read(self):
        self.assertEqual(rc._listed({"keys": KEYS}), KEYS)
        self.assertEqual(rc._listed(KEYS), KEYS)
        self.assertEqual(rc._listed({"keys": "x"}), [])


if __name__ == "__main__":
    unittest.main()
