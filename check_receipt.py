"""Check the signed receipt on a chat completion from EmpirioLabs AI.

Every successful chat completion from a model that EmpirioLabs serves on its own managed GPU infrastructure (a Native
Inference model) carries a receipt signed by EmpirioLabs. The receipt names the model and holds the SHA-256 of the
request body exactly as it was sent and of the response exactly as it was received, so anyone who kept those two
byte strings can show later what was asked, what was answered, and that EmpirioLabs served it.

    python check_receipt.py live  --model qwen3-5-9b        one request, not streamed and streamed, both checked
    python check_receipt.py files --request request.json --response response.bin

Python 3.9 or later, standard library only. `live` reads your API key from EMPIRIOLABS_API_KEY.

Where the receipt travels
    not streamed  the last member of the JSON body: ...,"receipt":{...}}
                  It covers the body with that member removed, without leading or trailing whitespace.
    streamed      the last event of the stream, an SSE comment after `data: [DONE]`: `: receipt {...}`
                  It covers every byte before that comment.

The receipt
    {"version": 1, "issuer": "api.empiriolabs.ai", "key_id": ..., "served_by": "empiriolabs-native", "model": ...,
     "issued_at": <unix seconds>, "request_sha256": ..., "response_sha256": ..., "signature": ...}
    The signature is Ed25519 (base64url without padding) over the receipt without "signature", written as JSON with
    sorted keys and no spaces.

Keys
    keys/receipt-keys.json in this repository lists every key that has signed receipts. The API publishes the same list
    at https://api.empiriolabs.ai/v1/receipts/keys; `--keys live` uses that list instead.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request

API = "https://api.empiriolabs.ai/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
PINNED_KEYS = os.path.join(HERE, "keys", "receipt-keys.json")
USER_AGENT = "verifiable-inference/1.0"
VERSION = 1
ISSUER = "api.empiriolabs.ai"
SERVED_BY = "empiriolabs-native"
JSON_MEMBER = b',"receipt":'
STREAM_COMMENT = b": receipt "
STREAM_END = b"data: [DONE]\n\n"
WHITESPACE = b" \t\r\n"
USABLE = ("current", "retired")  # a listed key in any other state signs nothing we accept

# ---------------------------------------------------------------------------------------------------------------
# Ed25519 verification (RFC 8032, section 5.1.7) in plain integers. Verification handles no secret, so nothing here
# needs to run in constant time. Any Ed25519 library gives the same answer.
# ---------------------------------------------------------------------------------------------------------------
_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _recover_x(y, sign):
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    return _P - x if (x & 1) != sign else x


def _point(encoded):
    """A curve point (X, Y, Z, T) from its 32 bytes, or None when they are not a point."""
    if len(encoded) != 32:
        return None
    y = int.from_bytes(encoded, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _add(p, q):
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * p[3] * q[3] * _D % _P
    d = 2 * p[2] * q[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _multiply(scalar, p):
    result = (0, 1, 1, 0)
    while scalar > 0:
        if scalar & 1:
            result = _add(result, p)
        p = _add(p, p)
        scalar >>= 1
    return result


_BASE_Y = 4 * pow(5, _P - 2, _P) % _P
_BASE_X = _recover_x(_BASE_Y, 0)
_BASE = (_BASE_X, _BASE_Y, 1, _BASE_X * _BASE_Y % _P)


def ed25519_verify(public_key, message, signature):
    """True when `signature` (64 bytes) is the Ed25519 signature of `message` under `public_key` (32 bytes)."""
    if len(public_key) != 32 or len(signature) != 64:
        return False
    a = _point(public_key)
    r = _point(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if a is None or r is None or s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public_key + message).digest(), "little") % _L
    left = _multiply(s, _BASE)
    right = _add(r, _multiply(h, a))
    return (left[0] * right[2] - right[0] * left[2]) % _P == 0 and (left[1] * right[2] - right[1] * left[2]) % _P == 0


# ---------------------------------------------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------------------------------------------
def b64url_decode(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def canonical_json(value):
    """The bytes a receipt's signature covers: sorted keys, no spaces, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def key_id(public_key):
    """A key's name: the first 16 hex characters of the SHA-256 of the raw 32-byte public key."""
    return hashlib.sha256(public_key).hexdigest()[:16]


def split_json(body):
    """(the bytes the receipt covers, the receipt) of a response that came whole, or None when it carries none."""
    body = body.strip(WHITESPACE)
    start = body.rfind(JSON_MEMBER + b"{")
    if start < 0 or not body.endswith(b"}}"):
        return None
    try:
        receipt = json.loads(body[start + len(JSON_MEMBER):-1])
    except ValueError:
        return None
    return (body[:start] + b"}", receipt) if isinstance(receipt, dict) else None


def split_stream(body):
    """(the bytes the receipt covers, the receipt) of a streamed response, or None when it carries none."""
    start = body.rfind(b"\n" + STREAM_COMMENT)
    if start < 0 or not body.endswith(b"\n\n"):
        return None
    covered, line = body[:start + 1], body[start + 1 + len(STREAM_COMMENT):-2]
    if b"\n" in line or b"\r" in line or not covered.endswith(STREAM_END):
        return None
    try:
        receipt = json.loads(line)
    except ValueError:
        return None
    return (covered, receipt) if isinstance(receipt, dict) else None


def check(request_body, response_body, keys):
    """Check one exchange against `keys`, the published list of {"key_id", "public_key", "status"}.

    Returns (True, the receipt) or (False, the reason). True means: the receipt is signed by one of `keys`, and it
    names exactly this request body and exactly this response."""
    streamed = response_body.lstrip(WHITESPACE)[:1] != b"{"
    parts = split_stream(response_body) if streamed else split_json(response_body)
    if parts is None:
        return False, "the response carries no receipt"
    covered, receipt = parts
    signature = receipt.get("signature")
    if receipt.get("version") != VERSION or not isinstance(signature, str):
        return False, "the receipt is not a version %d receipt" % VERSION
    if receipt.get("issuer") != ISSUER or receipt.get("served_by") != SERVED_BY:
        return False, "the receipt does not name this issuer"
    key = next((k for k in keys if k.get("key_id") == receipt.get("key_id") and k.get("status") in USABLE), None)
    if key is None:
        return False, "the receipt is signed with a key that is not listed (%s)" % receipt.get("key_id")
    unsigned = dict((name, value) for name, value in receipt.items() if name != "signature")
    try:
        public = b64url_decode(key["public_key"])
        valid = key_id(public) == key["key_id"] and ed25519_verify(public, canonical_json(unsigned),
                                                                 b64url_decode(signature))
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        return False, "the signature is not valid"
    if receipt.get("request_sha256") != hashlib.sha256(request_body).hexdigest():
        return False, "the receipt is for another request"
    if receipt.get("response_sha256") != hashlib.sha256(covered).hexdigest():
        return False, "the receipt is for another response"
    return True, receipt


# ---------------------------------------------------------------------------------------------------------------
# Keys and requests
# ---------------------------------------------------------------------------------------------------------------
def api_key():
    key = os.environ.get("EMPIRIOLABS_API_KEY") or os.environ.get("EMPIRIO_API_KEY")
    if not key:
        sys.exit("Set EMPIRIOLABS_API_KEY to an EmpirioLabs AI API key (https://platform.empiriolabs.ai/dashboard/api-keys).")
    return key


def _listed(data):
    keys = data.get("keys") if isinstance(data, dict) else data
    return [k for k in keys if isinstance(k, dict)] if isinstance(keys, list) else []


def live_keys():
    """The keys the API publishes."""
    request = urllib.request.Request(API + "/receipts/keys", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return _listed(json.loads(response.read()))


def load_keys(source="pinned"):
    """The keys to check against: `pinned` (keys/receipt-keys.json in this repository), `live` (the API's list), or
    the path of a JSON file of either shape."""
    if source == "live":
        return live_keys()
    path = PINNED_KEYS if source == "pinned" else source
    with open(path, encoding="utf-8") as handle:
        return _listed(json.load(handle))


def post(path, body, key, timeout=300):
    """Send one request; (the request bytes sent, the response bytes received)."""
    sent = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(API + path, data=sent, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return sent, response.read()


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------
def _live(args):
    key = api_key()
    keys = load_keys(args.keys)
    failed = 0
    for stream in (False, True):
        body = {"model": args.model, "max_tokens": 16, "stream": stream, "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name one fruit. Answer with the single word only."}]}
        sent, received = post("/chat/completions", body, key)
        ok, result = check(sent, received, keys)
        failed += not ok
        print("%s %-13s %s" % (args.model, "streamed:" if stream else "not streamed:",
                               "signed by EmpirioLabs, model %s, key %s" % (result["model"], result["key_id"]) if ok
                               else "NOT VERIFIED, " + result))
    return 1 if failed else 0


def _files(args):
    with open(args.request, "rb") as handle:
        sent = handle.read()
    with open(args.response, "rb") as handle:
        received = handle.read()
    ok, result = check(sent, received, load_keys(args.keys))
    print("signed by EmpirioLabs: model %s, issued at %s, key %s" % (result["model"], result["issued_at"], result["key_id"])
          if ok else "NOT VERIFIED: " + result)
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("live", help="send one request each way and check both receipts")
    one.add_argument("--model", required=True, help="a Native Inference chat model ID, for example qwen3-5-9b")
    one.add_argument("--keys", default="pinned", help="pinned (default), live, or a keys JSON file")
    two = sub.add_parser("files", help="check a saved request body and the response received for it")
    two.add_argument("--request", required=True, help="the request body, byte for byte as it was sent")
    two.add_argument("--response", required=True, help="the response body, byte for byte as it was received")
    two.add_argument("--keys", default="pinned", help="pinned (default), live, or a keys JSON file")
    args = parser.parse_args(argv)
    return _live(args) if args.command == "live" else _files(args)


if __name__ == "__main__":
    sys.exit(main())
