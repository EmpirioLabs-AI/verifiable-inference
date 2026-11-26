"""Check the signed receipt on an answer from EmpirioLabs AI.

Every successful answer from a model that EmpirioLabs runs on GPU infrastructure it manages (a Native Inference model)
carries a receipt signed by EmpirioLabs: chat, completions, Responses, Messages and Gemini answers, transcripts,
decisions, OCR, images, video, audio, speech and 3D. The receipt names the model and holds the SHA-256 of the request
body exactly as it was sent and of the response exactly as it was received, so anyone who kept those two byte strings
can show later what was asked, what was answered, and that EmpirioLabs served it.

    python check_receipt.py live  --model qwen3-5-9b       a chat model: one request, not streamed and streamed
    python check_receipt.py live  --model flux-2-klein-4b  an image model: one image, its file checked too
    python check_receipt.py files --request request.json --response response.bin [--header ...] [--output FILE ...]
    python check_receipt.py statement [--file statement.json]  EmpirioLabs' signed results of its own daily checks

Python 3.9 or later, standard library only. `live` reads your API key from EMPIRIOLABS_API_KEY.

Where the receipt travels
    JSON          the last member of the body: ...,"receipt":{...}}
                  It covers the body with that member removed, without leading or trailing whitespace.
    a stream      the last event, an SSE comment after the stream's own last event: `: receipt {...}`
                  It covers every byte before that comment.
    other bytes   audio, a transcript as text or subtitles: the X-Receipt response header holds the receipt.
                  It covers the whole body.

A job (an image, video, audio, 3D or OCR request that runs in the background) is answered later, by the request that
waited for it or by GET /v1/jobs/{id}. That answer's receipt names the POST that made the job: check it with the POST's
body as the request. Each generated file in the answer lists the SHA-256 of its bytes (`sha256`), which the receipt
covers; `--output` checks a downloaded file against it.

The receipt
    {"version": 1, "issuer": "api.empiriolabs.ai", "key_id": ..., "served_by": "empiriolabs-native", "model": ...,
     "issued_at": <unix seconds>, "request_sha256": ..., "response_sha256": ..., "signature": ...}
    The signature is Ed25519 (base64url without padding) over the receipt without "signature", written as JSON with
    sorted keys and no spaces.

The statement
    GET https://api.empiriolabs.ai/v1/verification (no API key needed) answers EmpirioLabs' latest results of its own
    checks, signed with the same key as every receipt and in the same way: each Native Inference model's last check
    against its public fingerprint (matches, does not match, or not checked; the date; whether every answer of that
    check was signed) and the privacy test (which models it covers, whether it is running, and how many of its test
    links have ever been opened). Dates only. `statement` fetches it, or reads one you saved, and checks the signature.
    {"type": "verification-statement", "version": 1, "issuer": "api.empiriolabs.ai", "key_id": ...,
     "issued_at": <unix seconds>, "models": [...], "privacy_test": {...}, "fingerprint_check": {...}, "signature": ...}
    A statement is never a receipt: it names no request or response, and no receipt names a "type".

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
import time
import urllib.parse
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
EVENT_END = b"\n\n"
HEADER = "X-Receipt"
WHITESPACE = b" \t\r\n"
USABLE = ("current", "retired")  # a listed key in any other state signs nothing we accept
STATEMENT_TYPE = "verification-statement"

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
    """(the bytes the receipt covers, the receipt) of a streamed response, or None when it carries none. The receipt
    is the stream's last event and follows a complete event: the bytes it covers end with a blank line."""
    start = body.rfind(b"\n" + STREAM_COMMENT)
    if start < 0 or not body.endswith(EVENT_END):
        return None
    covered, line = body[:start + 1], body[start + 1 + len(STREAM_COMMENT):-2]
    if b"\n" in line or b"\r" in line or not covered.endswith(EVENT_END):
        return None
    try:
        receipt = json.loads(line)
    except ValueError:
        return None
    return (covered, receipt) if isinstance(receipt, dict) else None


def split_header(body, header):
    """(the bytes the receipt covers, the receipt) of an answer whose X-Receipt header is `header`, or None."""
    try:
        receipt = json.loads(header)
    except (TypeError, ValueError):
        return None
    return (body, receipt) if isinstance(receipt, dict) else None


def signature_problem(signed, keys, what):
    """Why `signed` (a receipt or a statement) is not signed by one of `keys`, or None when it is: its key_id names a
    current or retired listed key whose name is its own hash, and its signature verifies over every other member."""
    key = next((k for k in keys if k.get("key_id") == signed.get("key_id") and k.get("status") in USABLE), None)
    if key is None:
        return "the %s is signed with a key that is not listed (%s)" % (what, signed.get("key_id"))
    unsigned = dict((name, value) for name, value in signed.items() if name != "signature")
    try:
        public = b64url_decode(key["public_key"])
        valid = key_id(public) == key["key_id"] and ed25519_verify(public, canonical_json(unsigned),
                                                                 b64url_decode(signed["signature"]))
    except (ValueError, KeyError, TypeError):
        valid = False
    return None if valid else "the signature is not valid"


def check(request_body, response_body, keys, header=None):
    """Check one exchange against `keys`, the published list of {"key_id", "public_key", "status"}. `header` is the
    answer's X-Receipt header, for an answer that is neither JSON nor a stream (audio, text, subtitles).

    Returns (True, the receipt) or (False, the reason). True means: the receipt is signed by one of `keys`, and it
    names exactly this request body and exactly this response."""
    if header is not None:
        parts = split_header(response_body, header)
    elif response_body.lstrip(WHITESPACE)[:1] == b"{":
        parts = split_json(response_body)
    else:
        parts = split_stream(response_body)
    if parts is None:
        return False, "the response carries no receipt"
    covered, receipt = parts
    signature = receipt.get("signature")
    if receipt.get("version") != VERSION or not isinstance(signature, str):
        return False, "the receipt is not a version %d receipt" % VERSION
    if receipt.get("issuer") != ISSUER or receipt.get("served_by") != SERVED_BY:
        return False, "the receipt does not name this issuer"
    why = signature_problem(receipt, keys, "receipt")
    if why:
        return False, why
    if receipt.get("request_sha256") != hashlib.sha256(request_body).hexdigest():
        return False, "the receipt is for another request"
    if receipt.get("response_sha256") != hashlib.sha256(covered).hexdigest():
        return False, "the receipt is for another response"
    return True, receipt


def report_receipts(model, verdicts):
    """Print a model check's receipt line from `verdicts`, one (ok, the receipt or why) per answer. Returns True when
    every answer carries a valid receipt naming `model`, False when one does not, None when none were checked."""
    if verdicts is None:
        print("  receipts: not checked")
        return None
    named = sum(1 for ok, result in verdicts if ok and isinstance(result, dict) and result.get("model") == model)
    signed = bool(verdicts) and named == len(verdicts)
    reasons = sorted(set(result for ok, result in verdicts if not ok))
    print("  receipts: %s" % ("every answer signed by EmpirioLabs for %s" % model if signed else
                              "%d of %d answers carry a valid receipt naming %s%s" % (
                                  named, len(verdicts), model, (": " + "; ".join(reasons[:3])) if reasons else "")))
    return signed


def check_statement(statement, keys):
    """Check a verification statement (GET /v1/verification) against `keys`. Returns (True, the statement) or
    (False, the reason). True means: EmpirioLabs signed exactly these results with one of `keys`."""
    if not isinstance(statement, dict):
        return False, "this is not a statement"
    if statement.get("type") != STATEMENT_TYPE or statement.get("version") != VERSION:
        return False, "this is not a version %d verification statement" % VERSION
    if not isinstance(statement.get("signature"), str):
        return False, "the statement is not signed"
    if statement.get("issuer") != ISSUER or "served_by" in statement:
        return False, "the statement does not name this issuer"
    why = signature_problem(statement, keys, "statement")
    return (False, why) if why else (True, statement)


def output_hashes(response_body):
    """The SHA-256 of every generated file a JSON answer lists (`sha256` of each item of `data`, also inside a job's
    `result`), in order. The receipt covers these values as part of the answer."""
    parts = split_json(response_body)
    try:
        answer = json.loads(parts[0] if parts else response_body.strip(WHITESPACE))
    except ValueError:
        return []
    found = []
    for holder in (answer, answer.get("result") if isinstance(answer, dict) else None):
        items = holder.get("data") if isinstance(holder, dict) else None
        for item in items if isinstance(items, list) else []:
            value = item.get("sha256") if isinstance(item, dict) else None
            if isinstance(value, str) and len(value) == 64:
                found.append(value.lower())
    return found


def check_outputs(response_body, contents):
    """(True, "") when every file in `contents` (bytes) is one the answer lists by its SHA-256, else (False, why)."""
    listed = output_hashes(response_body)
    for number, content in enumerate(contents, 1):
        if hashlib.sha256(content).hexdigest() not in listed:
            return False, "output %d is not a file this answer lists" % number
    return True, ""


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


def live_statement():
    """The statement the API publishes now."""
    request = urllib.request.Request(API + "/verification", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


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


def download(url, timeout=300):
    """The bytes of a generated file's link."""
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as response:
        return response.read()


def endpoints_of(model, key):
    """The endpoints the API lists for `model` (`supported_endpoints`), or [] when it cannot say."""
    request = urllib.request.Request(API + "/models/" + urllib.parse.quote(model, safe=""),
                                     headers={"Authorization": "Bearer " + key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            listed = json.loads(response.read()).get("supported_endpoints")
    except (OSError, ValueError, AttributeError):
        return []
    return [str(endpoint) for endpoint in listed] if isinstance(listed, list) else []


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------
def _verdict(model, label, ok, result):
    print("%s %-13s %s" % (model, label, "signed by EmpirioLabs, model %s, key %s" % (result["model"], result["key_id"])
                           if ok else "NOT VERIFIED, " + result))


def _live_chat(model, key, keys):
    failed = 0
    for stream in (False, True):
        body = {"model": model, "max_tokens": 16, "stream": stream, "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name one fruit. Answer with the single word only."}]}
        sent, received = post("/chat/completions", body, key)
        ok, result = check(sent, received, keys)
        failed += not ok
        _verdict(model, "streamed:" if stream else "not streamed:", ok, result)
    return 1 if failed else 0


def _live_image(model, key, keys):
    body = {"model": model, "prompt": "A red apple on a white table, soft studio light.", "n": 1}
    sent, received = post("/images/generations", body, key, timeout=900)
    ok, result = check(sent, received, keys)
    _verdict(model, "image:", ok, result)
    if not ok:
        return 1
    answer = json.loads(split_json(received)[0])
    links = [item["url"] for item in answer.get("data") or [] if isinstance(item, dict) and item.get("url")]
    good, why = check_outputs(received, [download(link) for link in links]) if links else (False, "the answer lists no file")
    print("%s %-13s %s" % (model, "file:", "%d file(s) match the SHA-256 the receipt covers" % len(links) if good
                           else "NOT VERIFIED, " + why))
    return 0 if good else 1


def _live(args):
    key = api_key()
    keys = load_keys(args.keys)
    endpoints = endpoints_of(args.model, key)
    if not endpoints or "POST /v1/chat/completions" in endpoints:
        return _live_chat(args.model, key, keys)
    if "POST /v1/images/generations" in endpoints:
        return _live_image(args.model, key, keys)
    print("%s answers on %s. Keep the request body you sent and the response you received, then run: "
          "python check_receipt.py files --request ... --response ..." % (args.model, ", ".join(endpoints)))
    return 2


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


def _files(args):
    sent, received = _read(args.request), _read(args.response)
    header = args.header
    if header is not None and header.startswith("@"):
        header = _read(header[1:]).decode("utf-8").strip()
    ok, result = check(sent, received, load_keys(args.keys), header=header)
    print("signed by EmpirioLabs: model %s, issued at %s, key %s" % (result["model"], result["issued_at"], result["key_id"])
          if ok else "NOT VERIFIED: " + result)
    if not ok or not args.output:
        return 0 if ok else 1
    good, why = check_outputs(received, [_read(path) for path in args.output])
    print("files: %d match the SHA-256 the receipt covers" % len(args.output) if good else "NOT VERIFIED: " + why)
    return 0 if good else 1


def _statement(args):
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            statement = json.load(handle)
    else:
        statement = live_statement()
    ok, result = check_statement(statement, load_keys(args.keys))
    if not ok:
        print("NOT VERIFIED: " + result)
        return 1
    issued = result.get("issued_at")
    age = (time.time() - issued) / 3600.0 if isinstance(issued, int) and not isinstance(issued, bool) else None
    print("signed by EmpirioLabs: issued at %s (%s), key %s" % (
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(issued)) if age is not None else issued,
        "%.1f hours ago" % age if age is not None else "no time", result.get("key_id")))
    if args.max_age is not None and (age is None or age > args.max_age):
        print("NOT CURRENT: issued more than %g hours ago" % args.max_age)
        return 1
    privacy = result.get("privacy_test") if isinstance(result.get("privacy_test"), dict) else {}
    checks = result.get("fingerprint_check") if isinstance(result.get("fingerprint_check"), dict) else {}
    print("privacy test: %s, last %s; test links opened since %s: %s%s; models: %s" % (
        "running" if privacy.get("running") is True else "NOT RUNNING", privacy.get("checked_on"),
        privacy.get("since"), privacy.get("links_opened"),
        " or more" if privacy.get("links_opened_is_minimum") is True else "",
        ", ".join(str(m) for m in privacy.get("models") or []) or "none"))
    print("model check:  %s, last %s" % ("running" if checks.get("running") is True else "NOT RUNNING",
                                         checks.get("checked_on")))
    for entry in result.get("models") or []:
        if isinstance(entry, dict):
            print("  %-24s %-13s %-14s %-10s %-10s %s" % (
                entry.get("model"), entry.get("kind"), entry.get("fingerprint"), entry.get("checked_on"),
                entry.get("schedule"), "answers signed" if entry.get("answers_signed") is True else "NOT ALL SIGNED"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("live", help="send a request and check its receipt (chat and image models)")
    one.add_argument("--model", required=True, help="a Native Inference model ID, for example qwen3-5-9b")
    one.add_argument("--keys", default="pinned", help="pinned (default), live, or a keys JSON file")
    two = sub.add_parser("files", help="check a saved request body and the response received for it")
    two.add_argument("--request", required=True,
                     help="the request body, byte for byte as it was sent (for a job's answer, the POST that made the job)")
    two.add_argument("--response", required=True, help="the response body, byte for byte as it was received")
    two.add_argument("--header", help="the answer's X-Receipt header, or @file holding it (audio, text, subtitles)")
    two.add_argument("--output", nargs="+", help="generated files you downloaded, checked against the answer's sha256")
    two.add_argument("--keys", default="pinned", help="pinned (default), live, or a keys JSON file")
    three = sub.add_parser("statement", help="check EmpirioLabs' signed results of its own daily checks")
    three.add_argument("--file", help="a statement you saved from /v1/verification (default: fetch it now)")
    three.add_argument("--keys", default="pinned", help="pinned (default), live, or a keys JSON file")
    three.add_argument("--max-age", type=float, metavar="HOURS",
                       help="fail a statement issued longer ago than this (a saved one shown as today's, for example)")
    args = parser.parse_args(argv)
    if args.command == "statement":
        return _statement(args)
    return _live(args) if args.command == "live" else _files(args)


if __name__ == "__main__":
    sys.exit(main())
