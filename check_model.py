"""Check that a model on EmpirioLabs AI is the public model it is said to be, without running any model.

Each file in fingerprints/ was made once, from the model's public weights: for a fixed list of questions, the 20
tokens the model finds most likely to begin its answer, with their log-probabilities. This script asks the live API
the same questions for one token each, with `logprobs` on, and compares. It also checks the signed receipt on every
answer (check_receipt.py), so the verdict covers both what answered and who served it.

    python check_model.py --model qwen3-5-9b          one model (fingerprints/qwen3-5-9b.json)
    python check_model.py --all                       every model that has a fingerprint

Python 3.9 or later, standard library only. Reads your API key from EMPIRIOLABS_API_KEY. A check of one model sends
50 requests of one output token each, billed to your account at the model's listed price.

Verdict
    matches          the live model's first-token probabilities agree with the public weights within the limits the
                     fingerprint states, and every prompt reached the model as it was sent
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
import urllib.error

import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
TOP = 20
PROMPT_SLACK = 3  # a chat template may render a prompt a token or two longer or shorter, the same on every question
DEFAULT_REQUEST = {"max_tokens": 1, "reasoning_effort": "none", "logprobs": True, "top_logprobs": TOP, "temperature": 1.0}
KINDS = ("chat-first-token",)


def load_fingerprint(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    kind = fingerprint.get("kind", "chat-first-token")
    if kind not in KINDS:
        raise SystemExit("%s: fingerprint kind %r is not one this script checks" % (os.path.basename(path), kind))
    return fingerprint


def ask(model, fingerprint, question, key, keys):
    """One question. Returns (the 20 most likely first tokens with their log-probabilities, the prompt length the API
    reported, the receipt verdict as (ok, receipt or reason))."""
    request = dict(DEFAULT_REQUEST, **(fingerprint.get("request") or {}))
    body = dict(request, model=model, messages=[{"role": "system", "content": fingerprint["system"]},
                                                 {"role": "user", "content": question}])
    for attempt in range(4):
        try:
            sent, received = check_receipt.post("/chat/completions", body, key, timeout=120)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    signed = check_receipt.check(sent, received, keys) if keys is not None else (None, "not checked")
    data = json.loads(received)
    piece = data["choices"][0]["logprobs"]["content"][0]
    top = [[alternative["token"], alternative["logprob"]] for alternative in piece["top_logprobs"]]
    return top, (data.get("usage") or {}).get("prompt_tokens"), signed


def compare(fingerprint, answers):
    """The verdict on a model's answers to every question of `fingerprint`. `answers` holds, per row in order, the
    API's (top tokens, prompt length)."""
    limits = fingerprint.get("limits") or {}
    max_gap = float(limits.get("typical_gap", 0.15))
    min_first = float(limits.get("first_token", 0.8))
    same_first, gaps, shared, offsets = 0, [], [], set()
    for row, (top, prompt_tokens) in zip(fingerprint["rows"], answers):
        offsets.add((prompt_tokens or 0) - row["prompt_tokens"])
        reference = dict((token, value) for token, value in row["top"])
        live = dict((token, value) for token, value in top)
        same_first += max(reference, key=reference.get) == max(live, key=live.get)
        common = [token for token in reference if token in live]
        shared.append(len(common))
        # Weighted by how likely the public weights find each token, so the tokens that carry the answer count most.
        weights = [2.718281828 ** reference[token] for token in common]
        if common:
            gaps.append(sum(w * abs(reference[t] - live[t]) for w, t in zip(weights, common)) / sum(weights))
        else:
            gaps.append(float("inf"))
    n = len(fingerprint["rows"])
    typical = statistics.median(gaps) if gaps else float("inf")
    same_prompt = len(offsets) == 1 and abs(next(iter(offsets))) <= PROMPT_SLACK
    matches = len(answers) == n and same_first / n >= min_first and typical <= max_gap and same_prompt
    return {"result": "matches" if matches else "DOES NOT MATCH", "questions": n, "same_first": same_first,
            "needs_first": int(min_first * n + 0.999), "typical_gap": typical, "limit": max_gap,
            "shared": sum(shared) / n if n else 0.0, "prompt_as_sent": same_prompt, "offsets": sorted(offsets)}


def check_model(model, path, key, keys):
    fingerprint = load_fingerprint(path)
    started = time.time()
    answers, receipts = [], []
    for row in fingerprint["rows"]:
        top, prompt_tokens, signed = ask(model, fingerprint, row["question"], key, keys)
        answers.append((top, prompt_tokens))
        receipts.append(signed)
    verdict = compare(fingerprint, answers)
    good = [r for ok, r in receipts if ok]
    named = sum(1 for r in good if r.get("model") == model)
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], verdict["result"]))
    print("  same most likely first token: %d of %d questions (needs %d)" % (
        verdict["same_first"], verdict["questions"], verdict["needs_first"]))
    print("  typical gap between the log-probabilities: %.3f (limit %.2f)" % (verdict["typical_gap"], verdict["limit"]))
    print("  tokens in common among the 20 most likely: %.1f on average" % verdict["shared"])
    print("  prompt as sent: %s" % ("yes" if verdict["prompt_as_sent"] else "NO, lengths differ by %s" % verdict["offsets"][:4]))
    if keys is None:
        print("  receipts: not checked")
        signed_ok = True
    else:
        signed_ok = named == len(receipts)
        reasons = sorted(set(r for ok, r in receipts if not ok))
        print("  receipts: %s" % ("every answer signed by EmpirioLabs for %s" % model if signed_ok else
                                  "%d of %d answers carry a valid receipt naming %s%s" % (
                                      named, len(receipts), model, (": " + "; ".join(reasons[:3])) if reasons else "")))
    print("  %.0f s, no model was run" % (time.time() - started))
    return verdict["result"] == "matches" and signed_ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", help="the model ID on EmpirioLabs AI, for example qwen3-5-9b")
    which.add_argument("--all", action="store_true", help="check every model that has a fingerprint")
    parser.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    parser.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    args = parser.parse_args(argv)
    key = check_receipt.api_key()
    keys = None if args.keys == "none" else check_receipt.load_keys(args.keys)
    if args.all:
        targets = [(os.path.basename(p)[:-len(".json")], p) for p in sorted(glob.glob(os.path.join(FINGERPRINTS, "*.json")))]
    else:
        targets = [(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"))]
    results = [check_model(model, path, key, keys) for model, path in targets]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
