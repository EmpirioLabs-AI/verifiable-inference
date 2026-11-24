"""Check that a model on EmpirioLabs AI is the public model it is said to be, without running any model.

Each file in fingerprints/ was made once, from the model's public weights. A chat model's file (kind
"chat-first-token") holds, for a fixed list of questions, the 20 tokens the model finds most likely to begin its
answer, with their log-probabilities. This script asks the live API the same questions for one token each, with
`logprobs` on, and compares. It also checks the signed receipt on every answer (check_receipt.py), so the verdict
covers both what answered and who served it.

A file's `kind` says which check reads it: "transcription" files are checked by check_transcription.py and "decision"
files by check_decision.py, each with the same verdict line. Their answers carry no receipt, so those checks print
"receipts: not on this endpoint" and their verdict reads the answers alone.

    python check_model.py --model qwen3-5-9b          one model (fingerprints/qwen3-5-9b.json)
    python check_model.py --all                       every model that has a fingerprint

Python 3.9 or later, standard library only. Reads your API key from EMPIRIOLABS_API_KEY. A check of one chat model
sends 50 requests of one output token each (four per question for a model whose API returns no list of alternatives),
billed to your account at the model's listed price.

Verdict
    matches          the live model's first-token probabilities agree with the public weights within the limits the
                     fingerprint states, and every prompt reached the model as it was sent
    DOES NOT MATCH   anything else
    CANNOT TELL      the questions this fingerprint holds are too certain to tell models apart without a limit on the
                     other alternatives, which this file does not set
    signed           every answer carries a valid receipt from EmpirioLabs that names this model
Exit status: 0 matches and signed, 1 anything else, 2 cannot tell.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
import urllib.error

import check_decision
import check_receipt
import check_transcription

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
# The check of every kind but chat, by the file's `kind`.
OTHER_KINDS = {check_transcription.KIND: check_transcription, check_decision.KIND: check_decision}
KINDS = ("chat-first-token",) + tuple(OTHER_KINDS)
TOP = 20
PROMPT_SLACK = 3  # a chat template may render a prompt a token or two longer or shorter, the same on every question
# An answer whose first tokens are its template's framing, which the API leaves out, holds no token at max_tokens 1.
# The question is then asked for this many tokens and the first one returned is read: the first token after the
# framing, which depends only on the prompt.
ROOM = 8
# Where the API returns fewer alternatives than asked, each question is asked this many times for one token: once at
# temperature 0, so the most likely token comes back, then sampled. Each answer carries its token's own log-probability.
SAMPLES = 4
# A question whose most likely token is above this log-probability (above 99%) says almost nothing through the
# weighted gap. When such questions alone could meet the first-token rule, only the other alternatives can tell two
# models apart, so the fingerprint needs a limit on them (`tail_gap`) before the check can say it matches.
NEAR_CERTAIN = -0.01
RETRY_STATUSES = (429, 500, 502, 503, 504)


def load_fingerprint(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    kind = fingerprint.get("kind", "chat-first-token")
    if kind not in KINDS:
        raise ValueError("%s: fingerprint kind %r is not one this script checks" % (os.path.basename(path), kind))
    return fingerprint


def request_body(model, fingerprint, question, max_tokens=1, temperature=1.0):
    """The chat completion a check sends for one question."""
    body = {"model": model, "max_tokens": max_tokens, "reasoning_effort": "none", "logprobs": True,
            "top_logprobs": TOP, "temperature": temperature}
    body.update(fingerprint.get("request") or {})
    body["messages"] = [{"role": "system", "content": fingerprint["system"]}, {"role": "user", "content": question}]
    return body


def read_answer(data):
    """(the first log-probability entry the answer holds or None when it holds none, the prompt length the API
    reported). Raises ValueError when the answer carries no log-probabilities at all."""
    logprobs = data["choices"][0].get("logprobs")
    entries = logprobs.get("content") if isinstance(logprobs, dict) else None
    if entries is None:
        raise ValueError("the answer carries no log-probabilities")
    return (entries[0] if entries else None), (data.get("usage") or {}).get("prompt_tokens")


def _post(body, key):
    for attempt in range(4):
        try:
            return check_receipt.post("/chat/completions", body, key, timeout=120)
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUSES or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


class Asker:
    """Sends a check's requests and keeps every answer's receipt verdict. `post(body, key)` returns (the request bytes
    sent, the response bytes received); the default retries a rate limit or a server error a few times."""

    def __init__(self, model, fingerprint, key, keys, post=None):
        self.model, self.fingerprint, self.key, self.keys = model, fingerprint, key, keys
        self.post = post or _post
        self.receipts = []

    def first_token(self, question, temperature=1.0):
        for max_tokens in (1, ROOM):
            sent, received = self.post(request_body(self.model, self.fingerprint, question, max_tokens, temperature), self.key)
            if self.keys is not None:
                self.receipts.append(check_receipt.check(sent, received, self.keys))
            entry, prompt_tokens = read_answer(json.loads(received))
            if entry is not None:
                return entry, prompt_tokens
        raise ValueError("no token came back within %d" % ROOM)

    def top(self, question):
        entry, prompt_tokens = self.first_token(question)
        return [[a["token"], a["logprob"]] for a in entry.get("top_logprobs") or []], prompt_tokens

    def sample(self, question, count=SAMPLES):
        """The first token asked `count` times (the first at temperature 0), each with the log-probability the API
        gives it, averaged where a token comes back again. The prompt length is None unless every answer reports the
        same one."""
        seen, lengths = {}, set()
        for attempt in range(count):
            entry, prompt_tokens = self.first_token(question, temperature=0.0 if attempt == 0 else 1.0)
            seen.setdefault(entry["token"], []).append(entry["logprob"])
            lengths.add(prompt_tokens)
        return [[token, sum(values) / len(values)] for token, values in seen.items()], (
            lengths.pop() if len(lengths) == 1 else None)


def score(rows, answers, limits):
    """The verdict on `answers` (for each row in order: [token, log-probability] pairs, and the prompt length) against
    a fingerprint's rows and limits."""
    max_gap = float(limits.get("typical_gap", 0.15))
    min_first = float(limits.get("first_token", 0.8))
    tail_limit = limits.get("tail_gap")
    same_first, near_certain = 0, 0
    gaps, tails, shared, offsets = [], [], [], set()
    for row, (got, prompt_tokens) in zip(rows, answers):
        offsets.add((prompt_tokens or 0) - row["prompt_tokens"])
        reference = dict((token, value) for token, value in row["top"])
        live = dict((token, value) for token, value in got)
        best = max(reference, key=reference.get)
        near_certain += reference[best] > NEAR_CERTAIN
        same_first += bool(live) and best == max(live, key=live.get)
        common = [token for token in reference if token in live]
        shared.append(len(common))
        # Weighted by how likely the public weights find each token, so the tokens that carry the answer count most.
        weights = [2.718281828 ** reference[token] for token in common]
        if common:
            gaps.append(sum(w * abs(reference[t] - live[t]) for w, t in zip(weights, common)) / sum(weights))
        others = [token for token in common if token != best]
        if others:
            tails.append(statistics.median(abs(reference[t] - live[t]) for t in others))
    n = len(rows)
    typical = statistics.median(gaps) if gaps else float("inf")
    tail = statistics.median(tails) if tails else None
    # The prompt the model saw has to be the prompt that was sent: a chat template may differ by the same token or two
    # on every question, and anything added to a prompt is longer than that.
    same_prompt = len(offsets) == 1 and abs(next(iter(offsets))) <= PROMPT_SLACK
    passed = len(answers) == n and same_first / n >= min_first and typical <= max_gap and same_prompt
    tail_read = tail_limit is not None and tail is not None
    if tail_read:
        passed = passed and tail <= float(tail_limit)
    decisive = tail_read or near_certain < min_first * n
    result = "matches" if passed and decisive else "CANNOT TELL" if passed else "DOES NOT MATCH"
    return {"result": result, "same_first": same_first, "questions": n, "needs": int(min_first * n + 0.999),
            "typical_gap": typical, "limit": max_gap, "tail_gap": tail, "tail_limit": tail_limit,
            "near_certain": near_certain, "shared": sum(shared) / n if n else 0.0, "prompt_as_sent": same_prompt,
            "offsets": sorted(offsets)[:4]}


def run_check(model, fingerprint, key, keys, samples=SAMPLES, post=None):
    """Ask every question; (the verdict, the receipt verdicts, how many questions were read from sampled answers)."""
    asker = Asker(model, fingerprint, key, keys, post=post)
    answers, sampled = [], 0
    for row in fingerprint["rows"]:
        got, prompt_tokens = asker.top(row["question"]) if not sampled else ([], None)
        if len(got) < TOP:
            # No full list of the most likely tokens: this question and every later one are read from sampled answers.
            got, prompt_tokens = asker.sample(row["question"], samples)
            sampled += 1
        answers.append((got, prompt_tokens))
    return score(fingerprint["rows"], answers, fingerprint.get("limits") or {}), asker.receipts, sampled


def report(model, fingerprint, verdict, receipts, keys, sampled, seconds):
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], verdict["result"]))
    print("  same most likely first token: %d of %d questions (needs %d)" % (
        verdict["same_first"], verdict["questions"], verdict["needs"]))
    print("  typical gap between the log-probabilities: %.3f (limit %.2f)" % (verdict["typical_gap"], verdict["limit"]))
    if verdict["tail_limit"] is not None or verdict["result"] == "CANNOT TELL":
        print("  typical gap among the other alternatives: %s (limit %s)" % (
            "none read" if verdict["tail_gap"] is None else "%.3f" % verdict["tail_gap"],
            "none set" if verdict["tail_limit"] is None else "%.2f" % float(verdict["tail_limit"])))
    print("  tokens in common among the 20 most likely: %.1f on average" % verdict["shared"])
    print("  prompt as sent: %s" % ("yes" if verdict["prompt_as_sent"] else "NO, lengths differ by %s" % verdict["offsets"]))
    if sampled:
        print("  the API returned no list of alternatives, so %d questions were read from sampled answers" % sampled)
    if keys is None:
        print("  receipts: not checked")
        signed = True
    else:
        named = sum(1 for ok, r in receipts if ok and r.get("model") == model)
        signed = named == len(receipts)
        reasons = sorted(set(r for ok, r in receipts if not ok))
        print("  receipts: %s" % ("every answer signed by EmpirioLabs for %s" % model if signed else
                                  "%d of %d answers carry a valid receipt naming %s%s" % (
                                      named, len(receipts), model, (": " + "; ".join(reasons[:3])) if reasons else "")))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, keys):
    """0 matches and signed, 1 anything else, 2 cannot tell. A transcription or a decision fingerprint is checked by
    its own module, which reads no receipt: its answers carry none."""
    started = time.time()
    try:
        fingerprint = load_fingerprint(path)
    except (OSError, ValueError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    kind = fingerprint.get("kind", "chat-first-token")
    if kind in OTHER_KINDS:
        return OTHER_KINDS[kind].check_one(model, path, key, fingerprint)
    try:
        verdict, receipts, sampled = run_check(model, fingerprint, key, keys)
    except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, verdict, receipts, keys, sampled, time.time() - started)
    if verdict["result"] == "CANNOT TELL":
        return 2
    return 0 if verdict["result"] == "matches" and signed else 1


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
    results = [check_one(model, path, key, keys) for model, path in targets]
    if not results or 1 in results:
        return 1
    return 2 if 2 in results else 0


if __name__ == "__main__":
    sys.exit(main())
