"""Check that a decision model on EmpirioLabs AI is the public model it is said to be, without running any model.

A fingerprint of kind "decision" (fingerprints/<model>.json) was made once, from the model's public weights and the
reference code published with them: the probabilities the model gives for a fixed set of decision questions (yes or
no, choice, score, not in the state, tool) about a fixed set of short states. The check sends the same requests to the
live API and compares every probability in the answers with the fingerprint: the same answers, and a typical largest
difference within its limit.

    python check_model.py --model aplomb-1               the usual way, as for every other model
    python check_decision.py check --model aplomb-1      the same check, --out keeps the answers
    python check_decision.py make --weights <folder> --name empiriolabsai/aplomb-1 --revision <commit> \
        --model aplomb-1 --out fingerprints/aplomb-1.json --same-answer <share> --typical-gap <limit>

The check uses only the Python standard library and reads your API key from EMPIRIOLABS_API_KEY. It sends each of the
fingerprint's requests once, billed to your account at the model's listed price. `make` runs the public weights with
the reference code published with them and needs torch and that code's own requirements.

Verdict
    matches          the same answer on at least the share of questions the fingerprint states, and the typical
                     largest difference between the probabilities within its limit
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model (check_receipt.py)
Exit status: 0 matches and signed, 1 anything else.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error

import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "decision"

YES_NO = {"type": "noul"}
REQUESTS = [
    {"state": "My package was supposed to arrive on Monday. It is now Thursday and the tracking page has not changed "
              "since Saturday. I need it for a birthday party this weekend.",
     "questions": {
         "urgent": {**YES_NO, "instructions": "Does the customer need this solved before the weekend?"},
         "refund": {**YES_NO, "instructions": "Is the customer asking for a refund?",
                    "criteria": {"true": "The message asks for money back.",
                                 "false": "The message reports a problem without asking for money back."}},
         "team": {"type": "choice", "instructions": "Which team should handle this message?",
                  "criteria": {"shipping": "Deliveries, tracking and lost parcels", "billing": "Charges, refunds and invoices",
                               "technical": "Bugs, errors and outages", "sales": "Plans, pricing and upgrades"}},
         "mood": {"type": "score", "instructions": "How upset is the customer?",
                  "criteria": ["Calm", "Annoyed", "Frustrated", "Very angry"]},
         "carrier": {"type": "choice", "instructions": "Which carrier is delivering the package?",
                     "criteria": {"ups": None, "fedex": None, "dhl": None, "usps": None}, "abstain": True},
         "order_number": {**YES_NO, "instructions": "Does the message give an order number?", "abstain": True}}},
    {"state": "The blender is powerful and crushes ice in seconds, but it is so loud that my neighbours complained, and "
              "the lid started leaking after two weeks.",
     "questions": {
         "sentiment": {"type": "choice", "instructions": "What is the overall sentiment of the review?",
                       "criteria": {"positive": None, "negative": None, "mixed": None, "neutral": None}},
         "recommend": {**YES_NO, "instructions": "Would the reviewer recommend this blender to a friend?"},
         "stars": {"type": "score", "instructions": "How many stars would this review most likely give?",
                   "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
         "defect": {**YES_NO, "instructions": "Does the review describe a defect in the product?"},
         "complaint": {"type": "choice", "instructions": "What does the reviewer complain about most?",
                       "criteria": {"power": None, "noise": None, "durability": None, "price": None}}}},
    {"state": {"order": {"id": "A-1009", "status": "delivered", "items": 3, "paid": True, "delivered_at": "2026-11-02",
                         "returns_open": 1}, "customer": {"tier": "gold", "member_since": 2019}},
     "questions": {
         "paid": {**YES_NO, "instructions": "Has the order been paid?"},
         "return_open": {**YES_NO, "instructions": "Has the customer opened a return for this order?", "abstain": True},
         "tier": {"type": "choice", "instructions": "Which loyalty tier is the customer in?",
                  "criteria": {"bronze": None, "silver": None, "gold": None, "platinum": None}},
         "tenure": {"type": "score", "instructions": "How long has the customer been a member?",
                    "criteria": ["Under a year", "One to three years", "Three to six years", "More than six years"]},
         "discount": {**YES_NO, "instructions": "Should the customer be offered a loyalty discount on the next order?"},
         "gift": {**YES_NO, "instructions": "Was the order sent as a gift?", "abstain": True}}},
    {"state": "Der Stadtrat hat am Dienstag beschlossen, die Buslinie 12 ab Januar alle zehn Minuten fahren zu lassen. "
              "Die Kosten von 1,2 Millionen Euro im Jahr sollen zum Teil durch höhere Parkgebühren gedeckt werden.",
     "questions": {
         "topic": {"type": "choice", "instructions": "What is the main topic of the text?",
                   "criteria": {"transport": None, "finance": None, "education": None, "health": None}},
         "parking": {**YES_NO, "instructions": "Will parking become more expensive?"},
         "start": {"type": "choice", "instructions": "When does the new timetable start?",
                   "criteria": {"january": None, "february": None, "march": None}, "abstain": True},
         "dispute": {"type": "score", "instructions": "How controversial is the decision likely to be?",
                     "criteria": ["Not controversial", "Somewhat controversial", "Very controversial"]}}},
    {"state": "Cliente: Hola, quiero cambiar mi plan al anual.\nAgente: Claro, ¿desea mantener el mismo método de pago?\n"
              "Cliente: No, prefiero pagar con transferencia bancaria.",
     "questions": {
         "intent": {"type": "choice", "instructions": "¿Qué quiere hacer el cliente?",
                    "criteria": {"upgrade": "Cambiar a un plan mayor o más largo", "cancel": "Darse de baja",
                                 "complaint": "Quejarse de un problema", "question": "Hacer una pregunta general"}},
         "new_payment": {**YES_NO, "instructions": "¿Quiere el cliente cambiar el método de pago?"},
         "satisfaction": {"type": "score", "instructions": "¿Qué tan satisfecho parece el cliente?",
                          "criteria": ["Insatisfecho", "Neutral", "Satisfecho"]},
         "language": {"type": "choice", "instructions": "In which language is the conversation?",
                      "criteria": {"spanish": None, "portuguese": None, "italian": None, "catalan": None}}}},
    {"state": "def average(values):\n    return sum(values) / len(values)\n",
     "questions": {
         "empty_fails": {**YES_NO, "instructions": "Does this function raise an error when the list is empty?"},
         "language": {"type": "choice", "instructions": "Which programming language is this?",
                      "criteria": {"python": None, "javascript": None, "ruby": None, "go": None}},
         "quality": {"type": "score", "instructions": "How ready is this function for production use?",
                     "criteria": ["Not ready", "Needs work", "Acceptable", "Good", "Excellent"]},
         "needs_test": {**YES_NO, "instructions": "Should this function get a unit test before it ships?"}}},
    {"state": "The train leaves at 14:10 and arrives at 16:55.",
     "questions": {
         "duration": {"type": "choice", "instructions": "How long is the trip?",
                      "criteria": {"2h15": "2 hours 15 minutes", "2h45": "2 hours 45 minutes",
                                   "2h55": "2 hours 55 minutes", "3h05": "3 hours 5 minutes"}},
         "over_two_hours": {**YES_NO, "instructions": "Is the trip longer than two hours?"},
         "weekday": {"type": "choice", "instructions": "On which day does the train leave?",
                     "criteria": {"monday": None, "tuesday": None, "wednesday": None, "thursday": None, "friday": None},
                     "abstain": True},
         "length": {"type": "score", "instructions": "How long is this trip for a daily commute?",
                    "criteria": ["Short", "Moderate", "Long", "Very long"]}}},
    {"state": "Your account will be closed in 24 hours unless you confirm your password at the link below. "
              "Click here to verify now.",
     "questions": {
         "phishing": {**YES_NO, "instructions": "Is this message likely a phishing attempt?"},
         "category": {"type": "choice", "instructions": "What kind of message is this?",
                      "criteria": {"phishing": None, "spam": None, "notice": "A real notice from a service",
                                   "marketing": None}},
         "risk": {"type": "score", "instructions": "How risky is it to follow the link?",
                  "criteria": ["Low", "Medium", "High", "Critical"]},
         "sender": {"type": "choice", "instructions": "Which company sent the message?",
                    "criteria": {"bank": None, "email provider": None, "online shop": None}, "abstain": True}}},
    {"state": "明日の会議は午後3時から4時までです。資料は前日までに共有してください。",
     "questions": {
         "time_of_day": {"type": "choice", "instructions": "When is the meeting?",
                         "criteria": {"morning": None, "afternoon": None, "evening": None}},
         "share_early": {**YES_NO, "instructions": "Must the documents be shared before the day of the meeting?"},
         "length": {"type": "score", "instructions": "How long is the meeting?",
                    "criteria": ["Under an hour", "About an hour", "Two hours or more"]}}},
    {"state": {"ticket": "The coffee machine I bought last month stopped heating. I would like a replacement, "
                         "not my money back."},
     "questions": {
         "next_step": {"type": "tool", "instructions": "Which tool should the support agent call next?",
                       "tools": [
                           {"type": "function", "function": {"name": "send_replacement", "parameters": {
                               "type": "object", "properties": {
                                   "reason": {"type": "string", "enum": ["defective", "damaged", "wrong_item"]},
                                   "express": {"type": "boolean"}}}}},
                           {"type": "function", "function": {"name": "issue_refund", "parameters": {
                               "type": "object", "properties": {"full_refund": {"type": "boolean"}}}}},
                           {"type": "function", "function": {"name": "escalate_to_human", "parameters": {
                               "type": "object", "properties": {}}}}]}}},
]


def numbers(answer):
    """One answer as (its discrete answer, every probability in it in a fixed order)."""
    kind = answer.get("type")
    values = []
    if kind == "noul":
        p = answer["noul"]
        label, values = ("yes" if p >= 0.5 else "no"), [p]
    elif kind in ("choice", "score"):
        probs = answer["probabilities"]
        keys = sorted(probs)
        label, values = max(keys, key=lambda k: probs[k]), [probs[k] for k in keys]
    elif "tool" in answer:
        probs = answer.get("probabilities") or {}
        keys = sorted(probs)
        label, values = str(answer.get("tool")), [probs[k] for k in keys]
        for name in sorted(answer.get("arguments") or {}):
            for arg in sorted(answer["arguments"][name]):
                a = answer["arguments"][name][arg]
                if "probability_true" in a:
                    values.append(a["probability_true"])
                for k in sorted(a.get("probabilities") or {}):
                    values.append(a["probabilities"][k])
    else:
        raise ValueError("an answer of an unknown shape: %s" % json.dumps(answer)[:200])
    if "abstain" in answer:
        values.append(answer["abstain"])
    return label, [float(v) for v in values]


# ----------------------------------------------------------------------------------------------------------- make
def reference(weights, dtype="float32", log=print):
    """Every request's answers from the reference code published with the weights (run_aplomb.py), with its
    debiasing on: every choice is also asked with its options in reversed order."""
    import torch

    sys.path.insert(0, os.path.abspath(weights))
    import run_aplomb
    from aplomb.adapter import DecisionAdapter
    from aplomb.decision_head import DecisionHead

    adapter = DecisionAdapter(weights, device="cpu", dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype])
    adapter.head = DecisionHead.for_model(adapter.model, adapter.noul_ids, adapter.digit_ids, adapter.option_ids)
    adapter.head.attach(adapter.model)
    adapter.model.eval()
    with open(os.path.join(weights, "edm_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    out = []
    for body in REQUESTS:
        started = time.time()
        answers = run_aplomb.decide(adapter, {"model": "reference", **body}, config.get("temperatures", {}), debias=True,
                                    abstain_bias=float(config.get("abstain_bias", 0.0) or 0.0))["answers"]
        out.append(answers)
        log("%d questions in %.0f s: %s" % (len(answers), time.time() - started,
                                             ", ".join("%s=%s" % (q, numbers(a)[0]) for q, a in answers.items())))
    return out


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(args, answers):
    weight_files = {}
    for name in sorted(os.listdir(args.weights)):
        if name.endswith(".safetensors") and name.startswith("model"):
            weight_files[name] = sha256_of(os.path.join(args.weights, name))
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": weight_files,
                   "made_with": "the reference code published with the weights, on cpu",
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "limits": {"same_answer": args.same_answer, "typical_gap": args.typical_gap},
                   "requests": [{**body, "answers": a} for body, a in zip(REQUESTS, answers)]}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)


def make(args):
    started = time.time()
    answers = reference(args.weights, args.dtype)
    write(args, answers)
    print("fingerprint of %d questions written in %.0f s" % (sum(len(a) for a in answers), time.time() - started))


# ---------------------------------------------------------------------------------------------------------- check
class Unanswered(Exception):
    """The API gave no answer: the check cannot be done, which says nothing about the model."""


def _post(body, key):
    return check_receipt.post("/decisions", body, key, timeout=120)


def ask(model, body, key, post=None, exchanges=None):
    """One request of the fingerprint, sent as it was made: the model, the state and the questions. The answers.
    `post(body, key)` sends the request body and returns (the request bytes sent, the response bytes received), as in
    check_model.py; the default sends it to /v1/decisions on the API. A list `exchanges` also gets those two byte
    strings, for the receipt check."""
    try:
        sent, received = (post or _post)({"model": model, "state": body["state"], "questions": body["questions"]}, key)
        if exchanges is not None:
            exchanges.append((sent, received))
        return json.loads(received)["answers"]
    except urllib.error.HTTPError as error:
        raise Unanswered("HTTP %d: %s" % (error.code, error.read()[:300].decode("utf-8", "replace"))) from None
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Unanswered("%s: %s" % (type(error).__name__, error)) from None


def verdict(fingerprint, live_answers):
    """live_answers: one {question id: answer} per request, in the fingerprint's order. Per question, the largest
    difference between any of its probabilities; the verdict reads the share of the same answers and the median."""
    limits = fingerprint["limits"]
    same, gaps, per_question = 0, [], []
    for body, live in zip(fingerprint["requests"], live_answers):
        for qid, reference in body["answers"].items():
            label, values = numbers(reference)
            try:
                live_label, live_values = numbers(live[qid])
            except (KeyError, TypeError, ValueError):
                live_label, live_values = None, []
            gap = 1.0
            if len(live_values) == len(values):
                gap = max((abs(a - b) for a, b in zip(values, live_values)), default=0.0)
            same += live_label == label
            gaps.append(gap)
            per_question.append((qid, label, live_label, gap))
    n = len(gaps)
    typical = statistics.median(gaps)
    ok = same / n >= limits["same_answer"] and typical <= limits["typical_gap"]
    return {"result": "matches" if ok else "DOES NOT MATCH", "questions": n, "same_answer": same,
            "typical_gap": typical, "largest_gap": max(gaps), "per_question": per_question}


def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not a %s fingerprint" % (os.path.basename(path), KIND))
    return fingerprint


def run_check(model, fingerprint, key, post=None, keys=None):
    """Every request sent once; (the verdict, the answers). Raises Unanswered when the API gives no answer. A caller can
    send the requests through its own `post` (ask's), for example to add its own rate limiting and retries. With
    `keys` (check_receipt.load_keys), every answer's receipt is checked too: the verdict's `receipts` holds one
    (True, the receipt) or (False, why) per answer."""
    if keys is None:
        live = [ask(model, body, key, post=post) for body in fingerprint["requests"]]
        return verdict(fingerprint, live), live
    exchanges = []
    live = [ask(model, body, key, post=post, exchanges=exchanges) for body in fingerprint["requests"]]
    v = verdict(fingerprint, live)
    v["receipts"] = [check_receipt.check(sent, received, keys) for sent, received in exchanges]
    return v, live


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    print("  same answer: %d of %d questions (needs %d)" % (v["same_answer"], v["questions"],
                                                           int(limits["same_answer"] * v["questions"] + 0.999)))
    print("  typical largest difference between the probabilities: %.4f (limit %.3f), largest %.4f" % (
        v["typical_gap"], limits["typical_gap"], v["largest_gap"]))
    signed = check_receipt.report_receipts(model, v.get("receipts"))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, fingerprint=None, out="", keys=None):
    """Check `model` against the fingerprint at `path` (already read when `fingerprint` is given), and with `keys` every
    answer's receipt. 0 matches (and signed, when checked), 1 anything else. `out` names a JSON file for the verdict
    and the answers."""
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        v, live = run_check(model, fingerprint, key, keys=keys)
    except Unanswered as error:
        print("%s could not be checked: the API did not answer: %s" % (model, error))
        return 1
    except (OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"]}
        record.update((name, v[name]) for name in v if name != "receipts")
        record["answers"] = live
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1)
    return 0 if v["result"] == "matches" and signed is not False else 1


def main(argv=None):
    # An option name the console cannot show is printed with a stand-in, never a crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example aplomb-1")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the verdict and the answers to this JSON file")
    c.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint")
    m.add_argument("--name", required=True, help="the public repository these weights are, written into the fingerprint")
    m.add_argument("--revision", required=True, help="the repository revision, written into the fingerprint")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, written into the fingerprint")
    m.add_argument("--out", required=True)
    m.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    m.add_argument("--same-answer", type=float, required=True)
    m.add_argument("--typical-gap", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "make":
        return make(args)
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out,
                     keys=None if args.keys == "none" else check_receipt.load_keys(args.keys))


if __name__ == "__main__":
    sys.exit(main() or 0)
