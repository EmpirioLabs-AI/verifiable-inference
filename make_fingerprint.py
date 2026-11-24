"""Make a model fingerprint from public weights, or compare two fingerprints.

EmpirioLabs makes each fingerprint once, from the public checkpoint the model's catalog entry names, and nobody who
checks a model needs to run this. It is here so that anyone can make a fingerprint again and compare it with ours.

    pip install torch transformers huggingface_hub safetensors
    python make_fingerprint.py make --weights Qwen/Qwen3.5-9B --revision <commit> --model qwen3-5-9b \
        --out mine.json
    python make_fingerprint.py compare fingerprints/qwen3-5-9b.json mine.json

`make` runs the public weights once per question with the model's own chat template and records the 20 most likely
first answer tokens. A checkpoint that the plain transformers code cannot run is made the same way with any stock
open-source inference server at its defaults; the fingerprint says which way it was made. Two honest ways of running
one checkpoint differ by a few hundredths in the typical gap, which `compare` reports.
"""
import argparse
import hashlib
import json
import os
import sys
import time

TOP = 20
SYSTEM = "You are a helpful assistant."


def make(args):
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    questions = json.load(open(args.questions, encoding="utf-8")) if args.questions else default_questions()
    source = {"revision": args.revision} if args.revision and not os.path.isdir(args.weights) else {}
    tokenizer = AutoTokenizer.from_pretrained(args.weights, **source)
    config = AutoConfig.from_pretrained(args.weights, **source)
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        args.weights, dtype=torch.bfloat16 if (args.half or device == "cuda") else torch.float32, **source)
    model.to(device)
    model.eval()
    rows = []
    started = time.time()
    for question in questions:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float().cpu()
        best = torch.topk(torch.log_softmax(logits, dim=-1), TOP)
        rows.append({"question": question, "prompt_tokens": len(ids),
                     "top": [[tokenizer.decode([int(i)]), round(float(v), 4)] for v, i in zip(best.values, best.indices)]})
    folder = args.weights
    if not os.path.isdir(folder):
        from huggingface_hub import snapshot_download
        folder = snapshot_download(args.weights, revision=args.revision or None, allow_patterns=["*.safetensors"])
    files = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".safetensors"):
            digest = hashlib.sha256()
            with open(os.path.join(folder, name), "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(chunk)
            files[name] = digest.hexdigest()
    fingerprint = {"kind": "chat-first-token", "model": args.model, "weights": args.name or args.weights,
                   "revision": args.revision, "weight_files_sha256": files, "system": SYSTEM,
                   "made_with": "plain transformers code on %s" % device,
                   "limits": dict({"typical_gap": args.limit_gap, "first_token": 0.8},
                                  **({"tail_gap": args.tail_gap} if args.tail_gap is not None else {})),
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=0)
    print("fingerprint of %d questions written in %.0f s" % (len(rows), time.time() - started))


def default_questions():
    """The questions of an existing fingerprint, so a fingerprint made again asks exactly what ours asked."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingerprints")
    for name in sorted(os.listdir(here)):
        if name.endswith(".json"):
            with open(os.path.join(here, name), encoding="utf-8") as handle:
                return [row["question"] for row in json.load(handle)["rows"]]
    raise SystemExit("no fingerprint to take the questions from; pass --questions <file.json>")


def compare(args):
    """How close two fingerprints of the same questions are, scored exactly as check_model.py scores a live model:
    the second file's numbers are read as if the API had returned them."""
    import check_model

    first = json.load(open(args.first, encoding="utf-8"))
    second = json.load(open(args.second, encoding="utf-8"))
    rows = dict((row["question"], row) for row in second["rows"])
    if [row["question"] for row in first["rows"]] != [row["question"] for row in second["rows"]]:
        raise SystemExit("the two fingerprints do not ask the same questions in the same order")
    answers = [(rows[row["question"]]["top"], rows[row["question"]]["prompt_tokens"]) for row in first["rows"]]
    verdict = check_model.score(first["rows"], answers, first.get("limits") or {})
    print("%s against %s: %s" % (args.second, args.first, verdict["result"]))
    print("same most likely first token: %d of %d questions" % (verdict["same_first"], verdict["questions"]))
    print("typical gap between the log-probabilities: %.3f" % verdict["typical_gap"])
    if verdict["tail_gap"] is not None:
        print("typical gap among the other alternatives: %.3f" % verdict["tail_gap"])
    print("prompt lengths differ by: %s" % verdict["offsets"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--weights", required=True, help="a local folder or a Hugging Face repository")
    m.add_argument("--revision", default="", help="the repository commit the fingerprint is made from")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI")
    m.add_argument("--name", default="", help="the public repository name to record, when --weights is a folder")
    m.add_argument("--out", required=True)
    m.add_argument("--questions", default="", help="a JSON list of questions (default: the questions of our fingerprints)")
    m.add_argument("--half", action="store_true", help="run in bfloat16 (always on a GPU)")
    m.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    m.add_argument("--limit-gap", type=float, default=0.15, help="twice the gap between two honest runs, at least 0.15")
    m.add_argument("--tail-gap", type=float, default=None,
                   help="a limit on the other alternatives, for a model whose answers are near-certain")
    c = sub.add_parser("compare", help="compare two fingerprints of the same questions")
    c.add_argument("first")
    c.add_argument("second")
    args = parser.parse_args(argv)
    return make(args) if args.command == "make" else compare(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
