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
import statistics
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
                   "limits": {"typical_gap": args.limit_gap, "first_token": 0.8},
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
    """How far two fingerprints of the same questions are apart, in the terms check_model.py uses."""
    first = json.load(open(args.first, encoding="utf-8"))
    second = json.load(open(args.second, encoding="utf-8"))
    pairs = [(a, b) for a, b in zip(first["rows"], second["rows"]) if a["question"] == b["question"]]
    if len(pairs) != len(first["rows"]):
        raise SystemExit("the two fingerprints do not ask the same questions in the same order")
    same_first, gaps, lengths = 0, [], set()
    for a, b in pairs:
        ra, rb = dict(map(tuple, a["top"])), dict(map(tuple, b["top"]))
        same_first += max(ra, key=ra.get) == max(rb, key=rb.get)
        common = [t for t in ra if t in rb]
        weights = [2.718281828 ** ra[t] for t in common]
        gaps.append(sum(w * abs(ra[t] - rb[t]) for w, t in zip(weights, common)) / sum(weights) if common else float("inf"))
        lengths.add(b["prompt_tokens"] - a["prompt_tokens"])
    print("same most likely first token: %d of %d questions" % (same_first, len(pairs)))
    print("typical gap between the log-probabilities: %.3f" % statistics.median(gaps))
    print("prompt lengths differ by: %s" % sorted(lengths))


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
    m.add_argument("--limit-gap", type=float, default=0.15)
    c = sub.add_parser("compare", help="compare two fingerprints of the same questions")
    c.add_argument("first")
    c.add_argument("second")
    args = parser.parse_args(argv)
    return make(args) if args.command == "make" else compare(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
