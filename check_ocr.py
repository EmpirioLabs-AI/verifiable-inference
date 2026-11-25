"""Check that a document OCR model on EmpirioLabs AI is the public model it is said to be, without running any model.

A fingerprint of kind "ocr" (fingerprints/<model>.json) was made once, from the model's public weights: for a fixed set
of short public-domain document pages kept beside it (fingerprints/<model>/), the text the public model writes in its
model card's grounding mode, greedy, with each block of the page labelled and boxed (coordinates 0 to 1000). The
check sends each page once to POST /v1/ocr and compares the answer's blocks with the fingerprint's: the text (its
character error rate after whitespace is normalized), the blocks' kinds in reading order, and their boxes. Another
model reads, splits or places the same page differently.

    python check_model.py --model lightonocr-3-4b                the usual way, as for every other model
    python check_ocr.py check --model lightonocr-3-4b            the same check, --out keeps the numbers read
    python check_ocr.py make --weights <folder> --name lightonai/LightOnOCR-3-4B --revision <commit> \
        --model lightonocr-3-4b --pages fingerprints/lightonocr-3-4b --sources <pages.json> \
        --out fingerprints/lightonocr-3-4b.json --page-cer <limit> --typical-cer <limit> --box-gap <limit> --close <share>

The check uses only the Python standard library and reads your API key from EMPIRIOLABS_API_KEY. It sends one page
per request, billed to your account at the model's listed price. The pages are read from the folder beside the
fingerprint and must have the SHA-256 the fingerprint records; a page that does not is never sent. `make` runs the
public weights with the inference code their model card publishes (transformers, Qwen3_5ForConditionalGeneration and
the checkpoint's own processor and chat template, thinking off) and needs torch, transformers and Pillow.

Verdict
    matches          on at least the share of pages the fingerprint states: a character error rate within the page
                     limit, the same kinds of blocks in the same order, and boxes within the box limit; and the
                     typical (median) character error rate and box gap within their limits
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model (check_receipt.py);
                     a page answered as a job carries it on the job's result, for the request that made it
Exit status: 0 matches and signed, 1 anything else.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "ocr"
ORIGIN = "%s://%s" % urllib.parse.urlsplit(check_receipt.API)[:2]
# The model card's grounding mode: the page image, then this prompt.
PROMPT = "grounding"
# The model card's page images: at most this many pixels on the longest edge (every page here is smaller).
LONGEST_EDGE = 2048
# Every page is asked with the API's blocks on (the default), the boxes in pixels of the page as sent.
REQUEST = {"include_blocks": True}
# `![label](x1,y1,x2,y2)` alone on a line opens a block; a `+` after the label continues the block before it.
PLACEHOLDER = re.compile(r"^!\[([a-z_]+)(\+?)\]\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)\s*(.*)$")
# Each label the model card lists, and the block type the API answers with for it.
BLOCK_TYPES = {"text": "text", "title": "title", "section": "title", "list": "list", "header": "header",
               "footer": "footer", "page_number": "footer", "footnote": "text", "caption": "caption",
               "formula": "equation", "code": "code", "table": "table", "image": "image", "chart": "image",
               "header_image": "image", "footer_image": "image", "aside_text": "aside_text"}
IMAGE_LABELS = ("image", "chart", "header_image", "footer_image")
BOX_SCALE = 1000


# ------------------------------------------------------------------------------------------------- the model's text
def parse_blocks(text):
    """The blocks of the model's grounding text, in reading order: [{"type", "box", "content"}], each box (x1, y1,
    x2, y2) from 0 to 1000. A continuation (`+`) joins the block before it of the same label (its box the union), a
    page number is a header in the top half of the page and a footer below, and a block with no text is dropped
    unless it is an image. Text before the first placeholder is one text block covering the page."""
    blocks, current, lines = [], None, []

    def close():
        if current is not None:
            current["content"] = "\n".join(lines).strip()
            blocks.append(current)

    for raw in (text or "").splitlines():
        match = PLACEHOLDER.match(raw.strip())
        if match:
            close()
            label, plus, x1, y1, x2, y2, rest = match.groups()
            left, right = sorted((min(BOX_SCALE, int(x1)), min(BOX_SCALE, int(x2))))
            top, bottom = sorted((min(BOX_SCALE, int(y1)), min(BOX_SCALE, int(y2))))
            current = {"label": label, "box": (left, top, right, bottom), "continued": bool(plus)}
            lines = [rest] if rest.strip() else []
            continue
        if current is None:
            current, lines = {"label": "text", "box": (0, 0, BOX_SCALE, BOX_SCALE), "continued": False}, []
        lines.append(raw)
    close()
    merged = []
    for block in blocks:
        if not (block["content"] or block["label"] in IMAGE_LABELS):
            continue
        if block["continued"] and merged and merged[-1]["label"] == block["label"]:
            previous = merged[-1]
            previous["content"] = (previous["content"] + "\n" + block["content"]).strip()
            previous["box"] = (min(previous["box"][0], block["box"][0]), min(previous["box"][1], block["box"][1]),
                               max(previous["box"][2], block["box"][2]), max(previous["box"][3], block["box"][3]))
            continue
        merged.append(block)
    out = []
    for block in merged:
        kind = BLOCK_TYPES.get(block["label"], "text")
        if block["label"] == "page_number":
            kind = "header" if (block["box"][1] + block["box"][3]) / 2 < BOX_SCALE / 2 else "footer"
        out.append({"type": kind, "box": block["box"], "content": block["content"]})
    return out


def answer_blocks(page):
    """The blocks of one page of the API's answer, in the same terms: each box scaled from the page's pixels back
    to 0 to 1000 with the page's own `dimensions`. None when the page lists no blocks or no size."""
    blocks = page.get("blocks") if isinstance(page, dict) else None
    size = page.get("dimensions") if isinstance(page, dict) else None
    if not isinstance(blocks, list) or not isinstance(size, dict):
        return None
    width, height = size.get("width"), size.get("height")
    if not (isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0):
        return None
    out = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        try:
            box = (block["top_left_x"] * BOX_SCALE / width, block["top_left_y"] * BOX_SCALE / height,
                   block["bottom_right_x"] * BOX_SCALE / width, block["bottom_right_y"] * BOX_SCALE / height)
        except (KeyError, TypeError):
            return None
        out.append({"type": str(block.get("type")), "box": box, "content": str(block.get("content") or "")})
    return out


def normalize(text):
    """The text with every run of whitespace (line breaks included) made one space, and none at the ends."""
    return " ".join((text or "").split())


def page_text(blocks):
    return normalize("\n".join(block["content"] for block in blocks))


def edit_distance(a, b):
    """Levenshtein distance between two strings (insertions, deletions and substitutions of one character each),
    with Myers' bit-vector algorithm on Python integers (Hyyro 2003): one pass over b, the pattern a as bits."""
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if m == 0:
        return len(b)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    pv, mv, score = mask, 0, m
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def character_error_rate(reference, live):
    """Edits needed to turn the reference text into the live one, per reference character."""
    return edit_distance(reference, live) / max(1, len(reference))


def compare(reference_blocks, live):
    """(character error rate, the same kinds of blocks in the same order, the box gap) of one page. The box gap is
    the mean absolute difference of all box coordinates (0 to 1000) when the kinds agree, else infinite."""
    if live is None:
        return 1.0, False, math.inf
    cer = character_error_rate(page_text(reference_blocks), page_text(live))
    same_layout = [b["type"] for b in reference_blocks] == [b["type"] for b in live]
    gap = math.inf
    if same_layout:
        gaps = [abs(r - x) for ref, got in zip(reference_blocks, live) for r, x in zip(ref["box"], got["box"])]
        gap = sum(gaps) / len(gaps) if gaps else 0.0
    return cer, same_layout, gap


def verdict(fingerprint, live_pages):
    """live_pages: one answer_blocks() per page, in the fingerprint's order."""
    limits = fingerprint["limits"]
    results = [compare(parse_blocks(row["output"]), live) for row, live in zip(fingerprint["pages"], live_pages)]
    close = sum(1 for cer, same, gap in results
                if cer <= limits["page_cer"] and same and gap <= limits["box_gap"])
    typical_cer = statistics.median(r[0] for r in results)
    typical_gap = statistics.median(r[2] for r in results)
    ok = (close / len(results) >= limits["close"] and typical_cer <= limits["typical_cer"]
          and typical_gap <= limits["box_gap"])
    return {"result": "matches" if ok else "DOES NOT MATCH", "pages": len(results), "close": close,
            "typical_cer": typical_cer, "typical_box_gap": typical_gap, "per_page": results}


# ------------------------------------------------------------------------------------------------------ the API
class Unanswered(Exception):
    """The API gave no page for a request: the check cannot be done, which says nothing about the model."""


def _send(method, path, key, body=None):
    """One request to the API at `path`; (the HTTP status, the JSON answer, the request bytes, the response bytes)."""
    sent = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(ORIGIN + path, data=sent, method=method,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                              "User-Agent": check_receipt.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            received = response.read()
            return response.status, json.loads(received), sent, received
    except urllib.error.HTTPError as error:
        raise Unanswered("HTTP %d: %s" % (error.code, error.read()[:300].decode("utf-8", "replace"))) from None
    except (OSError, ValueError) as error:
        raise Unanswered("%s: %s" % (type(error).__name__, error)) from None


def ask(model, page_bytes, key, request=None, exchanges=None, send=None):
    """The first page of the API's answer for one PNG page, sent inline as a data URI. A page still being read
    answers 202 with a job, polled on the API's own origin to its end. A list `exchanges` also gets (the request
    bytes, the bytes of the answer that carries the page), for the receipt check: a job's result is signed for the
    request that made it. `send` replaces the HTTP call (tests)."""
    send = send or _send
    body = {"model": model, "document": {"type": "image_url",
                                         "image_url": "data:image/png;base64," + base64.b64encode(page_bytes).decode()},
            **(REQUEST if request is None else request)}
    status, data, sent, received = send("POST", "/v1/ocr", key, body)
    if not isinstance(data, dict):
        raise Unanswered("the answer is not a JSON object")
    # The job is polled on the API's own origin only: a poll_url that is not a path under /v1/jobs/ is not used.
    poll = data.get("poll_url")
    if not (isinstance(poll, str) and poll.startswith("/v1/jobs/")):
        poll = "/v1/jobs/%s" % urllib.parse.quote(str(data.get("job_id") or data.get("id") or ""), safe="")
    deadline = time.time() + 900
    while status == 202 or (data.get("status") in ("queued", "processing", "running") and "pages" not in data):
        if time.time() > deadline:
            raise Unanswered("the OCR job did not finish in 15 minutes")
        time.sleep(2)
        status, job, _sent, received = send("GET", poll, key)
        if not isinstance(job, dict):
            raise Unanswered("the job's answer is not a JSON object")
        if job.get("status") in ("completed", "succeeded"):
            data = job.get("result") or {}
            break
        if job.get("status") == "failed":
            raise Unanswered("the OCR job failed: %s" % json.dumps(job.get("error"))[:300])
        data = job
    pages = [page for page in (data.get("pages") or []) if isinstance(page, dict)]
    if not pages:
        raise Unanswered("the answer holds no page")
    if exchanges is not None:
        exchanges.append((sent, received))
    return pages[0]


def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not an %s fingerprint" % (os.path.basename(path), KIND))
    return fingerprint


def read_pages(fingerprint, folder):
    """Every page's bytes, each checked against the SHA-256 the fingerprint records before any is sent."""
    pages = []
    for row in fingerprint["pages"]:
        with open(os.path.join(folder, row["file"]), "rb") as handle:
            data = handle.read()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("%s is not the page the fingerprint was made from" % row["file"])
        pages.append(data)
    return pages


def run_check(model, fingerprint, folder, key, keys=None, ask_one=None):
    """Every page sent once; (the verdict, each answer's blocks). With `keys` (check_receipt.load_keys) every
    answer's receipt is checked too: the verdict's `receipts` holds one (True, the receipt) or (False, why) per page.
    `ask_one(page_bytes)` replaces the request to the API (tests)."""
    pages = read_pages(fingerprint, folder)
    exchanges = [] if keys is not None and ask_one is None else None
    live = []
    for number, page in enumerate(pages, 1):
        try:
            answer = ask_one(page) if ask_one else ask(model, page, key, fingerprint.get("request"), exchanges)
        except Unanswered as error:
            raise Unanswered("the API did not answer page %d: %s" % (number, error)) from None
        live.append(answer_blocks(answer))
    v = verdict(fingerprint, live)
    if exchanges is not None:
        v["receipts"] = [check_receipt.check(sent, received, keys) for sent, received in exchanges]
    return v, live


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    print("  pages read the same (error rate within %.3f, the same blocks, boxes within %.1f): %d of %d (needs %d)" % (
        limits["page_cer"], limits["box_gap"], v["close"], v["pages"], int(limits["close"] * v["pages"] + 0.999)))
    print("  typical character error rate: %.4f (limit %.3f)" % (v["typical_cer"], limits["typical_cer"]))
    print("  typical box gap: %.2f (limit %.1f, of 1000)" % (v["typical_box_gap"], limits["box_gap"]))
    signed = check_receipt.report_receipts(model, v.get("receipts"))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, fingerprint=None, out="", keys=None):
    """Check `model` against the fingerprint at `path` (already read when `fingerprint` is given), and with `keys`
    every answer's receipt. 0 matches (and signed, when checked), 1 anything else."""
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        v, live = run_check(model, fingerprint, os.path.dirname(os.path.abspath(path)), key, keys=keys)
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"],
                  **{name: v[name] for name in v if name not in ("per_page", "receipts")},
                  "per_page": [{"file": row["file"], "cer": r[0], "same_blocks": r[1], "box_gap": r[2]}
                               for row, r in zip(fingerprint["pages"], v["per_page"])],
                  "answers": live}
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=1, default=str)
    return 0 if v["result"] == "matches" and signed is not False else 1


# ---------------------------------------------------------------------------------------------------------- make
def page_image(path):
    """The page as the model card prepares an image: RGB on white, at most LONGEST_EDGE pixels on its longest edge."""
    from PIL import Image

    image = Image.open(path)
    image.load()
    if image.mode != "RGB":
        if image.mode in ("RGBA", "LA", "P", "PA"):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            image = background
        else:
            image = image.convert("RGB")
    if max(image.size) > LONGEST_EDGE:
        image.thumbnail((LONGEST_EDGE, LONGEST_EDGE), Image.Resampling.LANCZOS)
    return image


def load_model(weights, dtype="float32", device="cpu"):
    """The checkpoint with the model card's code: Qwen3_5ForConditionalGeneration and its own processor."""
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(weights, dtype=torch_dtype).to(device).eval()
    return model, AutoProcessor.from_pretrained(weights)


def recognize(model, processor, image, max_new_tokens=4096, temperature=0.0, seed=0):
    """(the model's text, prompt tokens, generated tokens) for one page in grounding mode, thinking off: greedy at
    temperature 0, else sampled at that temperature from torch's generator seeded with `seed` (no top-k, no top-p)."""
    import torch

    conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False).to(model.device)
    options = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if temperature > 0:
        torch.manual_seed(seed)
        options = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_k": 0,
                   "top_p": 1.0}
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    prompt_length = inputs["input_ids"].shape[1]
    generated = output[0, prompt_length:]
    return processor.decode(generated, skip_special_tokens=True), int(prompt_length), int(generated.shape[0])


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make(args):
    with open(args.sources, encoding="utf-8") as handle:
        manifest = json.load(handle)
    folder = os.path.abspath(args.pages)
    model, processor = load_model(args.weights, args.dtype, args.device)
    rows, started = [], time.time()
    for page in manifest:
        path = os.path.join(folder, page["file"])
        image = page_image(path)
        t = time.time()
        text, prompt_tokens, generated = recognize(model, processor, image, args.max_new_tokens, args.temperature,
                                                   args.seed)
        rows.append({"file": "%s/%s" % (os.path.basename(folder), page["file"]), "sha256": sha256_of(path),
                     "source": page["source"], "license": page["license"], "processing": page["processing"],
                     "width": image.width, "height": image.height, "output": text})
        print("%-28s %4d prompt tokens %5d generated %6.0f s  %d blocks" % (
            page["file"], prompt_tokens, generated, time.time() - t, len(parse_blocks(text))), flush=True)
    weight_files = {name: sha256_of(os.path.join(args.weights, name))
                    for name in sorted(os.listdir(args.weights)) if name.endswith(".safetensors")}
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": weight_files,
                   "made_with": "plain transformers code on %s" % args.device.split(":")[0],
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request": REQUEST,
                   "limits": {"page_cer": args.page_cer, "typical_cer": args.typical_cer, "box_gap": args.box_gap,
                              "close": args.close},
                   "pages": rows}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)
    print("fingerprint of %d pages written in %.0f s" % (len(rows), time.time() - started))


def main(argv=None):
    # A page's text the console cannot show is printed with a stand-in, never a crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example lightonocr-3-4b")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the numbers read from each answer to this JSON file")
    c.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint")
    m.add_argument("--name", required=True, help="the public repository these weights are")
    m.add_argument("--revision", required=True, help="the repository revision")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI")
    m.add_argument("--pages", required=True, help="the folder beside the fingerprint that holds the pages")
    m.add_argument("--sources", required=True, help="each page's file, source, license and processing (JSON)")
    m.add_argument("--out", required=True)
    m.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    m.add_argument("--device", default="cpu", help="cpu or cuda")
    m.add_argument("--max-new-tokens", type=int, default=4096)
    m.add_argument("--temperature", type=float, default=0.0, help="0 (the default) decodes greedily")
    m.add_argument("--seed", type=int, default=0, help="the sampling seed, with a temperature above 0")
    m.add_argument("--page-cer", type=float, required=True)
    m.add_argument("--typical-cer", type=float, required=True)
    m.add_argument("--box-gap", type=float, required=True)
    m.add_argument("--close", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "make":
        return make(args)
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out,
                     keys=None if args.keys == "none" else check_receipt.load_keys(args.keys))


if __name__ == "__main__":
    sys.exit(main() or 0)
