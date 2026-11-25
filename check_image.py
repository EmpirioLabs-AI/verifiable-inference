"""Check that an image model on EmpirioLabs AI is the public model it is said to be, without running any model.

A fingerprint of kind "image" (fingerprints/<model>.json) was made once, from the model's public weights: for a fixed
set of prompts, each with its own seed, the image the pipeline published with the weights draws with the settings the
fingerprint states, kept as a 32 x 32 thumbnail (the average color of each block of the picture). The seed fixes the
starting noise on every computer (float32 numbers from torch's CPU generator, in the shape the pipeline draws them), so
the same request gives the same picture wherever the public weights run, up to the small rounding differences between
cards. The check asks the live API for the same images as PNG and compares
each with its thumbnail: the distance is one minus the correlation of their colors, 0 for the same picture even when
it is a little lighter or darker. A picture from another model, or from the same model with another seed, is laid out
differently and is far from it.

    python check_model.py --model flux-2-klein-4b                the usual way, as for every other model
    python check_image.py check --model flux-2-klein-4b          the same check, --out keeps the numbers read
    python check_image.py make --weights <folder> --name black-forest-labs/FLUX.2-klein-4B --revision <commit> \
        --model flux-2-klein-4b --probes <probes.json> --request '<API settings as JSON>' \
        --out fingerprints/flux-2-klein-4b.json --image-distance <limit> --typical-distance <limit> --close <share>

You choose what is checked: `make` takes any prompts and seeds (--probes), so you can make a fingerprint of your own
that we have never seen, from the public weights, and check the API with it. `replay` draws one image you already
received again from the public weights and compares the two.

The check uses only the Python standard library and reads your API key from EMPIRIOLABS_API_KEY. It sends each of the
fingerprint's prompts once, billed to your account at the model's listed price. `make` and `replay` run the public
weights with diffusers and need torch, diffusers, transformers and Pillow.

Verdict
    matches          at least the share of images the fingerprint states within its per-image distance, and the
                     typical (median) distance within its limit
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model, and the image
                     checked is the file that answer lists
Exit status: 0 matches and signed, 1 anything else.
"""
import argparse
import base64
import hashlib
import json
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "image"
ORIGIN = "%s://%s" % urllib.parse.urlsplit(check_receipt.API)[:2]
THUMBNAIL = 32
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# API settings the pipeline reads, by the name of the pipeline argument they set.
PIPELINE_ARGUMENTS = {"width": "width", "height": "height", "num_inference_steps": "num_inference_steps",
                      "guidance_scale": "guidance_scale", "max_sequence_length": "max_sequence_length"}


# ----------------------------------------------------------------------------------------------------- pictures
class NotAnImage(ValueError):
    """The bytes are not a PNG this module reads."""


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(data):
    """(width, height, rows) of a PNG: rows is a list of bytearrays, three bytes (red, green, blue) per pixel. Reads
    non-interlaced 8- and 16-bit grey, grey with alpha, color, color with alpha and 8-bit palette images; alpha is
    dropped."""
    if not data.startswith(PNG_SIGNATURE):
        raise NotAnImage("not a PNG file")
    position, header, palette, compressed = len(PNG_SIGNATURE), None, None, []
    while position + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"PLTE":
            palette = body
        elif kind == b"IDAT":
            compressed.append(body)
        elif kind == b"IEND":
            break
    if header is None or not compressed:
        raise NotAnImage("a PNG without an image header or image data")
    width, height, depth, color, _compression, _filter, interlace = header
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    if channels is None or interlace or depth not in (8, 16) or (color == 3 and (depth != 8 or palette is None)):
        raise NotAnImage("a PNG layout this check does not read (color type %d, depth %d, interlace %d)"
                         % (color, depth, interlace))
    step = channels * depth // 8
    stride = width * step
    raw = zlib.decompress(b"".join(compressed))
    if len(raw) != height * (stride + 1):
        raise NotAnImage("the PNG's image data has the wrong length")
    rows, previous = [], bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        method, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        if method == 1:
            for i in range(step, stride):
                line[i] = (line[i] + line[i - step]) & 255
        elif method == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 255
        elif method == 3:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 255
        elif method == 4:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                upper_left = previous[i - step] if i >= step else 0
                line[i] = (line[i] + _paeth(left, previous[i], upper_left)) & 255
        elif method != 0:
            raise NotAnImage("a PNG row with an unknown filter")
        previous = line
        if depth == 16:
            line = line[0::2]
        if color == 2:
            pixels = line
        elif color == 6:
            pixels = bytearray(len(line) * 3 // 4)
            for c in range(3):
                pixels[c::3] = line[c::4]
        elif color == 0:
            pixels = bytearray(len(line) * 3)
            for c in range(3):
                pixels[c::3] = line
        elif color == 4:
            pixels = bytearray(len(line) * 3 // 2)
            for c in range(3):
                pixels[c::3] = line[0::2]
        else:
            pixels = bytearray()
            for index in line:
                pixels += palette[index * 3:index * 3 + 3]
        rows.append(pixels)
    return width, height, rows


def thumbnail(data, size=THUMBNAIL):
    """The picture as size x size blocks, each the rounded average red, green and blue of its pixels: a list of
    3 * size * size integers, row by row."""
    return blocks(*decode_png(data), size=size)


def blocks(width, height, rows, size=THUMBNAIL):
    """The thumbnail of a picture given as `rows` (each row three bytes, red, green and blue, per pixel): size x size
    blocks, each the rounded average of its pixels, row by row. A video frame is read the same way (check_video.py)."""
    if width < size or height < size:
        raise NotAnImage("a picture smaller than its thumbnail")
    out = []
    for by in range(size):
        y0, y1 = by * height // size, (by + 1) * height // size
        sums = [[0, 0, 0] for _ in range(size)]
        for y in range(y0, y1):
            row = rows[y]
            for bx in range(size):
                x0, x1 = bx * width // size, (bx + 1) * width // size
                block = sums[bx]
                block[0] += sum(row[3 * x0 + 0:3 * x1:3])
                block[1] += sum(row[3 * x0 + 1:3 * x1:3])
                block[2] += sum(row[3 * x0 + 2:3 * x1:3])
        for bx in range(size):
            count = (y1 - y0) * ((bx + 1) * width // size - bx * width // size)
            out += [(total + count // 2) // count for total in sums[bx]]
    return out


def encode_thumbnail(values):
    return base64.b64encode(bytes(values)).decode("ascii")


def decode_thumbnail(text):
    return list(base64.b64decode(text))


def distance(reference, live):
    """How far two thumbnails are apart: one minus the correlation of their color values, from 0 (the same picture,
    even a little lighter or darker) to 2. Two pictures of the same scene laid out the same way correlate closely;
    another model's picture for the same prompt and seed is laid out differently and does not. A thumbnail with no
    variation at all has no correlation with anything and counts as 1."""
    if len(reference) != len(live):
        raise ValueError("thumbnails of different sizes")
    n = len(reference)
    mean_a, mean_b = sum(reference) / n, sum(live) / n
    da = [a - mean_a for a in reference]
    db = [b - mean_b for b in live]
    sa, sb = sum(x * x for x in da), sum(y * y for y in db)
    if sa == 0 or sb == 0:
        return 1.0
    return 1.0 - sum(x * y for x, y in zip(da, db)) / (sa * sb) ** 0.5


# ---------------------------------------------------------------------------------------------------------- make
def _pipeline_kwargs(request, settings, probe):
    kwargs = {PIPELINE_ARGUMENTS[k]: v for k, v in request.items() if k in PIPELINE_ARGUMENTS}
    kwargs.update({k: v for k, v in settings.items() if k != "sigmas"})
    kwargs.update({PIPELINE_ARGUMENTS[k]: v for k, v in probe.items() if k in PIPELINE_ARGUMENTS})
    return kwargs


def _merge_adapter(transformer, path, scale=1.0):
    """Add a low-rank adapter in the common `diffusion_model.<layer>.lora_down/lora_up/alpha` layout to the
    transformer's own weights: W + scale * alpha / rank * up @ down, in float32."""
    import torch
    from safetensors.torch import load_file

    adapter = load_file(path)
    params = dict(transformer.named_parameters())
    merged = 0
    for name in sorted(k[: -len(".lora_down.weight")] for k in adapter if k.endswith(".lora_down.weight")):
        down, up = adapter[name + ".lora_down.weight"].float(), adapter[name + ".lora_up.weight"].float()
        rank = down.shape[0]
        alpha = float(adapter[name + ".alpha"]) if name + ".alpha" in adapter else float(rank)
        target = name[len("diffusion_model."):] if name.startswith("diffusion_model.") else name
        weight = params.get(target + ".weight")
        if weight is None:
            raise SystemExit("the adapter's layer %s is not in the model" % target)
        with torch.no_grad():
            delta = (up.to(weight.device) @ down.to(weight.device)) * (scale * alpha / rank)
            weight.copy_((weight.float() + delta).to(weight.dtype))
        merged += 1
    return merged


def load_pipeline(weights, dtype="bfloat16", device="cuda", adapters=(), offload=False):
    """The pipeline published with the weights (a custom one when the checkpoint ships its own code), with each
    adapter (path, scale) merged into its transformer."""
    import torch
    from diffusers import DiffusionPipeline

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    custom = [name for name in os.listdir(weights) if name.startswith("pipeline_") and name.endswith(".py")]
    options = {"torch_dtype": torch_dtype}
    if custom:
        options = {"custom_pipeline": os.path.join(weights, custom[0]), "trust_remote_code": True,
                   "torch_dtype": {"default": torch_dtype, "vae": torch.float32}}
    pipe = DiffusionPipeline.from_pretrained(weights, **options)
    for path, scale in adapters:
        print("merged %d adapter layers from %s" % (_merge_adapter(pipe.transformer, path, scale), os.path.basename(path)))
    if offload == "group":
        # The large parts stream to the card one block at a time; the small ones (the decoder) move there whole.
        from diffusers.hooks import apply_group_offloading

        for component in pipe.components.values():
            if not isinstance(component, torch.nn.Module):
                continue
            if sum(p.numel() for p in component.parameters()) < 500_000_000:
                component.to(device)
            else:
                apply_group_offloading(component, onload_device=torch.device(device), offload_device=torch.device("cpu"),
                                       offload_type="block_level", num_blocks_per_group=1, use_stream=True)
    elif offload:
        pipe.enable_sequential_cpu_offload() if offload == "sequential" else pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _float32_noise(pipe):
    """Make the pipeline draw its starting noise as float32 normal numbers from the seed's CPU generator, cast to its
    own precision afterwards. Torch draws float32 numbers the same way on every release; a pipeline that asks for
    bfloat16 noise directly gets numbers that changed between torch releases, so nobody could draw them again."""
    import torch

    module = sys.modules.get(type(pipe).__module__)
    original = getattr(module, "randn_tensor", None)
    if original is None or getattr(original, "float32_noise", False):
        return

    def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
        noise = original(shape, generator=generator, device=device, dtype=torch.float32, layout=layout)
        return noise if dtype is None else noise.to(dtype)

    randn_tensor.float32_noise = True
    module.randn_tensor = randn_tensor


def draw(pipe, request, settings, probe):
    """One picture, as PNG bytes, from the noise the probe's seed draws on the CPU (float32 numbers from torch's CPU
    generator seeded with the probe's seed, in the shape the pipeline draws them)."""
    import io

    import torch

    _float32_noise(pipe)
    sigmas = settings.get("sigmas")
    if sigmas:
        scheduler = pipe.scheduler
        parent = type(scheduler).__mro__[1] if type(scheduler).__name__ != "FlowMatchEulerDiscreteScheduler" else type(scheduler)
        scheduler.set_timesteps = lambda num_inference_steps=None, device=None, **_: parent.set_timesteps(
            scheduler, device=device, sigmas=list(sigmas))
    kwargs = _pipeline_kwargs(request, settings, probe)
    with torch.inference_mode():
        image = pipe(prompt=probe["prompt"], generator=torch.Generator("cpu").manual_seed(int(probe["seed"])),
                     **kwargs).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(folder):
    found = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(".safetensors"):
                path = os.path.join(root, name)
                found[os.path.relpath(path, folder).replace(os.sep, "/")] = sha256_of(path)
    return dict(sorted(found.items()))


def _adapters(args):
    out = []
    for spec in args.adapter or []:
        fields = dict(part.split("=", 1) for part in spec.split(","))
        out.append(fields)
    return out


def make(args):
    with open(args.probes, encoding="utf-8") as handle:
        probes = json.load(handle)
    request = json.loads(args.request)
    settings = json.loads(args.settings or "{}")
    adapters = _adapters(args)
    pipe = load_pipeline(args.weights, args.dtype, args.device,
                         [(a["path"], float(a.get("scale", 1))) for a in adapters], args.offload)
    rows, started = [], time.time()
    for probe in probes:
        t = time.time()
        png = draw(pipe, request, settings, probe)
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            with open(os.path.join(args.keep, "%s-%d.png" % (args.model, probe["seed"])), "wb") as handle:
                handle.write(png)
        rows.append({**probe, "thumbnail": encode_thumbnail(thumbnail(png))})
        print("%-60s seed %-10d %.0f s" % (probe["prompt"][:60], probe["seed"], time.time() - t), flush=True)
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": _weight_files(args.weights),
                   "made_with": "the pipeline published with the weights, on %s" % args.device.split(":")[0],
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request": request,
                   "limits": {"image_distance": args.image_distance, "typical_distance": args.typical_distance,
                              "close": args.close},
                   "probes": rows}
    if settings:
        fingerprint["settings"] = settings
    if adapters:
        fingerprint["adapters"] = [{"weights": a["weights"], "revision": a["revision"], "file": a["file"],
                                    "sha256": sha256_of(a["path"]), "scale": float(a.get("scale", 1))} for a in adapters]
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)
    print("fingerprint of %d images written in %.0f s" % (len(rows), time.time() - started))


# --------------------------------------------------------------------------------------------------------- check
class Unanswered(Exception):
    """The API gave no image for a prompt: the check cannot be done, which says nothing about the model."""


def _send(method, path, key, body=None):
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


def ask(model, probe, request, key, exchanges=None, call=None):
    """One picture for the probe, as PNG bytes. An image that is still being drawn answers with a job, which is polled
    on the API's own origin to its end. A list `exchanges` also gets (the request bytes, the bytes of the answer that
    carries the picture, the picture), for the receipt check: a job's result is signed for the request that made it.
    `call(method, path, key, body=None)` sends one request and returns (the HTTP status, the JSON answer, the request
    bytes, the answer bytes), as _send does, which is the default: a caller can send the requests through its own
    transport, for example to add its own rate limiting and retries."""
    call = call or _send
    body = {"model": model, "prompt": probe["prompt"], "seed": probe["seed"], **request,
            **{k: v for k, v in probe.items() if k in PIPELINE_ARGUMENTS}}
    status, data, sent, received = call("POST", "/v1/images/generations", key, body)
    deadline = time.time() + 900
    while status == 202 or (data.get("status") in ("queued", "processing", "running") and not data.get("data")):
        if time.time() > deadline:
            raise Unanswered("the image job did not finish in 15 minutes")
        time.sleep(2)
        poll = data.get("poll_url")
        if not (isinstance(poll, str) and poll.startswith("/v1/jobs/")):
            poll = "/v1/jobs/%s" % urllib.parse.quote(str(data.get("job_id") or ""), safe="")
        status, job, _sent, received = call("GET", poll, key)
        if job.get("status") in ("completed", "succeeded"):
            data = job.get("result") or {}
            break
        if job.get("status") == "failed":
            raise Unanswered("the image job failed: %s" % json.dumps(job.get("error"))[:300])
        data = job
    items = [item for item in data.get("data") or [] if isinstance(item, dict)]
    if not items:
        raise Unanswered("the answer holds no image")
    if items[0].get("b64_json"):
        picture = base64.b64decode(items[0]["b64_json"])
    elif items[0].get("url"):
        picture = check_receipt.download(items[0]["url"])
    else:
        raise Unanswered("the answer's image has neither data nor a link")
    if exchanges is not None:
        exchanges.append((sent, received, picture))
    return picture


def verdict(fingerprint, live_thumbnails):
    limits = fingerprint["limits"]
    distances = [distance(decode_thumbnail(row["thumbnail"]), live)
                 for row, live in zip(fingerprint["probes"], live_thumbnails)]
    close = sum(1 for d in distances if d <= limits["image_distance"])
    typical = statistics.median(distances)
    ok = close / len(distances) >= limits["close"] and typical <= limits["typical_distance"]
    return {"result": "matches" if ok else "DOES NOT MATCH", "images": len(distances), "close": close,
            "typical_distance": typical, "distances": distances}


def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not an %s fingerprint" % (os.path.basename(path), KIND))
    return fingerprint


def run_check(model, fingerprint, key, keys=None, ask_one=None, call=None):
    """Every probe sent once; (the verdict, each live thumbnail). With `keys` (check_receipt.load_keys) every answer's
    receipt is checked too, and that the picture is the file the answer lists: the verdict's `receipts` holds one
    (True, the receipt) or (False, why) per image. `call` sends the requests through the caller's own transport (ask's;
    the caller then checks the receipts it sees), and `ask_one(probe)` replaces the request to the API (tests)."""
    exchanges = [] if keys is not None and ask_one is None else None
    thumbnails = []
    for number, probe in enumerate(fingerprint["probes"], 1):
        try:
            picture = ask_one(probe) if ask_one else ask(model, probe, fingerprint["request"], key, exchanges, call=call)
            thumbnails.append(thumbnail(picture))
        except (Unanswered, NotAnImage) as error:
            raise Unanswered("the API did not answer image %d: %s" % (number, error)) from None
    v = verdict(fingerprint, thumbnails)
    if exchanges is not None:
        receipts = []
        for sent, received, picture in exchanges:
            ok, result = check_receipt.check(sent, received, keys)
            if ok:
                good, why = check_receipt.check_outputs(received, [picture])
                if not good:
                    ok, result = False, why
            receipts.append((ok, result))
        v["receipts"] = receipts
    return v, thumbnails


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    print("  images within %.3f of their thumbnail: %d of %d (needs %d)" % (
        limits["image_distance"], v["close"], v["images"], int(limits["close"] * v["images"] + 0.999)))
    print("  typical distance: %.4f (limit %.3f)" % (v["typical_distance"], limits["typical_distance"]))
    signed = check_receipt.report_receipts(model, v.get("receipts"))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, fingerprint=None, out="", keys=None):
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        v, thumbnails = run_check(model, fingerprint, key, keys=keys)
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"],
                  **{name: v[name] for name in v if name != "receipts"},
                  "thumbnails": [encode_thumbnail(t) for t in thumbnails]}
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=1, default=str)
    return 0 if v["result"] == "matches" and signed is not False else 1


def replay(args):
    """Draw one picture you received again from the public weights, with the fingerprint's settings, and compare."""
    fingerprint = load(args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"))
    with open(args.image, "rb") as handle:
        received = thumbnail(handle.read())
    probe = {"prompt": args.prompt, "seed": args.seed}
    pipe = load_pipeline(args.weights, args.dtype, args.device,
                         [(path, float(a.get("scale", 1))) for a, path in zip(fingerprint.get("adapters") or [], args.adapter_file or [])],
                         args.offload)
    d = distance(thumbnail(draw(pipe, fingerprint["request"], fingerprint.get("settings") or {}, probe)), received)
    limit = fingerprint["limits"]["image_distance"]
    print("distance %.4f (limit %.3f): %s" % (d, limit, "the same picture" if d <= limit else "NOT the same picture"))
    return 0 if d <= limit else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example flux-2-klein-4b")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the numbers read from each answer to this JSON file")
    c.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    for name, helptext in (("make", "make a fingerprint from public weights"),
                           ("replay", "draw a picture you received again from the public weights and compare")):
        m = sub.add_parser(name, help=helptext)
        m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint")
        m.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
        m.add_argument("--device", default="cuda", help="cuda or cpu")
        m.add_argument("--offload", default="", choices=("", "group", "model", "sequential"),
                       help="keep the weights in memory and move them to the card as needed (small cards)")
        if name == "make":
            m.add_argument("--name", required=True, help="the public repository these weights are")
            m.add_argument("--revision", required=True, help="the repository revision")
            m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI")
            m.add_argument("--probes", required=True, help="a JSON list of {prompt, seed}: any prompts and seeds you choose")
            m.add_argument("--request", required=True, help="the API settings every image is asked with (JSON)")
            m.add_argument("--settings", default="", help="pipeline settings the API fixes for this model (JSON)")
            m.add_argument("--adapter", action="append",
                           help="weights=<repo>,revision=<commit>,file=<name>,path=<local file>[,scale=<s>]")
            m.add_argument("--out", required=True)
            m.add_argument("--keep", default="", help="also keep each picture in this folder")
            m.add_argument("--image-distance", type=float, required=True)
            m.add_argument("--typical-distance", type=float, required=True)
            m.add_argument("--close", type=float, required=True)
        else:
            m.add_argument("--model", required=True)
            m.add_argument("--fingerprint", default="")
            m.add_argument("--adapter-file", action="append", help="the local file of each adapter the fingerprint lists")
            m.add_argument("--prompt", required=True)
            m.add_argument("--seed", type=int, required=True)
            m.add_argument("--image", required=True, help="the PNG the API returned")
    args = parser.parse_args(argv)
    if args.command == "make":
        return make(args)
    if args.command == "replay":
        return replay(args)
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out,
                     keys=None if args.keys == "none" else check_receipt.load_keys(args.keys))


if __name__ == "__main__":
    sys.exit(main() or 0)
