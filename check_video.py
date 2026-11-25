"""Check that a video model on EmpirioLabs AI is the public model it is said to be, without running any model.

A fingerprint of kind "video" (fingerprints/<model>.json) was made once, from the model's public weights: for a fixed
set of probes (a prompt and a seed, with a starting image for a model that animates one), a few frames of the clip the
public pipeline makes, each kept as a 32 x 32 thumbnail, the average color of each block of the frame, exactly as an
image fingerprint keeps a picture (check_image.py). The seed's noise is drawn as float32 numbers from torch's CPU
generator, which draws the same numbers on every computer, so the same request gives the same clip wherever the public
weights run, up to the small rounding differences between graphics cards. The check asks the live API for the same
clips, reads the same frames and compares each with its thumbnail: the distance is one minus the correlation of their
colors, 0 for the same frame even when it is a little lighter or darker, and a clip's distance is the mean over its
frames. A clip from another model, or from the same model with another seed, is laid out differently and is far from it.

    python check_model.py --model hunyuan-video-1-5               the usual way, as for every other model
    python check_video.py check --model hunyuan-video-1-5         the same check, --out keeps the numbers read
    python check_video.py signature clip.mp4                      the thumbnails of a clip you have
    python check_video.py compare clip.mp4 other.mp4              how far two clips are, frame by frame
    python check_video.py make --weights <folder> --name OpenMOSS-Team/MOVA-360p --revision <commit> \
        --model moss-video-and-audio --probes <probes.json> --request '<API settings as JSON>' --pipeline mova \
        --settings '<pipeline settings as JSON>' --out fingerprints/moss-video-and-audio.json \
        --clip-distance <limit> --typical-distance <limit> --close <share>

Decoding needs ffmpeg. The Python standard library cannot decode the video in an MP4 file, so this check runs ffmpeg
(https://ffmpeg.org/download.html) to turn each clip into frames, and for nothing else: the `ffmpeg` on your PATH, or
the program the environment variable VERIFIABLE_INFERENCE_FFMPEG names. It reads only local files, and only the video
stream. Everything else is the standard library. Without ffmpeg the check says so and stops before it sends any request.

The check reads your API key from EMPIRIOLABS_API_KEY. It asks for each of the fingerprint's clips once, billed to your
account at the model's listed price (with the shortest and cheapest settings the model takes), sends every request
before it waits for the first, and polls each job on the API's own origin until it ends. A starting image is sent from
the folder beside the fingerprint, and only when its SHA-256 is the one the fingerprint records.

You choose what is checked: `make` takes any prompts, seeds and starting images (--probes), so you can make a
fingerprint of your own that we have never seen, from the public weights, and check the API with it. `make` draws each
clip with the pipeline published with the weights in this process (--pipeline mova: the MOVA code; --pipeline
diffusers: a diffusers video pipeline), or reads the clips another public program drew, one MP4 per probe named
<seed>.mp4 (--clips <folder>). It keeps the frames at --positions, shares of the clip (default a quarter, a half, three
quarters and the end). It needs torch and the pipeline's own requirements, and ffmpeg for --clips.

Verdict
    matches          at least the share of clips the fingerprint states within its per-clip distance, each the length
                     its fingerprint states, and the typical (median) distance within its limit
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model, and the clip checked
                     is the file that answer lists
Exit status: 0 matches and signed, 1 anything else.
"""
import argparse
import base64
import contextlib
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import check_image
import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "video"
ENDPOINT = "/v1/videos/generations"
ORIGIN = "%s://%s" % urllib.parse.urlsplit(check_receipt.API)[:2]
FFMPEG_ENV = "VERIFIABLE_INFERENCE_FFMPEG"
# make's frames, as shares of the clip: index round(share * (frames - 1)). The very first frame of a clip made from a
# starting image is that image, whatever the model, so it is not among them.
POSITIONS = (0.25, 0.5, 0.75, 1.0)
# How long a clip may take: the API's own wait for a video, plus room for a worker that has to start first.
JOB_WAIT_SECONDS = 4200
POLL_SECONDS = 10
RUNNING = ("queued", "processing", "running", "pending", "in_progress")
REFUSED_AGAIN = (5, 15, 30)  # waits after a 429 before the request is sent again (a 429 is never a job)
MADE_WITH = ("the pipeline published with the weights, on cuda", "the pipeline published with the weights, on cpu",
             "the reference code published with the weights, on cuda",
             "the reference code published with the weights, on cpu")
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


# ------------------------------------------------------------------------------------------------------- frames
class NotAVideo(ValueError):
    """The bytes are not a clip ffmpeg decodes into frames."""


class NoDecoder(RuntimeError):
    """ffmpeg is not here: nothing can be decoded, so nothing is asked."""


def ffmpeg_program():
    """The ffmpeg this check runs: VERIFIABLE_INFERENCE_FFMPEG, else the one on PATH, else None."""
    named = os.environ.get(FFMPEG_ENV, "").strip()
    if named:
        return named if os.path.isfile(named) else None
    return shutil.which("ffmpeg")


def require_ffmpeg():
    program = ffmpeg_program()
    if program is None:
        raise NoDecoder("ffmpeg is not installed here, and this check needs it to read a clip's frames: install it "
                        "(https://ffmpeg.org/download.html) or set %s to its path. Nothing was sent." % FFMPEG_ENV)
    return program


def _read_exact(stream, size):
    chunks, left = [], size
    while left > 0:
        chunk = stream.read(left)
        if not chunk:
            break
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def _ppm_header(stream):
    """(width, height) of the next binary PPM frame of `stream`, or None at its end. The header is P6, the width, the
    height and the largest value (255), each after white space; one white space byte ends it."""
    tokens, token = [], b""
    while len(tokens) < 4:
        byte = stream.read(1)
        if not byte:
            if tokens or token:
                raise NotAVideo("the decoder stopped inside a frame header")
            return None
        if byte == b"#" and not token:
            while byte not in (b"\n", b""):
                byte = stream.read(1)
            continue
        if byte.isspace():
            if token:
                tokens.append(token)
                token = b""
            continue
        token += byte
    magic, width, height, largest = tokens
    if magic != b"P6" or not (width.isdigit() and height.isdigit()) or largest != b"255":
        raise NotAVideo("the decoder returned frames this check does not read")
    return int(width), int(height)


def read_frames(stream, wanted):
    """(the number of frames, {index: (width, height, bytes)} for each index in `wanted`) of a stream of binary PPM
    frames, each frame read and dropped unless it is wanted, so one frame at a time is in memory."""
    count, kept = 0, {}
    while True:
        header = _ppm_header(stream)
        if header is None:
            return count, kept
        width, height = header
        body = _read_exact(stream, 3 * width * height)
        if len(body) != 3 * width * height:
            raise NotAVideo("the decoder stopped inside a frame")
        if count in wanted:
            kept[count] = (width, height, body)
        count += 1


def frames(data, wanted=(), program=None):
    """(the number of frames, {index: (width, height, red-green-blue bytes)}) of the MP4 clip `data`, decoded by ffmpeg:
    the clip is written to a new temporary folder, only the local file protocol and the MP4 demuxer are allowed, and
    only its first video stream is read."""
    program = program or require_ffmpeg()
    folder = tempfile.mkdtemp(prefix="verifiable-inference-")
    try:
        clip = os.path.join(folder, "clip.mp4")
        with open(clip, "wb") as handle:
            handle.write(data)
        log = os.path.join(folder, "decoder.log")
        command = [program, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
                   "-f", "mov", "-i", clip, "-map", "0:v:0", "-an", "-sn", "-dn",
                   "-f", "image2pipe", "-c:v", "ppm", "-pix_fmt", "rgb24", "pipe:1"]
        with open(log, "wb") as errors:
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
            except OSError as error:
                raise NoDecoder("ffmpeg could not be started (%s: %s)" % (type(error).__name__, error)) from None
            try:
                count, kept = read_frames(process.stdout, set(wanted))
            finally:
                process.stdout.close()
                code = process.wait()
        if code != 0 or count == 0:
            with open(log, "rb") as handle:
                said = handle.read(300).decode("utf-8", "replace").strip().splitlines()
            raise NotAVideo("ffmpeg could not read the clip%s" % (": " + said[0] if said else ""))
        return count, kept
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def frame_thumbnail(width, height, raw):
    """The 32 x 32 thumbnail of one frame of red, green and blue bytes (check_image.blocks)."""
    stride = 3 * width
    return check_image.blocks(width, height, [raw[y * stride:(y + 1) * stride] for y in range(height)])


def frame_indices(count, positions=POSITIONS):
    """The frames at `positions` (shares of the clip) of a clip of `count` frames, in order, each once."""
    return sorted({min(count - 1, max(0, int(round(share * (count - 1))))) for share in positions})


def signature(data, indices=None, positions=POSITIONS, program=None):
    """{"frame_count", "frames", "thumbnails", "width", "height"} of a clip: the thumbnails of the frames at `indices`
    (those the clip has), or at `positions` when no indices are given."""
    if indices is None:
        count, _ = frames(data, (), program)
        indices = frame_indices(count, positions)
    count, kept = frames(data, indices, program)
    shown = [index for index in indices if index in kept]
    size = kept[shown[0]][:2] if shown else (0, 0)
    return {"frame_count": count, "frames": shown, "thumbnails": [frame_thumbnail(*kept[i]) for i in shown],
            "width": size[0], "height": size[1]}


def pictures_signature(pictures, positions=POSITIONS):
    """The same signature from frames a pipeline returned in this process (PIL images), before any encoding."""
    indices = frame_indices(len(pictures), positions)
    thumbnails = []
    for index in indices:
        picture = pictures[index].convert("RGB")
        thumbnails.append(frame_thumbnail(picture.width, picture.height, picture.tobytes()))
    return {"frame_count": len(pictures), "frames": indices, "thumbnails": thumbnails,
            "width": pictures[0].width, "height": pictures[0].height}


def clip_distance(reference, live):
    """(the mean distance over the reference's frames, each frame's distance) of a live signature against a
    fingerprint probe, or (None, why) when the live clip is not the length the fingerprint states."""
    if live["frame_count"] != reference["frame_count"]:
        return None, "%d frames, the fingerprint's clip %d" % (live["frame_count"], reference["frame_count"])
    found = dict(zip(live["frames"], live["thumbnails"]))
    gaps = []
    for index, thumbnail in zip(reference["frames"], reference["thumbnails"]):
        if index not in found:
            return None, "no frame %d" % index
        gaps.append(check_image.distance(check_image.decode_thumbnail(thumbnail), found[index]))
    return sum(gaps) / len(gaps), gaps


def verdict(fingerprint, lives):
    """The verdict on live signatures, one per probe in order."""
    limits = fingerprint["limits"]
    distances, frame_gaps, other_length = [], [], 0
    for probe, live in zip(fingerprint["probes"], lives):
        mean, gaps = clip_distance(probe, live)
        distances.append(mean)
        frame_gaps.append(gaps if mean is not None else [])
        other_length += mean is None
    read = [d for d in distances if d is not None]
    close = sum(1 for d in read if d <= limits["clip_distance"])
    typical = statistics.median(read) if read else float("inf")
    ok = bool(distances) and close / len(distances) >= limits["close"] and typical <= limits["typical_distance"]
    return {"result": "matches" if ok else "DOES NOT MATCH", "clips": len(distances), "close": close,
            "typical_distance": typical, "distances": distances, "frame_distances": frame_gaps,
            "other_length": other_length}


# ------------------------------------------------------------------------------------------------------ asking
class Unanswered(Exception):
    """The API gave no clip for a probe: the check cannot be done, which says nothing about the model."""


def _send(method, path, key, body=None):
    sent = None if body is None else json.dumps(body).encode()
    for attempt in range(len(REFUSED_AGAIN) + 1):
        request = urllib.request.Request(ORIGIN + path, data=sent, method=method,
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                                  "User-Agent": check_receipt.USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=600 if method == "POST" else 120) as response:
                received = response.read()
                return response.status, json.loads(received), sent, received
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < len(REFUSED_AGAIN):
                time.sleep(REFUSED_AGAIN[attempt])
                continue
            raise Unanswered("HTTP %d: %s" % (error.code, error.read()[:300].decode("utf-8", "replace"))) from None
        except (OSError, ValueError) as error:
            raise Unanswered("%s: %s" % (type(error).__name__, error)) from None
    raise Unanswered("the API kept refusing the request (HTTP 429)")


def start_image(folder, image):
    """The bytes of a probe's starting image beside its fingerprint, refused unless they are the file it records."""
    path = os.path.join(folder, *str(image["file"]).split("/"))
    if os.path.splitext(path)[1].lower() not in IMAGE_TYPES:
        raise ValueError("the starting image %s is not a PNG or JPEG file" % image["file"])
    with open(path, "rb") as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != image["sha256"]:
        raise ValueError("the starting image %s is not the file this fingerprint records" % image["file"])
    return data


def request_body(model, probe, request, image=None, image_name=""):
    """The request a probe sends: the fingerprint's settings, the probe's prompt and seed, and its starting image."""
    body = {"model": model, **request, "prompt": probe["prompt"], "seed": probe["seed"]}
    if image is not None:
        kind = IMAGE_TYPES.get(os.path.splitext(image_name)[1].lower(), "image/png")
        body["image"] = "data:%s;base64,%s" % (kind, base64.b64encode(image).decode("ascii"))
    return body


def _running(status, data):
    return status == 202 or (data.get("status") in RUNNING and not data.get("data"))


def _poll_path(data):
    poll = data.get("poll_url")
    if isinstance(poll, str) and poll.startswith("/v1/jobs/"):
        return poll
    return "/v1/jobs/%s" % urllib.parse.quote(str(data.get("job_id") or data.get("id") or ""), safe="")


def submit(model, probe, request, key, call, image=None, image_name=""):
    """Send one probe; its state: the request bytes, and either the answer that holds the clip or the job to poll."""
    status, data, sent, received = call("POST", ENDPOINT, key, request_body(model, probe, request, image, image_name))
    state = {"sent": sent, "received": received, "data": data, "done": not _running(status, data),
             "started": time.time()}
    if not state["done"]:
        state["poll"] = _poll_path(data)
    return state


def poll(state, key, call):
    """Read a job once; the state is done when the job has ended with its clip."""
    _status, job, _sent, received = call("GET", state["poll"], key)
    status = job.get("status")
    if status in ("completed", "succeeded"):
        state.update(data=job.get("result") or {}, received=received, done=True)
    elif status not in RUNNING and status is not None:
        raise Unanswered("the video job ended %s: %s" % (status, json.dumps(job.get("error"))[:300]))
    elif time.time() - state["started"] > JOB_WAIT_SECONDS:
        raise Unanswered("the video job did not finish in %d minutes" % (JOB_WAIT_SECONDS // 60))


def clip_of(data):
    """The MP4 bytes an answer holds or links."""
    items = [item for item in data.get("data") or [] if isinstance(item, dict)]
    if not items:
        raise Unanswered("the answer holds no clip")
    if items[0].get("b64_json"):
        return base64.b64decode(items[0]["b64_json"])
    if items[0].get("url"):
        return check_receipt.download(items[0]["url"])
    raise Unanswered("the answer's clip has neither data nor a link")


def ask_all(model, fingerprint, key, folder, call=None):
    """Every probe's clip, as MP4 bytes, in order, with each (request bytes, answer bytes, clip) for the receipt check.
    Every probe is sent first, then the jobs are polled together. `call(method, path, key, body=None)` sends one
    request and returns (the HTTP status, the JSON answer, the request bytes, the answer bytes), as _send does, which
    is the default: a caller can send the requests through its own transport. A GET of /v1/jobs/{id} reads the job
    that the POST answered with that id made, and that POST's bytes are what its answer's receipt names."""
    call = call or _send
    states = []
    for number, probe in enumerate(fingerprint["probes"], 1):
        image = probe.get("image")
        data = start_image(folder, image) if image else None
        try:
            states.append(submit(model, probe, fingerprint["request"], key, call, data,
                                 image["file"] if image else ""))
        except Unanswered as error:
            raise Unanswered("the API did not take clip %d: %s" % (number, error)) from None
    while not all(state["done"] for state in states):
        time.sleep(POLL_SECONDS)
        for number, state in enumerate(states, 1):
            if not state["done"]:
                try:
                    poll(state, key, call)
                except Unanswered as error:
                    raise Unanswered("the API did not answer clip %d: %s" % (number, error)) from None
    clips = []
    for number, state in enumerate(states, 1):
        try:
            clips.append((state["sent"], state["received"], clip_of(state["data"])))
        except (Unanswered, OSError) as error:
            raise Unanswered("the API did not answer clip %d: %s" % (number, error)) from None
    return clips


# ------------------------------------------------------------------------------------------------------- check
def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not a %s fingerprint" % (os.path.basename(path), KIND))
    return fingerprint


def run_check(model, fingerprint, key, keys=None, folder=FINGERPRINTS, ask_one=None, call=None, exchanges=None):
    """Every probe asked once; (the verdict, each live signature). With `keys` (check_receipt.load_keys) every answer's
    receipt is checked, and that the clip is the file the answer lists: the verdict's `receipts` holds one (True, the
    receipt) or (False, why) per clip. `call` sends the requests through the caller's own transport (ask_all's), a list
    `exchanges` gets each (request bytes, answer bytes, clip) for a caller that checks the receipts itself, and
    `ask_one(probe)` replaces the request to the API (tests)."""
    require_ffmpeg()
    if ask_one is not None:
        clips = [(None, None, ask_one(probe)) for probe in fingerprint["probes"]]
    else:
        clips = ask_all(model, fingerprint, key, folder, call=call)
    lives = []
    for number, ((_sent, _received, clip), probe) in enumerate(zip(clips, fingerprint["probes"]), 1):
        try:
            lives.append(signature(clip, indices=probe["frames"]))
        except NotAVideo as error:
            raise Unanswered("clip %d could not be read: %s" % (number, error)) from None
    if exchanges is not None:
        exchanges.extend(clips)
    v = verdict(fingerprint, lives)
    if keys is not None and ask_one is None:
        receipts = []
        for sent, received, clip in clips:
            ok, result = check_receipt.check(sent, received, keys)
            if ok:
                good, why = check_receipt.check_outputs(received, [clip])
                if not good:
                    ok, result = False, why
            receipts.append((ok, result))
        v["receipts"] = receipts
    return v, lives


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    print("  clips within %.3f of their frames: %d of %d (needs %d)" % (
        limits["clip_distance"], v["close"], v["clips"], int(limits["close"] * v["clips"] + 0.999)))
    print("  typical distance: %.4f (limit %.3f)" % (v["typical_distance"], limits["typical_distance"]))
    print("  frames compared: %d per clip%s" % (
        len(fingerprint["probes"][0]["frames"]),
        ", every clip the length its fingerprint states" if not v["other_length"]
        else ", %d clips of another length" % v["other_length"]))
    signed = check_receipt.report_receipts(model, v.get("receipts"))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, fingerprint=None, out="", keys=None):
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        v, lives = run_check(model, fingerprint, key, keys=keys, folder=os.path.dirname(os.path.abspath(path)))
    except (NoDecoder, Unanswered, OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"],
                  **{name: v[name] for name in v if name != "receipts"},
                  "signatures": [{**live, "thumbnails": [check_image.encode_thumbnail(t) for t in live["thumbnails"]]}
                                 for live in lives]}
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=1, default=str)
    return 0 if v["result"] == "matches" and signed is not False else 1


# -------------------------------------------------------------------------------------------------------- make
@contextlib.contextmanager
def seeded_noise(module, seed):
    """While the block runs, the pipeline module's randn_tensor draws float32 normal numbers from torch's CPU generator
    seeded with `seed` (the generator the pipeline passes, when it passes a CPU one), in the order and the shapes the
    pipeline asks, then moves and casts them to the device and precision it asks. A pipeline that draws its noise
    without a generator (MOVA) gets the seed's numbers all the same, and a pipeline that asks for bfloat16 noise gets
    the float32 numbers cast, which every torch release draws alike."""
    import torch

    original = getattr(module, "randn_tensor", None)
    if original is None:
        raise SystemExit("the pipeline's module %s draws no noise through randn_tensor" % module.__name__)
    source = torch.Generator("cpu").manual_seed(int(seed))

    def draw(shape, generator, layout):
        if isinstance(generator, torch.Generator) and generator.device.type == "cpu":
            return torch.randn(shape, generator=generator, dtype=torch.float32, layout=layout)
        return torch.randn(shape, generator=source, dtype=torch.float32, layout=layout)

    def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
        shape, layout = tuple(shape), layout or torch.strided
        if isinstance(generator, (list, tuple)) and len(generator) > 1:
            noise = torch.cat([draw((1,) + shape[1:], one, layout) for one in generator], dim=0)
        else:
            noise = draw(shape, generator[0] if isinstance(generator, (list, tuple)) else generator, layout)
        return noise.to(device=device if device is not None else "cpu",
                        dtype=dtype if dtype is not None else torch.float32)

    module.randn_tensor = randn_tensor
    try:
        yield
    finally:
        module.randn_tensor = original


def _mova(weights, dtype, device, offload):
    """MOVA (the code published with OpenMOSS-Team/MOVA-360p): a pipeline object and the function that draws a clip."""
    import torch
    import torch.distributed as dist

    for name, value in (("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29500"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                        ("LOCAL_RANK", "0")):
        os.environ.setdefault(name, value)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")  # MOVA reads its rank from torch.distributed
    from mova.datasets.transforms.custom import crop_and_resize
    from mova.diffusion.pipelines.pipeline_mova import MOVA
    from PIL import Image

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = MOVA.from_pretrained(weights, torch_dtype=torch_dtype)
    if offload:
        pipe.enable_group_offload(onload_device=torch.device(device), offload_device=torch.device("cpu"),
                                  offload_type="leaf_level", use_stream=True)
    else:
        pipe.to(device)

    def draw_clip(probe, request, settings, image_path):
        height, width = int(settings["height"]), int(settings["width"])
        image = crop_and_resize(Image.open(image_path).convert("RGB"), height, width)
        with seeded_noise(sys.modules[type(pipe).__module__], probe["seed"]):
            video, _audio = pipe(prompt=probe["prompt"], image=image, negative_prompt=request.get("negative_prompt", ""),
                                 seed=probe["seed"], height=height, width=width,
                                 num_frames=int(settings["num_frames"]),
                                 num_inference_steps=int(settings["num_inference_steps"]),
                                 cfg_scale=float(settings["cfg_scale"]), sigma_shift=float(settings["sigma_shift"]))
        return list(video[0])
    return draw_clip


def _diffusers(weights, dtype, device, offload):
    """A video pipeline published in diffusers' format: a pipeline object and the function that draws a clip."""
    import inspect

    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = DiffusionPipeline.from_pretrained(weights, torch_dtype=torch_dtype)
    if offload:
        pipe.enable_sequential_cpu_offload() if offload == "sequential" else pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    accepted = set(inspect.signature(pipe.__call__).parameters)

    def draw_clip(probe, request, settings, image_path):
        unknown = sorted(set(settings) - accepted)
        if unknown:
            raise SystemExit("this pipeline takes no setting %s" % ", ".join(unknown))
        options = dict(settings)
        if image_path:
            options["image"] = Image.open(image_path).convert("RGB")
        if "negative_prompt" in request and "negative_prompt" in accepted:
            options["negative_prompt"] = request["negative_prompt"]
        with seeded_noise(sys.modules[type(pipe).__module__], probe["seed"]), torch.inference_mode():
            output = pipe(prompt=probe["prompt"], generator=torch.Generator("cpu").manual_seed(int(probe["seed"])),
                          output_type="pil", **options)
        return list(output.frames[0])
    return draw_clip


PIPELINES = {"mova": _mova, "diffusers": _diffusers}


def _positions(text):
    shares = tuple(float(part) for part in str(text).split(",") if part.strip())
    if not shares or any(not 0 <= share <= 1 for share in shares):
        raise SystemExit("--positions takes shares of the clip from 0 to 1, for example 0.25,0.5,0.75,1")
    return shares


def _recorded_image(image, model, out_folder):
    """A probe's starting image copied beside the fingerprint (fingerprints/<model>/<name>), as the file records it."""
    name = os.path.basename(image["path"])
    target = os.path.join(out_folder, model)
    os.makedirs(target, exist_ok=True)
    shutil.copyfile(image["path"], os.path.join(target, name))
    recorded = {"file": "%s/%s" % (model, name), "sha256": check_image.sha256_of(image["path"]),
                "source": image["source"], "license": image["license"]}
    if image.get("processing"):
        recorded["processing"] = image["processing"]
    return recorded


def _components(specs):
    """Each --component role=...,weights=<repo>,revision=<commit>[,file=<name>,path=<local file>][,scale=<s>]: a public
    repository the clip was made from besides the weights (a base model, an adapter merged at a scale, an encoder)."""
    out = []
    for spec in specs or []:
        fields = dict(part.split("=", 1) for part in spec.split(","))
        item = {"role": fields["role"], "weights": fields["weights"], "revision": fields["revision"]}
        if fields.get("file"):
            item["file"] = fields["file"]
            item["sha256"] = check_image.sha256_of(fields["path"])
        if fields.get("scale"):
            item["scale"] = float(fields["scale"])
        out.append(item)
    return out


def make(args):
    with open(args.probes, encoding="utf-8") as handle:
        probes = json.load(handle)
    request = json.loads(args.request)
    settings = json.loads(args.settings or "{}")
    positions = _positions(args.positions)
    out_folder = os.path.dirname(os.path.abspath(args.out))
    if args.clips:
        require_ffmpeg()
        draw_clip, made_with = None, args.made_with or MADE_WITH[0]
    else:
        draw_clip = PIPELINES[args.pipeline](args.weights, args.dtype, args.device, args.offload)
        made_with = args.made_with or "the pipeline published with the weights, on %s" % args.device.split(":")[0]
    if made_with not in MADE_WITH:
        raise SystemExit("--made-with is one of: %s" % "; ".join(MADE_WITH))
    rows, started = [], time.time()
    for probe in probes:
        t = time.time()
        image = probe.get("image")
        if draw_clip is None:
            with open(os.path.join(args.clips, "%d.mp4" % probe["seed"]), "rb") as handle:
                made = signature(handle.read(), positions=positions)
        else:
            made = pictures_signature(draw_clip(probe, request, settings, image["path"] if image else None), positions)
        row = {"prompt": probe["prompt"], "seed": probe["seed"]}
        if image:
            row["image"] = _recorded_image(image, args.model, out_folder)
        row.update(frame_count=made["frame_count"], width=made["width"], height=made["height"], frames=made["frames"],
                   thumbnails=[check_image.encode_thumbnail(t) for t in made["thumbnails"]])
        rows.append(row)
        print("%-50s seed %-10d %d frames %.0f s" % (probe["prompt"][:50], probe["seed"], made["frame_count"],
                                                    time.time() - t), flush=True)
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": check_image._weight_files(args.weights), "made_with": made_with,
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request": request,
                   "limits": {"clip_distance": args.clip_distance, "typical_distance": args.typical_distance,
                              "close": args.close},
                   "probes": rows}
    if settings:
        fingerprint["settings"] = settings
    components = _components(args.component)
    if components:
        fingerprint["components"] = components
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)
    print("fingerprint of %d clips written in %.0f s" % (len(rows), time.time() - started))


# ------------------------------------------------------------------------------------------- local clips
def _signature_command(args):
    with open(args.clip, "rb") as handle:
        data = handle.read()
    indices = [int(i) for i in args.frames.split(",")] if args.frames else None
    made = signature(data, indices=indices, positions=_positions(args.positions))
    print(json.dumps({**made, "thumbnails": [check_image.encode_thumbnail(t) for t in made["thumbnails"]]}, indent=1))
    return 0


def _compare_command(args):
    with open(args.clip, "rb") as handle:
        first = signature(handle.read(), positions=_positions(args.positions))
    with open(args.other, "rb") as handle:
        second = signature(handle.read(), indices=first["frames"])
    mean, gaps = clip_distance({**first, "thumbnails": [check_image.encode_thumbnail(t) for t in first["thumbnails"]]},
                               second)
    if mean is None:
        print("not the same clip: %s" % gaps)
        return 1
    print("frames %s: distances %s, mean %.4f" % (first["frames"], ", ".join("%.4f" % g for g in gaps), mean))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example hunyuan-video-1-5")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the numbers read from each answer to this JSON file")
    c.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    s = sub.add_parser("signature", help="print the thumbnails of a clip you have")
    s.add_argument("clip")
    s.add_argument("--frames", default="", help="frame indices, for example 6,12,18,24 (default: --positions)")
    s.add_argument("--positions", default=",".join(str(p) for p in POSITIONS))
    p = sub.add_parser("compare", help="how far two clips are, at the same frames")
    p.add_argument("clip")
    p.add_argument("other")
    p.add_argument("--positions", default=",".join(str(p) for p in POSITIONS))
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint (its files are hashed)")
    m.add_argument("--name", required=True, help="the public repository these weights are")
    m.add_argument("--revision", required=True, help="the repository revision")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI")
    m.add_argument("--probes", required=True, help="a JSON list of {prompt, seed[, image: {path, source, license, "
                                                   "processing}]}: any prompts, seeds and starting images you choose")
    m.add_argument("--request", required=True, help="the API settings every clip is asked with (JSON)")
    m.add_argument("--settings", default="", help="pipeline settings the API fixes for this model (JSON)")
    m.add_argument("--component", action="append", help="role=<role>,weights=<repo>,revision=<commit>"
                                                         "[,file=<name>,path=<local file>][,scale=<s>]")
    source = m.add_mutually_exclusive_group(required=True)
    source.add_argument("--pipeline", choices=sorted(PIPELINES), help="draw each clip in this process")
    source.add_argument("--clips", help="a folder of <seed>.mp4 clips another public program drew")
    m.add_argument("--made-with", default="", help="how the clips were made: one of the published phrasings")
    m.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    m.add_argument("--device", default="cuda", help="cuda or cpu")
    m.add_argument("--offload", default="", choices=("", "group", "model", "sequential"),
                   help="keep the weights in memory and move them to the card as needed (small cards)")
    m.add_argument("--positions", default=",".join(str(p) for p in POSITIONS))
    m.add_argument("--out", required=True)
    m.add_argument("--clip-distance", type=float, required=True)
    m.add_argument("--typical-distance", type=float, required=True)
    m.add_argument("--close", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make":
            return make(args)
        if args.command == "signature":
            return _signature_command(args)
        if args.command == "compare":
            return _compare_command(args)
    except (NoDecoder, NotAVideo) as error:
        print(error)
        return 1
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out,
                     keys=None if args.keys == "none" else check_receipt.load_keys(args.keys))


if __name__ == "__main__":
    sys.exit(main() or 0)
