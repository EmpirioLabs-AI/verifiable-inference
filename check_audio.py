"""Check that an audio model on EmpirioLabs AI (music, sound effects or speech) is the public model it is said to be,
without running any model.

A fingerprint of kind "audio" (fingerprints/<model>.json) was made once, from the model's public weights: for a fixed
set of probes (a prompt and a seed, with lyrics where the model sings, or a text and a voice for speech), the clip the
pipeline published with the weights makes from that seed, kept as a compact spectral signature: the energy, in dB, of
30 frequency bands from 50 Hz to 12 kHz in every quarter second of the clip, from 70 dB under a full-scale tone up. The seed's noise is drawn as float32
numbers from torch's CPU generator, which draws the same numbers on every computer and every torch release, so the same
request gives the same clip wherever the public weights run, up to the small rounding differences between cards. The
check asks the live API for the same clips as WAV, computes the same signature in pure Python and compares the two: the
mean absolute difference of their dB values. A clip from another model, or from the same model with another seed, is
far from it. A probe may hold several reference clips of the model (`references`): a live clip is then compared with the
closest of them. A file whose `made_with` is "the model as served" holds clips recorded from the model EmpirioLabs
serves: its check confirms that the model is still the one that was recorded, and `make` doesn't rebuild it.

    python check_model.py --model ace-step-1-5-xl                the usual way, as for every other model
    python check_audio.py check --model ace-step-1-5-xl          the same check, --out keeps the numbers read
    python check_audio.py make --pipeline ace-step --weights <folder> --name ACE-Step/acestep-v15-xl-turbo-diffusers \
        --revision <commit> --model ace-step-1-5-xl --probes <probes.json> --request '{"format": "wav"}' \
        --out fingerprints/ace-step-1-5-xl.json --clip-distance <dB> --typical-distance <dB> --close <share>
    python check_audio.py signature clip.wav                     the signature of a clip you have

You choose what is checked: `make` takes any prompts and seeds (--probes), so you can make a fingerprint of your own
from the public weights and check the API with it.

The check uses only the Python standard library and reads your API key from EMPIRIOLABS_API_KEY. It sends each probe
once, billed to your account at the model's listed price. `make` runs the public weights with the pipeline published
with them (diffusers' AceStepPipeline for ACE-Step, the stable-audio-3 library for Stable Audio 3) and needs torch, the
pipeline's own requirements and numpy.

Verdict
    matches          at least the share of clips the fingerprint states within its per-clip distance (dB), and the
                     typical (median) distance within its limit
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model, and the clip checked
                     is the file that answer lists (or, for audio answered as bytes, the very bytes the receipt covers)
Exit status: 0 matches and signed, 1 anything else.
"""
import argparse
import array
import base64
import cmath
import hashlib
import io
import json
import math
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "audio"
ORIGIN = "%s://%s" % urllib.parse.urlsplit(check_receipt.API)[:2]
# The signature: BANDS log-spaced bands from LOW_HZ to HIGH_HZ, in cells of CELL_SECONDS, each value the band's mean
# energy over the cell's frames in dB relative to a full-scale tone, kept to half a dB (one byte a value) and never
# under FLOOR_DB, so the rounding noise of near-silent passages cannot weigh in. Frames are the largest power of two of
# samples within FRAME_SECONDS, Hann-windowed, side by side.
BANDS = 30
LOW_HZ = 50.0
HIGH_HZ = 12000.0
CELL_SECONDS = 0.25
FRAME_SECONDS = 0.1
FLOOR_DB = -70.0
STEP_DB = 0.5
# Request fields a probe may set, by endpoint; every other setting comes from the fingerprint's `request`.
MUSIC_FIELDS = ("prompt", "lyrics", "audio_duration", "num_inference_steps", "guidance_scale", "shift", "seed",
                "instrumental")
SPEECH_FIELDS = ("input", "voice", "speed", "sample_rate", "model_quality", "volume", "seed", "voice_model",
                 "voice_s1", "voice_s2", "voice_s3", "voice_s4", "temperature", "top_k", "top_p", "repetition_penalty")
ENDPOINTS = {"/v1/audio/generations": MUSIC_FIELDS, "/v1/audio/speech": SPEECH_FIELDS}


# ------------------------------------------------------------------------------------------------------- decoding
class NotAudio(ValueError):
    """The bytes are not a WAV file this module reads."""


def _chunks(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise NotAudio("not a WAV file")
    found, position = {}, 12
    while position + 8 <= len(data):
        name, size = struct.unpack("<4sI", data[position:position + 8])
        body = data[position + 8:] if name == b"data" and size in (0, 0xFFFFFFFF) else data[position + 8:position + 8 + size]
        found.setdefault(name, body)
        position += 8 + size + (size & 1)
    return found


def _float_wav(data):
    """(rate, channels, samples) of a WAV of 32- or 64-bit floating-point samples, which the wave module does not
    read: a RIFF walk to its format and data chunks."""
    chunks = _chunks(data)
    form, body = chunks.get(b"fmt "), chunks.get(b"data")
    if form is None or body is None or len(form) < 16:
        raise NotAudio("a WAV without a format or data chunk")
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack("<HHIIHH", form[:16])
    if tag == 0xFFFE and len(form) >= 26:
        tag = struct.unpack("<H", form[24:26])[0]  # WAVE_FORMAT_EXTENSIBLE: the subformat's tag
    if tag != 3 or bits not in (32, 64) or channels < 1:
        raise NotAudio("a WAV layout this check does not read (format %d, %d bits)" % (tag, bits))
    samples = array.array("f" if bits == 32 else "d")
    samples.frombytes(body[:len(body) - len(body) % samples.itemsize])
    if sys.byteorder != "little":
        samples.byteswap()
    return rate, channels, samples


def _pcm(raw, width):
    """Integer PCM bytes (little-endian, signed but for 8 bits) as floats from -1 to 1."""
    if width == 1:
        return [(value - 128) / 128.0 for value in raw]
    if width == 3:  # widen each sample to four bytes, the low one zero, so the sign stays in the top byte
        wide = bytearray(len(raw) // 3 * 4)
        wide[1::4], wide[2::4], wide[3::4] = raw[0::3], raw[1::3], raw[2::3]
        raw, width = bytes(wide), 4
    samples = array.array({2: "h", 4: "i"}[width])
    samples.frombytes(raw[:len(raw) - len(raw) % width])
    if sys.byteorder != "little":
        samples.byteswap()
    scale = 1.0 / (1 << (8 * width - 1))
    return [value * scale for value in samples]


def decode_wav(data):
    """(sample rate, mono samples as floats) of a WAV file: integer PCM through the standard library's wave module,
    floating point through a RIFF walk; channels are averaged."""
    try:
        with wave.open(io.BytesIO(data)) as clip:
            channels, width, rate = clip.getnchannels(), clip.getsampwidth(), clip.getframerate()
            raw = clip.readframes(clip.getnframes())
        if width not in (1, 2, 3, 4):
            raise NotAudio("a WAV of %d-byte samples" % width)
        samples = _pcm(raw, width)
    except (wave.Error, EOFError):
        rate, channels, samples = _float_wav(data)
    if channels == 1:
        return rate, list(samples)
    if channels == 2:
        return rate, [(a + b) * 0.5 for a, b in zip(samples[0::2], samples[1::2])]
    columns = [samples[c::channels] for c in range(channels)]
    return rate, [sum(frame) / channels for frame in zip(*columns)]


def encode_wav(rate, channels):
    """16-bit PCM WAV bytes of float channels (a list of equal-length sequences), each sample rounded and clipped as
    a 16-bit writer does."""
    frames = array.array("h", (max(-32768, min(32767, int(round(value * 32767.0))))
                               for frame in zip(*channels) for value in frame))
    if sys.byteorder != "little":
        frames.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(len(channels))
        clip.setsampwidth(2)
        clip.setframerate(int(rate))
        clip.writeframes(frames.tobytes())
    return buffer.getvalue()


# ----------------------------------------------------------------------------------------------------- signature
_TWIDDLES = {}


def _fft(values):
    """The discrete Fourier transform of a list of complex numbers whose length is a power of two."""
    n = len(values)
    if n <= 2:
        return values if n == 1 else [values[0] + values[1], values[0] - values[1]]
    even, odd = _fft(values[0::2]), _fft(values[1::2])
    twiddles = _TWIDDLES.get(n)
    if twiddles is None:
        twiddles = _TWIDDLES[n] = [cmath.exp(-2j * math.pi * k / n) for k in range(n // 2)]
    turned = [w * o for w, o in zip(twiddles, odd)]
    return [e + t for e, t in zip(even, turned)] + [e - t for e, t in zip(even, turned)]


def frame_length(rate):
    n = 256
    while n * 2 <= rate * FRAME_SECONDS:
        n *= 2
    return n


def band_bins(rate, n):
    """[(first bin, end bin)] of each band: the bins whose frequency is in [low, high), at least one each."""
    edges = [LOW_HZ * (HIGH_HZ / LOW_HZ) ** (b / BANDS) for b in range(BANDS + 1)]
    out = []
    for low, high in zip(edges, edges[1:]):
        first = int(math.ceil(low * n / rate))
        end = max(first + 1, int(math.ceil(high * n / rate)))
        out.append((min(first, n // 2), min(end, n // 2 + 1)))
    return out


def signature(rate, samples):
    """The clip's signature: for every CELL_SECONDS of it, each band's mean energy over the cell's frames in dB
    (rows of BANDS values, the cells in order). Two frames go through one complex transform."""
    n = frame_length(rate)
    window = [0.5 - 0.5 * math.cos(2 * math.pi * k / n) for k in range(n)]
    scale = (sum(window) / 2) ** 2
    bins = band_bins(rate, n)
    cell = CELL_SECONDS * rate
    cells = max(1, int(math.ceil(len(samples) / cell)))
    energy = [[0.0] * BANDS for _ in range(cells)]
    frames = [0] * cells
    starts = list(range(0, max(1, len(samples)), n))

    def spectrum_of(power, start):
        where = min(cells - 1, int((start + n / 2) / cell))
        frames[where] += 1
        row = energy[where]
        for b, (first, end) in enumerate(bins):
            row[b] += sum(power[first:end])

    for i in range(0, len(starts), 2):
        pair = starts[i:i + 2]
        x = samples[pair[0]:pair[0] + n]
        x = [s * w for s, w in zip(x, window)] + [0.0] * (n - len(x))
        y = [0.0] * n
        if len(pair) == 2:
            y = samples[pair[1]:pair[1] + n]
            y = [s * w for s, w in zip(y, window)] + [0.0] * (n - len(y))
        z = _fft([complex(a, b) for a, b in zip(x, y)])
        half = n // 2
        conj = [z[0].conjugate()] + [z[n - k].conjugate() for k in range(1, half + 1)]
        px = [abs((z[k] + conj[k]) * 0.5) ** 2 for k in range(half + 1)]
        spectrum_of(px, pair[0])
        if len(pair) == 2:
            py = [abs((z[k] - conj[k]) * 0.5) ** 2 for k in range(half + 1)]
            spectrum_of(py, pair[1])
    out = []
    for row, count in zip(energy, frames):
        out.append([max(FLOOR_DB, 10 * math.log10(max(value / max(count, 1) / scale, 1e-30))) for value in row])
    return out


def encode_signature(rows):
    """One byte a value: its half-dB steps above FLOOR_DB, capped at 255."""
    return base64.b64encode(bytes(max(0, min(255, int(round((v - FLOOR_DB) / STEP_DB)))) for row in rows
                                  for v in row)).decode("ascii")


def decode_signature(text):
    values = [FLOOR_DB + STEP_DB * byte for byte in base64.b64decode(text)]
    if len(values) % BANDS:
        raise ValueError("a signature whose length is not a whole number of cells")
    return [values[i:i + BANDS] for i in range(0, len(values), BANDS)]


def distance(reference, live):
    """The mean absolute difference, in dB, over every band of every cell of the longer of the two; a cell one of
    them lacks (a shorter clip) counts as silence (FLOOR_DB)."""
    cells = max(len(reference), len(live))
    silence = [FLOOR_DB] * BANDS
    total = 0.0
    for i in range(cells):
        ref = reference[i] if i < len(reference) else silence
        got = live[i] if i < len(live) else silence
        total += sum(abs(a - b) for a, b in zip(ref, got))
    return total / (cells * BANDS)


def clip_signature(data):
    """The signature of a WAV file's bytes, at the half-dB steps a fingerprint keeps."""
    return decode_signature(encode_signature(signature(*decode_wav(data))))


# --------------------------------------------------------------------------------------------------------- check
class Unanswered(Exception):
    """The API gave no clip for a probe: the check cannot be done, which says nothing about the model."""


def _send(method, path, key, body=None, accept="application/json"):
    """One request to the API at `path`; (status, headers, the response bytes, the request bytes sent)."""
    sent = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(ORIGIN + path, data=sent, method=method,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                              "Accept": accept, "User-Agent": check_receipt.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, dict(response.headers.items()), response.read(), sent
    except urllib.error.HTTPError as error:
        raise Unanswered("HTTP %d: %s" % (error.code, error.read()[:300].decode("utf-8", "replace"))) from None
    except OSError as error:
        raise Unanswered("%s: %s" % (type(error).__name__, error)) from None


def _json(raw):
    try:
        value = json.loads(raw)
    except ValueError:
        raise Unanswered("the answer is not JSON") from None
    if not isinstance(value, dict):
        raise Unanswered("the answer is not a JSON object")
    return value


def _clip_of(answer, download):
    items = [item for item in answer.get("data") or [] if isinstance(item, dict)]
    if not items:
        raise Unanswered("the answer holds no clip")
    if items[0].get("b64_json"):
        return base64.b64decode(items[0]["b64_json"])
    if items[0].get("url"):
        return download(items[0]["url"])
    raise Unanswered("the answer's clip has neither data nor a link")


def ask(model, probe, request, endpoint, key, exchanges=None, send=None, download=None):
    """One clip for the probe, as WAV bytes. Music and sound answer with a job, polled on the API's own origin to its
    end; speech is asked for the audio bytes themselves (Accept: audio/wav), which carry their receipt in the
    X-Receipt header. A list `exchanges` also gets (the request bytes, the bytes of the answer that carries the clip,
    its X-Receipt header or None, the clip, whether the clip must be a file the answer lists). `send` and `download`
    replace the HTTP calls (tests)."""
    send, download = send or _send, download or check_receipt.download
    fields = ENDPOINTS[endpoint]
    body = {"model": model, **request, **{name: probe[name] for name in fields if name in probe}}
    speech = endpoint == "/v1/audio/speech"
    status, headers, received, sent = send("POST", endpoint, key, body, "audio/wav" if speech else "application/json")
    content_type = {k.lower(): v for k, v in headers.items()}.get("content-type", "")
    if speech and not content_type.startswith("application/json"):
        header = {k.lower(): v for k, v in headers.items()}.get("x-receipt")
        if exchanges is not None:
            exchanges.append((sent, received, header, received, False))
        return received
    data = _json(received)
    # The job is polled on the API's own origin only: a poll_url that is not a path under /v1/jobs/ is not used.
    poll = data.get("poll_url")
    if not (isinstance(poll, str) and poll.startswith("/v1/jobs/")):
        poll = "/v1/jobs/%s" % urllib.parse.quote(str(data.get("job_id") or data.get("id") or ""), safe="")
    deadline = time.time() + 1800
    while status == 202 or (data.get("status") in ("queued", "processing", "running") and not data.get("data")):
        if time.time() > deadline:
            raise Unanswered("the audio job did not finish in 30 minutes")
        time.sleep(3)
        status, _headers, received, _sent = send("GET", poll, key)
        job = _json(received)
        if job.get("status") in ("completed", "succeeded"):
            data = job.get("result") or {}
            break
        if job.get("status") == "failed":
            raise Unanswered("the audio job failed: %s" % json.dumps(job.get("error"))[:300])
        data = job
    clip = _clip_of(data, download)
    if exchanges is not None:
        exchanges.append((sent, received, None, clip, True))
    return clip


def references(row):
    """The reference signatures of one probe, decoded: its `signature`, or each signature in its `references`."""
    if "references" in row:
        return [decode_signature(item["signature"]) for item in row["references"]]
    return [decode_signature(row["signature"])]


def verdict(fingerprint, live_signatures):
    limits = fingerprint["limits"]
    distances = [min(distance(reference, live) for reference in references(row))
                 for row, live in zip(fingerprint["probes"], live_signatures)]
    close = sum(1 for d in distances if d <= limits["clip_distance"])
    typical = statistics.median(distances)
    ok = close / len(distances) >= limits["close"] and typical <= limits["typical_distance"]
    return {"result": "matches" if ok else "DOES NOT MATCH", "clips": len(distances), "close": close,
            "typical_distance": typical, "distances": distances}


def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not an %s fingerprint" % (os.path.basename(path), KIND))
    if fingerprint.get("endpoint") not in ENDPOINTS:
        raise ValueError("%s: an endpoint this check does not ask" % os.path.basename(path))
    return fingerprint


def run_check(model, fingerprint, key, keys=None, ask_one=None):
    """Every probe sent once; (the verdict, each live signature). With `keys` (check_receipt.load_keys) every answer's
    receipt is checked too, and that the clip is the file the answer lists: the verdict's `receipts` holds one
    (True, the receipt) or (False, why) per clip. `ask_one(probe)` replaces the request to the API (tests)."""
    exchanges = [] if keys is not None and ask_one is None else None
    signatures = []
    for number, probe in enumerate(fingerprint["probes"], 1):
        try:
            clip = ask_one(probe) if ask_one else ask(model, probe, fingerprint["request"], fingerprint["endpoint"],
                                                      key, exchanges)
            signatures.append(clip_signature(clip))
        except (Unanswered, NotAudio) as error:
            raise Unanswered("the API did not answer clip %d: %s" % (number, error)) from None
    v = verdict(fingerprint, signatures)
    if exchanges is not None:
        receipts = []
        for sent, received, header, clip, listed in exchanges:
            ok, result = check_receipt.check(sent, received, keys, header=header)
            if ok and listed:
                good, why = check_receipt.check_outputs(received, [clip])
                if not good:
                    ok, result = False, why
            receipts.append((ok, result))
        v["receipts"] = receipts
    return v, signatures


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    print("  clips within %.1f dB of their signature: %d of %d (needs %d)" % (
        limits["clip_distance"], v["close"], v["clips"], int(limits["close"] * v["clips"] + 0.999)))
    print("  typical distance: %.2f dB (limit %.1f)" % (v["typical_distance"], limits["typical_distance"]))
    signed = check_receipt.report_receipts(model, v.get("receipts"))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, fingerprint=None, out="", keys=None):
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        if fingerprint.get("endpoint") not in ENDPOINTS:
            raise ValueError("an endpoint this check does not ask")
        v, signatures = run_check(model, fingerprint, key, keys=keys)
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"],
                  **{name: v[name] for name in v if name != "receipts"},
                  "signatures": [encode_signature(s) for s in signatures]}
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=1, default=str)
    return 0 if v["result"] == "matches" and signed is not False else 1


# ---------------------------------------------------------------------------------------------------------- make
class SeededTorch:
    """torch as a pipeline's own modules see it during one generation: `randn` and `randn_like` draw float32 normal
    numbers from one CPU generator, in the shape asked, then move and cast them to the device and precision asked;
    every other name is torch itself, and a call that names its own generator keeps it."""

    def __init__(self, torch, generator):
        self._torch, self._generator, self.draws = torch, generator, []

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def _draw(self, shape, device, dtype):
        noise = self._torch.randn(tuple(int(n) for n in shape), generator=self._generator, dtype=self._torch.float32)
        noise = noise.to(device=device, dtype=dtype)
        self.draws.append((tuple(noise.shape), str(noise.device), str(noise.dtype).replace("torch.", "")))
        return noise

    def randn(self, *size, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad=False,
              pin_memory=False, names=None):
        torch = self._torch
        if generator is not None or out is not None or names is not None or (layout is not None and layout != torch.strided):
            options = {"generator": generator, "out": out, "dtype": dtype, "device": device,
                       "requires_grad": requires_grad, "pin_memory": pin_memory}
            if layout is not None:
                options["layout"] = layout
            if names is not None:
                options["names"] = names
            return torch.randn(*size, **options)
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        noise = self._draw(shape, torch.get_default_device() if device is None else device,
                           torch.get_default_dtype() if dtype is None else dtype)
        return noise.requires_grad_(True) if requires_grad else noise

    def randn_like(self, input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
        if layout is not None and layout != self._torch.strided:
            return self._torch.randn_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        noise = self._draw(input.shape, input.device if device is None else device,
                           input.dtype if dtype is None else dtype)
        return noise.requires_grad_(True) if requires_grad else noise


class cpu_noise:
    """Inside the block, every randn and randn_like of the named modules draws float32 numbers from
    torch.Generator("cpu").manual_seed(seed), in the order the pipeline draws them."""

    def __init__(self, seed, modules):
        import torch

        self.modules = [sys.modules[name] for name in modules]
        self.seeded = SeededTorch(torch, torch.Generator(device="cpu").manual_seed(int(seed)))

    def __enter__(self):
        self.saved = [(module, module.torch) for module in self.modules]
        for module in self.modules:
            module.torch = self.seeded
        return self.seeded

    def __exit__(self, *exc):
        for module, original in self.saved:
            module.torch = original
        return False


def seed_latents(seed, frames, channels):
    """(1, frames, channels) float32 normal numbers from torch's CPU generator seeded with `seed`."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((1, int(frames), int(channels)), generator=generator, dtype=torch.float32)


class AceStep:
    """diffusers' AceStepPipeline (the ACE-Step checkpoints in diffusers format). A text-to-music request draws its
    noise once, the starting latents, which are passed in: float32 numbers from the seed's CPU generator in the
    shape the pipeline draws, ceil(duration * latents per second) frames of the transformer's acoustic width. The
    Euler flow-matching steps and the decoder draw nothing. The output is peak-normalized by the pipeline itself."""

    endpoint = "/v1/audio/generations"

    def __init__(self, weights, dtype, device):
        import torch
        from diffusers import AceStepPipeline

        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.pipe = AceStepPipeline.from_pretrained(weights, torch_dtype=torch_dtype).to(device)
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=True)

    def clip(self, request, probe):
        import torch

        options = {**request, **probe}
        duration = float(options.get("audio_duration", 30.0))
        frames = math.ceil(duration * self.pipe.latents_per_second)
        latents = seed_latents(options["seed"], frames, self.pipe.transformer.config.audio_acoustic_hidden_dim)
        kwargs = {"prompt": options["prompt"], "lyrics": options.get("lyrics", ""), "audio_duration": duration,
                  "num_inference_steps": int(options.get("num_inference_steps", 8)),
                  "guidance_scale": float(options.get("guidance_scale", 1.0)), "latents": latents, "generator": None}
        if options.get("shift") is not None:
            kwargs["shift"] = float(options["shift"])
        with torch.inference_mode():
            audio = self.pipe(**kwargs).audios[0]
        return self.pipe.sample_rate, [channel for channel in audio.float().cpu().numpy()]


class StableAudio3:
    """The stable-audio-3 library (StableAudioModel.generate, as published), built from a local copy of the
    checkpoint, with the library's own requirements installed (Medium runs on a CUDA card). Every random draw of a
    generation (the starting noise, one per ping-pong sampler step, and the autoencoder's per decoded chunk) comes
    from the seed's CPU generator as float32 numbers (cpu_noise over the library's modules that draw)."""

    endpoint = "/v1/audio/generations"
    NOISE_MODULES = ("stable_audio_3.model", "stable_audio_3.inference.sampling", "stable_audio_3.models.bottleneck",
                     "stable_audio_3.models.autoencoders")

    def __init__(self, weights, dtype, device):
        import torch
        import stable_audio_3.inference.sampling  # noqa: F401
        import stable_audio_3.models.autoencoders  # noqa: F401
        import stable_audio_3.models.bottleneck  # noqa: F401
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        with open(os.path.join(weights, "model_config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        for conditioner in config["model"]["conditioning"]["configs"]:
            if conditioner.get("type") == "t5gemma":
                conditioner["config"]["model_path"] = weights  # its text encoder from the same local copy
        half = dtype == "float16"
        model = load_diffusion_cond(config, os.path.join(weights, "model.safetensors"), device=device, model_half=half)
        self.model = StableAudioModel(model, config, device, half)
        self.sample_size = config["sample_size"]
        self.torch = torch

    def clip(self, request, probe):
        options = {**request, **probe}
        with cpu_noise(options["seed"], self.NOISE_MODULES):
            audio = self.model.generate(prompt=options["prompt"], duration=float(options.get("audio_duration", 30.0)),
                                        steps=int(options.get("num_inference_steps", 8)), seed=int(options["seed"]),
                                        batch_size=1, sample_size=self.sample_size, disable_tqdm=True)
        return int(self.model.model.sample_rate), [channel for channel in audio[0].float().cpu().numpy()]


PIPELINES = {"ace-step": AceStep, "stable-audio-3": StableAudio3}


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


def make(args):
    with open(args.probes, encoding="utf-8") as handle:
        probes = json.load(handle)
    request = json.loads(args.request)
    pipeline = PIPELINES[args.pipeline](args.weights, args.dtype, args.device)
    fields = ENDPOINTS[pipeline.endpoint]
    rows, started = [], time.time()
    for probe in probes:
        unknown = set(probe) - set(fields)
        if unknown:
            raise SystemExit("a probe sets fields %s, which %s does not take" % (sorted(unknown), pipeline.endpoint))
        t = time.time()
        rate, channels = pipeline.clip(request, probe)
        data = encode_wav(rate, channels)  # the 16-bit WAV the API answers with, read back as the check reads it
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            with open(os.path.join(args.keep, "%s-%d.wav" % (args.model, probe["seed"])), "wb") as handle:
                handle.write(data)
        rate, samples = decode_wav(data)
        rows.append({**probe, "seconds": round(len(samples) / rate, 3),
                     "signature": encode_signature(signature(rate, samples))})
        print("%-50s seed %-10d %.0f s" % (str(probe.get("prompt") or probe.get("input"))[:50], probe["seed"],
                                           time.time() - t), flush=True)
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": _weight_files(args.weights),
                   "made_with": "the pipeline published with the weights, on %s" % args.device.split(":")[0],
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "endpoint": pipeline.endpoint,
                   "request": request,
                   "limits": {"clip_distance": args.clip_distance, "typical_distance": args.typical_distance,
                              "close": args.close},
                   "probes": rows}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)
    print("fingerprint of %d clips written in %.0f s" % (len(rows), time.time() - started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example ace-step-1-5-xl")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the signatures read from each answer to this JSON file")
    c.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    s = sub.add_parser("signature", help="print the signature of a WAV file, and its distance from another")
    s.add_argument("clip")
    s.add_argument("--against", default="", help="a second WAV file: print the distance between the two")
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--pipeline", required=True, choices=sorted(PIPELINES))
    m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint")
    m.add_argument("--name", required=True, help="the public repository these weights are")
    m.add_argument("--revision", required=True, help="the repository revision")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI")
    m.add_argument("--probes", required=True, help="a JSON list of probes: a prompt (or input) and a seed, and any "
                                                   "other field the endpoint takes, such as lyrics or audio_duration")
    m.add_argument("--request", default='{"format": "wav"}', help="the API settings every clip is asked with (JSON)")
    m.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    m.add_argument("--device", default="cpu", help="cpu or cuda")
    m.add_argument("--out", required=True)
    m.add_argument("--keep", default="", help="also keep each clip in this folder")
    m.add_argument("--clip-distance", type=float, required=True)
    m.add_argument("--typical-distance", type=float, required=True)
    m.add_argument("--close", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "make":
        return make(args)
    if args.command == "signature":
        with open(args.clip, "rb") as handle:
            mine = clip_signature(handle.read())
        print(encode_signature(mine))
        if args.against:
            with open(args.against, "rb") as handle:
                print("distance %.2f dB" % distance(mine, clip_signature(handle.read())))
        return 0
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out,
                     keys=None if args.keys == "none" else check_receipt.load_keys(args.keys))


if __name__ == "__main__":
    sys.exit(main() or 0)
