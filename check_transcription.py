"""Check that a speech-to-text model on EmpirioLabs AI is the public model it is said to be, without running any model.

A fingerprint of kind "transcription" (fingerprints/<model>.json) was made once, from the model's public weights: for
a fixed set of short public-domain clips kept beside it (fingerprints/<model>/), the model's own numbers for each clip
with the most deterministic settings the API takes (one temperature of 0, a beam of one, no voice filter, no
timestamps): the log-probabilities of the ten languages it finds most likely, the transcript's token ids and their
average log-probability (and, recorded but not compared, the no-speech probability). The check sends the same clips to
the live API with the same settings and compares the numbers in the answers with the fingerprint.

    python check_model.py --model whisper-large-v3-turbo               the usual way, as for every other model
    python check_transcription.py check --model whisper-large-v3-turbo  the same check, --out keeps the numbers read
    python check_transcription.py make --weights <folder> --name openai/whisper-large-v3-turbo --revision <commit> \
        --model whisper-large-v3-turbo --clips fingerprints/whisper-large-v3-turbo --sources <sources.json> \
        --out fingerprints/whisper-large-v3-turbo.json --same-transcript <share> --language-gap <limit> \
        --logprob-gap <limit>

The check uses only the Python standard library and reads your API key from EMPIRIOLABS_API_KEY. It sends one
transcription per clip, billed to your account at the model's listed price. The clips are read from the folder beside
the fingerprint and must have the SHA-256 the fingerprint records; a clip that does not is never sent. `make` runs the
public weights with plain transformers code and needs numpy, soundfile, torch and transformers.

Verdict
    matches          the same language and transcript tokens on at least the share of clips the fingerprint states,
                     and the typical gaps between the language log-probabilities and between the transcripts' average
                     log-probabilities within its limits
    DOES NOT MATCH   anything else
Transcription answers carry no receipt. Exit status: 0 matches, 1 anything else.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "transcription"
# The API's origin: a transcription is sent to /v1/audio/transcriptions, and a job's poll_url is a path on it.
ORIGIN = "%s://%s" % urllib.parse.urlsplit(check_receipt.API)[:2]
# Every setting is a published parameter of the model; together they leave nothing to chance: one temperature of 0
# and a beam of one, no voice filter, no text carried between windows, no timestamps, no clip ever set aside as
# silence (the threshold is a probability, so 1 is never exceeded), and the token ids in the answer.
REQUEST = {"temperature": "0", "beam_size": 1, "vad_filter": False, "without_timestamps": True,
           "condition_on_previous_text": False, "no_speech_threshold": 1, "include_tokens": True,
           "response_format": "verbose_json"}
TOP_LANGUAGES = 10
# A probability under 1e-8, a 0 included, counts as 1e-8, so a language one side gives no weight stays a finite gap.
LOG_FLOOR = math.log(1e-8)
SAMPLE_RATE = 16000


# ----------------------------------------------------------------------------------------------------------- make
def _clip_samples(path):
    import numpy as np
    import soundfile

    samples, rate = soundfile.read(path, dtype="int16")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise SystemExit("%s: a clip is 16 kHz mono" % path)
    return samples.astype(np.float32) / 32768.0


def _features(extractor, audio):
    """Log-mel features of the clip and a 160-sample tail, then zeros to 30 seconds: Whisper's own input for a clip
    shorter than one window. Returns the window the language is read from (every frame) and the window the
    transcript is decoded from (the clip's frames)."""
    import numpy as np
    import torch

    feats = extractor(np.pad(audio, (0, 160)), sampling_rate=SAMPLE_RATE, padding="longest", return_tensors="np").input_features[0]
    frames = feats.shape[-1] - 1

    def window(part):
        out = np.zeros((feats.shape[0], 3000), dtype=np.float32)
        out[:, : part.shape[-1]] = part[:, :3000]
        return torch.from_numpy(out)[None]

    return window(feats), window(feats[:, :frames])


def reference(weights, clip_paths, dtype="float32", log=print, top_languages=TOP_LANGUAGES):
    """The numbers the fingerprint holds, for each clip, from plain transformers code (every language with
    top_languages=None, as the API answers)."""
    import torch
    from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = WhisperForConditionalGeneration.from_pretrained(weights, dtype=torch_dtype).eval()
    extractor = WhisperFeatureExtractor.from_pretrained(weights)
    tokenizer = WhisperTokenizer.from_pretrained(weights)
    generation = GenerationConfig.from_pretrained(weights)
    sot = generation.decoder_start_token_id
    eot = generation.eos_token_id
    no_speech = tokenizer.convert_tokens_to_ids("<|nospeech|>")
    transcribe = generation.task_to_id["transcribe"]
    no_timestamps = generation.no_timestamps_token_id
    languages = sorted(generation.lang_to_id.items(), key=lambda item: item[1])
    language_ids = torch.tensor([token_id for _, token_id in languages])
    suppress = torch.tensor(generation.suppress_tokens)
    suppress_first = torch.tensor(generation.begin_suppress_tokens)
    encoder = model.get_encoder()
    rows = []
    for path in clip_paths:
        started = time.time()
        audio = _clip_samples(path)
        language_window, transcript_window = _features(extractor, audio)
        with torch.inference_mode():
            # The language: one step after the start token, the distribution over the language tokens only.
            encoded = encoder(language_window.to(torch_dtype)).last_hidden_state
            logits = model(encoder_outputs=(encoded,), decoder_input_ids=torch.tensor([[sot]])).logits[0, -1].float()
            language_logprobs = torch.log_softmax(logits[language_ids], dim=-1)
            order = torch.argsort(language_logprobs, descending=True)
            ranked = [(languages[int(i)][0][2:-2], float(language_logprobs[int(i)])) for i in order]
            language_token = languages[int(order[0])][1]
            # The transcript: greedy decoding after start, language, transcribe and no-timestamps, with the
            # checkpoint's own suppressed tokens; the average log-probability counts the end token too.
            encoded = encoder(transcript_window.to(torch_dtype)).last_hidden_state
            prompt = [sot, language_token, transcribe, no_timestamps]
            out = model(encoder_outputs=(encoded,), decoder_input_ids=torch.tensor([prompt]), use_cache=True)
            no_speech_prob = float(torch.softmax(out.logits[0, 0].float(), dim=-1)[no_speech])
            cache, logits, tokens, total = out.past_key_values, out.logits[0, -1].float(), [], 0.0
            while True:
                logits = logits.clone()
                logits[suppress] = -math.inf
                if not tokens:
                    logits[suppress_first] = -math.inf
                logprobs = torch.log_softmax(logits, dim=-1)
                token = int(torch.argmax(logprobs))
                total += float(logprobs[token])
                if token == eot or len(prompt) + len(tokens) + 1 >= generation.max_length:
                    break
                tokens.append(token)
                out = model(encoder_outputs=(encoded,), decoder_input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache, logits = out.past_key_values, out.logits[0, -1].float()
        rows.append({"language": ranked[0][0], "languages": [[code, round(lp, 4)] for code, lp in ranked[:top_languages]],
                     "tokens": tokens, "text": tokenizer.decode(tokens), "avg_logprob": round(total / (len(tokens) + 1), 6),
                     "no_speech_prob": round(no_speech_prob, 6)})
        log("%-40s %-3s %6.3f  %3d tokens  %.0f s  %s" % (os.path.basename(path), ranked[0][0], rows[-1]["avg_logprob"],
                                                        len(tokens), time.time() - started, rows[-1]["text"][:70]))
    return rows


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(args):
    with open(args.sources, encoding="utf-8") as handle:
        return json.load(handle)


def make(args):
    manifest = _manifest(args)
    folder = os.path.abspath(args.clips)
    started = time.time()
    write(args, reference(args.weights, [os.path.join(folder, clip["file"]) for clip in manifest], args.dtype))
    print("fingerprint of %d clips written in %.0f s" % (len(manifest), time.time() - started))


def write(args, rows):
    manifest = _manifest(args)
    folder = os.path.abspath(args.clips)
    paths = [os.path.join(folder, clip["file"]) for clip in manifest]
    clips = []
    for clip, path, row in zip(manifest, paths, rows):
        clips.append({"file": "%s/%s" % (os.path.basename(folder), clip["file"]), "sha256": sha256_of(path),
                      "seconds": clip["seconds"], "source": clip["source"], "license": clip["license"],
                      "cut": clip["cut"], "processing": clip["processing"], **row})
    weight_files = {name: sha256_of(os.path.join(args.weights, name))
                    for name in sorted(os.listdir(args.weights)) if name.endswith(".safetensors")}
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": weight_files, "made_with": "plain transformers code on cpu",
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request": REQUEST,
                   "limits": {"same_transcript": args.same_transcript, "language_gap": args.language_gap,
                              "logprob_gap": args.logprob_gap},
                   "clips": clips}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------------------------------------- check
class Unanswered(Exception):
    """The API gave no transcript for a clip: the check cannot be done, which says nothing about the model."""


def _call(method, path, key, body=None):
    """One request to the API at `path` (a path that starts with /v1); (the HTTP status, the JSON answer)."""
    request = urllib.request.Request(ORIGIN + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                              "User-Agent": check_receipt.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise Unanswered("HTTP %d: %s" % (error.code, error.read()[:300].decode("utf-8", "replace"))) from None
    except (OSError, ValueError) as error:
        raise Unanswered("%s: %s" % (type(error).__name__, error)) from None


def ask(model, clip_bytes, suffix, key, settings=None):
    """One transcription with the fingerprint's settings, the audio sent inline as a customer may send it. A
    transcription that is still running answers 202 with a job, which is polled to its end."""
    body = {"model": model, "audio_base64": base64.b64encode(clip_bytes).decode(), "audio_suffix": suffix,
            **(REQUEST if settings is None else settings)}
    status, data = _call("POST", "/v1/audio/transcriptions", key, body)
    deadline = time.time() + 900
    while status == 202 or data.get("status") in ("queued", "processing", "running"):
        if time.time() > deadline:
            raise Unanswered("the transcription job did not finish in 15 minutes")
        time.sleep(3)
        status, job = _call("GET", data.get("poll_url") or "/v1/jobs/%s" % data["job_id"], key)
        if job.get("status") in ("completed", "succeeded"):
            data = job.get("result") or {}
            break
        if job.get("status") == "failed":
            raise Unanswered("the transcription job failed: %s" % json.dumps(job.get("error"))[:300])
    return data


def read_answer(data):
    """The answer's numbers in the fingerprint's terms. A field that is missing comes back as None.

    Reads only `language`, every language's probability (`all_language_probs`, at the top level or in `info`), and the
    segments' `tokens` and `avg_logprob`. Nothing else in the answer counts."""
    segments = [s for s in (data.get("segments") or []) if isinstance(s, dict)]
    probs = data.get("all_language_probs") or (data.get("info") or {}).get("all_language_probs")
    languages = None
    if isinstance(probs, list):
        languages = {str(pair[0]): (math.log(pair[1]) if pair[1] > 0 else -math.inf) for pair in probs
                     if isinstance(pair, (list, tuple)) and len(pair) == 2 and isinstance(pair[1], (int, float))}
    tokens = None
    if segments and all(isinstance(s.get("tokens"), list) for s in segments):
        tokens = [int(t) for s in segments for t in s["tokens"]]
    logprob = segments[0].get("avg_logprob") if len(segments) == 1 else None
    return {"language": data.get("language"), "languages": languages, "tokens": tokens,
            "avg_logprob": logprob if isinstance(logprob, (int, float)) else None}


def compare(row, live):
    """How far one live answer is from the fingerprint's row: the same transcript, the typical gap between the
    log-probabilities of the most likely languages (weighted by how likely the reference finds each), and the gap
    between the transcripts' average log-probabilities."""
    same = live["tokens"] == row["tokens"] and live["language"] == row["language"]
    language_gap = math.inf
    if live["languages"]:
        reference = dict((code, lp) for code, lp in row["languages"])
        weights = [math.exp(lp) for lp in reference.values()]
        floor = lambda lp: max(lp, LOG_FLOOR)
        gaps = [abs(floor(lp) - floor(live["languages"].get(code, -math.inf))) for code, lp in reference.items()]
        language_gap = sum(w * g for w, g in zip(weights, gaps)) / sum(weights)
    logprob_gap = abs(live["avg_logprob"] - row["avg_logprob"]) if live["avg_logprob"] is not None else math.inf
    return same, language_gap, logprob_gap


def verdict(fingerprint, answers):
    """answers: one read_answer() per clip, in the fingerprint's order."""
    limits = fingerprint["limits"]
    results = [compare(row, live) for row, live in zip(fingerprint["clips"], answers)]
    n = len(results)
    same = sum(1 for r in results if r[0])
    language_gap = statistics.median(r[1] for r in results)
    logprob_gap = statistics.median(r[2] for r in results)
    ok = same / n >= limits["same_transcript"] and language_gap <= limits["language_gap"] and logprob_gap <= limits["logprob_gap"]
    return {"result": "matches" if ok else "DOES NOT MATCH", "clips": n, "same_transcript": same,
            "language_gap": language_gap, "logprob_gap": logprob_gap, "per_clip": results}


def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not a %s fingerprint" % (os.path.basename(path), KIND))
    return fingerprint


def run_check(model, fingerprint, folder, key):
    """Every clip, each checked against the SHA-256 the fingerprint records before any is sent, then sent once;
    (the verdict, the numbers read from each answer). Raises ValueError for a clip that is not the one the fingerprint
    was made from and Unanswered when the API gives no transcript."""
    clips = []
    for row in fingerprint["clips"]:
        path = os.path.join(folder, row["file"])
        with open(path, "rb") as handle:
            clip = handle.read()
        if hashlib.sha256(clip).hexdigest() != row["sha256"]:
            raise ValueError("%s is not the clip the fingerprint was made from" % row["file"])
        clips.append((clip, os.path.splitext(path)[1]))
    answers = []
    for clip, suffix in clips:
        try:
            answers.append(read_answer(ask(model, clip, suffix, key, fingerprint["request"])))
        except Unanswered as error:
            raise Unanswered("the API did not answer clip %d: %s" % (len(answers) + 1, error)) from None
    return verdict(fingerprint, answers), answers


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    print("  same language and transcript: %d of %d clips (needs %d)" % (
        v["same_transcript"], v["clips"], int(limits["same_transcript"] * v["clips"] + 0.999)))
    print("  typical gap between the language log-probabilities: %.4f (limit %.3f)" % (v["language_gap"], limits["language_gap"]))
    print("  typical gap between the transcripts' average log-probabilities: %.4f (limit %.3f)" % (
        v["logprob_gap"], limits["logprob_gap"]))
    print("  receipts: not on this endpoint")
    print("  %.0f s, no model was run" % seconds)


def check_one(model, path, key, fingerprint=None, out=""):
    """Check `model` against the fingerprint at `path` (already read when `fingerprint` is given). 0 matches, 1
    anything else. `out` names a JSON file for the numbers read from each answer."""
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        v, answers = run_check(model, fingerprint, os.path.dirname(os.path.abspath(path)), key)
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"]}
        record.update((name, v[name]) for name in v if name != "per_clip")
        record["per_clip"] = [{"file": row["file"], "same": r[0], "language_gap": r[1], "logprob_gap": r[2]}
                              for row, r in zip(fingerprint["clips"], v["per_clip"])]
        record["answers"] = answers
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=1, default=str)
    return 0 if v["result"] == "matches" else 1


def main(argv=None):
    # A transcript the console cannot show is printed with a stand-in, never a crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example whisper-large-v3-turbo")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the numbers read from each answer to this JSON file")
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint")
    m.add_argument("--name", required=True, help="the public repository these weights are, written into the fingerprint")
    m.add_argument("--revision", required=True, help="the repository revision, written into the fingerprint")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, written into the fingerprint")
    m.add_argument("--clips", required=True, help="the folder beside the fingerprint that holds the clips")
    m.add_argument("--sources", required=True, help="each clip's file, seconds, source, license, cut and processing (JSON)")
    m.add_argument("--out", required=True)
    m.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    m.add_argument("--same-transcript", type=float, required=True)
    m.add_argument("--language-gap", type=float, required=True)
    m.add_argument("--logprob-gap", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "make":
        return make(args)
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out)


if __name__ == "__main__":
    sys.exit(main() or 0)
