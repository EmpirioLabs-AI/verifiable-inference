<div align="center">

<a href="https://empiriolabs.ai">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/EmpirioLabs-AI/assets/main/logo-white.png" />
    <img src="https://raw.githubusercontent.com/EmpirioLabs-AI/assets/main/logo-dark.png" alt="EmpirioLabs AI" width="280" />
  </picture>
</a>

# Verifiable inference on EmpirioLabs AI

Check which model answered and that EmpirioLabs served it, for the models EmpirioLabs runs on GPU infrastructure it manages.

<p>
  <a href="https://docs.empiriolabs.ai/verifiable-inference"><img alt="Documentation" src="https://img.shields.io/badge/Docs-Verifiable%20inference-0a66c2?style=flat-square"></a>
  <a href="https://platform.empiriolabs.ai/dashboard/api-keys"><img alt="Get an API key" src="https://img.shields.io/badge/API%20key-Get%20one-0b8a5a?style=flat-square"></a>
  <a href="LICENSE"><img alt="License: MIT-0" src="https://img.shields.io/badge/License-MIT--0-555555?style=flat-square"></a>
</p>

</div>

Two checks, each a single Python file that needs nothing beyond Python itself:

- **Check a response.** Every successful answer from a Native Inference model comes with a receipt signed by EmpirioLabs: chat, completions, Responses, Messages and Gemini requests, transcription, decisions, OCR, and image, video, audio, speech and 3D generation. `check_receipt.py` confirms the signature and that the receipt covers your exact request and response, and any generated file you downloaded.
- **Check a model.** `check_model.py` asks the live API a fixed list of questions and compares the answers with a fingerprint made once from the model's public weights. It takes under two minutes and you don't run any model yourself.

## Get started

1. Create an API key at https://platform.empiriolabs.ai/dashboard/api-keys and set it: `export EMPIRIOLABS_API_KEY="sk-empiriolabs-..."`.
2. Check a model: `python check_model.py --model qwen3-5-9b`, or every model with a fingerprint: `python check_model.py --all`.
3. Check a response: `python check_receipt.py live --model qwen3-5-9b` sends a request streamed and not streamed and checks both receipts (for an image model it makes one image and checks its file too). To check a response you kept, `python check_receipt.py files --request request.json --response response.bin`, adding `--header` for an answer that carries its receipt in the `X-Receipt` header and `--output` for files you downloaded.

Python 3.9 or later. A model check sends 50 requests of one output token each (200 for a model whose API returns no list of alternatives), billed to your account at the model's listed price (https://empiriolabs.ai/pricing). A transcription or decision check sends each clip or request in its file once, billed the same way.

`check_model.py` prints one verdict per model:

- `matches`: the answers agree with the public weights within the limits the fingerprint states, every prompt reached the model as it was sent, and every answer carries a valid receipt naming the model.
- `DOES NOT MATCH`: anything else. The lines below the verdict say which part failed.
- `CANNOT TELL`: the fingerprint's questions are too certain to tell models apart without a limit on the other alternatives, and the file sets none.

## Fingerprints

<!-- fingerprints:start -->
| Model ID | Kind | Public weights | Revision | Made |
| --- | --- | --- | --- | --- |
| [`aplomb-1`](fingerprints/aplomb-1.json) | Decision | [empiriolabsai/aplomb-1](https://huggingface.co/empiriolabsai/aplomb-1/tree/1225d382f51a29d04c888cdadc8b94f21261c7b2) | `1225d382f51a` | 2026-11-24 |
| [`gemma-4-26b-a4b`](fingerprints/gemma-4-26b-a4b.json) | Chat | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52) | `4d7ae4984b7d` | 2026-11-23 |
| [`muse-glimmer-30b`](fingerprints/muse-glimmer-30b.json) | Chat | [RadixArk/Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4/tree/3f7428bda949d17f478e89b920374d622a4b9bb0) | `3f7428bda949` | 2026-11-24 |
| [`qwen3-5-4b`](fingerprints/qwen3-5-4b.json) | Chat | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806ef` | 2026-11-23 |
| [`qwen3-5-9b`](fingerprints/qwen3-5-9b.json) | Chat | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) | `c20223623576` | 2026-11-23 |
| [`qwen3-6-35b-a3b`](fingerprints/qwen3-6-35b-a3b.json) | Chat | [EschaLabs/Qwen3.6-35B-A3B-Escha-W2](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2/tree/6072a54913552616b64599c30f5e8a3c8fecad9c) | `6072a5491355` | 2026-11-24 |
| [`qwen3-8-27b`](fingerprints/qwen3-8-27b.json) | Chat | [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/tree/6128240ebaf4eaa7bad2b3d1c72c37d677c5f462) | `6128240ebaf4` | 2026-11-23 |
| [`whisper-large-v3-turbo`](fingerprints/whisper-large-v3-turbo.json) | Transcription | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | `41f01f3fe87f` | 2026-11-24 |
<!-- fingerprints:end -->

Each file names the public repository, the exact revision and the SHA-256 of every weight file it was made from, and holds nothing about how EmpirioLabs serves the model. A chat file holds the questions with the 20 most likely first answer tokens and their log-probabilities. `make_fingerprint.py` makes a chat fingerprint again from the same public weights, and `make_fingerprint.py compare` reports how close two fingerprints are.

**Transcription.** A transcription file holds short public-domain clips, kept in the folder of the same name beside it with each clip's source and license, and for each clip the ten languages the public weights find most likely with their log-probabilities, the transcript's tokens and their average log-probability. The check sends each clip to `POST /v1/audio/transcriptions` with the settings the file states and says `matches` when the language and every transcript token are the same on the share of clips the file states, and the typical gaps between the language log-probabilities and between the average log-probabilities are within its limits. `check_transcription.py make` makes the file again from the same public weights.

**Decision.** A decision file holds fixed decision requests and the probabilities the public weights, run with the reference code published with them, give for every question. The check sends each request to `POST /v1/decisions` and says `matches` when the same answer comes back on the share of questions the file states and the typical largest difference between the probabilities is within its limit. `check_decision.py make` makes the file again from the same public weights.

## Receipts

A receipt is a JSON object signed with Ed25519:

```json
{"version": 1, "issuer": "api.empiriolabs.ai", "key_id": "...", "served_by": "empiriolabs-native",
 "model": "qwen3-5-9b", "issued_at": 1795651200, "request_sha256": "...", "response_sha256": "...", "signature": "..."}
```

- **JSON:** the last field of the response body, `...,"receipt":{...}}`. It covers the body without that field.
- **Streams:** a final SSE comment after the stream's last event, `: receipt {...}`. It covers every byte before it.
- **Raw bytes:** audio returned as a file, and transcripts returned as text or subtitles, carry it in the `X-Receipt` response header. It covers the whole body.
- **Jobs:** a request that runs as a job (images, video, audio, 3D, OCR, transcription) gets its receipt on the answer that returns the result, the response to the original request or `GET /v1/jobs/{id}`. It covers that original request, so check it against the body you first sent. Each generated file lists its SHA-256 in `sha256`, which the receipt covers too.
- **Signature:** over the receipt without `signature`, written as JSON with sorted keys and no spaces, encoded as base64url without padding.
- **Keys:** [`keys/receipt-keys.json`](keys/receipt-keys.json) lists every key that has signed receipts, and https://api.empiriolabs.ai/v1/receipts/keys publishes the same list. A key is never removed from this file; a key that no longer signs is marked `retired`.

To check a receipt you need the request body exactly as you sent it and the response exactly as you received it.

## What the checks cover

- Receipts cover every Native Inference model. Third-party models don't carry one.
- Text-to-video requests to `moss-video-and-audio` and `svi-2-0-pro` don't carry a receipt; send a starting image to get one. Streams from the Gemini endpoint `streamGenerateContent` don't either; use `generateContent` when you need one.
- Fingerprints cover the models in the table above. Every model check also checks the receipt on every answer.
- A model check confirms the model at the time it runs. The questions are fixed and public, so the receipt on every response is what ties each answer to a model and to EmpirioLabs.
- A fingerprint confirms the model, not the numeric precision it runs at.
- A receipt shows what EmpirioLabs returned for a request. It doesn't encrypt anything, and it doesn't show whether an answer is correct.

## Report a problem

Email support@empiriolabs.ai with "Security" in the subject line. A check that says `DOES NOT MATCH` or `NOT VERIFIED` is worth reporting: include the model ID, the time and the script's output.
