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

Two checks, each a single Python file that uses only the standard library:

- **Check a model.** `check_model.py` asks the live API a fixed list of questions for one token each and compares the token probabilities in the answers with a fingerprint made once from the model's public weights. It takes about half a minute and runs no model.
- **Check a response.** Every successful chat completion from a Native Inference model carries a receipt signed by EmpirioLabs. `check_receipt.py` confirms the signature and that the receipt names your exact request and response.

## Get started

1. Create an API key at https://platform.empiriolabs.ai/dashboard/api-keys and set it: `export EMPIRIOLABS_API_KEY="sk-empiriolabs-..."`.
2. Check a model: `python check_model.py --model qwen3-5-9b`, or every model with a fingerprint: `python check_model.py --all`.
3. Check a response: `python check_receipt.py live --model qwen3-5-9b` sends one request, not streamed and streamed, and checks both receipts. To check a response you kept, `python check_receipt.py files --request request.json --response response.bin`.

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

- **Not streamed:** the last member of the response body, `...,"receipt":{...}}`. It covers the body with that member removed.
- **Streamed:** one SSE comment after `data: [DONE]`, `: receipt {...}`. It covers every byte before that comment.
- **Signature:** over the receipt without `signature`, written as JSON with sorted keys and no spaces, encoded as base64url without padding.
- **Keys:** [`keys/receipt-keys.json`](keys/receipt-keys.json) lists every key that has signed receipts, and https://api.empiriolabs.ai/v1/receipts/keys publishes the same list. A key is never removed from this file; a key that no longer signs is marked `retired`.

To check a receipt you need the request body exactly as it was sent and the response exactly as it was received.

## What the checks cover

- Native Inference chat models on `POST /v1/chat/completions`. The fingerprints cover the models in the table above.
- The transcription and decision models in the table above, on `POST /v1/audio/transcriptions` and `POST /v1/decisions`, with the model check only. Their answers carry no receipt, and the model check prints `receipts: not on this endpoint` for them.
- A model check confirms the model when the check runs. The questions are fixed and public, so the receipt on every response is what ties each answer to a model and to EmpirioLabs.
- A fingerprint confirms the model, not the numeric precision it runs at.
- A receipt shows what EmpirioLabs returned for a request. It does not encrypt anything and does not show whether an answer is correct.

## Report a problem

Email support@empiriolabs.ai with "Security" in the subject line. A check that says `DOES NOT MATCH` or `NOT VERIFIED` is worth reporting: include the model ID, the time and the script's output.
