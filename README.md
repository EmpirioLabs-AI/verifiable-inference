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
- **Check a model.** `check_model.py` asks the live API a fixed list of questions and compares the answers with the model's fingerprint, made once from its public weights (for three speech and music models, recorded from the model we serve). It takes a few minutes at most and you don't run any model yourself.

## Get started

1. Create an API key at https://platform.empiriolabs.ai/dashboard/api-keys and set it: `export EMPIRIOLABS_API_KEY="sk-empiriolabs-..."`.
2. Check a model: `python check_model.py --model qwen3-5-9b`, or every model with a fingerprint: `python check_model.py --all`.
3. Check a response: `python check_receipt.py live --model qwen3-5-9b` sends a request streamed and not streamed and checks both receipts (for an image model it makes one image and checks its file too). To check a response you kept, `python check_receipt.py files --request request.json --response response.bin`, adding `--header` for an answer that carries its receipt in the `X-Receipt` header and `--output` for files you downloaded.

Python 3.9 or later. A model check sends 50 requests of one output token each (200 for a model whose API returns no list of alternatives), billed to your account at the model's listed price (https://empiriolabs.ai/pricing). A transcription or decision check sends each clip or request in its file once, an image check asks for each picture in its file once, an OCR check sends each page in its file once, an audio check asks for each clip in its file once, a video check for each clip and a 3D check for each mesh, billed the same way. A video check also needs ffmpeg (https://ffmpeg.org) to read the frames of each clip.

`check_model.py` prints one verdict per model:

- `matches`: the answers agree with the fingerprint within the limits it states, every prompt reached the model as it was sent, and every answer carries a valid receipt naming the model.
- `DOES NOT MATCH`: anything else. The lines below the verdict say which part failed.
- `CANNOT TELL`: the fingerprint's questions are too certain to tell models apart without a limit on the other alternatives, and the file sets none.

## Fingerprints

<!-- fingerprints:start -->
| Model ID | Kind | Public weights | Revision | Made |
| --- | --- | --- | --- | --- |
| [`ace-step-1-5-xl`](fingerprints/ace-step-1-5-xl.json) | Audio | [ACE-Step/acestep-v15-xl-turbo-diffusers](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers/tree/200ba991ae448051e14b0183157e35c2d27c9fb0) | `200ba991ae44` | 2026-11-25 |
| [`aplomb-1`](fingerprints/aplomb-1.json) | Decision | [empiriolabsai/aplomb-1](https://huggingface.co/empiriolabsai/aplomb-1/tree/1225d382f51a29d04c888cdadc8b94f21261c7b2) | `1225d382f51a` | 2026-11-24 |
| [`flux-2-klein-4b`](fingerprints/flux-2-klein-4b.json) | Image | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) | `e7b7dc27f91d` | 2026-11-25 |
| [`gemma-4-26b-a4b`](fingerprints/gemma-4-26b-a4b.json) | Chat | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52) | `4d7ae4984b7d` | 2026-11-23 |
| [`glm-tts`](fingerprints/glm-tts.json) | Audio | [zai-org/GLM-TTS](https://huggingface.co/zai-org/GLM-TTS/tree/b3feb691fa4cf590eec4418f6275f95220e58844) | `b3feb691fa4c` | 2026-11-25 |
| [`hunyuan-video-1-5`](fingerprints/hunyuan-video-1-5.json) | Video | [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038) | `9b49404b3f5d` | 2026-11-25 |
| [`lightonocr-3-4b`](fingerprints/lightonocr-3-4b.json) | OCR | [lightonai/LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B/tree/b71010f095dc3735de7ee6b9969bff8f8f50b39a) | `b71010f095dc` | 2026-11-25 |
| [`minimax-music-3`](fingerprints/minimax-music-3.json) | Audio | [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3/tree/9c47cc42eb579455170bd868a600ac7096d7e5e0) | `9c47cc42eb57` | 2026-11-25 |
| [`moss-image-turbo`](fingerprints/moss-image-turbo.json) | Image | [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9) | `f972bd4b4b9e` | 2026-11-25 |
| [`moss-video-and-audio`](fingerprints/moss-video-and-audio.json) | Video | [OpenMOSS-Team/MOVA-360p](https://huggingface.co/OpenMOSS-Team/MOVA-360p/tree/eab4aa91d6d5eb515e259a0c8533c90062b117a5) | `eab4aa91d6d5` | 2026-11-25 |
| [`muse-glimmer-30b`](fingerprints/muse-glimmer-30b.json) | Chat | [RadixArk/Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4/tree/3f7428bda949d17f478e89b920374d622a4b9bb0) | `3f7428bda949` | 2026-11-24 |
| [`qwen3-5-4b`](fingerprints/qwen3-5-4b.json) | Chat | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806ef` | 2026-11-23 |
| [`qwen3-5-9b`](fingerprints/qwen3-5-9b.json) | Chat | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) | `c20223623576` | 2026-11-23 |
| [`qwen3-6-35b-a3b`](fingerprints/qwen3-6-35b-a3b.json) | Chat | [EschaLabs/Qwen3.6-35B-A3B-Escha-W2](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2/tree/6072a54913552616b64599c30f5e8a3c8fecad9c) | `6072a5491355` | 2026-11-24 |
| [`qwen3-8-27b`](fingerprints/qwen3-8-27b.json) | Chat | [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/tree/6128240ebaf4eaa7bad2b3d1c72c37d677c5f462) | `6128240ebaf4` | 2026-11-23 |
| [`soulx-podcast`](fingerprints/soulx-podcast.json) | Audio | [Soul-AILab/SoulX-Podcast-1.7B](https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B/tree/7d381e3aa4f2ac9e23201b57957fd0499f8f2fd1) | `7d381e3aa4f2` | 2026-11-25 |
| [`stable-audio-3-medium`](fingerprints/stable-audio-3-medium.json) | Audio | [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium/tree/d4d3579395a8ac0df4ab13b5b1d7138ce7290f81) | `d4d3579395a8` | 2026-11-25 |
| [`svi-2-0-pro`](fingerprints/svi-2-0-pro.json) | Video | [epfl-vita/svi-model](https://huggingface.co/epfl-vita/svi-model/tree/f9bdd50c57e4c28f0d088b6bad8afa3fdc3bc1ac) | `f9bdd50c57e4` | 2026-11-25 |
| [`trellis-2-4b`](fingerprints/trellis-2-4b.json) | 3D | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d) | `af44b45f2e35` | 2026-11-25 |
| [`whisper-large-v3-turbo`](fingerprints/whisper-large-v3-turbo.json) | Transcription | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | `41f01f3fe87f` | 2026-11-24 |
<!-- fingerprints:end -->

Each file names the public repository, the exact revision and the SHA-256 of the model's weight files, and holds nothing about how EmpirioLabs serves the model. A chat file holds the questions with the 20 most likely first answer tokens and their log-probabilities. `make_fingerprint.py` makes a chat fingerprint again from the same public weights, and `make_fingerprint.py compare` reports how close two fingerprints are.

**Transcription.** A transcription file holds short public-domain clips, kept in the folder of the same name beside it with each clip's source and license, and for each clip the ten languages the public weights find most likely with their log-probabilities, the transcript's tokens and their average log-probability. The check sends each clip to `POST /v1/audio/transcriptions` with the settings the file states and says `matches` when the language and every transcript token are the same on the share of clips the file states, and the typical gaps between the language log-probabilities and between the average log-probabilities are within its limits. `check_transcription.py make` makes the file again from the same public weights.

**Decision.** A decision file holds fixed decision requests and the probabilities the public weights, run with the reference code published with them, give for every question. The check sends each request to `POST /v1/decisions` and says `matches` when the same answer comes back on the share of questions the file states and the typical largest difference between the probabilities is within its limit. `check_decision.py make` makes the file again from the same public weights.

**Image.** An image file holds a list of prompts, each with its own seed, and a 32 × 32 thumbnail of the picture the public weights draw for it with the settings the file states. The seed fixes the starting noise, so the same request gives the same picture on any computer, up to small rounding differences between graphics cards. The check asks `POST /v1/images/generations` for the same pictures as PNG and says `matches` when the share of pictures the file states are within its distance of their thumbnails and the typical distance is within its limit. The distance is one minus the correlation of the two thumbnails' colors, so a picture that is only a little lighter or darker still counts as the same picture. A picture from another model, or from the same model with another seed, is laid out differently and is far from it. `check_image.py make` makes the file again from the same public weights and takes any prompts and seeds you choose, so you can make a fingerprint of your own that we have never seen and check the API with it. `check_image.py replay` draws a picture you received again from the public weights and tells you whether it's the same picture.

**OCR.** An OCR file holds document pages kept as PNG in the folder of the same name beside it, and for each page the text the public weights write in their model card's grounding mode, decoded greedily: every block of the page with its kind and its box. The pages were written for this repository (CC0 1.0) and rendered by `scripts/document_pages.py`, so no model has read them before: display mathematics, dense tables, columns with footnotes, accented and non-Latin scripts, a form and chemistry. The check sends each page to `POST /v1/ocr` and says `matches` when, on the share of pages the file states, the character error rate of the text (with whitespace normalized) is within the file's page limit, the blocks are of the same kinds in the same order and their boxes are within its box limit, and the typical (median) error rate and box gap are within their own limits. `check_ocr.py make` makes the file again from the same public weights.

**Audio.** An audio file holds a list of probes (a prompt and a seed, with lyrics for a model that sings, or a text and a voice for speech) and a compact signature of the clip the public weights make for each: the energy of 30 frequency bands in every quarter second. The seed's noise is drawn as float32 numbers on the CPU, so the same request gives the same clip on any computer, up to small rounding differences between graphics cards. The check asks for the same clips as WAV, computes the same signature with nothing but Python, and says `matches` when the share of clips the file states are within its distance (the mean difference in dB) and the typical distance is within its limit. `check_audio.py make` makes the file again from the same public weights and takes any prompts and seeds you choose, and `check_audio.py signature` prints the signature of a clip you have.

The files of `glm-tts`, `minimax-music-3` and `soulx-podcast` hold clips recorded from the models EmpirioLabs serves, several for each probe, and the check compares a live clip with the closest of them. It confirms the model is still the one that was recorded.

**Video.** A video file holds a list of probes, each a prompt and a seed, with a starting image for a model that animates a picture, kept in the folder of the same name beside the file with its source and license. For each probe it holds 32 × 32 thumbnails of a few frames of the clip the public weights make with the settings the file states. The seed fixes the starting noise, so the same request gives the same clip on any computer, up to small rounding differences between graphics cards. The check asks `POST /v1/videos/generations` for the same clips, reads the same frames with ffmpeg and says `matches` when the share of clips the file states have the length it records and are within its distance of their thumbnails, and the typical distance is within its limit. A clip's distance is the mean over its frames, each measured as for an image. `check_video.py make` makes the file again from the same public weights and takes any prompts, seeds and starting images you choose. `check_video.py compare` tells you how far two clips you have are.

**3D.** A 3D file holds a list of probes, each a picture of an object on a transparent background, kept in the folder of the same name beside the file with its source and license, and a seed. For each probe it holds a coarse signature of the GLB the public weights make: the share of the mesh's surface in each cell of a 16 × 16 × 16 grid over its bounding box, the box's proportions, and the average color of the surface in coarser cells. Remeshing, simplification and another texture layout move vertices but hardly any surface between cells, so a mesh keeps its signature, while another model shapes the object's hidden sides and depth differently. The check asks `POST /v1/3d/generations` for the same meshes, reads each GLB with nothing but Python and says `matches` when the share of meshes the file states are within its limits and the typical shape distance is within its limit. `check_mesh.py make` makes the file again from the same public weights and takes any pictures and seeds you choose. `check_mesh.py compare` tells you how far two GLB files you have are.

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

## Our daily results, signed

EmpirioLabs publishes the latest results of its own checks of every Native Inference model at https://api.empiriolabs.ai/v1/verification (no API key needed), signed with the same key as every receipt. For each model: the result of its last check against its fingerprint (`matches`, `does not match`, or `not checked` when the last check gave no confirmed result), the date, whether every answer of that check was signed, and whether the model is checked daily or on request. For the privacy test: the models it covers, whether it is running, and how many of its test links have ever been opened.

`python check_receipt.py statement` fetches the results and checks the signature; `--file statement.json` checks results you saved. The signature covers the statement without `signature`, written as JSON with sorted keys and no spaces, as for a receipt.

## What the checks cover

- Receipts cover every Native Inference model. Third-party models don't carry one.
- Every kind of answer that carries no receipt is listed on [the docs page](https://docs.empiriolabs.ai/verifiable-inference).
- Text-to-video requests to `moss-video-and-audio` and `svi-2-0-pro` don't carry a receipt; send a starting image to get one. Streams from the Gemini endpoint `streamGenerateContent` don't either; use `generateContent` when you need one.
- Fingerprints cover the models in the table above. Every model check also checks the receipt on every answer.
- A model check confirms the model at the time it runs. The questions are fixed and public. The receipt on every response is EmpirioLabs' signed statement of which model answered it. For image, video and 3D models and for `ace-step-1-5-xl` and `stable-audio-3-medium` you can also choose your own prompts, seeds and pictures, and for OCR models your own pages, so a check can use requests we have never seen.
- The fingerprints of `glm-tts`, `minimax-music-3` and `soulx-podcast` were recorded from the models we serve, so their checks confirm that the model hasn't changed since.
- A fingerprint confirms the model, not the numeric precision it runs at.
- A receipt shows what EmpirioLabs returned for a request. It doesn't encrypt anything, and it doesn't show whether an answer is correct.
- The daily results are EmpirioLabs' signed statement of its own checks. The model checks above let you confirm them yourself.

## Report a problem

Email support@empiriolabs.ai with "Security" in the subject line. A check that says `DOES NOT MATCH` or `NOT VERIFIED` is worth reporting: include the model ID, the time and the script's output.
