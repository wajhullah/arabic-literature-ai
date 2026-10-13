# 📚 Classical Arabic Literature AI Tutor

> A cross-lingual Retrieval-Augmented Generation (RAG) tutor that lets a student learn
> **Classical Arabic literature** by asking questions in **English** or **Tamil** — and
> get grounded, citation-backed answers that quote the original Arabic source verbatim.

<p align="center">
  <img src="https://img.shields.io/badge/LLM-DeepSeek--R1--Distill--Qwen--14B-5B8DEE?style=for-the-badge" alt="Model badge">
  <img src="https://img.shields.io/badge/Fine--tuning-Unsloth%20QLoRA-ED7D31?style=for-the-badge" alt="Fine-tuning badge">
  <img src="https://img.shields.io/badge/Serving-vLLM%20(RunPod)-7C3AED?style=for-the-badge" alt="Serving badge">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge" alt="UI badge">
  <img src="https://img.shields.io/badge/RAG-Qdrant%20%2B%20FastEmbed-2D9CDB?style=for-the-badge" alt="RAG badge">
</p>

---

## 🧭 Overview & Problem Statement

Classical Arabic texts such as *Al-ʿAwāmil* (العوامل) by Al-Jurjānī are the traditional
gateway to Arabic grammar and literature, but they are dense, written in a high literary
register, and almost impossible to approach without a fluent instructor.

This project solves that for a student who thinks in **English** and **Tamil** — two
very different languages, one of them a genuinely low-resource language for ML tooling:

- The student asks a question in English or Tamil (or Arabic).
- A **cross-lingual embedding model** maps the question into the same vector space as the
  Arabic source text, so Tamil and English queries retrieve the *correct Arabic passage*
  without any machine translation step.
- The retrieved passage is bundled with the question and sent to a **fine-tuned 14B
  reasoning model**, which answers in the student's language, **quoting the Arabic
  verbatim** and labelling each quote with its `[n]` citation — so nothing is invented.

The result is an always-available, citation-grounded tutor that teaches classical Arabic
the way it was originally written.

---

## 🏗️ System Architecture

```
                              ┌──────────────────────────────────────────────────────────┐
                              │                      Student (Browser)                   │
                              │        Question in English / Tamil / Arabic              │
                              └───────────────────────────────┬──────────────────────────┘
                                                              │
                                                              ▼
                        ┌───────────────────────────────────────────────────────────────┐
                        │                     Streamlit UI  (app.py)                     │
                        │   chat history · retrieved-context expander · RTL Arabic view  │
                        └───────────────┬───────────────────────────────┬───────────────┘
                                        │                               │
                     ┌──────────────────▼──────────────────┐   ┌────────▼─────────────────────────┐
                     │    Cross-Lingual RAG (LOCAL)        │   │   Inference (REMOTE · RunPod)    │
                     │  ┌───────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
                     │  │ FastEmbed                    │  │   │  │ vLLM · OpenAI-compatible    │  │
                     │  │ intfloat/multilingual-e5-large│  │   │  │ /v1/chat/completions        │  │
                     │  │  "query: …" prefix            │  │   │  │ served-model: arabic-tutor  │  │
                     │  └───────────────┬───────────────┘  │   │  └──────────────┬──────────────┘  │
                     │                  │ top-3            │   │                 │                 │
                     │  ┌───────────────▼───────────────┐  │   │  ┌──────────────▼──────────────┐  │
                     │  │ Qdrant (disk-backed)          │  │   │  │ Fine-tuned 14B model        │  │
                     │  │ collection: arabic_literature │  │   │  │ Wajhullah/Waj               │  │
                     │  │ stored in /vector_db          │  │   │  │ (QLoRA of DeepSeek-R1-      │  │
                     │  └───────────────┬───────────────┘  │   │  │  Distill-Qwen-14B)          │  │
                     │                  │                   │   │  │ <think>…</think> filtered  │  │
                     │                  │ Arabic passages   │   │  └──────────────┬──────────────┘  │
                     └──────────────────┼───────────────────┘   └─────────────────┼────────────────┘
                                        │                                           │
                                        └───────────────────┬───────────────────────┘
                                                            │ prompt = system + question + [1..n] Arabic context
                                                            ▼
                                           ┌────────────────────────────────────┐
                                           │  Citation-grounded answer in the    │
                                           │  student's language (EN / TA / AR) │
                                           └────────────────────────────────────┘
```

### Offline / build-time pipeline

The runtime diagram above depends on three offline steps that build the corpus and the model:

```
  data/raw_pdfs/*.pdf ──▶ extract_text.py ──▶ data/processed_text/*.txt
          │                    │                          │
          │              pdfplumber (+easyocr)      rag_search.py (embed + index)
          │                                              │
          ▼                                              ▼
  distill_data.py ──▶ dataset/dataset.json ──▶ vector_db/   (Qdrant on disk)
     (DeepSeek API)      (EN + TA pairs)
          │
          ▼
  train.py (Unsloth QLoRA) ──▶ merged 16-bit model ──▶ Hugging Face Hub (Wajhullah/Waj)
                                                              │
                                                              ▼
                                                    vLLM serves it on RunPod
```

---

## ✨ Core Features

| Feature | What it does |
|---|---|
| 🌍 **Multilingual embeddings** | `intfloat/multilingual-e5-large` via FastEmbed embeds English, Tamil and Arabic into one shared vector space, so retrieval is language-agnostic. |
| 🗣️ **Low-resource language matching** | Tamil queries retrieve Arabic passages *directly* — no translation layer — using the E5 `query:`/`passage:` task prefixes for optimal cross-lingual quality. |
| 🎯 **Parameter-efficient fine-tuning** | QLoRA (4-bit, `r=16`, `α=16`) fine-tunes a 14B model on a single GPU, then merges back to a 16-bit vLLM-ready checkpoint. |
| 📖 **Citation-grounded answers** | Every answer quotes the retrieved Arabic source verbatim with `[n]` markers, plus an in-app expander to inspect the exact passages retrieved. |
| 🧠 **Reasoning-aware streaming** | The model's hidden `<think>…</think>` chain-of-thought is filtered out, so students only see the tutor's answer. |
| ⚡ **GPU-free local retrieval** | Embeddings and Qdrant run locally on CPU, so the entire RAG half of the system works offline. |
| 📚 **Full classical corpus pipeline** | PDF ingestion → text cleaning → dataset distillation → embedding → indexing, all reproducible via `scripts/`. |

---

## 📁 Repository Structure

```
arabic-literature-ai/
├── app.py                      # Streamlit UI + cross-lingual RAG + vLLM streaming client
├── requirements.txt            # Python dependencies
├── scripts/
│   ├── extract_text.py         # PDF → clean UTF-8 Arabic text (pdfplumber + optional easyocr)
│   ├── distill_data.py         # DeepSeek API → bilingual (EN/TA) instruction dataset
│   └── rag_search.py           # chunk → embed (FastEmbed) → index/search (Qdrant)
├── data/
│   ├── raw_pdfs/               # Source PDFs (git-ignored — see .gitignore)
│   └── processed_text/         # Cleaned Arabic text
├── dataset/
│   └── dataset.json            # 80 instruction/response pairs (English + Tamil)
└── vector_db/                  # Disk-backed Qdrant store (git-ignored — see .gitignore)
```

---

## 🚀 Installation Guide

### Part 1 — Local sandbox (Windows / Linux / macOS)

The local stack covers **ingestion, dataset building, and retrieval** — everything
except the 14B model itself. It runs on CPU.

#### 1. Prerequisites

- Python **3.10+** (the repo targets 3.10)
- Git
- *(Optional, for OCR)* — Tesseract or enough RAM for EasyOCR

#### 2. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/arabic-literature-ai.git
cd arabic-literature-ai

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure environment variables

Create a `.env` file from the template (this file is git-ignored):

```bash
# Only needed for dataset distillation (distill_data.py)
DEEPSEEK_API_KEY=sk-...
```

#### 5. Ingest the textbook

Drop your PDF into `data/raw_pdfs/`, then extract clean Arabic text:

```bash
python scripts/extract_text.py            # digital text only
python scripts/extract_text.py --ocr      # include OCR for scanned pages
```

#### 6. Build the vector index

```bash
python scripts/rag_search.py              # index the text + run built-in demo queries
```

This embeds the text with `intfloat/multilingual-e5-large` and persists a Qdrant
collection under `vector_db/`.

#### 7. (Optional) Generate the training dataset

```bash
python scripts/distill_data.py --max-chunks 2    # dry-run on 2 chunks
python scripts/distill_data.py                    # full distillation
```

#### 8. Run the app

```bash
streamlit run app.py
```

The sidebar walks through the pipeline; the first question warms up the local
retrieval stack.

---

### Part 2 — Cloud deployment (RunPod + vLLM)

The fine-tuned model is public on the Hugging Face Hub as
**[`Wajhullah/Waj`](https://huggingface.co/Wajhullah/Waj)** — a 16-bit merge of the
QLoRA adapter, ready for vLLM. You don't need to train to serve it.

#### 1. GPU sizing

| GPUs | VRAM | Notes |
|---|---|---|
| 1 × L40S / A40 / RTX 6000 Ada / A100 | ≥ 48 GB | ✅ Recommended — fits the 28 GB weights + KV cache. |
| 2 × 24 GB (4090 / A5000) | 2 × 24 GB | Works with `--tensor-parallel-size 2`. |

#### 2. Create the RunPod pod

- **Image:** `vllm/vllm-openai:latest` (OpenAI-compatible server, vLLM 0.31.0)
- **GPU:** an L40S (48 GB) is the default choice (~$1.09/hr)
- **Container disk:** ≥ 40 GB (the model is pulled fresh from the Hub)
- **Ports:** expose `8000/http`
- **Start command:**
  ```bash
  vllm serve Wajhullah/Waj \
    --served-model-name arabic-tutor \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
  ```

> ⚠️ **Driver gotcha (learned the hard way).** Some RunPod hosts ship older CUDA
> drivers. If you hit `CUDA unknown error` or `cudaErrorDevicesUnavailable` (error 46),
> pin `gpu.allowedCudaVersions: ["13.2"]` when creating the pod.

#### 3. Verify the endpoint

Cold start pulls the image + the ~28 GB model from the Hub (expect a few minutes):

```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/models
curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arabic-tutor","messages":[{"role":"user","content":"من هو أبو العلاء المعري؟"}]}'
```

Expect a fluent, professor-style Arabic answer — English questions are also
answered in Arabic.

#### 4. Point the app at your endpoint

Edit the constants at the top of [`app.py`](app.py):

```python
VLLM_BASE_URL = "https://<pod-id>-8000.proxy.runpod.net/v1"
VLLM_API_KEY  = "runpod-vllm-placeholder-key"   # any non-empty placeholder works
MODEL_NAME    = "arabic-tutor"
```

Then run the local app as in Part 1, step 8.

---

### Part 3 — Reproduce the fine-tune (optional, advanced)

Training was done with **Unsloth** on the pre-quantized base
`unsloth/deepseek-r1-distill-qwen-14b-unsloth-bnb-4bit`.

```python
# train.py — configuration summary (see the handoff / session files for the full script)
model = "unsloth/deepseek-r1-distill-qwen-14b-unsloth-bnb-4bit"
# LoRA: r=16, alpha=16
# target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# Trainer: batch 2 × grad-accum 4 · lr 2e-4 · adamw_8bit · linear scheduler
# 80 examples → max_steps 60 (≈ 6 epochs) · loss 0.86 → 0.61
```

> **Install Unsloth from source**, not via `requirements.txt` — follow the
> [official Unsloth installation guide](https://docs.unsloth.ai). Use a pod with
> `gpu.allowedCudaVersions: ["13.2"]` and mount a network volume, because the
> container disk on Unsloth pods is ephemeral.

---

## 🧰 Tech Stack Checklist

**User interface**
- [x] **Streamlit** — chat UI with RTL Arabic rendering and retrieval inspection

**Retrieval / RAG**
- [x] **Qdrant** — disk-backed vector store (`/vector_db/`)
- [x] **ChromaDB** — listed as an alternative store in `requirements.txt`
- [x] **FastEmbed** — local CPU/GPU embeddings (`intfloat/multilingual-e5-large`)

**Fine-tuning**
- [x] **Unsloth** — QLoRA fine-tuning of DeepSeek-R1-Distill-Qwen-14B (4-bit)

**Inference serving**
- [x] **vLLM** — OpenAI-compatible serving on RunPod (`--served-model-name arabic-tutor`)

**Dataset distillation**
- [x] **DeepSeek API** — generates bilingual (EN/TA) instruction pairs

**Ingestion & utilities**
- [x] **pdfplumber**, **PyMuPDF**, **easyocr** — PDF → Arabic text extraction
- [x] **openai**, **httpx** — OpenAI-compatible clients (vLLM + DeepSeek)
- [x] **torch / transformers / peft / bitsandbytes / trl / accelerate** — training stack
- [x] **python-dotenv**, **numpy**, **pandas** — config & data handling

---

## ⚙️ Configuration Reference

| Constant | Location | Meaning |
|---|---|---|
| `VLLM_BASE_URL` | `app.py` | OpenAI-compatible vLLM endpoint (RunPod proxy) |
| `MODEL_NAME` | `app.py` | vLLM `--served-model-name` alias (`arabic-tutor`) |
| `EMBEDDING_MODEL` | `app.py`, `rag_search.py` | `intfloat/multilingual-e5-large` |
| `COLLECTION_NAME` | `app.py`, `rag_search.py` | Qdrant collection (`arabic_literature`) |
| `TOP_K` | `app.py` | Number of passages retrieved per query (default 3) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `rag_search.py` | 500 / 50 characters per text window |
| `DEEPSEEK_API_KEY` | `.env` | Key for dataset distillation |

---

## ⚠️ Notes & Disclaimer

- **Model weights are not in this repo.** The LoRA adapter, merged checkpoints, and
  training outputs are git-ignored; the public model lives at
  [`Wajhullah/Waj`](https://huggingface.co/Wajhullah/Waj).
- **Source PDFs are not in this repo.** Place your own copies under `data/raw_pdfs/`
  (git-ignored) — do not commit copyrighted textbooks.
- The tutor is grounded in the retrieved passages and instructed to refuse to answer
  when the context is insufficient, but it is an LLM: verify quotations against the
  original text before academic use.

---

## 📄 License

Add your chosen license here (e.g. MIT). The model on the Hub is subject to its own
license terms from DeepSeek.

---

<p align="center">
  Built with ❤️ for students of Classical Arabic — English and Tamil speakers alike.
</p>
