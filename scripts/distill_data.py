"""
Distill Classical Arabic textbook text into bilingual (English + Tamil)
instruction-response pairs for fine-tuning a QLoRA tutor model.

Pipeline step 2 — Dataset generation (model distillation):
    1. Read extracted text from data/processed_text/Awaamil.txt.
    2. Split the text into semantic chunks of roughly 1,000 words each,
       preferring paragraph and sentence boundaries.
    3. Send each chunk to DeepSeek (via the OpenAI-compatible client) with a
       distillation prompt that produces exactly 5 instruction-response pairs
       (2 English + 3 Tamil) grounded in — and quoting — the Arabic source text.
    4. Parse DeepSeek's JSON response, compile all pairs across all chunks, and
       append/save them to dataset/dataset.json.

Output format (dataset/dataset.json) — a JSON array of objects with exactly
two keys each:

    {"instruction": "<question or task>", "output": "<explanation>"}

Environment:
    DEEPSEEK_API_KEY    Required. API key for the DeepSeek endpoint.

Usage:
    python distill_data.py                      # distill every chunk
    python distill_data.py --max-chunks 2       # dry-run the first 2 chunks
    python distill_data.py --base-url https://api.deepseek.com
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the script works from any CWD.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed_text" / "Awaamil.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "dataset" / "dataset.json"

# ---------------------------------------------------------------------------
# DeepSeek / OpenAI-compatible API configuration.
# ---------------------------------------------------------------------------
# DeepSeek exposes an OpenAI-compatible endpoint. The official docs commonly
# use https://api.deepseek.com; https://deepseek.com is kept here as requested.
DEEPSEEK_BASE_URL = "https://deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

# ---------------------------------------------------------------------------
# Distillation parameters.
# ---------------------------------------------------------------------------
TARGET_CHUNK_WORDS = 1_000   # approximate words per semantic chunk
ENGLISH_PAIR_COUNT = 2       # instruction + output both in English
TAMIL_PAIR_COUNT = 3         # instruction + output both in Tamil
PAIRS_PER_CHUNK = ENGLISH_PAIR_COUNT + TAMIL_PAIR_COUNT   # always 5
REQUEST_DELAY_SECONDS = 1.0  # polite pause between API calls (rate-limit safety)
MAX_ATTEMPTS_PER_CHUNK = 3   # retries for API errors / malformed responses

# Sentence terminators used when a paragraph must be split further.
# \u061f = Arabic question mark (؟), \u06d4 = Arabic full stop (۔).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u061f\u06d4])\s+")
# Tamil Unicode block (U+0B80–U+0BFF) — used only for a language sanity check.
_TAMIL_RE = re.compile(r"[\u0b80-\u0bff]")

SYSTEM_PROMPT = (
    "You are an expert in Classical Arabic grammar (nahw) and literature. "
    "You produce high-quality instruction-tuning datasets for a student who "
    "learns best in English and Tamil. You always answer with raw JSON only."
)


# ---------------------------------------------------------------------------
# Chunking — group text into semantic chunks of ~TARGET_CHUNK_WORDS words.
# ---------------------------------------------------------------------------
def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs on blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_unit_by_words(unit: str, target_words: int) -> list[str]:
    """Hard-split an oversized unit into word-count-sized slices."""
    words = unit.split()
    return [" ".join(words[i : i + target_words]) for i in range(0, len(words), target_words)]


def chunk_text(text: str, target_words: int = TARGET_CHUNK_WORDS) -> list[str]:
    """Split ``text`` into semantic chunks of roughly ``target_words`` words.

    The algorithm prefers paragraph boundaries, then sentence boundaries, and
    only falls back to a raw word-count split for a single pathologically long
    paragraph that contains no sentence punctuation.
    """
    if not text.strip():
        return []

    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph.split()) > target_words:
            units.extend(s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip())
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        word_count = len(unit.split())

        # A single unit larger than a full chunk can only be hard-split.
        if word_count > target_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            chunks.extend(_split_long_unit_by_words(unit, target_words))
            continue

        if current and current_words + word_count > target_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(unit)
        current_words += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------
def build_user_prompt(chunk: str) -> str:
    """Build the per-chunk distillation prompt.

    The prompt forces a raw JSON array (no markdown fences, no surrounding
    prose) so the response can be handed to ``json.loads`` after only light
    cleanup. It also fixes the 2-English / 3-Tamil split that the student needs.
    """
    return (
        "Analyze the following excerpt from a Classical Arabic grammar textbook "
        "(Al-Awaamil, the 'amil in nahw).\n\n"
        f"<textbook_excerpt>\n{chunk}\n</textbook_excerpt>\n\n"
        f"Generate EXACTLY {PAIRS_PER_CHUNK} distinct instruction-response pairs "
        f"based only on the content of this excerpt:\n"
        f"- {ENGLISH_PAIR_COUNT} pairs: the instruction is in English and the output "
        f"is a clear, academic explanation written in fluent English that explicitly "
        f"references and quotes the original Arabic terms from the excerpt.\n"
        f"- {TAMIL_PAIR_COUNT} pairs: the instruction is in Tamil and the output is a "
        f"clear, academic explanation written in fluent Tamil that explicitly "
        f"references and quotes the original Arabic terms from the excerpt.\n\n"
        "Return ONLY a raw JSON array of objects. Each object must have exactly "
        'two keys: "instruction" and "output". Do not wrap the array in code '
        "fences, and do not add any text before or after the array.\n\n"
        "Example format:\n"
        '[{"instruction": "...", "output": "..."}]'
    )


# ---------------------------------------------------------------------------
# DeepSeek client + response handling.
# ---------------------------------------------------------------------------
def build_client(base_url: str = DEEPSEEK_BASE_URL, api_key: str | None = None):
    """Create an OpenAI client pointed at DeepSeek's compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    api_key = api_key or os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise SystemExit(
            f"Missing API key: set the {API_KEY_ENV_VAR} environment variable."
        )

    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)


def _call_chat(client, chunk: str, model: str) -> str:
    """Send one chat completion request and return the assistant's text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(chunk)},
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def extract_json_array(content: str) -> list | None:
    """Parse the model's output as a JSON array, tolerating common wrappers.

    DeepSeek occasionally wraps the array in triple-backtick fences or returns
    a JSON object instead. We strip fences, locate the first '[' / last ']',
    and finally fall back to scanning for the first list value in an object.
    """
    if not content:
        return None

    text = content.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed

    # Fallback: the model returned a JSON object — look for the first list.
    obj_start = text.find("{")
    if obj_start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[obj_start:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    return value

    return None


def normalize_pairs(raw_items: list | None) -> list[dict[str, str]]:
    """Keep only well-formed ``{"instruction", "output"}`` objects."""
    pairs: list[dict[str, str]] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "") or "").strip()
        output = str(item.get("output", "") or "").strip()
        if instruction and output:
            pairs.append({"instruction": instruction, "output": output})
    return pairs


def _warn_language_balance(pairs: list[dict[str, str]]) -> None:
    """Log a warning if the 2-English / 3-Tamil split is off."""
    tamil = sum(1 for p in pairs if _TAMIL_RE.search(p["instruction"]))
    english = len(pairs) - tamil
    if (english, tamil) != (ENGLISH_PAIR_COUNT, TAMIL_PAIR_COUNT):
        print(
            f"    [WARN] language split is {english} EN / {tamil} TA "
            f"(expected {ENGLISH_PAIR_COUNT} EN / {TAMIL_PAIR_COUNT} TA)"
        )


def generate_pairs(client, chunk: str, model: str) -> list[dict[str, str]]:
    """Distill one chunk into exactly ``PAIRS_PER_CHUNK`` pairs, with retries.

    Raises ``RuntimeError`` if the model never produces a well-formed response
    containing the required number of pairs.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS_PER_CHUNK + 1):
        try:
            content = _call_chat(client, chunk, model)
            pairs = normalize_pairs(extract_json_array(content))
            if len(pairs) == PAIRS_PER_CHUNK:
                _warn_language_balance(pairs)
                return pairs
            last_error = RuntimeError(
                f"expected {PAIRS_PER_CHUNK} pairs but parsed {len(pairs)}"
            )
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_error = exc

        if attempt < MAX_ATTEMPTS_PER_CHUNK:
            backoff = 2 ** attempt
            print(
                f"    attempt {attempt}/{MAX_ATTEMPTS_PER_CHUNK} failed "
                f"({last_error}); retrying in {backoff}s..."
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"chunk failed after {MAX_ATTEMPTS_PER_CHUNK} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Dataset persistence.
# ---------------------------------------------------------------------------
def load_existing_dataset(path: Path) -> list[dict]:
    """Load previously saved pairs, or start fresh if the file is unusable."""
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] {path} is not valid JSON; starting fresh.")
        return []

    if not isinstance(data, list):
        print(f"[WARN] {path} is not a JSON array; starting fresh.")
        return []

    return [item for item in data if isinstance(item, dict)]


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    """Drop exact (instruction, output) duplicates while preserving order."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for pair in pairs:
        key = (str(pair.get("instruction")), str(pair.get("output")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def save_dataset(path: Path, pairs: list[dict]) -> None:
    """Atomically write the dataset as pretty-printed, UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def read_input(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Input text file not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill Awaamil textbook text into bilingual (English + Tamil) "
            "instruction-response pairs for QLoRA fine-tuning."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_FILE,
        help="Extracted text file to distill (default: data/processed_text/Awaamil.txt).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE,
        help="Dataset file to append/save results to (default: dataset/dataset.json).",
    )
    parser.add_argument(
        "--chunk-words", type=int, default=TARGET_CHUNK_WORDS,
        help=f"Approximate words per semantic chunk (default: {TARGET_CHUNK_WORDS}).",
    )
    parser.add_argument(
        "--base-url", default=DEEPSEEK_BASE_URL,
        help=f"DeepSeek base URL (default: {DEEPSEEK_BASE_URL}).",
    )
    parser.add_argument(
        "--model", default=DEEPSEEK_MODEL,
        help=f"Model name (default: {DEEPSEEK_MODEL}).",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SECONDS,
        help=f"Seconds to sleep between API calls (default: {REQUEST_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Only distill the first N chunks (useful for dry-runs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    text = read_input(args.input)
    chunks = chunk_text(text, target_words=args.chunk_words)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    print(f"Input text     : {args.input}")
    print(f"Output dataset : {args.output}")
    print(f"Chunk size     : ~{args.chunk_words} words")
    print(f"Total chunks   : {len(chunks)}")
    if not chunks:
        raise SystemExit("No text to distill — input file is empty.")

    client = build_client(base_url=args.base_url)

    pairs = dedupe_pairs(load_existing_dataset(args.output))
    if pairs:
        print(f"Resuming with {len(pairs)} existing pair(s) in the dataset.")

    failed_chunks = 0
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Distilling chunk {index} of {total}...")
        try:
            new_pairs = generate_pairs(client, chunk, model=args.model)
        except Exception as exc:  # noqa: BLE001 — keep processing remaining chunks
            failed_chunks += 1
            print(f"  [ERROR] skipped chunk {index}: {exc}")
        else:
            pairs.extend(new_pairs)
            pairs = dedupe_pairs(pairs)
            save_dataset(args.output, pairs)
            print(f"  Generated {len(new_pairs)} pair(s); dataset total = {len(pairs)}.")

        if index < total:
            time.sleep(args.delay)

    # Final save guards against a run where every chunk failed but the file
    # still needs to reflect the (possibly unchanged) accumulated dataset.
    save_dataset(args.output, pairs)

    print("Done.")
    print(f"Dataset saved to {args.output} ({len(pairs)} total pairs).")
    if failed_chunks:
        print(f"[WARN] {failed_chunks} chunk(s) failed and were skipped.")


if __name__ == "__main__":
    main()
