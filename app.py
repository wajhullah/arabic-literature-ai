"""
Streamlit web UI for the Classical Arabic Literature AI Tutor.

Architecture
------------
* Cross-lingual RAG (local) — FastEmbed's ``intfloat/multilingual-e5-large``
  embeds the user's English / Tamil / Arabic question, and a disk-backed
  Qdrant store (``vector_db/``) returns the top-3 Arabic textbook passages.
* Inference (remote) — the fine-tuned 14B model ``Wajhullah/Waj`` (a QLoRA
  fine-tune of DeepSeek-R1-Distill-Qwen-14B) served through vLLM's
  OpenAI-compatible endpoint on RunPod.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# vLLM (RunPod) backend configuration
# ---------------------------------------------------------------------------
# Swap this whenever you relaunch your RunPod pod. It is the OpenAI-compatible
# base URL of the vLLM server (container port 8000 exposed over the proxy).
VLLM_BASE_URL = "https://cgecyfpq5h3vno-8000.proxy.runpod.net/v1"

# The RunPod proxy does not require a key; any non-empty placeholder satisfies
# the OpenAI client. Replace it if you put an auth layer in front of vLLM.
VLLM_API_KEY = "runpod-vllm-placeholder-key"

# The alias the vLLM pod serves the merged 14B model under (set via
# ``--served-model-name``). Requests must use this alias, not the HF repo path.
MODEL_NAME = "arabic-tutor"

# Academic accuracy: a low temperature keeps the tutor from improvising.
TEMPERATURE = 0.3
# Budget must cover the model's hidden <think> block plus the visible answer.
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Local RAG configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
E5_QUERY_PREFIX = "query: "          # required by the E5 family at search time
COLLECTION_NAME = "arabic_literature"
TOP_K = 3

# The reasoning model closes its hidden chain-of-thought with this tag; text
# after it is the visible answer we keep for the student.
REASONING_END = "</think>"

PROJECT_ROOT = Path(__file__).resolve().parent

# Expert tutor persona — answers in the student's language and must quote the
# retrieved Arabic so nothing is invented (hallucination guard).
SYSTEM_PROMPT = (
    "You are an expert Classical Arabic Literature tutor. "
    "Answer in the same language as the student's question (for example English "
    "or Tamil). Base every answer strictly on the retrieved Arabic textbook "
    "passages provided in the user message. Quote the relevant Arabic phrases "
    "verbatim and translate them so the student can verify the source, labelling "
    "each quote with its [n] citation marker. If the passages do not contain the "
    "answer, say so clearly instead of guessing — never hallucinate. Explain "
    "concepts step by step, keep a formal didactic tone, and prioritize academic "
    "accuracy."
)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved Arabic passage returned by the local Qdrant store."""

    rank: int
    score: float
    text: str
    source: str | None = None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_vector_db_path() -> Path:
    """Locate the disk-backed Qdrant store.

    The docs refer to the store as ``../vector_db`` (relative to ``scripts/``);
    it actually sits at the project root, next to this file. Check both spots
    and keep the first one that contains a real Qdrant database.
    """
    candidates = (
        PROJECT_ROOT / "vector_db",
        PROJECT_ROOT.parent / "vector_db",
    )
    for path in candidates:
        if (path / "meta.json").is_file() or (path / "collection").is_dir():
            return path
    return PROJECT_ROOT / "vector_db"  # documented default


# ---------------------------------------------------------------------------
# Cached resources (created once per Streamlit session, shared across users)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Multilingual E5 embedding model via FastEmbed (local CPU/GPU inference)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    """Local, disk-backed Qdrant client rooted at ``vector_db/``."""
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(resolve_vector_db_path()))


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    """OpenAI-compatible client pointed at the remote vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_context(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Embed ``query`` (with the E5 ``query:`` prefix) and search Qdrant."""
    query = query.strip()
    if not query:
        return []

    model = get_embedding_model()
    client = get_qdrant_client()

    query_vector = list(
        model.embed([f"{E5_QUERY_PREFIX}{query}"], batch_size=1)
    )[0].tolist()

    try:
        # qdrant-client >= 1.10 exposes query_points; older releases use search.
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    except AttributeError:  # pragma: no cover - fallback for older clients
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    results: list[SearchResult] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        results.append(
            SearchResult(
                rank=rank,
                score=float(hit.score),
                text=str(payload.get("text", "")),
                source=payload.get("source"),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Prompt construction + inference
# ---------------------------------------------------------------------------
def build_chat_messages(
    question: str, contexts: list[SearchResult]
) -> list[dict[str, str]]:
    """Bundle the question and the retrieved Arabic passages into one prompt."""
    context_block = "\n\n".join(
        f"[{ctx.rank}] (source: {ctx.source or 'unknown'})\n{ctx.text.strip()}"
        for ctx in contexts
    )
    user_content = (
        f"Student question (answer in this language):\n{question.strip()}\n\n"
        f"Retrieved Arabic textbook context:\n{context_block}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def filter_reasoning_stream(pieces: Iterator[str]) -> Iterator[str]:
    """Drop the model's ``<think>…</think>`` prefix, keeping only the answer.

    The tokenizer may or may not include the opening ``<think>`` tag in the
    stream, but it always closes the block with ``</think>``, so we buffer the
    stream and emit everything after that marker. Leading whitespace on the
    first visible piece is trimmed; if a turn never opens a reasoning block,
    the whole text is emitted unchanged.
    """
    buffer = ""
    phase = "reasoning"  # → "answer" once the closing tag is seen
    lead_trimmed = False
    for piece in pieces:
        if phase == "reasoning":
            buffer += piece
            idx = buffer.find(REASONING_END)
            if idx == -1:
                continue
            buffer = buffer[idx + len(REASONING_END):]
            phase = "answer"
            if not buffer:
                continue
            piece = buffer
            buffer = ""
        if phase == "answer":
            if not lead_trimmed:
                piece = piece.lstrip()
                if not piece:
                    continue
                lead_trimmed = True
            yield piece
    if phase == "reasoning" and buffer.strip():
        yield buffer


def stream_llm(messages: list[dict[str, str]]) -> Iterator[str]:
    """Stream the vLLM completion, hiding the model's chain-of-thought.

    The fine-tuned base is DeepSeek-R1-Distill-Qwen, a reasoning model that
    emits an internal ``<think>…</think>`` block before the real answer; we
    filter it so students see only the tutor's answer.
    """
    client = get_openai_client()
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    def pieces() -> Iterator[str]:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    yield from filter_reasoning_stream(pieces())


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Explain the Cross-Lingual RAG mechanics so students can follow along."""
    with st.sidebar:
        st.header("🧭 How this works")
        st.markdown(
            """
            **Cross-Lingual RAG pipeline**

            1. Your question — English, Tamil, or Arabic — is embedded locally
               with `intfloat/multilingual-e5-large` (FastEmbed).
            2. The top **3** most similar Arabic textbook passages are pulled
               from the disk-backed Qdrant store (`vector_db/`) — texts from
               classical works such as *Mejan* or *Ajnas*.
            3. The question and the retrieved Arabic context are bundled into a
               structured prompt and sent to the fine-tuned 14B model
               (`Wajhullah/Waj`) served by vLLM on RunPod.
            4. The tutor answers in your language and quotes the Arabic source
               phrases verbatim, so nothing is invented.
            """
        )
        st.divider()
        st.caption(f"**Backend:** {VLLM_BASE_URL}")
        st.caption(f"**Model:** {MODEL_NAME}")
        st.caption(f"**Embeddings:** {EMBEDDING_MODEL}")
        st.caption(f"**Collection:** `{COLLECTION_NAME}` · top {TOP_K}")
        st.caption(f"**Temperature:** {TEMPERATURE}")


def render_contexts(contexts: list[SearchResult]) -> None:
    """Show the exact Arabic paragraphs the database retrieved for a turn."""
    if not contexts:
        st.info("No passages were retrieved (the vector database may be empty).")
        return
    for ctx in contexts:
        st.markdown(
            f"**[{ctx.rank}]** · `{ctx.source or 'unknown'}` · score `{ctx.score:.3f}`"
        )
        # Right-to-left block so the Arabic renders naturally.
        st.markdown(
            f"<div dir='rtl' lang='ar' style='text-align:right;"
            f"background:#f7f7f7;padding:10px;border-radius:8px'>"
            f"{html.escape(ctx.text)}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Classical Arabic Literature AI Tutor",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 Classical Arabic Literature AI Tutor")
    st.caption(
        "Cross-lingual retrieval over the classical textbook corpus, answered "
        "by the fine-tuned 14B tutor."
    )

    render_sidebar()

    # Warm up the local retrieval stack now so the first question is fast.
    with st.spinner("Loading the local retrieval stack (embeddings + vector DB)…"):
        try:
            get_embedding_model()
            get_qdrant_client()
        except Exception as exc:  # deps missing or DB unreadable
            st.warning(f"Local RAG could not be initialized: {exc}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render the conversation, with retrieved context available on each answer.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("contexts"):
                with st.expander("📚 View Retrieved Textbook Context"):
                    render_contexts(msg["contexts"])

    # New question → retrieve → bundle → stream the tutor's answer.
    if prompt := st.chat_input("Ask about Classical Arabic literature…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving textbook passages…"):
                    contexts = retrieve_context(prompt)
                with st.expander("📚 View Retrieved Textbook Context"):
                    render_contexts(contexts)

                answer = st.write_stream(
                    stream_llm(build_chat_messages(prompt, contexts))
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "contexts": contexts}
                )
            except Exception as exc:
                st.error(
                    f"Inference failed: {exc}\n\n"
                    f"Check that the vLLM server is running at `{VLLM_BASE_URL}`."
                )


if __name__ == "__main__":
    main()
