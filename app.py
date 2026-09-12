"""
CyberLawGPT — A RAG-based assistant for Pakistan's Prevention of
Electronic Crimes Act (PECA), 2016.

Stack: Streamlit (UI) + FAISS (vector search) + Sentence-Transformers
(local, free embeddings) + Groq API (fast LLM inference).

Run locally:      streamlit run app.py
Run on Colab:      see README.md
Deploy to cloud:   see README.md (add GROQ_API_KEY to Secrets)
"""

import os
import re
import pickle
import hashlib
from typing import List, Dict

import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================================
# CONFIG
# ============================================================================
APP_TITLE = "⚖️ CyberLawGPT"
APP_SUBTITLE = "AI Assistant for Pakistan's Prevention of Electronic Crimes Act (PECA), 2016"
APP_VERSION = "1.0.0"

# Google Drive share link -> file id, used to auto-download the source Act on first run
GDRIVE_FILE_ID = "1Py8v_3YVskhHMsG6Rq55kThDIs5ihWCk"

DATA_DIR = "cyberlaw_data"
PDF_PATH = os.path.join(DATA_DIR, "peca_2016.pdf")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "llama3-70b-8192",
]

LEVEL_INSTRUCTIONS = {
    "beginner": (
        "Explain in simple, plain everyday language that a non-lawyer can understand. "
        "Avoid jargon; when you must use a legal term, briefly define it in parentheses."
    ),
    "intermediate": (
        "Explain clearly for an educated general reader (e.g. a student, journalist, or "
        "small business owner). You may use standard legal terminology but briefly "
        "explain any specialised terms."
    ),
    "expert": (
        "Answer with the precision expected by a lawyer or legal professional. Use exact "
        "statutory terminology, section numbers, and a formal legal register, without "
        "simplifying the concepts."
    ),
}

LENGTH_INSTRUCTIONS = {
    "concise": "Keep the answer short and to the point (roughly 3-5 sentences).",
    "balanced": (
        "Give a clear, well-organised answer of moderate length (roughly 1-2 short "
        "paragraphs) with explanation."
    ),
    "detailed": (
        "Give a thorough, comprehensive answer covering relevant sub-clauses, "
        "punishments, exceptions and related provisions where applicable. Use headings "
        "or bullet points if useful."
    ),
}

LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in clear English.",
    "Roman Urdu": (
        "Respond in Roman Urdu (Urdu written using the Latin/English alphabet), while "
        "keeping legal terms and Section numbers in English."
    ),
    "Bilingual": (
        "Respond first in English, then add a short Roman Urdu summary of the key point "
        "at the end."
    ),
}

MAX_TOKENS_MAP = {"concise": 450, "balanced": 900, "detailed": 1700}

SYSTEM_PROMPT_TEMPLATE = """You are CyberLawGPT, a specialised legal-information assistant \
focused strictly on Pakistan's Prevention of Electronic Crimes Act (PECA), 2016, as passed \
by the Majlis-e-Shoora (Parliament).

Rules you must always follow:
1. Answer ONLY using the "CONTEXT" excerpts provided below, which are retrieved directly \
from the official Act. Do not invent sections, penalties, or provisions that are not \
present in the context.
2. Wherever possible, cite the relevant Section number(s) (e.g. "Section 21 - Cyber \
stalking") and the page number of the excerpt you used.
3. If the answer is not covered by the provided context, clearly say that PECA 2016 (as \
indexed here) does not appear to address this, and suggest the user consult a qualified \
lawyer or check for amendments/other applicable laws. Do NOT guess or fabricate.
4. You are not a lawyer and this is not legal advice. Where appropriate, remind the user \
this is general legal information, not a substitute for professional legal counsel.
5. Stay strictly on-topic: Pakistani cyber law / PECA 2016 and closely related legal \
concepts. Politely decline unrelated requests.

Style instructions for this response:
- Technicality level: {level_instruction}
- Length: {length_instruction}
- Language: {language_instruction}
"""

EXAMPLE_QUESTIONS = [
    "What is cyber terrorism under PECA 2016?",
    "What is the punishment for unauthorized access to an information system?",
    "What does the Act say about cyber stalking?",
    "Explain electronic forgery under this law.",
    "What is a 'critical infrastructure information system'?",
]


# ============================================================================
# HELPERS: PDF acquisition, cleaning, chunking, indexing
# ============================================================================
def ensure_pdf() -> str:
    """Download the PECA 2016 PDF (from Google Drive) if not already present,
    with a manual-upload fallback."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
        return PDF_PATH

    download_ok = False
    last_error = None
    try:
        import gdown

        # Try a few call styles for compatibility across gdown versions.
        attempts = [
            lambda: gdown.download(id=GDRIVE_FILE_ID, output=PDF_PATH, quiet=False),
            lambda: gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&export=download",
                PDF_PATH,
                quiet=False,
            ),
            lambda: gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", PDF_PATH, quiet=False
            ),
        ]
        for attempt in attempts:
            try:
                result = attempt()
                if result and os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
                    download_ok = True
                    break
            except Exception as e:
                last_error = e
                continue
    except Exception as e:
        last_error = e

    if not download_ok:
        st.warning(f"Automatic download from Google Drive failed ({last_error}).")

    if not os.path.exists(PDF_PATH) or os.path.getsize(PDF_PATH) < 1000:
        st.warning("Please upload the PECA 2016 PDF manually to continue.")
        uploaded = st.file_uploader("Upload PECA 2016 PDF", type=["pdf"])
        if uploaded is not None:
            with open(PDF_PATH, "wb") as f:
                f.write(uploaded.read())
            st.rerun()
        else:
            st.stop()

    return PDF_PATH


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(pdf_path: str) -> List[tuple]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


def chunk_pages(pages: List[tuple], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict]:
    chunks = []
    for page_num, raw in pages:
        text = clean_text(raw)
        if not text:
            continue
        start, n = 0, len(text)
        while start < n:
            end = min(start + chunk_size, n)
            piece = text[start:end].strip()
            if len(piece) > 40:
                chunks.append({"text": piece, "page": page_num})
            if end == n:
                break
            start = end - overlap
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def build_knowledge_base(pdf_path: str, _file_hash: str):
    """Builds (or loads cached) FAISS index + chunk metadata.
    `_file_hash` is only used as a cache key so this reruns automatically
    if the source PDF changes."""
    embedder = load_embedder()

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks, embedder

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError("No extractable text found in the PDF.")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks, embedder


def retrieve(query: str, index, chunks, embedder, k=4) -> List[Dict]:
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        item = dict(chunks[idx])
        item["score"] = float(score)
        results.append(item)
    return results


# ============================================================================
# HELPERS: prompt building + Groq call
# ============================================================================
def build_messages(query, contexts, level, length, language, history):
    context_block = "\n\n".join(
        f"[Excerpt {i + 1} | Page {c['page']}]\n{c['text']}" for i, c in enumerate(contexts)
    ) or "(no relevant excerpts found)"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        level_instruction=LEVEL_INSTRUCTIONS[level],
        length_instruction=LENGTH_INSTRUCTIONS[length],
        language_instruction=LANGUAGE_INSTRUCTIONS[language],
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append(turn)
    messages.append(
        {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION:\n{query}"}
    )
    return messages


def call_groq(api_key: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_default_api_key() -> str:
    try:
        val = st.secrets.get("GROQ_API_KEY", "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="CyberLawGPT", page_icon="⚖️", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("### 🔑 Groq API Key")
    api_key_input = st.text_input(
        "Enter your Groq API key",
        value=st.session_state.get("groq_api_key", get_default_api_key()),
        type="password",
        help="Free key at https://console.groq.com/keys. You can also set it as "
        "GROQ_API_KEY in Streamlit secrets or an environment variable.",
    )
    st.session_state["groq_api_key"] = api_key_input

    st.markdown("---")
    st.markdown("### 🧠 Model")
    model_name = st.selectbox(
        "Groq model",
        DEFAULT_GROQ_MODELS,
        index=0,
        help="If a model errors as deprecated, pick another or check "
        "console.groq.com/docs/models for the current list.",
    )

    st.markdown("### 🎯 Answer style")
    level_label = st.select_slider(
        "Technicality level",
        options=["🟢 Beginner", "🟡 Intermediate", "🔴 Expert / Legal Professional"],
        value="🟡 Intermediate",
    )
    level = {
        "🟢 Beginner": "beginner",
        "🟡 Intermediate": "intermediate",
        "🔴 Expert / Legal Professional": "expert",
    }[level_label]

    length_label = st.select_slider(
        "Response length",
        options=["✂️ Concise", "📄 Balanced", "📚 Detailed"],
        value="📄 Balanced",
    )
    length = {"✂️ Concise": "concise", "📄 Balanced": "balanced", "📚 Detailed": "detailed"}[
        length_label
    ]

    language = st.selectbox("Response language", ["English", "Roman Urdu", "Bilingual"], index=0)

    st.markdown("### 🔍 Retrieval")
    top_k = st.slider("Number of Act excerpts to retrieve", 2, 10, 4)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    show_sources = st.checkbox("Show cited excerpts", value=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("♻️ Rebuild KB", use_container_width=True):
            for p in [INDEX_PATH, CHUNKS_PATH]:
                if os.path.exists(p):
                    os.remove(p)
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** CyberLawGPT provides general information about the "
        "Prevention of Electronic Crimes Act (PECA), 2016 for educational purposes only. "
        "It is **not legal advice**. Always consult a licensed lawyer for advice on a "
        "specific matter."
    )
    st.caption(f"CyberLawGPT v{APP_VERSION}")

# ---- Header ----
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.info(
    "This assistant answers strictly from the text of PECA 2016 as passed by the "
    "Majlis-e-Shoora (Parliament). It is an educational tool, **not a substitute for "
    "professional legal advice**.",
    icon="⚖️",
)

# ---- Build / load knowledge base ----
with st.spinner("📚 Preparing the PECA 2016 knowledge base (first run only)..."):
    pdf_path = ensure_pdf()
    fhash = file_md5(pdf_path)
    try:
        index, chunks, embedder = build_knowledge_base(pdf_path, fhash)
    except Exception as e:
        st.error(f"Failed to build knowledge base: {e}")
        st.stop()

with st.sidebar:
    st.markdown("### 📊 Knowledge Base")
    n_pages = len(set(c["page"] for c in chunks))
    st.info(f"{len(chunks)} indexed excerpts from {n_pages} pages of the Act.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Example questions (shown before first message) ----
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, ex in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(ex, use_container_width=True):
            st.session_state["pending_query"] = ex

# ---- Render chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            with st.expander("📖 View cited Act excerpts"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']} · similarity {s['score']:.2f}**")
                    st.write(s["text"])
                    st.divider()

# ---- Chat input ----
query = st.chat_input("Ask a question about Pakistan's Cyber Crime Law (PECA 2016)...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    if not st.session_state["groq_api_key"]:
        st.error("Please enter your Groq API key in the sidebar to start chatting.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching PECA 2016 and thinking..."):
            contexts = retrieve(query, index, chunks, embedder, k=top_k)
            best_score = max((c["score"] for c in contexts), default=0.0)

            history_for_model = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            messages = build_messages(query, contexts, level, length, language, history_for_model)

            try:
                answer = call_groq(
                    st.session_state["groq_api_key"],
                    model_name,
                    messages,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS_MAP[length],
                )
            except Exception as e:
                answer = f"⚠️ Error calling Groq API: {e}"

        if best_score < 0.25 and contexts:
            st.caption(
                "⚠️ Low retrieval confidence — this topic may not be directly covered "
                "by PECA 2016 as indexed here."
            )

        st.markdown(answer)
        if contexts and show_sources:
            with st.expander("📖 View cited Act excerpts"):
                for s in contexts:
                    st.markdown(f"**Page {s['page']} · similarity {s['score']:.2f}**")
                    st.write(s["text"])
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": contexts}
    )

# ---- Footer: download transcript ----
if st.session_state.messages:
    transcript = "\n\n".join(
        f"**{m['role'].upper()}:** {m['content']}" for m in st.session_state.messages
    )
    st.download_button(
        "⬇️ Download chat transcript",
        data=transcript,
        file_name="cyberlawgpt_chat.md",
        mime="text/markdown",
    )

st.markdown("---")
st.caption(
    "Built with Streamlit, FAISS, Sentence-Transformers & Groq · Unofficial educational "
    "tool · Not affiliated with the Government of Pakistan."
)
