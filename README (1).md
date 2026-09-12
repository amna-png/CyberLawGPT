# ⚖️ CyberLawGPT

**A Retrieval-Augmented Generation (RAG) assistant for Pakistan's Prevention of Electronic Crimes Act (PECA), 2016.**

CyberLawGPT lets you ask natural-language questions ("Can I be jailed for hacking someone's email?", "What's the punishment for cyber stalking?") and answers **strictly grounded in the official text of PECA 2016**, with page-level citations — not from the model's general knowledge.

> ⚠️ **Disclaimer:** This is an educational/informational tool, not a substitute for professional legal advice. Always consult a licensed lawyer for advice on a specific matter. Verify against the official Gazette text for anything consequential.

---

## ✨ Features

- **Grounded answers only** — the LLM is instructed to answer only from retrieved excerpts of the Act, and to say so clearly when a topic isn't covered.
- **Adjustable technicality level** — Beginner / Intermediate / Expert (legal-professional) phrasing.
- **Adjustable response length** — Concise / Balanced / Detailed.
- **Response language** — English / Roman Urdu / Bilingual.
- **Configurable retrieval** — number of excerpts (top-k) and model creativity (temperature).
- **Model choice** — pick between several free Groq-hosted models.
- **Page-level citations** — expand any answer to see exactly which page(s) of the Act it was based on, with similarity scores.
- **Low-confidence warning** — flags answers built on weak retrieval matches.
- **Example questions**, **chat history**, **one-click rebuild of the knowledge base**, and **downloadable chat transcript**.
- **Auto-setup** — downloads the source PDF and builds the FAISS index automatically on first run; cached afterwards.

---

## 🧱 Architecture

```
User question
    │
    ▼
Sentence-Transformers (all-MiniLM-L6-v2) → query embedding
    │
    ▼
FAISS similarity search over PECA 2016 chunks  →  top-k relevant excerpts
    │
    ▼
Prompt = system instructions (style/level/length) + excerpts + question + short history
    │
    ▼
Groq API (Llama 3 / Gemma) → grounded answer with citations
    │
    ▼
Streamlit chat UI
```

**Files (exactly 3, as requested):**

| File               | Purpose                                                   |
|--------------------|------------------------------------------------------------|
| `app.py`           | Full Streamlit app: PDF ingestion, chunking, FAISS index, Groq calls, UI |
| `requirements.txt` | Python dependencies                                       |
| `README.md`        | This file                                                  |

On first run, `app.py` automatically:
1. Downloads the PECA 2016 PDF from Google Drive (falls back to a manual upload widget if the download fails).
2. Extracts text page-by-page (`pypdf`), cleans it, and splits it into ~900-character overlapping chunks.
3. Embeds every chunk locally with `sentence-transformers/all-MiniLM-L6-v2` (no API cost).
4. Builds a `faiss.IndexFlatIP` (cosine similarity via normalized vectors) and caches it to disk (`cyberlaw_data/`) plus in-memory via `st.cache_resource`, so it isn't rebuilt on every rerun.

---

## 🔑 Getting a Groq API key

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys) and sign up (free tier available).
2. Create an API key.
3. Paste it into the sidebar text field when you run the app — **or** set it as an environment variable / Streamlit secret named `GROQ_API_KEY` so you don't have to paste it every time.

If a listed model name is ever deprecated by Groq, just pick a different one from the sidebar dropdown, or check the current list at [console.groq.com/docs/models](https://console.groq.com/docs/models).

> **Note on model access:** Groq's lineup changes over time, and some models are gated to Enterprise-tier keys only (a regular/free API key gets a `404 model_not_found` error when calling them, even though the model still exists). As of this writing, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `groq/compound`, and `groq/compound-mini` are available on the standard developer plan, while `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` currently require an Enterprise plan. If you hit a 404, just switch models in the sidebar.

---

## 💻 Run locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), paste your Groq API key in the sidebar, and start asking questions.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo/branch and set **Main file path** to `app.py`.
3. Before (or after) deploying, open **App → Settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. Deploy. The first load will take a bit longer (downloading the PDF + embedding model + building the index); subsequent visits are fast because of caching.

> 💡 The `requirements.txt` pins the CPU-only PyTorch wheel index so the install stays small and fast enough for the free tier. If you ever hit a resource/build-timeout limit, you can swap `all-MiniLM-L6-v2` for an even smaller embedding model in `app.py` (`EMBED_MODEL_NAME`).

---

## 🧪 Run on Google Colab

```python
# Cell 1 — get the files (either clone your repo or upload the 3 files)
!git clone <your-repo-url> cyberlawgpt
%cd cyberlawgpt

# Cell 2 — install dependencies
!pip install -r requirements.txt -q

# Cell 3 — expose the Streamlit app publicly with localtunnel
!streamlit run app.py --server.headless true --server.port 8501 &>/content/logs.txt &
!npx --yes localtunnel --port 8501
```

`localtunnel` will print a public URL (e.g. `https://xxxx.loca.lt`). Open it, and if prompted for a "tunnel password," it's the output of:

```python
!wget -q -O - https://loca.lt/mytunnelpassword
```

Paste your Groq API key into the sidebar of the opened app and start chatting. (You can alternatively use `pyngrok` instead of `localtunnel` if you prefer.)

---

## ⚙️ Sidebar options explained

| Setting                     | What it does                                                                 |
|------------------------------|-------------------------------------------------------------------------------|
| Groq API key                 | Your key for LLM generation (kept only in session memory, never logged)      |
| Groq model                   | Which hosted LLM answers your question                                       |
| Technicality level           | Beginner / Intermediate / Expert phrasing of the answer                      |
| Response length              | Concise / Balanced / Detailed answer                                         |
| Response language            | English / Roman Urdu / Bilingual                                             |
| Number of Act excerpts (top-k)| How many chunks of the Act are retrieved and shown to the model per question |
| Creativity (temperature)     | Lower = more literal/consistent; higher = more free-form phrasing            |
| Show cited excerpts          | Toggle the "sources" panel under each answer                                 |
| Clear chat                   | Wipes the conversation history                                               |
| Rebuild KB                   | Forces re-download/re-embedding of the Act (useful after updating the PDF)   |

---

## 🗂️ Data source

The app is built on the text of the **Prevention of Electronic Crimes Act, 2016**, as passed by the Majlis-e-Shoora (Parliament) of Pakistan, sourced from the PDF provided for this project. Because the source PDF is a scanned/OCR document, extracted text may occasionally contain minor artefacts — the retrieval step is designed to be resilient to this, but for anything legally significant, please cross-check the cited page(s) against an official gazette copy of the Act.

---

## 📄 License

MIT — use, modify, and redistribute freely. Not affiliated with, endorsed by, or representing the Government of Pakistan or any official body.
