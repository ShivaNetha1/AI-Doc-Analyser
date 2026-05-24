# ✨ AI Doc Analyzer — Intelli Doc Analyser

![AI Doc Analyzer Hero](https://img.shields.io/badge/AI%20Doc%20Analyzer-📄%20🔍-blue)

AI Doc Analyzer (aka `Intelli Doc Analyser`) is a lightweight Streamlit-powered application that lets organizations index, search, and extract knowledge from document collections using embeddings and a vector store. Use it to turn large document repositories into searchable, actionable knowledge. 🚀

Live demo: [https://ai-doc-analyser.streamlit.app/](https://ai-doc-analyser.streamlit.app/) ✅

## 🔎 Features
- **Search:** Semantic search across PDFs and text sources.
- **Embeddings:** Uses embedding vectors to find similar content.
- **Retrieval & QA:** Ask natural-language questions over your documents.
- **Extensible:** Modular agents in `agents/` and core utilities in `core/`.

## 🖼️ Preview
Below are a few screenshots from the app showing common responses and follow-up prompts.

![3 Ques Answered Correctly](3%20ques%20answered%20correctly.jpg)

![AI Couldn't Find The Answer](ai%20couldnt%20find%20the%20answer.jpg)

![Follow Up Question](follow%20up%20question.jpg)

## 🚀 Quickstart
Requirements: Python 3.9+ and `pip`.

1. Clone the repo and open the project folder.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your API key(s) to a `.env` file in the project root (example provided):

```
GROQ_API_KEY=your_api_key_here
```

5. Run the app locally:

```bash
streamlit run app.py
```

Open http://localhost:8501 to view the UI.

## 📁 Project Structure
- **app.py**: Streamlit entrypoint and UI glue.
- **agents/**: Modular agents (calculator, retrieval, router, validator).
- **core/**: Embeddings, vectorstore, memory and prompt utilities.
- **utils/**: Helpers and document loaders for PDF/text.
- **zip/**: (zipped exports) — ignored by git via `.gitignore`.

## 🧩 How it helps industry
- **Knowledge discovery:** Quickly find relevant policies, contracts, or product docs.
- **Faster onboarding:** New hires search internal docs without hunting folders.
- **Compliance & audit:** Pinpoint clauses and evidence across many documents.
- **Customer support:** Surface relevant KB articles and reduce resolution time.

## 🔧 How to use (recommended workflow)
1. Place your documents (PDFs, text) in a folder and use the UI to ingest.
2. The system creates embeddings and stores vectors in the configured `vectorstore`.
3. Use the Search/Ask interface to query the corpus conversationally.
4. Export results or snapshots into the `zip/` folder for sharing.

## ♻️ Future Improvements
- Add multi-language support and OCR pre-processing for scanned PDFs.
- Add role-based access control and multi-tenant separation..
- Add automated document change detection and incremental updates.
- Improve UI with result highlighting, source tracing, and drill-down analytics.

## 🤝 Contributing
Contributions welcome — open an issue or a PR. Please follow standard GitHub PR etiquette.

---
