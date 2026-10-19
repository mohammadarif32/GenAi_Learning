# NovaTech Company Policy Chatbot

A Streamlit chatbot that answers questions from company policy documents using retrieval-augmented generation (RAG).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
streamlit run app.py
```

## Customize policies

Edit or add files in `policies/` (`.md`, `.txt`, or `.pdf`), then click **Rebuild knowledge** in the sidebar.
