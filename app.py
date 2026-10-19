
import os

import streamlit as st
from dotenv import load_dotenv

from src.policy_rag import POLICIES_DIR, build_vectorstore, answer_policy_question

load_dotenv()

st.set_page_config(
    page_title="NovaTech Policy Chatbot",
    page_icon="📋",
    layout="centered",
)

st.title("NovaTech Company Policy Chatbot")
st.caption(
    "Ask about leave, conduct, IT security, travel, and expenses. "
    "Answers come from the policy files in `policies/`."
)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
1. Put your OpenAI key in `.env` as `OPENAI_API_KEY`.
2. Add or edit files in `policies/` (Markdown, text, or PDF).
3. Click **Rebuild knowledge** after changing policies.
        """
    )

    if POLICIES_DIR.exists():
        files = sorted(
            p.name for p in POLICIES_DIR.iterdir() if p.is_file()
        )
        st.subheader("Loaded policies")
        for name in files:
            st.write(f"- {name}")

    if st.button("Rebuild knowledge"):
        with st.spinner("Indexing policies..."):
            st.session_state.store = build_vectorstore(force_rebuild=True)
        st.success("Policy index rebuilt.")

api_key = os.getenv("OPENAI_API_KEY", "").strip()

if not api_key or api_key.startswith("your_"):
    st.warning(
        "Add your OpenAI API key to `.env` as `OPENAI_API_KEY`, "
        "then restart the app."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    with st.spinner("Loading policy knowledge..."):
        st.session_state.store = build_vectorstore()

for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Ask a company policy question...")

if question:
    st.session_state.messages.append(("user", question))

    with st.chat_message("user"):
        st.markdown(question)

    history = [
        (role, text)
        for role, text in st.session_state.messages[:-1]
    ]

    normalized_question = question.strip().lower().rstrip("!.?")

    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }

    with st.chat_message("assistant"):
        try:
            if normalized_question in greetings:
                answer = (
                    "Hi! 👋 Welcome to the NovaTech Company Policy Chatbot. "
                    "You can ask me about leave, conduct, IT security, "
                    "travel, and expenses."
                )
            else:
                answer = answer_policy_question(
                    question,
                    history,
                    st.session_state.store,
                )

        except Exception as exc:
            answer = (
                "I could not answer that yet. "
                "Check your API key and policy files.\n\n"
                f"Details: {exc}"
            )

        st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))
