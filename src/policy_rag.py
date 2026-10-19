import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"
INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"

SYSTEM_PROMPT = """You are NovaTech's Company Policy Assistant.
Answer only from the provided policy excerpts.
If the policies do not contain the answer, say you do not know and suggest contacting People Operations.
Quote the policy name when possible.
Keep answers short, clear, and practical.
Do not invent rules, numbers, or exceptions.
"""


def load_policy_documents() -> list[Document]:
    if not POLICIES_DIR.exists():
        raise FileNotFoundError(f"Missing policies folder: {POLICIES_DIR}")

    docs: list[Document] = []
    for path in sorted(POLICIES_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loaded = PyPDFLoader(str(path)).load()
            for doc in loaded:
                doc.metadata["source"] = path.name
            docs.extend(loaded)
        else:
            docs.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
    if not docs:
        raise ValueError("No policy files found. Add .md, .txt, or .pdf files to policies/.")
    return docs


def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    if INDEX_DIR.exists() and not force_rebuild:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(load_policy_documents())
    store = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    return store


def answer_policy_question(question: str, chat_history: list[tuple[str, str]], store: FAISS) -> str:
    retrieved = store.similarity_search(question, k=4)
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in retrieved
    )
    history_text = "\n".join(f"{role.title()}: {text}" for role, text in chat_history[-6:])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Conversation so far:\n{history}\n\nPolicy excerpts:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    response = (prompt | llm).invoke(
        {
            "history": history_text or "None",
            "context": context,
            "question": question,
        }
    )
    sources = sorted({doc.metadata.get("source", "unknown") for doc in retrieved})
    return f"{response.content}\n\n_Sources: {', '.join(sources)}_"
