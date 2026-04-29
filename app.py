import csv
import io
import os
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Chatbot PDF RAG"
VECTOR_DB_DIR = Path("vectorstores") / "faiss"
FAISS_INDEX_DIR = VECTOR_DB_DIR / "current_index"


def ensure_directories() -> None:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_text(pdf_docs) -> tuple[str, list[str]]:
    text_parts = []
    names = []

    for pdf in pdf_docs:
        names.append(pdf.name)
        reader = PdfReader(pdf)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")

    return "\n".join(text_parts), names


def get_text_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_text(text)


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )


def build_vector_store(text_chunks: list[str], api_key: str) -> FAISS:
    return FAISS.from_texts(text_chunks, embedding=get_embeddings(api_key))


def persist_vector_store(vector_store: FAISS) -> None:
    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
    vector_store.save_local(str(FAISS_INDEX_DIR))


def format_llm_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        joined = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                joined.append(item["text"])
            else:
                joined.append(str(item))
        return "\n".join(joined)
    return str(content)


def answer_question(question: str, vector_store: FAISS, api_key: str) -> str:
    docs = vector_store.similarity_search(question, k=4)
    if not docs:
        return "Khong tim thay ngu canh phu hop trong tai lieu da nap."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_template(
        """
Ban la tro ly RAG cho tai lieu PDF.
Hay tra loi dua tren context ben duoi.

Yeu cau:
1) Tra loi ro rang, dung trong pham vi context.
2) Neu context khong du thong tin, noi ro ban khong tim thay trong tai lieu.
3) Tra loi bang tieng Viet, ngan gon va co cau truc.

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key,
    )
    response = llm.invoke(prompt.format_messages(context=context, question=question))
    return format_llm_content(response.content).strip()


def build_csv(history: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["timestamp", "question", "answer", "pdf_names"],
    )
    writer.writeheader()
    writer.writerows(history)
    return buffer.getvalue()


def init_state() -> None:
    defaults = {
        "conversation_history": [],
        "vector_store": None,
        "processed_pdf_names": [],
        "index_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_history() -> None:
    for item in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])


def main() -> None:
    load_dotenv()
    ensure_directories()
    init_state()

    st.set_page_config(page_title=APP_TITLE, page_icon=":books:", layout="wide")
    st.title("Chatbot PDF RAG")
    st.caption("Tai lieu duoc vector hoa va luu local trong thu muc vectorstores/faiss")

    with st.sidebar:
        st.header("Cau hinh")
        model_name = st.radio("Model", ["Google AI"], index=0)
        default_api_key = os.getenv("GOOGLE_API_KEY", "")
        api_key = st.text_input("Google API Key", value=default_api_key, type="password")
        st.markdown("Lay API key tai: https://ai.google.dev/")
        st.divider()

        uploaded_pdfs = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )

        process_clicked = st.button("Submit and Process", type="primary")
        clear_chat = st.button("Clear Chat")
        reset_knowledge = st.button("Reset Knowledge Base")

        if clear_chat:
            st.session_state.conversation_history = []
            st.success("Da xoa lich su hoi dap.")

        if reset_knowledge:
            st.session_state.vector_store = None
            st.session_state.processed_pdf_names = []
            st.session_state.index_ready = False
            if FAISS_INDEX_DIR.exists():
                shutil.rmtree(FAISS_INDEX_DIR)
            st.success("Da reset vector database.")

    if model_name != "Google AI":
        st.error("Model hien tai chua duoc ho tro.")
        st.stop()

    if process_clicked:
        if not api_key:
            st.error("Vui long nhap Google API Key truoc khi xu ly.")
        elif not uploaded_pdfs:
            st.error("Vui long upload it nhat 1 file PDF.")
        else:
            with st.spinner("Dang doc PDF, tao embeddings va luu FAISS..."):
                try:
                    text, pdf_names = get_pdf_text(uploaded_pdfs)
                    if not text.strip():
                        st.error("Khong trich xuat duoc text tu PDF. Hay thu file khac.")
                    else:
                        chunks = get_text_chunks(text)
                        if not chunks:
                            st.error("Khong tao duoc text chunks tu PDF.")
                        else:
                            vector_store = build_vector_store(chunks, api_key)
                            persist_vector_store(vector_store)
                            st.session_state.vector_store = vector_store
                            st.session_state.processed_pdf_names = pdf_names
                            st.session_state.index_ready = True
                            st.success("Xu ly thanh cong. Ban co the bat dau dat cau hoi.")
                except Exception as exc:
                    st.exception(exc)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Hoi dap")
        render_history()
        question = st.chat_input("Dat cau hoi ve cac file PDF da xu ly")

        if question:
            if not api_key:
                st.warning("Vui long nhap API key.")
            elif not st.session_state.index_ready or st.session_state.vector_store is None:
                st.warning("Ban can bam Submit and Process truoc khi hoi.")
            else:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Dang truy van vector database..."):
                        answer = answer_question(question, st.session_state.vector_store, api_key)
                    st.write(answer)

                st.session_state.conversation_history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": answer,
                        "pdf_names": ", ".join(st.session_state.processed_pdf_names),
                    }
                )

    with col2:
        st.subheader("Trang thai")
        st.write(f"Index folder: {FAISS_INDEX_DIR}")
        st.write(f"Index san sang: {'Yes' if st.session_state.index_ready else 'No'}")
        st.write(
            "PDF da xu ly: "
            + (
                ", ".join(st.session_state.processed_pdf_names)
                if st.session_state.processed_pdf_names
                else "Chua co"
            )
        )

        if st.session_state.conversation_history:
            csv_data = build_csv(st.session_state.conversation_history)
            st.download_button(
                "Download conversation CSV",
                data=csv_data,
                file_name="conversation_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()