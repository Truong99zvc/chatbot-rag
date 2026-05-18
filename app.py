import csv
import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Chatbot PDF RAG"
VECTOR_DB_DIR = Path("vectorstores") / "faiss"
FAISS_INDEX_DIR = VECTOR_DB_DIR / "current_index"
MEMORY_FILE = VECTOR_DB_DIR / "conversation_memory.json"


def ensure_directories() -> None:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def load_conversation_memory() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    cleaned_history = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cleaned_history.append(
            {
                "timestamp": str(item.get("timestamp", "")),
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "sources": str(item.get("sources", "")),
                "pdf_names": str(item.get("pdf_names", "")),
            }
        )
    return cleaned_history


def persist_conversation_memory(history: list[dict]) -> None:
    MEMORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_pdf_documents(pdf_docs) -> tuple[list[Document], list[str]]:
    """Extract text from PDFs, returning LangChain Documents with page metadata."""
    documents = []
    names = []
    for pdf in pdf_docs:
        names.append(pdf.name)
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf.name, "page": page_num},
                    )
                )
    return documents, names


def get_document_chunks(documents: list[Document]) -> list[Document]:
    """Split Documents into smaller chunks, preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(documents)


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


def build_vector_store(chunks: list[Document], api_key: str) -> FAISS:
    return FAISS.from_documents(chunks, embedding=get_embeddings(api_key))


def persist_vector_store(vector_store: FAISS) -> None:
    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
    vector_store.save_local(str(FAISS_INDEX_DIR))


def load_vector_store(api_key: str) -> FAISS | None:
    """Load existing FAISS index from disk if available."""
    if not FAISS_INDEX_DIR.exists():
        return None
    try:
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings=get_embeddings(api_key),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


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


def build_memory_context(history: list[dict], max_turns: int = 6) -> str:
    if not history:
        return "Không có lịch sử hội thoại trước đó."
    recent_turns = history[-max_turns:]
    lines = []
    for idx, turn in enumerate(recent_turns, start=1):
        lines.append(f"Lượt {idx} | Người dùng: {turn.get('question', '')}")
        lines.append(f"Lượt {idx} | Trợ lý: {turn.get('answer', '')}")
    return "\n".join(lines)


def format_sources(docs: list[Document]) -> str:
    """Format unique source citations from retrieved documents."""
    seen: set[tuple] = set()
    lines = []
    for doc in docs:
        source = doc.metadata.get("source", "Không rõ")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"- **{source}**, trang {page}")
    return "\n".join(lines)


def answer_question(
    question: str,
    vector_store: FAISS,
    api_key: str,
    history: list[dict],
) -> tuple[str, str]:
    """Returns (answer, sources_markdown)."""
    docs = vector_store.similarity_search(question, k=4)
    if not docs:
        return "Không tìm thấy ngữ cảnh phù hợp trong tài liệu đã nạp.", ""

    context = "\n\n".join(doc.page_content for doc in docs)
    memory_context = build_memory_context(history)
    sources_text = format_sources(docs)

    prompt = ChatPromptTemplate.from_template(
        """Bạn là trợ lý RAG cho tài liệu PDF.
Hãy trả lời dựa trên ngữ cảnh bên dưới và xem lịch sử hội thoại để giữ mạch hỏi đáp.

Yêu cầu:
1. Trả lời rõ ràng, đúng trong phạm vi ngữ cảnh.
2. Nếu ngữ cảnh không đủ thông tin, có thể dùng lịch sử hội thoại để hiểu rõ câu hỏi tiếp nối.
3. Nếu vẫn không đủ thông tin, nói rõ bạn không tìm thấy trong tài liệu.
4. Trả lời bằng tiếng Việt, ngắn gọn và có cấu trúc.

Lịch sử hội thoại:
{memory_context}

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=api_key,
    )
    response = llm.invoke(
        prompt.format_messages(
            context=context,
            question=question,
            memory_context=memory_context,
        )
    )
    answer = format_llm_content(response.content).strip()
    return answer, sources_text


def build_csv(history: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["timestamp", "question", "answer", "sources", "pdf_names"],
    )
    writer.writeheader()
    writer.writerows(history)
    return buffer.getvalue()


def init_state() -> None:
    memory_history = load_conversation_memory()
    defaults = {
        "conversation_history": memory_history,
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
            if item.get("sources"):
                with st.expander("📄 Nguồn tham khảo"):
                    st.markdown(item["sources"])


def main() -> None:
    load_dotenv()
    ensure_directories()
    init_state()

    st.set_page_config(page_title=APP_TITLE, page_icon=":books:", layout="wide")
    st.title("📚 Chatbot PDF RAG")
    st.caption("Tài liệu được vector hóa và lưu local trong thư mục vectorstores/faiss")

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        default_api_key = os.getenv("GOOGLE_API_KEY", "")
        api_key = st.text_input("Google API Key", value=default_api_key, type="password")
        st.markdown("Lấy API key tại: https://ai.google.dev/")
        st.divider()

        uploaded_pdfs = st.file_uploader(
            "📂 Upload PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )

        process_clicked = st.button("▶️ Submit and Process", type="primary")
        load_existing = st.button("🔄 Load Existing Index")
        clear_chat = st.button("🗑️ Clear Chat")
        reset_knowledge = st.button("⚠️ Reset Knowledge Base")

        if clear_chat:
            st.session_state.conversation_history = []
            persist_conversation_memory(st.session_state.conversation_history)
            st.success("Đã xóa lịch sử hỏi đáp.")

        if reset_knowledge:
            st.session_state.vector_store = None
            st.session_state.processed_pdf_names = []
            st.session_state.index_ready = False
            if FAISS_INDEX_DIR.exists():
                shutil.rmtree(FAISS_INDEX_DIR)
            st.success("Đã reset vector database.")

        if load_existing:
            if not api_key:
                st.error("Vui lòng nhập Google API Key trước.")
            else:
                with st.spinner("Đang tải FAISS index từ disk..."):
                    vs = load_vector_store(api_key)
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.index_ready = True
                    st.success("Đã tải index thành công.")
                else:
                    st.error("Không tìm thấy index đã lưu. Hãy upload và process PDF trước.")

    if process_clicked:
        if not api_key:
            st.error("Vui lòng nhập Google API Key trước khi xử lý.")
        elif not uploaded_pdfs:
            st.error("Vui lòng upload ít nhất 1 file PDF.")
        else:
            with st.spinner("Đang đọc PDF, tạo embeddings và lưu FAISS..."):
                try:
                    documents, pdf_names = get_pdf_documents(uploaded_pdfs)
                    if not documents:
                        st.error("Không trích xuất được text từ PDF. Hãy thử file khác.")
                    else:
                        chunks = get_document_chunks(documents)
                        if not chunks:
                            st.error("Không tạo được text chunks từ PDF.")
                        else:
                            vector_store = build_vector_store(chunks, api_key)
                            persist_vector_store(vector_store)
                            st.session_state.vector_store = vector_store
                            st.session_state.processed_pdf_names = pdf_names
                            st.session_state.index_ready = True
                            st.success(
                                f"Xử lý thành công {len(chunks)} chunks từ {len(pdf_names)} file."
                            )
                except Exception as exc:
                    st.exception(exc)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("💬 Hỏi đáp")
        render_history()
        question = st.chat_input("Đặt câu hỏi về các file PDF đã xử lý")

        if question:
            if not api_key:
                st.warning("Vui lòng nhập API key.")
            elif not st.session_state.index_ready or st.session_state.vector_store is None:
                st.warning("Bạn cần bấm Submit and Process (hoặc Load Existing Index) trước khi hỏi.")
            else:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Đang truy vấn vector database..."):
                        answer, sources = answer_question(
                            question,
                            st.session_state.vector_store,
                            api_key,
                            st.session_state.conversation_history,
                        )
                    st.write(answer)
                    if sources:
                        with st.expander("📄 Nguồn tham khảo"):
                            st.markdown(sources)

                st.session_state.conversation_history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "pdf_names": ", ".join(st.session_state.processed_pdf_names),
                    }
                )
                persist_conversation_memory(st.session_state.conversation_history)

    with col2:
        st.subheader("📊 Trạng thái")
        st.write(f"Index folder: `{FAISS_INDEX_DIR}`")
        index_exists = FAISS_INDEX_DIR.exists()
        st.write(f"Index trên disk: {'✅ Có' if index_exists else '❌ Chưa có'}")
        st.write(
            f"Index đang dùng: {'✅ Sẵn sàng' if st.session_state.index_ready else '⏳ Chưa load'}"
        )
        st.write(
            "PDF đã xử lý: "
            + (
                ", ".join(st.session_state.processed_pdf_names)
                if st.session_state.processed_pdf_names
                else "Chưa có"
            )
        )

        if st.session_state.conversation_history:
            csv_data = build_csv(st.session_state.conversation_history)
            st.download_button(
                "⬇️ Download conversation CSV",
                data=csv_data,
                file_name="conversation_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
