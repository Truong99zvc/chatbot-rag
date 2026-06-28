import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.documents import Document

from app.database.connection import Base
from app.database.models import ChatSession, ChatTurn
from app.rag.agent import UITAcademicAgent, parse_json_from_text
from app.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Database Tests
# ---------------------------------------------------------------------------

def test_db_models_creation():
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        # Create session
        session = ChatSession(session_id="test-session-123")
        db.add(session)
        db.commit()
        
        # Add turn
        turn = ChatTurn(
            session_id="test-session-123",
            question="Xin chào?",
            answer="Chào bạn, tôi có thể giúp gì cho bạn?",
            sources="- quy_che.pdf, trang 1"
        )
        db.add(turn)
        db.commit()
        
        # Query and verify
        db_session = db.query(ChatSession).filter(ChatSession.session_id == "test-session-123").first()
        assert db_session is not None
        assert len(db_session.turns) == 1
        assert db_session.turns[0].question == "Xin chào?"
        assert db_session.turns[0].sources == "- quy_che.pdf, trang 1"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# JSON Parsing Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"choice": "chat"}', {"choice": "chat"}),
        ('```json\n{"query": "hello"}\n```', {"query": "hello"}),
        ('Some pre-text\n{\n  "is_relevant": "yes"\n}\nSome post-text', {"is_relevant": "yes"}),
        ('invalid json text', {}),
        ('', {}),
    ]
)
def test_parse_json_from_text(text, expected):
    assert parse_json_from_text(text) == expected


# ---------------------------------------------------------------------------
# Agent Flow Tests (Mocked)
# ---------------------------------------------------------------------------

def test_agent_routing_chat():
    mock_retriever = MagicMock(spec=Retriever)
    mock_generator = MagicMock()
    
    # Configure generator responses
    # First call: router -> choice: chat
    # Second call: chat_direct -> "Chào em"
    mock_generator.generate.side_effect = [
        '{"choice": "chat"}',
        "Chào em, anh giúp gì được cho em?"
    ]
    
    agent = UITAcademicAgent(mock_retriever, mock_generator)
    result = agent.run("hello", history=[])
    
    assert result["answer"] == "Chào em, anh giúp gì được cho em?"
    assert result["sources"] == ""
    assert result["route"] == "chat"


def test_agent_routing_rag_success():
    mock_retriever = MagicMock(spec=Retriever)
    mock_generator = MagicMock()
    
    # Mock retriever to return some documents
    docs = [Document(page_content="Điều 15. Điều kiện xét tốt nghiệp đại học chính quy UIT.", metadata={"source": "q.pdf", "page": 1})]
    mock_retriever.retrieve.return_value = docs
    mock_retriever.format_context.return_value = "Điều 15. Điều kiện..."
    mock_retriever.format_sources.return_value = "- q.pdf, trang 1"
    
    # Configure generator calls for the Agent Graph:
    # 1. Router -> RAG
    # 2. Query Rewriter -> "điều kiện tốt nghiệp"
    # 3. Document Grader -> is_relevant: yes
    # 4. Generator -> "Cần 130 tín chỉ"
    # 5. Hallucination Grader -> is_grounded: yes
    # 6. Answer Grader -> is_useful: yes
    mock_generator.generate.side_effect = [
        '{"choice": "rag"}',
        '{"query": "điều kiện tốt nghiệp"}',
        '{"is_relevant": "yes"}',
        "Cần tích lũy đủ số tín chỉ quy định.",
        '{"is_grounded": "yes"}',
        '{"is_useful": "yes"}'
    ]
    
    agent = UITAcademicAgent(mock_retriever, mock_generator)
    result = agent.run("ra trường cần gì", history=[])
    
    assert result["answer"] == "Cần tích lũy đủ số tín chỉ quy định."
    assert result["sources"] == "- q.pdf, trang 1"
    assert result["route"] == "rag"


# ---------------------------------------------------------------------------
# Retriever Hybrid Search test
# ---------------------------------------------------------------------------

def test_hybrid_search_deduplication():
    # Setup mock store and mock docstore dict
    mock_store = MagicMock()
    doc1 = Document(page_content="Văn bản A", metadata={"source": "file.pdf", "page": 1})
    doc2 = Document(page_content="Văn bản B", metadata={"source": "file.pdf", "page": 2})
    
    # Return doc1 & doc2 on vector search
    mock_store.similarity_search.return_value = [doc1, doc2]
    mock_store.docstore._dict = {"1": doc1, "2": doc2}
    
    # Create retriever
    retriever = Retriever(mock_store)
    
    # Mock BM25 using a plain MagicMock to avoid Pydantic __setattr__ issues
    mock_bm25 = MagicMock()
    mock_bm25.invoke.return_value = [doc2, doc1]
    retriever._bm25 = mock_bm25
        
    results = retriever.retrieve("truy vấn test", k=5)
    
    # Verify results are deduplicated
    assert len(results) == 2
    assert results[0].page_content == "Văn bản A"
    assert results[1].page_content == "Văn bản B"
