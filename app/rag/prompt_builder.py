from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """Bạn là trợ lý RAG cho tài liệu PDF.
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


def build_rag_prompt() -> ChatPromptTemplate:
    """Return the configured RAG ChatPromptTemplate."""
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)


def build_memory_context(history: list[dict], max_turns: int = 6) -> str:
    """Serialize recent conversation history into a plain-text context block."""
    if not history:
        return "Không có lịch sử hội thoại trước đó."
    lines = []
    for idx, turn in enumerate(history[-max_turns:], start=1):
        lines.append(f"Lượt {idx} | Người dùng: {turn.get('question', '')}")
        lines.append(f"Lượt {idx} | Trợ lý: {turn.get('answer', '')}")
    return "\n".join(lines)
