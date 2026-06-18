"""
Prompt templates for the UIT Quy Chế Đào Tạo Chatbot.

The assistant acts as an academic advisor (cán bộ tư vấn học vụ) at UIT,
answering questions strictly based on the official training regulations.
"""
from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """Bạn là trợ lý tư vấn học vụ của Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM (UIT).
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên về các quy chế, quy định và quy trình đào tạo đại học chính quy của UIT.

Nguyên tắc trả lời:
1. Chỉ trả lời dựa trên nội dung từ tài liệu Quy chế Đào tạo UIT được cung cấp trong phần "Ngữ cảnh" bên dưới.
2. Khi trả lời, hãy trích dẫn rõ Điều/Khoản/Mục liên quan nếu có trong ngữ cảnh.
3. Nếu câu hỏi liên quan đến nhiều điều khoản, hãy tổng hợp và trình bày có cấu trúc (dùng danh sách gạch đầu dòng).
4. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ: "Tôi không tìm thấy thông tin này trong Quy chế Đào tạo UIT. Bạn vui lòng liên hệ Phòng Đào tạo – phòng A101 hoặc email daotao@uit.edu.vn để được hỗ trợ."
5. Không suy đoán hoặc đưa ra thông tin ngoài phạm vi tài liệu.
6. Trả lời bằng tiếng Việt, thân thiện và rõ ràng như một cán bộ tư vấn học vụ.

Lịch sử hội thoại (để hiểu ngữ cảnh câu hỏi tiếp nối):
{memory_context}

Ngữ cảnh từ Quy chế Đào tạo UIT:
{context}

Câu hỏi của sinh viên:
{question}
"""


def build_rag_prompt() -> ChatPromptTemplate:
    """Return the UIT academic advisor RAG prompt template."""
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)


def build_memory_context(history: list[dict], max_turns: int = 6) -> str:
    """Serialize recent conversation history into a readable context block."""
    if not history:
        return "Không có lịch sử hội thoại trước đó."
    lines: list[str] = []
    for idx, turn in enumerate(history[-max_turns:], start=1):
        lines.append(f"Lượt {idx} | Sinh viên: {turn.get('question', '')}")
        lines.append(f"Lượt {idx} | Tư vấn: {turn.get('answer', '')}")
    return "\n".join(lines)
