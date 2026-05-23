from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config.settings import settings


class Generator:
    """Thin wrapper around the Google Gemini LLM for answer generation."""

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def generate(self, prompt: ChatPromptTemplate, **kwargs: str) -> str:
        """Render *prompt* with **kwargs and return the model's text response."""
        messages = prompt.format_messages(**kwargs)
        response = self._llm.invoke(messages)
        return self._parse_content(response.content)

    @staticmethod
    def _parse_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            )
        return str(content)
