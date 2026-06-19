from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

from app.config.settings import settings


class Generator:
    """
    LLM wrapper using HuggingFace Inference API (Mistral-7B-Instruct).

    - Calls the HF Inference API endpoint (no local GPU needed).
    - ChatHuggingFace wraps HuggingFaceEndpoint to support chat-style
      message formatting (system/human/ai turns) compatible with
      LangChain's ChatPromptTemplate.
    """

    def __init__(self) -> None:
        if not settings.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set. Add it to your .env file.\n"
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.LLM_MODEL,
            huggingfacehub_api_token=settings.HF_TOKEN,
            temperature=settings.LLM_TEMPERATURE,
            max_new_tokens=1024,
            task="text-generation",
        )
        self._llm = ChatHuggingFace(llm=endpoint)

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
