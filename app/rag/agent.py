import json
import logging
import re
from typing import Any, Dict, List, Literal, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from app.config.settings import settings
from app.rag.generator import Generator
from app.rag.prompt_builder import (
    build_answer_grader_prompt,
    build_doc_grader_prompt,
    build_hallucination_grader_prompt,
    build_memory_context,
    build_query_rewriter_prompt,
    build_rag_prompt,
    build_router_prompt,
)
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from LLM raw text, handling code blocks."""
    if not text:
        return {}
    cleaned = text.strip()
    # Remove markdown formatting
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    # Try to grab the JSON object substring
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except Exception as e:
        logger.warning("Failed to parse JSON: %s, raw text: %s", e, text)
        return {}


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    history: List[dict]
    memory_context: str
    documents: List[Document]
    generation: str
    sources: str
    route: str             # "chat" | "rag" | "article"
    is_relevant: str       # "yes" | "no"
    is_grounded: str       # "yes" | "no"
    is_useful: str         # "yes" | "no"
    loop_count: int


class UITAcademicAgent:
    """
    LangGraph-based Agentic RAG system for UIT Training Regulations.
    It routes, rewrites queries, retrieves, grades documents, and self-checks for hallucinations.
    """

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

        # Build prompt templates
        self.router_prompt = build_router_prompt()
        self.query_rewriter_prompt = build_query_rewriter_prompt()
        self.doc_grader_prompt = build_doc_grader_prompt()
        self.hallucination_grader_prompt = build_hallucination_grader_prompt()
        self.answer_grader_prompt = build_answer_grader_prompt()
        self.rag_prompt = build_rag_prompt()

        # Construct state graph
        self.workflow = self._create_workflow()

    def run(self, question: str, history: List[dict]) -> Dict[str, Any]:
        """Run the agent workflow for a given question and history."""
        # Clean history format if needed
        memory_context = build_memory_context(history, max_turns=settings.MAX_HISTORY_TURNS)
        initial_state = AgentState(
            question=question,
            rewritten_question=question,
            history=history,
            memory_context=memory_context,
            documents=[],
            generation="",
            sources="",
            route="",
            is_relevant="",
            is_grounded="",
            is_useful="",
            loop_count=0,
        )

        app = self.workflow.compile()
        # Enable Langfuse tracing via configuration if callback is passed
        config = {}

        # Execute Graph
        final_state = app.invoke(initial_state, config)
        return {
            "answer": final_state.get("generation"),
            "sources": final_state.get("sources"),
            "route": final_state.get("route"),
        }

    # ------------------------------------------------------------------
    # Graph Nodes
    # ------------------------------------------------------------------

    def route_question(self, state: AgentState) -> Dict[str, Any]:
        """Classify the user question into 'chat', 'article', or 'rag'."""
        logger.info("--- NODE: ROUTE QUESTION ---")
        question = state["question"]
        memory_context = state["memory_context"]

        # Invoke Router LLM
        response = self.generator.generate(
            self.router_prompt,
            question=question,
            memory_context=memory_context,
        )
        parsed = parse_json_from_text(response)
        choice = parsed.get("choice", "rag").lower()
        if choice not in ["chat", "article", "rag"]:
            choice = "rag"

        logger.info("   -> Routed to: %s", choice)
        return {"route": choice}

    def chat_direct(self, state: AgentState) -> Dict[str, Any]:
        """Generate a response directly for general chitchat."""
        logger.info("--- NODE: CHAT DIRECT ---")
        question = state["question"]
        memory_context = state["memory_context"]

        # Fast direct answer template
        chat_prompt = ChatPromptTemplate.from_template(
            "Bạn là trợ lý tư vấn học vụ UIT thân thiện. Hãy trả lời câu hỏi xã giao này một cách ngắn gọn, tự nhiên:\n"
            "Lịch sử hội thoại:\n{memory_context}\n\n"
            "Câu hỏi sinh viên: {question}"
        )
        response = self.generator.generate(
            chat_prompt,
            question=question,
            memory_context=memory_context,
        )
        return {"generation": response, "sources": ""}

    def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """Optimize user query for vector database keyword search."""
        logger.info("--- NODE: REWRITE QUERY ---")
        question = state["question"]
        memory_context = state["memory_context"]

        response = self.generator.generate(
            self.query_rewriter_prompt,
            question=question,
            memory_context=memory_context,
        )
        parsed = parse_json_from_text(response)
        rewritten = parsed.get("query", question)
        logger.info("   -> Rewritten query: '%s'", rewritten)
        return {"rewritten_question": rewritten}

    def retrieve_docs(self, state: AgentState) -> Dict[str, Any]:
        """Fetch matching documents using Hybrid Search."""
        logger.info("--- NODE: RETRIEVE DOCS ---")
        rewritten_question = state["rewritten_question"]
        docs = self.retriever.retrieve(rewritten_question)
        logger.info("   -> Retrieved %d doc chunks", len(docs))
        return {"documents": docs}

    def retrieve_article(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve direct content for a specific article."""
        logger.info("--- NODE: RETRIEVE ARTICLE ---")
        question = state["question"]

        # Try to parse the article number (e.g. "Điều 15" -> "15")
        match = re.search(r"Điều\s+(\d+)", question, re.IGNORECASE)
        if not match:
            # Fallback regex for pure numbers
            match = re.search(r"\b(\d+)\b", question)

        article_number = match.group(1) if match else "1"
        docs = self.retriever.search_by_article(article_number)

        content = self.retriever.format_article_results(docs, article_number)
        sources = self.retriever.format_sources(docs) if docs else ""

        return {"generation": content, "sources": sources}

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate if retrieved documents are relevant to the query."""
        logger.info("--- NODE: GRADE DOCUMENTS ---")
        question = state["rewritten_question"]
        documents = state["documents"]

        filtered_docs = []
        is_relevant = "no"

        for doc in documents:
            response = self.generator.generate(
                self.doc_grader_prompt,
                question=question,
                document=doc.page_content,
            )
            parsed = parse_json_from_text(response)
            grade = parsed.get("is_relevant", "no").lower()
            if grade == "yes":
                filtered_docs.append(doc)
                is_relevant = "yes"

        logger.info("   -> Grading result: %s (%d/%d relevant chunks)", is_relevant, len(filtered_docs), len(documents))
        return {"documents": filtered_docs, "is_relevant": is_relevant}

    def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        """Generate final answer using retrieved context."""
        logger.info("--- NODE: GENERATE ANSWER ---")
        question = state["question"]
        memory_context = state["memory_context"]
        documents = state["documents"]

        if not documents:
            fallback = (
                "Tôi không tìm thấy thông tin liên quan trong Quy chế Đào tạo UIT. "
                "Bạn vui lòng liên hệ Phòng Đào tạo – phòng A101 hoặc email daotao@uit.edu.vn để được hỗ trợ."
            )
            return {"generation": fallback, "sources": ""}

        context = self.retriever.format_context(documents)
        sources = self.retriever.format_sources(documents)

        response = self.generator.generate(
            self.rag_prompt,
            context=context,
            question=question,
            memory_context=memory_context,
        )
        return {"generation": response, "sources": sources}

    def grade_hallucination(self, state: AgentState) -> Dict[str, Any]:
        """Determine if generated response is grounded in retrieved documents."""
        logger.info("--- NODE: GRADE HALLUCINATION ---")
        documents = state["documents"]
        generation = state["generation"]

        if not documents:
            return {"is_grounded": "yes"}

        context = self.retriever.format_context(documents)
        response = self.generator.generate(
            self.hallucination_grader_prompt,
            context=context,
            generation=generation,
        )
        parsed = parse_json_from_text(response)
        is_grounded = parsed.get("is_grounded", "yes").lower()
        logger.info("   -> Grounded: %s", is_grounded)
        return {"is_grounded": is_grounded}

    def grade_answer(self, state: AgentState) -> Dict[str, Any]:
        """Determine if generated response solves the original user question."""
        logger.info("--- NODE: GRADE ANSWER ---")
        question = state["question"]
        generation = state["generation"]

        response = self.generator.generate(
            self.answer_grader_prompt,
            question=question,
            generation=generation,
        )
        parsed = parse_json_from_text(response)
        is_useful = parsed.get("is_useful", "yes").lower()
        logger.info("   -> Useful answer: %s", is_useful)
        return {"is_useful": is_useful}

    # ------------------------------------------------------------------
    # Graph Construction Helper
    # ------------------------------------------------------------------

    def _create_workflow(self) -> StateGraph:
        """Define state graph structure, nodes, edges, and transitions."""
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("chat_direct", self.chat_direct)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve_docs", self.retrieve_docs)
        workflow.add_node("retrieve_article", self.retrieve_article)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("grade_hallucination", self.grade_hallucination)
        workflow.add_node("grade_answer", self.grade_answer)

        # Set Entry Point
        workflow.set_entry_point("route_question")

        # Define Conditional Router Edge
        workflow.add_conditional_edges(
            "route_question",
            lambda state: state["route"],
            {
                "chat": "chat_direct",
                "article": "retrieve_article",
                "rag": "rewrite_query",
            }
        )

        # Terminal Edges
        workflow.add_edge("chat_direct", END)
        workflow.add_edge("retrieve_article", END)

        # Main Flow Edges
        workflow.add_edge("rewrite_query", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "grade_documents")

        # Grader Decisions
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_after_doc_grading,
            {
                "generate": "generate_answer",
                "retry_rewrite": "rewrite_query",
                "stop_fallback": "generate_answer",
            }
        )

        workflow.add_conditional_edges(
            "generate_answer",
            lambda state: "grade" if state["documents"] else "stop",
            {
                "grade": "grade_hallucination",
                "stop": END,
            }
        )

        workflow.add_conditional_edges(
            "grade_hallucination",
            self._decide_after_hallucination_grading,
            {
                "useful_test": "grade_answer",
                "regenerate": "generate_answer",
                "stop": END,
            }
        )

        workflow.add_conditional_edges(
            "grade_answer",
            self._decide_after_answer_grading,
            {
                "useful": END,
                "retry_search": "rewrite_query",
                "stop": END,
            }
        )

        return workflow

    # ------------------------------------------------------------------
    # Conditional Edges Logic
    # ------------------------------------------------------------------

    def _decide_after_doc_grading(self, state: AgentState) -> Literal["generate", "retry_rewrite", "stop_fallback"]:
        """Decide what to do after checking documents relevance."""
        if state["is_relevant"] == "yes":
            return "generate"

        loop_count = state.get("loop_count", 0)
        if loop_count < 1:
            logger.info("   -> [Doc Grader] No relevant docs found. Retrying query rewrite...")
            state["loop_count"] = loop_count + 1
            return "retry_rewrite"

        logger.info("   -> [Doc Grader] No relevant docs found and max retry reached. Proceeding to fallback answer.")
        return "stop_fallback"

    def _decide_after_hallucination_grading(self, state: AgentState) -> Literal["useful_test", "regenerate", "stop"]:
        """Decide what to do after checking hallucination."""
        if state["is_grounded"] == "yes":
            return "useful_test"

        loop_count = state.get("loop_count", 0)
        if loop_count < 2:
            logger.info("   -> [Hallucination Grader] Response not grounded! Regenerating... (Retry count: %d)", loop_count + 1)
            state["loop_count"] = loop_count + 1
            return "regenerate"

        logger.info("   -> [Hallucination Grader] Max retries reached. Stopping.")
        return "stop"

    def _decide_after_answer_grading(self, state: AgentState) -> Literal["useful", "retry_search", "stop"]:
        """Decide what to do after checking if answer is useful."""
        if state["is_useful"] == "yes":
            return "useful"

        loop_count = state.get("loop_count", 0)
        if loop_count < 2:
            logger.info("   -> [Answer Grader] Answer not useful. Retrying search flow... (Retry count: %d)", loop_count + 1)
            state["loop_count"] = loop_count + 1
            return "retry_search"

        logger.info("   -> [Answer Grader] Max retries reached. Stopping.")
        return "stop"
