"""
tests/evaluation/ragas_eval.py
================================
Step 2 of RAGAS evaluation pipeline.

Reads evaluated_answers.json (output of generate_answers.py) and runs
RAGAS evaluation with 4 metrics:

  1. Faithfulness        — Does the answer contain only information from the context?
                           Most critical for a legal/regulatory chatbot.
  2. Answer Relevancy    — Is the answer relevant and complete for the question?
  3. Context Precision   — Are the retrieved chunks relevant (no noisy chunks)?
  4. Context Recall      — Did we retrieve all necessary information?
                           (Requires ground_truth)

Uses Google Gemini as the evaluator LLM (same API key as the main app).

Usage:
    uv run python tests/evaluation/ragas_eval.py
    # or
    make evaluate

Output:
    tests/evaluation/evaluation_results.json  — per-question scores
    tests/evaluation/evaluation_summary.json  — aggregated metric averages
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ragas_eval")

ANSWERS_FILE = Path(__file__).parent / "evaluated_answers.json"
RESULTS_FILE = Path(__file__).parent / "evaluation_results.json"
SUMMARY_FILE = Path(__file__).parent / "evaluation_summary.json"


# ---------------------------------------------------------------------------
# RAGAS imports (deferred to give a clear error message if not installed)
# ---------------------------------------------------------------------------

def _import_ragas():
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        return (
            EvaluationDataset,
            SingleTurnSample,
            evaluate,
            LangchainLLMWrapper,
            LangchainEmbeddingsWrapper,
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
    except ImportError as exc:
        logger.error(
            "RAGAS is not installed. Run `uv add ragas datasets` first.\n%s", exc
        )
        sys.exit(1)


def build_evaluator_llm_and_embeddings():
    """Set up HuggingFace as RAGAS evaluator (reuses the app's HF token)."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

    if not settings.HF_TOKEN:
        logger.error("HF_TOKEN is not set in .env")
        sys.exit(1)

    endpoint = HuggingFaceEndpoint(
        repo_id=settings.LLM_MODEL,
        huggingfacehub_api_token=settings.HF_TOKEN,
        temperature=0.0,  # deterministic for evaluation
        max_new_tokens=1024,
        task="text-generation",
    )
    llm = ChatHuggingFace(llm=endpoint)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, embeddings


def load_answers() -> list[dict]:
    if not ANSWERS_FILE.exists():
        logger.error(
            "Answers file not found: '%s'.\n"
            "Run `make generate-eval-answers` first.",
            ANSWERS_FILE,
        )
        sys.exit(1)
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_ragas_dataset(answers: list[dict], SingleTurnSample, EvaluationDataset):
    """Convert evaluated_answers.json into a RAGAS EvaluationDataset."""
    samples = []
    for item in answers:
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item["contexts"],
                reference=item.get("ground_truth") or None,
            )
        )
    return EvaluationDataset(samples=samples)


def print_results_table(results_df) -> None:
    """Pretty-print per-question scores."""
    metrics = [c for c in results_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    print("\n" + "=" * 80)
    print(f"{'Question':<45} " + "  ".join(f"{m[:15]:>15}" for m in metrics))
    print("-" * 80)
    for _, row in results_df.iterrows():
        question_short = str(row.get("user_input", ""))[:43] + ".."
        scores = "  ".join(
            f"{float(row[m]):>15.3f}" if row[m] is not None else f"{'N/A':>15}"
            for m in metrics
        )
        print(f"{question_short:<45} {scores}")
    print("=" * 80)


def main() -> None:
    logger.info("Loading RAGAS modules...")
    (
        EvaluationDataset,
        SingleTurnSample,
        evaluate,
        LangchainLLMWrapper,
        LangchainEmbeddingsWrapper,
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    ) = _import_ragas()

    logger.info("Loading answers from '%s'...", ANSWERS_FILE)
    answers = load_answers()
    logger.info("Found %d samples to evaluate.", len(answers))

    logger.info("Setting up Gemini as RAGAS evaluator LLM...")
    llm, embeddings = build_evaluator_llm_and_embeddings()
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ------------------------------------------------------------------
    # Define metrics
    # ------------------------------------------------------------------
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    logger.info(
        "Running RAGAS evaluation with metrics: %s",
        [type(m).__name__ for m in metrics],
    )
    logger.info("This may take several minutes (one LLM call per metric per sample)...")

    # ------------------------------------------------------------------
    # Build dataset and evaluate
    # ------------------------------------------------------------------
    dataset = build_ragas_dataset(answers, SingleTurnSample, EvaluationDataset)

    start = time.perf_counter()
    result = evaluate(dataset=dataset, metrics=metrics)
    elapsed = time.perf_counter() - start

    logger.info("Evaluation completed in %.1fs", elapsed)

    # ------------------------------------------------------------------
    # Process and save results
    # ------------------------------------------------------------------
    results_df = result.to_pandas()

    # Print table
    print_results_table(results_df)

    # Per-question results
    metric_names = [type(m).__name__ for m in metrics]
    per_question = []
    for _, row in results_df.iterrows():
        per_question.append(
            {
                "question": row.get("user_input", ""),
                "scores": {
                    name: round(float(row[name]), 4) if row.get(name) is not None else None
                    for name in metric_names
                    if name in row
                },
            }
        )

    # Aggregate summary
    summary = {}
    for name in metric_names:
        col_name = name
        if col_name in results_df.columns:
            values = results_df[col_name].dropna().tolist()
            summary[name] = {
                "mean": round(sum(values) / len(values), 4) if values else None,
                "min": round(min(values), 4) if values else None,
                "max": round(max(values), 4) if values else None,
                "n_samples": len(values),
            }

    # Print summary
    print("\n📊 RAGAS Evaluation Summary — UIT Academic Policies Chatbot")
    print("-" * 50)
    for metric, stats in summary.items():
        mean = stats["mean"]
        bar = "█" * int((mean or 0) * 20) + "░" * (20 - int((mean or 0) * 20))
        print(f"  {metric:<22} {mean:.3f}  [{bar}]")
    print("-" * 50)
    print(f"  Total samples : {len(answers)}")
    print(f"  Eval duration : {elapsed:.1f}s")
    print()

    # Save files
    RESULTS_FILE.write_text(
        json.dumps(per_question, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    SUMMARY_FILE.write_text(
        json.dumps({"metrics": summary, "total_samples": len(answers), "duration_seconds": round(elapsed, 1)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to:")
    logger.info("  Per-question : %s", RESULTS_FILE)
    logger.info("  Summary      : %s", SUMMARY_FILE)


if __name__ == "__main__":
    main()
