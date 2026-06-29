"""
RAG evaluation script (Point E in the study guide).

Runs the hand-built test set through the REAL retrieval + reranking +
generation pipeline (not a mock), then scores the results using RAGAS:

- context_precision: of the retrieved chunks, how many were relevant?
- context_recall:    did retrieval surface what was needed to answer fully?
- faithfulness:      does the generated answer's claims trace back to the
                      retrieved context (hallucination check)?
- answer_relevancy:  does the generated answer actually address the question?

Run with:  python evaluate_rag.py
Requires: a real uploaded "Evolution of Man" PDF already ingested for
the USER_ID below (see eval_dataset.py for the questions this assumes).
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.services.retrieval import hybrid_search
from app.services.reranker import rerank
from app.services.llm import generate_answer
from eval_dataset import EVAL_QUESTIONS

# Use the same real user_id whose document was used for manual testing.
USER_ID = "60f8e992-980d-4b69-bd06-e0a4b44005c6"


def run_pipeline_for_eval(question: str) -> dict:
    """
    Runs the REAL retrieval -> rerank -> generation pipeline for one
    question, returning what RAGAS needs: the answer text and the
    raw context strings used to produce it.
    """
    candidates = hybrid_search(query=question, user_id=USER_ID, limit=20)
    top_chunks = rerank(query=question, chunks=candidates, top_n=5)

    answer = generate_answer(query=question, chunks=top_chunks)
    contexts = [c["chunk_text"] for c in top_chunks]

    return {"answer": answer, "contexts": contexts}


def build_eval_dataset() -> Dataset:
    """
    Runs every question in EVAL_QUESTIONS through the real pipeline,
    and assembles the result into the format RAGAS expects.
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in EVAL_QUESTIONS:
        print(f"Running: {item['question']}")
        result = run_pipeline_for_eval(item["question"])

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def main():
    print("Building evaluation dataset by running the real pipeline...\n")
    dataset = build_eval_dataset()

    print("\nRunning RAGAS evaluation (this calls the LLM judge for each metric)...\n")

    # RAGAS needs an LLM to act as the "judge" for faithfulness/relevancy,
    # and an embedding model for semantic comparisons — we reuse Groq and
    # a free HuggingFace embedding model, so this stays free, same as the
    # rest of the project.
    judge_llm = LangchainLLMWrapper(ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ))

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n=== EVALUATION RESULTS ===\n")
    print(results)

    df = results.to_pandas()
    df.to_csv("eval_results.csv", index=False)
    print("\nFull per-question results saved to eval_results.csv")


if __name__ == "__main__":
    main()