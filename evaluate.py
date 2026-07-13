"""
RAG evaluation harness.

Measures the two stages of the pipeline separately against golden_dataset.json:

Stage 1 — Retrieval (cheap, no LLM):
  - recall@k: was the gold chunk in the top-k results?
  - relevance score distributions for answerable vs unanswerable questions,
    used to calibrate the guardrail threshold in agents.py.

Stage 2 — Generation (costs LLM calls, opt-in via --generation):
  - correctness: does the agent's answer contain the gold fact?
  - faithfulness: LLM-as-judge — is the answer supported by the retrieved context?
  - guardrail check: does the agent refuse unanswerable questions?

Usage:
    python evaluate.py                # retrieval eval only
    python evaluate.py --generation   # full eval (runs the agent + judge)
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from retriever import retrieve_with_scores

TOP_K = 5


def load_dataset(path="golden_dataset.json"):
    with open(path) as f:
        return json.load(f)


def eval_retrieval(dataset):
    """Recall@k + relevance score distributions. Returns suggested threshold."""
    print("=" * 70)
    print(f"STAGE 1 — RETRIEVAL EVALUATION (recall@{TOP_K})")
    print("=" * 70)

    hits = 0
    answerable_best_scores = []
    for case in dataset["answerable"]:
        results = retrieve_with_scores(case["question"], top_k=TOP_K)
        best = max(score for _, score in results) if results else 0.0
        answerable_best_scores.append(best)
        found = any(
            case["gold_chunk_keyword"].lower() in doc.page_content.lower()
            for doc, _ in results
        )
        hits += found
        status = "HIT " if found else "MISS"
        print(f"  [{status}] best_score={best:.3f}  {case['question']}")

    recall = hits / len(dataset["answerable"])
    print(f"\n  Recall@{TOP_K}: {hits}/{len(dataset['answerable'])} = {recall:.0%}")

    print("\n" + "-" * 70)
    print("GUARDRAIL CALIBRATION — score distributions")
    print("-" * 70)

    unanswerable_best_scores = []
    for case in dataset["unanswerable"]:
        results = retrieve_with_scores(case["question"], top_k=TOP_K)
        best = max(score for _, score in results) if results else 0.0
        unanswerable_best_scores.append(best)
        print(f"  [OFF-TOPIC] best_score={best:.3f}  {case['question']}")

    lo_answerable = min(answerable_best_scores)
    hi_unanswerable = max(unanswerable_best_scores)
    print(f"\n  Answerable   best-score range: "
          f"{lo_answerable:.3f} – {max(answerable_best_scores):.3f}")
    print(f"  Unanswerable best-score range: "
          f"{min(unanswerable_best_scores):.3f} – {hi_unanswerable:.3f}")

    if lo_answerable > hi_unanswerable:
        threshold = (lo_answerable + hi_unanswerable) / 2
        print(f"\n  ✅ Distributions separate cleanly."
              f" Suggested RELEVANCE_THRESHOLD = {threshold:.2f}")
    else:
        threshold = hi_unanswerable
        print(f"\n  ⚠️ Distributions overlap — threshold {threshold:.2f} favours"
              f" recall; some off-topic queries may pass the guardrail.")
    return threshold


def judge_faithfulness(llm, answer, context):
    """LLM-as-judge: is every claim in the answer supported by the context?"""
    from langchain_core.messages import HumanMessage

    prompt = f"""You are a strict evaluator. Below is a Context (retrieved document
excerpts) and an Answer produced by a RAG system.

Context:
{context}

Answer:
{answer}

Is every factual claim in the Answer supported by the Context? Statements like
"this is not in the documents" count as SUPPORTED. Reply with exactly one word:
SUPPORTED or UNSUPPORTED."""
    result = llm.invoke([HumanMessage(content=prompt)])
    return "UNSUPPORTED" not in result.content.upper()


def query_with_backoff(agent, question, max_attempts=2, wait_s=20):
    """Run agent.query, retrying on rate-limit errors so infra noise
    doesn't pollute the quality metrics."""
    import time

    t0 = time.time()
    for attempt in range(max_attempts):
        answer = agent.query(question)["answer"]
        if "429" not in answer and "Rate limit" not in answer:
            print(f"      ({time.time() - t0:.1f}s)", flush=True)
            return answer
        if attempt < max_attempts - 1:
            print(f"      (rate limited, waiting {wait_s}s...)", flush=True)
            time.sleep(wait_s)
    print(f"      ({time.time() - t0:.1f}s, still rate limited)", flush=True)
    return answer


def eval_generation(dataset):
    """Run the full agent on the golden set; judge correctness + faithfulness."""
    from agents import AgenticRAG

    # Models are env-overridable so the eval runs with whichever API key is
    # valid (agent model must be one AgenticRAG supports: claude-* or gpt-*).
    agent_model = os.getenv("EVAL_AGENT_MODEL", "gpt-4o-mini")
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")

    print("\n" + "=" * 70)
    print(f"STAGE 2 — GENERATION EVALUATION (agent={agent_model}, "
          f"judge={judge_model})")
    print("=" * 70)

    if judge_model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        judge = ChatAnthropic(model=judge_model, temperature=0.0, max_tokens=10)
    else:
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(model=judge_model, temperature=0.0, max_tokens=10)

    correct = faithful = 0
    answerable = dataset["answerable"]
    for case in answerable:
        print(f"  -> {case['question']}", flush=True)
        agent = AgenticRAG(model_name=agent_model, temperature=0.0, top_k=TOP_K, verbose=False)
        answer = query_with_backoff(agent, case["question"])

        is_correct = case["gold_answer_contains"].lower() in answer.lower()
        context = "\n".join(
            doc.page_content
            for doc, _ in retrieve_with_scores(case["question"], top_k=TOP_K)
        )
        is_faithful = judge_faithfulness(judge, answer, context)

        correct += is_correct
        faithful += is_faithful
        c = "✓" if is_correct else "✗"
        f = "✓" if is_faithful else "✗"
        print(f"  correct:{c} faithful:{f}  {case['question']}")
        if not is_correct:
            print(f"      expected to contain: {case['gold_answer_contains']!r}")
            print(f"      got: {answer[:150]!r}")

    print(f"\n  Correctness:  {correct}/{len(answerable)} = "
          f"{correct/len(answerable):.0%}")
    print(f"  Faithfulness: {faithful}/{len(answerable)} = "
          f"{faithful/len(answerable):.0%}")

    print("\n" + "-" * 70)
    print("GUARDRAIL CHECK — off-topic questions must be refused")
    print("-" * 70)
    refusal_markers = ("not", "no relevant", "doesn't", "does not", "unable",
                       "couldn't find", "don't have", "can't", "can’t",
                       "cannot")
    refused = 0
    for case in dataset["unanswerable"]:
        print(f"  -> {case['question']}", flush=True)
        agent = AgenticRAG(model_name=agent_model, temperature=0.0, top_k=TOP_K, verbose=False)
        answer = query_with_backoff(agent, case["question"])
        is_refusal = any(m in answer.lower() for m in refusal_markers)
        refused += is_refusal
        r = "✓ refused" if is_refusal else "✗ ANSWERED ANYWAY"
        print(f"  [{r}] {case['question']}")
        if not is_refusal:
            print(f"      got: {answer[:150]!r}")
    n = len(dataset["unanswerable"])
    print(f"\n  Refusal rate: {refused}/{n} = {refused/n:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--generation", action="store_true",
                        help="also run the agent + LLM judge (costs API calls)")
    args = parser.parse_args()

    dataset = load_dataset()
    eval_retrieval(dataset)
    if args.generation:
        eval_generation(dataset)
    else:
        print("\n(run with --generation for correctness/faithfulness/guardrail eval)")
