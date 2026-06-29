import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve
from generate import generate

load_dotenv()

#=== Config =================================================#
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 5
RESULTS_PATH = "rag/eval_results.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#=== Test queries ===========================================#
EVAL_QUERIES = [
    "What CSF biomarkers are used to diagnose Alzheimer's disease?",
    "What is the role of amyloid PET imaging in Alzheimer's diagnosis?",
    "How are blood-based biomarkers being developed for Alzheimer's detection?",
    "What neuroimaging findings are characteristic of early Alzheimer's disease?",
    "How does the MMSE score relate to Alzheimer's disease staging?",
    "How does amyloid beta accumulation lead to neuronal death?",
    "What is the relationship between tau phosphorylation and neurodegeneration?",
    "How does neuroinflammation contribute to Alzheimer's disease progression?",
    "What is the cholinergic hypothesis of Alzheimer's disease?",
    "How do APOE4 gene variants increase Alzheimer's risk?",
    "What FDA-approved drugs exist for Alzheimer's disease treatment?",
    "How do acetylcholinesterase inhibitors work in Alzheimer's disease?",
    "What immunotherapy approaches are being studied for Alzheimer's?",
    "What lifestyle interventions reduce the risk of Alzheimer's disease?",
    "What is lecanemab and how does it work in Alzheimer's treatment?",
    "What are the main risk factors for developing Alzheimer's disease?",
    "How does age affect the prevalence of Alzheimer's disease?",
    "What is the global prevalence of Alzheimer's disease?",
    "How does education level relate to Alzheimer's disease risk?",
    "What is the relationship between diabetes and Alzheimer's disease?",
    "What are the symptoms of Parkinson's disease?",
    "How is type 2 diabetes treated with insulin?",
]

OUT_OF_SCOPE = {
    "What are the symptoms of Parkinson's disease?",
    "How is type 2 diabetes treated with insulin?"
}

#=== Retrieval relevance judge ==============================#
def judge_retrieval_relevance(query, hits):
    abstracts_text = "\n\n".join([
        f"[{i+1}] {hit['text'][:500]}"
        for i, hit in enumerate(hits)
    ])

    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved abstracts:\n{abstracts_text}\n\n"
        "For each abstract [1]-[5], rate its relevance to the query:\n"
        "2 = directly relevant (contains information that answers the query)\n"
        "1 = partially relevant (related topic but doesn't directly answer)\n"
        "0 = not relevant\n\n"
        "Respond ONLY with a JSON object like: "
        "{\"scores\": [2, 1, 0, 2, 1], \"reasoning\": \"brief reason\"}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    scores = result.get("scores", [0] * 5)
    avg_score = sum(scores) / len(scores)
    return avg_score, scores, result.get("reasoning", "")

#=== Citation rate check (deterministic) ===================#
def check_citation(answer):
    """Returns True if the answer contains at least one inline citation [N]."""
    return bool(re.search(r'\[\d+\]', answer))

#=== Refusal check (deterministic) =========================#
REFUSAL_PHRASES = [
    "do not contain",
    "does not contain",
    "cannot provide",
    "cannot answer",
    "no information",
    "not mentioned",
    "not specified",
    "insufficient information",
    "not enough information",
    "abstracts do not",
    "i cannot",
]

def check_refusal(answer):
    """Returns True if the answer explicitly refuses to answer due to missing context."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

#=== Run full evaluation ====================================#
def run_evaluation():
    results = []
    total = len(EVAL_QUERIES)

    print(f"Running evaluation on {total} queries...\n")

    for i, query in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{total}] {query[:70]}...")

        try:
            hits = retrieve(query, top_k=TOP_K)
            result = generate(query, top_k=TOP_K)
            answer = result["answer"]

            relevance_avg, relevance_scores, relevance_reasoning = judge_retrieval_relevance(
                query, hits
            )
            cited = check_citation(answer)
            refused = check_refusal(answer)

            results.append({
                "query": query,
                "answer": answer,
                "out_of_scope": query in OUT_OF_SCOPE,
                "retrieval_score_avg": relevance_avg,
                "retrieval_scores": relevance_scores,
                "retrieval_reasoning": relevance_reasoning,
                "cited": cited,
                "refused": refused,
                "sources": result["sources"]
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"query": query, "error": str(e)})

    return results

#=== Summarize ==============================================#
def summarize(results):
    valid = [r for r in results if "error" not in r]
    in_scope = [r for r in valid if not r.get("out_of_scope")]
    out_scope = [r for r in valid if r.get("out_of_scope")]

    avg_relevance_in = sum(r["retrieval_score_avg"] for r in in_scope) / len(in_scope)
    avg_relevance_out = sum(r["retrieval_score_avg"] for r in out_scope) / len(out_scope) if out_scope else 0

    cited_count = sum(1 for r in in_scope if r.get("cited"))
    citation_rate = cited_count / len(in_scope)

    refused_when_low = [
        r for r in in_scope
        if r["retrieval_score_avg"] < 1.0 and r.get("refused")
    ]
    low_relevance = [r for r in in_scope if r["retrieval_score_avg"] < 1.0]
    refusal_rate = len(refused_when_low) / len(low_relevance) if low_relevance else 0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries evaluated:              {len(valid)}")
    print(f"In-scope queries:                     {len(in_scope)}")
    print(f"Out-of-scope queries:                 {len(out_scope)}")
    print(f"\nRetrieval relevance (in-scope):       {avg_relevance_in:.3f} / 2.0")
    print(f"Retrieval relevance (out-of-scope):   {avg_relevance_out:.3f} / 2.0")
    print(f"\nCitation rate (in-scope):             {citation_rate:.1%}")
    print(f"  Answers with citations:             {cited_count} / {len(in_scope)}")
    print(f"\nRefusal rate (low-relevance queries): {refusal_rate:.1%}")
    print(f"  Correctly refused:                  {len(refused_when_low)} / {len(low_relevance)}")
    print("=" * 60)

    return {
        "total": len(valid),
        "avg_retrieval_relevance_in_scope": avg_relevance_in,
        "avg_retrieval_relevance_out_scope": avg_relevance_out,
        "citation_rate": citation_rate,
        "refusal_rate": refusal_rate
    }

#=== Main ===================================================#
if __name__ == "__main__":
    results = run_evaluation()

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")

    summary = summarize(results)

    with open("rag/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to rag/eval_summary.json")