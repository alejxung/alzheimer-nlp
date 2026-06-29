import os
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve

load_dotenv()

#=== Config =================================================#
OPENAI_MODEL = "gpt-4o-mini"  # cheaper than gpt-4o, more than enough for RAG generation
TOP_K = 5

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#=== Build context from retrieved abstracts =================#
def build_context(hits):
    context_parts = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        context_parts.append(
            f"[{i+1}] PMID: {meta['pmid']} ({meta['year']}) — {meta['title']}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(context_parts)

#=== Generate answer using retrieved context ================#
def generate(query, top_k=TOP_K):
    hits = retrieve(query, top_k=top_k)
    context = build_context(hits)

    system_prompt = (
        "You are a medical research assistant specializing in Alzheimer's disease. "
        "Answer the user's question using ONLY the provided research abstracts as context. "
        "Cite sources by their index number [1], [2], etc. "
        "If a claim is not explicitly stated in the provided abstracts, do not include it. "
        "If the abstracts do not contain sufficient information to answer the question, "
        "say so explicitly rather than speculating or drawing on outside knowledge. "
        "Do not use any knowledge outside the provided abstracts."
    )

    user_prompt = (
        f"Research abstracts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer citing relevant abstracts by index number:"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # low temperature for factual, consistent answers
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "index": i + 1,
                "pmid": hit["metadata"]["pmid"],
                "title": hit["metadata"]["title"],
                "year": hit["metadata"]["year"],
                "journal": hit["metadata"]["journal"],
                "score": hit["score"]
            }
            for i, hit in enumerate(hits)
        ],
        "model": OPENAI_MODEL
    }

#=== Pretty print ===========================================#
def print_result(result):
    print(f"\nQuestion: {result['query']}")
    print("=" * 70)
    print(f"Answer:\n{result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] PMID {s['pmid']} ({s['year']}) — {s['title'][:70]}...")
        print(f"       Relevance score: {s['score']:.4f}")

#=== Test ===================================================#
if __name__ == "__main__":
    test_queries = [
        "What are the early biomarkers of Alzheimer's disease?",
        "How does tau protein relate to cognitive decline?",
        "What treatments have shown promise for slowing Alzheimer's progression?"
    ]

    for query in test_queries:
        result = generate(query)
        print_result(result)
        print("\n" + "-" * 70)