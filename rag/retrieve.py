import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

#=== Config =================================================#
CHROMA_DIR = "rag/chroma_db"
COLLECTION_NAME = "alzheimer_abstracts"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

#=== Load ChromaDB + embedding model ========================#
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)

def retrieve(query, top_k=TOP_K):
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "pmid": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i]  # cosine similarity
        })

    return hits

#=== Test ====================================================#
if __name__ == "__main__":
    test_queries = [
        "What are the early biomarkers of Alzheimer's disease?",
        "How does tau protein relate to cognitive decline?",
        "What is the role of amyloid beta in Alzheimer's progression?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        hits = retrieve(query)
        for i, hit in enumerate(hits):
            meta = hit["metadata"]
            print(f"[{i+1}] Score: {hit['score']:.4f}")
            print(f"     PMID:    {meta['pmid']}")
            print(f"     Year:    {meta['year']}")
            print(f"     Journal: {meta['journal']}")
            print(f"     Title:   {meta['title'][:80]}...")