import os
import time
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

#=== Config =================================================#
CHROMA_DIR = "rag/chroma_db"
COLLECTION_NAME = "alzheimer_abstracts"
EMBED_MODEL = "all-MiniLM-L6-v2"
QUERY = "Alzheimer's disease"
MAX_RESULTS = 10000
FETCH_BATCH = 200 # max PMIDs per efetch call
EMBED_BATCH = 64 # sentence-transformers batch size
CHROMA_BATCH = 100 # ChromaDB upsert batch size
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#=== Get PMIDs ==============================================#
def get_pmids(query, max_results):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(ESEARCH_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    pmids = data["esearchresult"]["idlist"]
    total = int(data["esearchresult"]["count"])
    print(f"Total PubMed results for '{query}': {total:,}")
    print(f"Fetching {len(pmids):,} PMIDs")
    return pmids

#=== Fetch abstracts in batches =============================#
def fetch_abstracts(pmids):
    records = []
    batches = [pmids[i:i+FETCH_BATCH] for i in range(0, len(pmids), FETCH_BATCH)]

    for batch in tqdm(batches, desc="Fetching abstracts from PubMed"):
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml"
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        resp = requests.get(EFETCH_URL, params=params)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            year_el = article.find(".//PubDate/Year")
            journal_el = article.find(".//Journal/Title")

            if pmid_el is None or abstract_el is None:
                continue

            abstract_text = "".join(abstract_el.itertext())
            if not abstract_text.strip():
                continue

            title_text = "".join(title_el.itertext()) if title_el is not None else ""

            records.append({
                "pmid": pmid_el.text,
                "title": title_text,
                "abstract": abstract_text,
                "year": year_el.text if year_el is not None else "",
                "journal": journal_el.text if journal_el is not None else "",
                "text": f"{title_text}\n\n{abstract_text}"
            })

        # Rate limit: 3 req/sec without NCBI key, 10/sec with
        time.sleep(0.1 if NCBI_API_KEY else 0.34)

    print(f"Successfully fetched {len(records):,} abstracts with text")
    return records

#=== Embed + store in ChromaDB ==============================#
def embed_and_store(records):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' for clean rebuild")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    texts = [r["text"] for r in records]
    ids = [r["pmid"] for r in records]
    metadatas = [
        {
            "pmid": r["pmid"],
            "title": r["title"],
            "year": r["year"],
            "journal": r["journal"]
        }
        for r in records
    ]

    print(f"Embedding {len(texts):,} documents...")
    embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding"):
        batch_embeddings = embedder.encode(
            texts[i:i+EMBED_BATCH],
            show_progress_bar=False
        ).tolist()
        embeddings.extend(batch_embeddings)

    print("Storing in ChromaDB...")
    for i in tqdm(range(0, len(records), CHROMA_BATCH), desc="Storing"):
        collection.upsert(
            ids=ids[i:i+CHROMA_BATCH],
            embeddings=embeddings[i:i+CHROMA_BATCH],
            documents=texts[i:i+CHROMA_BATCH],
            metadatas=metadatas[i:i+CHROMA_BATCH]
        )

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count():,} documents")

#=== Main ===================================================#
if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)
    pmids = get_pmids(QUERY, MAX_RESULTS)
    records = fetch_abstracts(pmids)
    embed_and_store(records)
    print("Ingestion complete.")