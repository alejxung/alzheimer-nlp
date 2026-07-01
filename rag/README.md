# Alzheimer's Literature Q&A (RAG Pipeline)

An end-to-end retrieval-augmented generation (RAG) pipeline over 7,470 PubMed
Alzheimer's abstracts, enabling grounded Q&A with cited answers and a custom
evaluation framework measuring retrieval relevance and citation rate.

## Motivation

Large language models answer from parametric memory, which has a knowledge cutoff
and can hallucinate citations. This pipeline retrieves real PubMed abstracts first,
then generates answers grounded only in the retrieved evidence. Every claim in the
answer is citable back to a specific paper, making the system auditable in a way
a plain LLM response is not. The Alzheimer's literature focus connects directly to
Project 1 in this series, extending the classifier pipeline into a research Q&A tool.

## What it does

Given a clinical or research question about Alzheimer's disease, the pipeline
retrieves the most semantically relevant abstracts from a local vector store and
generates a cited, source-grounded answer.

Request:
```bash
python rag/generate.py
```

Example output:
```
Question: How does tau protein relate to cognitive decline?

Answer:
In abstract [2], protein tau concentration is significantly elevated in the
cerebrospinal fluid of patients with Alzheimer's disease compared to cognitively
healthy controls, even in those with very mild dementia [2]. However, no
correlation was found between tau levels and MMSE severity scores, suggesting
that elevated tau marks the disease without directly tracking its progression [2].
Tau oligomers have also been identified in serum of Alzheimer's patients [3].

Sources:
  [1] PMID 36012440 (2022) - Acetylated Tau Protein: A New Piece in the Puzzle...
  [2] PMID 8843109  (1996) - Cerebrospinal protein tau is elevated in early AD...
  [3] PMID 28453485 (2017) - Tau Oligomers in Sera of Patients with AD...
```

## Architecture

1. **Ingest** (`ingest.py`): fetches Alzheimer's abstracts from PubMed via the NCBI
   E-utilities API, embeds them with `sentence-transformers`, and stores the vectors
   in a local ChromaDB collection.
2. **Retrieve** (`retrieve.py`): embeds an incoming query with the same model and
   performs cosine similarity search against the ChromaDB collection to return
   the top-k most relevant abstracts.
3. **Generate** (`generate.py`): passes the retrieved abstracts as context to
   GPT-4o-mini with a strict grounding prompt. The model cites sources by index
   number and explicitly refuses to answer when the retrieved context is insufficient.
4. **Evaluate** (`evaluate.py`): runs 22 test queries across five categories and
   measures retrieval relevance (LLM-as-judge, 0-2 scale) and citation rate
   (deterministic regex check).

## Corpus

| | Value |
|---|---|
| Source | PubMed via NCBI E-utilities API |
| Search query | "Alzheimer's disease" |
| PMIDs fetched | 9,999 |
| Abstracts with text | 7,470 |
| Embedding model | all-MiniLM-L6-v2 |
| Vector store | ChromaDB (local, persistent) |
| Similarity metric | Cosine |

The gap between PMIDs fetched and abstracts stored reflects papers without abstract
text (editorials, letters, conference titles). Only papers with substantive abstract
content are indexed.

## Evaluation results

Evaluated across 22 queries spanning five categories: biomarkers/diagnosis,
mechanisms, treatment, epidemiology, and intentional out-of-scope queries
(Parkinson's disease, type 2 diabetes) to stress-test retrieval specificity.

| Metric | Score |
|---|---|
| Retrieval relevance (in-scope, 20 queries) | 1.45 / 2.0 (72.5%) |
| Retrieval relevance (out-of-scope, 2 queries) | 0.10 / 2.0 (5%) |
| Citation rate (in-scope) | 90% (18 / 20 answers) |

**Retrieval relevance** was scored by an LLM judge on a 0-2 scale per retrieved
abstract (0 = not relevant, 1 = partially relevant, 2 = directly relevant),
averaged across the top-5 results per query.

**Citation rate** measures whether the generated answer contains at least one inline
citation `[N]`. Answers that correctly refused to answer due to insufficient context
are the primary source of uncited responses and are expected behavior, not failures.

**On hallucination detection:** an initial LLM-as-judge hallucination metric was
developed and discarded. The judge (GPT-4o-mini) shared parametric knowledge with
the generator (GPT-4o-mini), making it unable to reliably distinguish claims sourced
from retrieved abstracts versus the model's own training data. The metric produced
unreliable results across runs (45% to 100% hallucination rate with minor prompt
changes). It was replaced with deterministic citation rate and refusal rate metrics.
A more reliable approach would use NLI-based textual entailment (e.g.
`cross-encoder/nli-deberta-v3-base`) to verify each answer claim is entailed by a
specific retrieved passage, without requiring the judge to have independent knowledge
of the claim's factual accuracy.

## Project structure

```
rag/
├── ingest.py        # PubMed fetch, embed, store in ChromaDB
├── retrieve.py      # cosine similarity search over ChromaDB
├── generate.py      # context-grounded generation with GPT-4o-mini
├── evaluate.py      # retrieval relevance + citation rate evaluation
└── chroma_db/       # local ChromaDB vector store (not redistributed)
```

## Running it

### Setup

```bash
# from repo root: alzheimer-nlp/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in OPENAI_API_KEY and NCBI_API_KEY
```

### Ingest (run once)

```bash
python rag/ingest.py
```

Downloads ~10K PMIDs, fetches abstracts in batches of 200, embeds with
sentence-transformers, and stores in ChromaDB. Takes roughly 2-3 minutes.
Requires an NCBI API key for higher rate limits (10 req/sec vs 3 req/sec).

### Query the pipeline

```bash
python rag/generate.py
```

Runs three example queries and prints cited answers with source metadata.

### Evaluate

```bash
python rag/evaluate.py
```

Runs the full 22-query evaluation suite and saves results to
`rag/eval_results.json` and `rag/eval_summary.json`.

## Tech stack

sentence-transformers, ChromaDB, OpenAI GPT-4o-mini, NCBI E-utilities API,
LangChain (optional orchestration layer, available as a drop-in replacement
for the direct OpenAI calls in `generate.py`)

## API keys required

| Key | Where to get it | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | platform.openai.com | LLM generation |
| `NCBI_API_KEY` | ncbi.nlm.nih.gov/account | PubMed rate limit (optional but recommended) |

See `.env.example` at the repo root for the expected format.