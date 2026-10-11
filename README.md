# RAG Docs Assistant

Ask questions about your own PDFs and get answers **only from those documents**, with page citations and an automatic **grounding check** that flags any sentence the sources do not support.

Built with **LangChain**, **ChromaDB**, **sentence-transformers** (local embeddings) and the **OpenAI API**, plus an evaluation script that measures retrieval quality (hit@k, MRR), grounding and refusal on out-of-scope questions.

## Why
LLMs answer confidently even when they are wrong. For technical, legal or regulatory documents that is not acceptable. This project applies the same idea as my [AI Act Compliance Copilot](https://github.com/AdityaShekhawat0504): never trust a generated sentence unless it can be traced back to a source.

## How it works
```
PDFs ─► PyPDFLoader ─► RecursiveCharacterTextSplitter (500 chars, 150 overlap)
     ─► all-MiniLM-L6-v2 embeddings ─► ChromaDB (persistent, stable chunk IDs)

Question ─► top-k retrieval with relevance scores
         ─► below threshold? ─► refuse ("I could not find this in the documents.")
         ─► LLM answers only from numbered passages, cites [n] after every sentence
         ─► grounding check: every sentence cited + enough word overlap with its cited passage
```

| File | What it does |
|---|---|
| `rag/ingest.py` | Loads PDFs, splits into chunks, stores them in Chroma with source and page metadata |
| `rag/store.py` | Embedding model and vector store setup |
| `rag/qa.py` | Retrieval, prompt, LLM call, refusal logic and grounding check |
| `rag/cli.py` | Ask questions from the terminal |
| `app.py` | Streamlit UI |
| `eval/evaluate.py` | hit@k, MRR, grounding pass rate, out-of-scope refusal rate |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your OPENAI_API_KEY in .env
```

Sample document (EU AI Act, Regulation (EU) 2024/1689):
```bash
curl -L -o data/eu_ai_act.pdf "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
```

## Run
```bash
python -m rag.ingest --reset
python -m rag.cli "What is the maximum fine for prohibited AI practices?"
streamlit run app.py
```

## Evaluate
```bash
python -m eval.evaluate            # retrieval only, free
python -m eval.evaluate --llm      # + grounding and refusal, uses the API
```

## Results
Retrieval on the 12 questions in `eval/questions_ai_act.jsonl` (EU AI Act, 144 pages, overlap 150):

| Chunk size | Chunks | hit@3 | MRR (k=3) | hit@5 | MRR (k=5) |
|---|---|---|---|---|---|
| **500 (default)** | 1770 | **100%** | **0.88** | **100%** | **0.88** |
| 900 | 871 | 92% | 0.78 | 92% | 0.78 |
| 1500 | 502 | 92% | 0.78 | 92% | 0.78 |

Chunk size 500 is the only setting that finds every expected passage, so it is the default in `rag/config.py`. k=3 and k=5 score the same on retrieval; the default stays at k=5.

End to end at the default setting (chunk 500 / overlap 150, k=5, `gpt-4o-mini`):

| Grounding pass rate | Out-of-scope refusal rate |
|---|---|
| 83% (10 of 12 answers) | 100% (3 of 3 questions) |

## Demo
```
$ python -m rag.cli "What is the maximum fine for prohibited AI practices?"

The maximum fine for prohibited AI practices is up to EUR 35,000,000 or, if the offender is an undertaking, up to 7% of its total worldwide annual turnover for the preceding financial year, whichever is higher [1].

Sources:
  [1] eu_ai_act.pdf, p. 115  (relevance 0.66)
  [2] eu_ai_act.pdf, p. 117  (relevance 0.585)
  [3] eu_ai_act.pdf, p. 51  (relevance 0.519)
  [4] eu_ai_act.pdf, p. 117  (relevance 0.499)
  [5] eu_ai_act.pdf, p. 89  (relevance 0.487)

Grounding check: every sentence is cited and supported.
```

## Design decisions
- **Local embeddings** (all-MiniLM-L6-v2): free, fast on a laptop, no data leaves the machine for indexing.
- **Stable chunk IDs** (file, page, chunk, content hash): re-running ingest does not duplicate data.
- **Refusal threshold**: if nothing relevant is retrieved, the system says so instead of guessing.
- **Grounding check without a second LLM call**: cheap, deterministic, and easy to explain. It catches uncited sentences and sentences with little overlap with their source.
- **Evaluation first**: chunk size, overlap and k can be tuned against hit@k and MRR instead of by gut feeling.

## Next steps
- Hybrid search (BM25 + vectors) and a cross-encoder re-ranker
- LLM-as-judge faithfulness check as a second layer
- Docling for tables and complex layouts
