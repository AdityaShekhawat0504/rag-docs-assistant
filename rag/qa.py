"""Retrieval-augmented question answering with citations and a grounding check.

Pipeline
1. Retrieve the top-k chunks from Chroma (with relevance scores).
2. If nothing is relevant enough, refuse instead of guessing.
3. Ask the LLM to answer ONLY from the numbered context and cite [n] after every sentence.
4. Grounding check (no extra LLM call): every sentence must cite a real chunk, and the
   cited chunk must share enough content words with the sentence. Weak sentences are flagged.
"""
import re
import warnings
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag import config
from rag.store import get_store

# Chroma's L2-based relevance can dip below 0 for unrelated questions; those are refused anyway.
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

MIN_RELEVANCE = 0.25  # below this the question is treated as out of scope
SUPPORT_THRESHOLD = 0.5  # share of a sentence's content words found in its cited chunks

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the numbered context passages. "
     "After every sentence put the passage number(s) you used, like [1] or [2][3]. "
     "If the context does not contain the answer, reply exactly: "
     "\"I could not find this in the documents.\" Do not use outside knowledge."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

STOP = set("""a an the and or of to in on for with by is are was were be been this that these those it its as at from
which who whom what when where how not no shall may must can will would should could under into than then there their
any all each such other also per""".split())


@dataclass
class Answer:
    question: str
    text: str
    sources: list = field(default_factory=list)   # [(n, source, page, score)]
    flags: list = field(default_factory=list)     # sentences that failed the grounding check
    refused: bool = False


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", text) if s.strip()]


def grounding_flags(answer: str, chunks: list) -> list:
    flags = []
    for sent in _sentences(answer):
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
        body = re.sub(r"\[\d+\]", "", sent)
        if not cited:
            flags.append(("no citation", body))
            continue
        if any(n < 1 or n > len(chunks) for n in cited):
            flags.append(("cites a passage that does not exist", body))
            continue
        words = _words(body)
        if not words:
            continue
        support_text = " ".join(chunks[n - 1].page_content for n in cited)
        support = len(words & _words(support_text)) / len(words)
        if support < SUPPORT_THRESHOLD:
            flags.append((f"weak support ({support:.0%})", body))
    return flags


def ask(question: str, k: int | None = None) -> Answer:
    k = k or config.TOP_K
    hits = get_store().similarity_search_with_relevance_scores(question, k=k)
    if not hits or max(score for _, score in hits) < MIN_RELEVANCE:
        return Answer(question, "I could not find this in the documents.", refused=True)

    chunks = [doc for doc, _ in hits]
    context = "\n\n".join(
        f"[{i}] ({d.metadata.get('source')}, p. {d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(chunks, 1)
    )
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    text = (PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})

    sources = [(i, d.metadata.get("source"), d.metadata.get("page"), round(s, 3))
               for i, (d, s) in enumerate(hits, 1)]
    refused = text.strip().startswith("I could not find")
    flags = [] if refused else grounding_flags(text, chunks)
    return Answer(question, text, sources, flags, refused)
