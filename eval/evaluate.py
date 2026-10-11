"""Evaluate the pipeline.

Retrieval only (free, no API key):
    python -m eval.evaluate
End to end with the LLM (needs OPENAI_API_KEY):
    python -m eval.evaluate --llm

Metrics
- hit@k: share of questions where a top-k chunk contains the expected text
- MRR:   mean reciprocal rank of the first correct chunk
- grounding pass rate (--llm): answers where every sentence is cited and supported
- refusal rate on out-of-scope questions (--llm): should be 100%
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from rag import config
from rag.store import get_store


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ").replace(" ", " ")).lower()


def load(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def retrieval_eval(questions: list, k: int) -> dict:
    store = get_store()
    hits, rr, rows = 0, [], []
    for item in questions:
        docs = store.similarity_search(item["q"], k=k)
        rank = next((i for i, d in enumerate(docs, 1) if norm(item["expect"]) in norm(d.page_content)), None)
        hits += rank is not None
        rr.append(1 / rank if rank else 0.0)
        rows.append((item["q"], rank))
    return {"hit@k": hits / len(questions), "mrr": statistics.mean(rr), "rows": rows}


def llm_eval(questions: list, oos: list) -> dict:
    from rag.qa import ask
    grounded = sum(1 for item in questions if not (a := ask(item["q"])).refused and not a.flags)
    refused = sum(1 for item in oos if ask(item["q"]).refused)
    return {"grounding_pass_rate": grounded / len(questions), "oos_refusal_rate": refused / len(oos)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions_ai_act.jsonl")
    ap.add_argument("--oos", default="eval/out_of_scope.jsonl")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    qs = load(args.questions)
    r = retrieval_eval(qs, args.k)
    print(f"Retrieval on {len(qs)} questions (k={args.k}, chunk={config.CHUNK_SIZE}/{config.CHUNK_OVERLAP})")
    for q, rank in r["rows"]:
        print(f"  {'OK ' if rank else 'MISS'} rank={rank or '-':<2} {q}")
    print(f"hit@{args.k}: {r['hit@k']:.0%}   MRR: {r['mrr']:.2f}")

    if args.llm:
        m = llm_eval(qs, load(args.oos))
        print(f"grounding pass rate: {m['grounding_pass_rate']:.0%}   "
              f"out-of-scope refusal rate: {m['oos_refusal_rate']:.0%}")


if __name__ == "__main__":
    main()
