"""Ask questions from the terminal:  python -m rag.cli "What does Article 11 require?" """
import sys

from rag.qa import ask


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m rag.cli "your question"')
    a = ask(" ".join(sys.argv[1:]))
    print("\n" + a.text + "\n")
    if a.sources:
        print("Sources:")
        for n, src, page, score in a.sources:
            print(f"  [{n}] {src}, p. {page}  (relevance {score})")
    if a.flags:
        print("\nGrounding check flagged:")
        for why, sent in a.flags:
            print(f"  - {why}: {sent}")
    elif not a.refused:
        print("\nGrounding check: every sentence is cited and supported.")


if __name__ == "__main__":
    main()
