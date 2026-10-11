"""Load every PDF in data/, split it into overlapping chunks and index them in Chroma.

Usage:  python -m rag.ingest            (adds new PDFs)
        python -m rag.ingest --reset    (rebuilds the index from scratch)
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.store import get_store


def chunk_id(source: str, page: int, idx: int, text: str) -> str:
    """Stable ID so re-running ingest does not create duplicates."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{Path(source).name}:p{page}:c{idx}:{h}"


def load_and_split(pdf: Path):
    pages = PyPDFLoader(str(pdf)).load()  # one Document per page, metadata has 'page'
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["source"] = pdf.name
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1  # human page numbers
        c.metadata["chunk"] = i
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete and rebuild the index")
    args = ap.parse_args()

    if args.reset and Path(config.DB_DIR).exists():
        shutil.rmtree(config.DB_DIR)

    pdfs = sorted(Path(config.DATA_DIR).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.DATA_DIR}/ - add some first.")

    store = get_store()
    total = 0
    for pdf in pdfs:
        chunks = load_and_split(pdf)
        ids = [chunk_id(pdf.name, c.metadata["page"], c.metadata["chunk"], c.page_content) for c in chunks]
        store.add_documents(chunks, ids=ids)
        total += len(chunks)
        print(f"indexed {pdf.name}: {len(chunks)} chunks")
    print(f"done: {total} chunks in '{config.DB_DIR}' (collection '{config.COLLECTION}')")


if __name__ == "__main__":
    main()
