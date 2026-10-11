"""Embedding model + Chroma vector store, shared by ingest, query and eval."""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag import config


def get_embeddings() -> HuggingFaceEmbeddings:
    # normalize_embeddings=True -> cosine similarity behaves well in Chroma
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.DB_DIR,
    )
