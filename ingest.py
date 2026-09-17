"""
ingest.py
---------
Step 1 of the GATE PYQ Assistant pipeline.

What this does:
1. Reads all PDFs from a folder (your GATE PYQs / notes).
2. Splits the text into overlapping chunks.
3. Converts each chunk into an embedding (using a free local model).
4. Stores everything in a local ChromaDB vector database.

Run:
    python ingest.py --pdf_dir ./pdfs --db_dir ./chroma_db
"""

import os
import argparse
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a single PDF and returns its full text."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def load_all_pdfs(pdf_dir: str) -> list[dict]:
    """Loads every PDF in a folder. Returns list of {filename, text}."""
    docs = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            print(f"Reading: {fname}")
            text = extract_text_from_pdf(path)
            docs.append({"filename": fname, "text": text})
    return docs


def chunk_documents(docs: list[dict], chunk_size=800, chunk_overlap=150) -> list[dict]:
    """Splits each document's text into overlapping chunks.
    Overlap helps preserve context across chunk boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{doc['filename']}_chunk_{i}",
                "text": piece,
                "source": doc["filename"],
            })
    return chunks


def store_in_chromadb(chunks: list[dict], db_dir: str, collection_name="gate_pyqs"):
    """Embeds each chunk and stores it in a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=db_dir)

    # Free, local embedding model — no API key needed.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{db_dir}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./pdfs", help="Folder containing your PDFs")
    parser.add_argument("--db_dir", default="./chroma_db", help="Where to save the vector DB")
    args = parser.parse_args()

    if not os.path.isdir(args.pdf_dir):
        print(f"Folder not found: {args.pdf_dir}")
        print("Create it and put your GATE PYQ / notes PDFs inside, then re-run.")
        return

    docs = load_all_pdfs(args.pdf_dir)
    if not docs:
        print("No PDFs found. Add some .pdf files to the pdf_dir first.")
        return

    chunks = chunk_documents(docs)
    store_in_chromadb(chunks, args.db_dir)


if __name__ == "__main__":
    main()
