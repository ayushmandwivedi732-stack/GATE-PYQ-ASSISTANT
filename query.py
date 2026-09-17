"""
query.py
--------
Step 2 of the GATE PYQ Assistant pipeline.

What this does:
1. Takes your question.
2. Finds the most relevant chunks from ChromaDB (retrieval).
3. Sends those chunks + your question to a free LLM (Groq/Llama) to
   generate a grounded answer (generation).

Setup:
    1. Get a free API key from https://console.groq.com
    2. Create a file named `.env` in this folder with:
           GROQ_API_KEY=your_key_here

Run:
    python query.py --db_dir ./chroma_db --question "Explain normalization in DBMS"
"""

import os
import argparse
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def retrieve_chunks(question: str, db_dir: str, collection_name="gate_pyqs", top_k=4):
    """Finds the top_k most relevant chunks for the question."""
    client = chromadb.PersistentClient(path=db_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name=collection_name, embedding_function=embed_fn)

    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def generate_answer(question: str, chunks: list[str]) -> str:
    """Sends the retrieved context + question to the LLM."""
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a GATE exam tutor. Answer the student's question using
ONLY the context below. If the context doesn't contain the answer, say so honestly
instead of guessing. Explain clearly, like a GATE examiner would.

Context:
{context}

Question: {question}

Answer:"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    chunks, sources = retrieve_chunks(args.question, args.db_dir)
    print("Retrieved from:", set(sources))
    print("\nGenerating answer...\n")

    answer = generate_answer(args.question, chunks)
    print(answer)


if __name__ == "__main__":
    main()
