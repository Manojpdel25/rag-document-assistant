import numpy as np
import ollama

from chunking import chunk_pages
from load_pdf import load_pdf

EMBED_MODEL = "nomic-embed-text"


def embed(texts, batch_size=32):
    """Turn texts into normalized embeddings with a local Ollama model."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = ollama.embed(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(response["embeddings"])
    arr = np.array(vectors, dtype=float)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def build_index(chunks):
    """Embed every chunk."""
    return embed([c["text"] for c in chunks])


def search(question, chunks, embeddings, top_k=3):
    """Return the chunks whose meaning is closest to the question."""
    q = embed([question])[0]
    scores = embeddings @ q
    best = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in best]


if __name__ == "__main__":
    pages = load_pdf("data/sample.pdf")
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    embeddings = build_index(chunks)

    question = input("Ask a question about the document: ")
    for score, chunk in search(question, chunks, embeddings):
        print(f"\n[page {chunk['page']}] score {score:.2f}")
        print(chunk["text"][:300])