import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_pages
from load_pdf import load_pdf

EMBED_MODEL = "nomic-embed-text"


def build_index(chunks):
    """Turn every chunk into an embedding (a list of numbers)."""
    return model.encode([c["text"] for c in chunks], normalize_embeddings=True)


def search(question, chunks, embeddings, top_k=3):
    """Return the chunks whose meaning is closest to the question."""
    q = model.encode([question], normalize_embeddings=True)[0]
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