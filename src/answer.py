import ollama

from chunking import chunk_pages
from load_pdf import load_pdf
from search import build_index, search

SYSTEM = (
    "Answer the question using only the passages provided. "
    "If the answer is not in the passages, say you don't know. "
    "Cite the page like (page 3)."
)


def answer(question, chunks, embeddings):
    hits = search(question, chunks, embeddings, top_k=3)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for _, c in hits)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    pages = load_pdf("data/sample.pdf")
    chunks = chunk_pages(pages)
    embeddings = build_index(chunks)

    question = input("Ask a question about the document: ")
    print("\n" + answer(question, chunks, embeddings))