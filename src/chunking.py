def chunk_pages(pages, chunk_size=800, overlap=100):
    """Split each page's text into overlapping chunks, keeping the page number."""
    chunks = []
    for page in pages:
        text = page["text"].strip()
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"page": page["page"], "text": text[start:end]})
            if end >= len(text):
                break
            start = end - overlap
    return chunks