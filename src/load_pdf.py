from pypdf import PdfReader


def load_pdf(path):
    """Read a PDF and return a list with the text of each page."""
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": number, "text": text})
    return pages


if __name__ == "__main__":
    pages = load_pdf("data/sample.pdf")
    print(f"Loaded {len(pages)} pages")
    print(pages[0]["text"][:500])