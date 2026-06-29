"""
Document loading service.

Extracts text from uploaded PDF or DOCX files, keeping track of which
page each piece of text came from — this page-level metadata is what
lets us cite sources accurately later (a gap the original naive-RAG
project had: it only tagged chunks with a generic "uploaded_pdf" label).
"""

from pypdf import PdfReader
from docx import Document as DocxDocument


def load_pdf(file_path: str) -> list[dict]:
    """
    Extracts text from a PDF, page by page.

    Returns a list of dicts like:
        [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}]
    """
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages
            pages.append({"page_number": page_number, "text": text})

    return pages


def load_docx(file_path: str) -> list[dict]:
    """
    Extracts text from a DOCX file.

    DOCX files don't have a native concept of "pages" the way PDFs do
    (page breaks depend on rendering, not file structure), so we treat
    the whole document as a single "page" for metadata purposes.
    """
    doc = DocxDocument(file_path)
    full_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    if not full_text.strip():
        return []

    return [{"page_number": 1, "text": full_text}]


def load_document(file_path: str, filename: str) -> list[dict]:
    """
    Dispatches to the right loader based on file extension.
    """
    lowered = filename.lower()

    if lowered.endswith(".pdf"):
        return load_pdf(file_path)
    elif lowered.endswith(".docx"):
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {filename}")