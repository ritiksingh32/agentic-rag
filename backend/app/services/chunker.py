"""
Chunking service.

Splits extracted page text into smaller chunks using LangChain's
RecursiveCharacterTextSplitter — the same approach from the original
naive-RAG project (Point A in the study guide), but now applied
per-page so every chunk retains accurate page-number metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Takes the output of document_loader (a list of {"page_number", "text"}),
    and returns a flat list of chunks, each tagged with its page number
    and a sequential chunk index.

    Returns:
        [
            {"chunk_index": 0, "page_number": 1, "text": "..."},
            {"chunk_index": 1, "page_number": 1, "text": "..."},
            {"chunk_index": 2, "page_number": 2, "text": "..."},
            ...
        ]
    """
    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            all_chunks.append({
                "chunk_index": chunk_index,
                "page_number": page["page_number"],
                "text": chunk_text,
            })
            chunk_index += 1

    return all_chunks