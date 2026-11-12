from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader

def read_pdf_text(pdf_path: str | Path) -> str:
    """read text from a PDF file and return it as a string"""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_multiple_pdfs(pdf_paths: List[str | Path]) -> List[Document]:
    """read text from multiple PDF files and return a list of Document objects"""
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        text = read_pdf_text(pdf_path)
        documents.append(Document(page_content=text, metadata={"source": str(pdf_path)}))
    return documents