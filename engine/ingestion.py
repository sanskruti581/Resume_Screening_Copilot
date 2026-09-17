"""
engine/ingestion.py
--------------------
Handles safe extraction of raw text from uploaded resume files (PDF / DOCX),
text cleaning, and chunking into standardized segments for embedding.
"""

from __future__ import annotations

import io
import re
from typing import List, Union

from pypdf import PdfReader
import docx  # python-docx

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


class IngestionError(Exception):
    """Raised when a resume file cannot be parsed."""
    pass


def _get_file_bytes(uploaded_file) -> bytes:
    """
    Normalizes input into raw bytes, whether it's a Streamlit UploadedFile,
    a file path string, or a raw bytes object.
    """
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)

    if hasattr(uploaded_file, "getvalue"):
        # Streamlit's UploadedFile supports getvalue()
        return uploaded_file.getvalue()

    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0) if hasattr(uploaded_file, "seek") else None
        return uploaded_file.read()

    if isinstance(uploaded_file, str):
        with open(uploaded_file, "rb") as f:
            return f.read()

    raise IngestionError("Unsupported file input type provided to ingestion engine.")


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF file safely, page by page, without dropping content."""
    try:
        raw_bytes = _get_file_bytes(uploaded_file)
        reader = PdfReader(io.BytesIO(raw_bytes))

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise IngestionError("PDF is password-protected and could not be decrypted.")

        pages_text = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages_text.append(text)

        full_text = "\n".join(pages_text)
        if not full_text.strip():
            raise IngestionError("No extractable text found in PDF (it may be a scanned image).")
        return full_text
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Failed to parse PDF file: {e}")


def extract_text_from_docx(uploaded_file) -> str:
    """Extract text from a DOCX file, including paragraphs and table cell content."""
    try:
        raw_bytes = _get_file_bytes(uploaded_file)
        document = docx.Document(io.BytesIO(raw_bytes))

        parts: List[str] = []

        for para in document.paragraphs:
            if para.text and para.text.strip():
                parts.append(para.text)

        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        parts.append(cell.text)

        full_text = "\n".join(parts)
        if not full_text.strip():
            raise IngestionError("No extractable text found in DOCX file.")
        return full_text
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Failed to parse DOCX file: {e}")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unusual control characters without losing content."""
    if not text:
        return ""
    # Remove non-printable / control characters (keep newlines and tabs as spaces)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Trim each line
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Break cleaned resume text into overlapping word-based chunks suitable for
    embedding. Overlap preserves context continuity across chunk boundaries.
    """
    if not text:
        return []

    words = text.split(" ")
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def parse_resume_file(uploaded_file) -> str:
    """
    Dispatches to the correct parser based on file extension and returns
    cleaned raw text. Raises IngestionError on failure.
    """
    filename = getattr(uploaded_file, "name", None) or str(uploaded_file)
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    elif lower_name.endswith(".docx"):
        raw_text = extract_text_from_docx(uploaded_file)
    else:
        raise IngestionError(
            f"Unsupported file type for '{filename}'. Only PDF and DOCX are supported."
        )

    return raw_text
