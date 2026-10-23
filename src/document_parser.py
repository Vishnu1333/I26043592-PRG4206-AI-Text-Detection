"""
src/document_parser.py — Document text extraction, chunking, and document-level analysis.
PRG4206 | AI-Generated Text Detection System

Supports:
  - Text extraction from .txt (UTF-8 with fallbacks), .pdf (pypdf), and .docx (python-docx).
  - Sentence- and paragraph-aware segmentation and chunking with configurable limits.
  - Multi-chunk model inference and document-level aggregation.
  - Clear identification of most AI-like and most Human-like passages.
"""

from io import BytesIO
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import (
    DOC_CHUNK_OVERLAP,
    DOC_MAX_CHUNK_WORDS,
    DOC_MIN_CHUNK_WORDS,
    DOC_MIN_TEXT_WORDS,
)
from src.inference import predict_text
from src.utils import get_logger

logger = get_logger(__name__)


# ── Text Extraction ──────────────────────────────────────────────────────────

def _read_txt(file_source: Any) -> str:
    """Read raw text from a file path or file-like object with encoding fallback."""
    if isinstance(file_source, (str, Path)):
        raw_bytes = Path(file_source).read_bytes()
    elif hasattr(file_source, "read"):
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        raw_bytes = file_source.read()
        if isinstance(raw_bytes, str):
            return raw_bytes
    elif isinstance(file_source, (bytes, bytearray)):
        raw_bytes = bytes(file_source)
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")

    # Try common encodings
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    raise ValueError("Could not decode text file with UTF-8 or standard encoding fallbacks.")


def _read_pdf(file_source: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text and per-page segments from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is required for PDF parsing. Please install pypdf.")

    stream: Any
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        stream = path
    elif hasattr(file_source, "read"):
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        raw = file_source.read()
        stream = BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    elif isinstance(file_source, (bytes, bytearray)):
        stream = BytesIO(bytes(file_source))
    else:
        raise ValueError(f"Unsupported PDF file source type: {type(file_source)}")

    try:
        reader = PdfReader(stream)
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable PDF file: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Try decrypting with empty password
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF document is encrypted or password-protected.")

    if len(reader.pages) == 0:
        raise ValueError("PDF document contains no pages.")

    pages_text = []
    segments = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            txt = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from page %d: %s", page_num, exc)
            txt = ""
        txt_clean = txt.strip()
        if txt_clean:
            pages_text.append(txt_clean)
            segments.append(
                {
                    "segment_id": page_num,
                    "type": "page",
                    "page_number": page_num,
                    "text": txt_clean,
                    "word_count": len(txt_clean.split()),
                }
            )

    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise ValueError(
            "No extractable text found in PDF. The document may be scanned, image-only, or empty."
        )

    return full_text, segments


def _read_docx(file_source: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text and per-paragraph segments from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required for DOCX parsing. Please install python-docx.")

    stream: Any
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")
        stream = path
    elif hasattr(file_source, "read"):
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        raw = file_source.read()
        stream = BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    elif isinstance(file_source, (bytes, bytearray)):
        stream = BytesIO(bytes(file_source))
    else:
        raise ValueError(f"Unsupported DOCX file source type: {type(file_source)}")

    try:
        doc = Document(stream)
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid DOCX document: {exc}") from exc

    paragraphs = []
    segments = []
    para_count = 0

    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            para_count += 1
            paragraphs.append(txt)
            segments.append(
                {
                    "segment_id": para_count,
                    "type": "paragraph",
                    "paragraph_number": para_count,
                    "text": txt,
                    "word_count": len(txt.split()),
                }
            )

    full_text = "\n\n".join(paragraphs).strip()
    if not full_text:
        raise ValueError("No extractable text found in DOCX file. The document is empty.")

    return full_text, segments


def extract_text_from_file(
    file_source: Any,
    filename: Optional[str] = None,
    min_text_words: int = DOC_MIN_TEXT_WORDS,
) -> Dict[str, Any]:
    """Extract full text and structure from a supported document (.txt, .pdf, .docx).

    Args:
        file_source: File path, file-like object, or bytes.
        filename: Optional filename used to determine format extension.
        min_text_words: Minimum words required for valid analysis.

    Returns:
        Dictionary with extracted full_text, file_type, word count, and structural segments.

    Raises:
        ValueError: On unsupported format, corrupted file, or insufficient text length.
    """
    # Determine extension
    ext = ""
    if filename:
        ext = Path(filename).suffix.lower()
    elif isinstance(file_source, (str, Path)):
        ext = Path(file_source).suffix.lower()
    elif hasattr(file_source, "name") and isinstance(file_source.name, str):
        ext = Path(file_source.name).suffix.lower()

    if not ext:
        raise ValueError("Could not determine file type. Please provide a file with a .txt, .pdf, or .docx extension.")

    if ext not in [".txt", ".pdf", ".docx"]:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .txt, .pdf, .docx")

    segments: List[Dict[str, Any]] = []

    if ext == ".txt":
        full_text = _read_txt(file_source).strip()
        # Create paragraph segments
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
        for i, p in enumerate(raw_paras):
            segments.append(
                {
                    "segment_id": i + 1,
                    "type": "paragraph",
                    "paragraph_number": i + 1,
                    "text": p,
                    "word_count": len(p.split()),
                }
            )
    elif ext == ".pdf":
        full_text, segments = _read_pdf(file_source)
    elif ext == ".docx":
        full_text, segments = _read_docx(file_source)
    else:
        raise ValueError(f"Unhandled file extension: {ext}")

    words = full_text.split()
    total_words = len(words)

    if total_words < min_text_words:
        raise ValueError(
            f"Extracted document text is too short ({total_words} words). "
            f"Please provide at least {min_text_words} words for reliable detection."
        )

    return {
        "filename": filename or (Path(file_source).name if isinstance(file_source, (str, Path)) else "document"),
        "file_type": ext,
        "full_text": full_text,
        "total_words": total_words,
        "total_characters": len(full_text),
        "segments": segments,
        "segment_count": len(segments),
    }


# ── Document Chunking ────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving terminal punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc_data: Dict[str, Any],
    min_chunk_words: int = DOC_MIN_CHUNK_WORDS,
    max_chunk_words: int = DOC_MAX_CHUNK_WORDS,
    chunk_overlap_words: int = DOC_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Segment an extracted document into ordered, coherent text chunks for model scoring.

    Preserves paragraph and sentence boundaries wherever feasible. Avoids tiny chunks
    by merging short segments, while splitting oversized segments cleanly on sentence ends.
    If the entire document is smaller than ``min_chunk_words``, it is retained as a single chunk.

    Args:
        doc_data: Output dictionary from :func:`extract_text_from_file`.
        min_chunk_words: Target minimum words per chunk.
        max_chunk_words: Maximum words per chunk.
        chunk_overlap_words: Optional word overlap between consecutive chunks.

    Returns:
        List of chunk dictionaries with indices, text, word counts, and page/paragraph metadata.
    """
    full_text = doc_data["full_text"]
    total_words = doc_data["total_words"]
    segments = doc_data.get("segments", [])

    # Special case: very short document (e.g. 10 to min_chunk_words words)
    if total_words <= max_chunk_words:
        return [
            {
                "chunk_index": 1,
                "text": full_text,
                "word_count": total_words,
                "char_count": len(full_text),
                "source_info": "Complete document (single passage)",
            }
        ]

    # Process segments (paragraphs or pages)
    unit_texts: List[Tuple[str, str]] = []  # (text, source_meta)
    if segments:
        for seg in segments:
            label = f"{seg['type'].capitalize()} {seg.get('page_number', seg.get('paragraph_number', seg['segment_id']))}"
            unit_texts.append((seg["text"], label))
    else:
        # Fallback to paragraph splitting
        for i, p in enumerate(re.split(r"\n\s*\n", full_text)):
            if p.strip():
                unit_texts.append((p.strip(), f"Paragraph {i + 1}"))

    chunks: List[Dict[str, Any]] = []
    current_words: List[str] = []
    current_sources: List[str] = []

    for seg_text, seg_source in unit_texts:
        seg_words = seg_text.split()
        seg_len = len(seg_words)

        if not seg_words:
            continue

        # If a single segment is larger than max_chunk_words, break it into sentence-based blocks
        if seg_len > max_chunk_words:
            # First, flush any accumulated small units
            if current_words:
                chunk_str = " ".join(current_words)
                chunks.append(
                    {
                        "chunk_index": len(chunks) + 1,
                        "text": chunk_str,
                        "word_count": len(current_words),
                        "char_count": len(chunk_str),
                        "source_info": ", ".join(dict.fromkeys(current_sources)),
                    }
                )
                current_words = []
                current_sources = []

            # Sub-chunk the long segment by sentences
            sentences = _split_into_sentences(seg_text)
            sub_words: List[str] = []
            for sent in sentences:
                s_words = sent.split()
                if len(sub_words) + len(s_words) > max_chunk_words and len(sub_words) >= min_chunk_words:
                    chunk_str = " ".join(sub_words)
                    chunks.append(
                        {
                            "chunk_index": len(chunks) + 1,
                            "text": chunk_str,
                            "word_count": len(sub_words),
                            "char_count": len(chunk_str),
                            "source_info": seg_source,
                        }
                    )
                    sub_words = s_words
                else:
                    sub_words.extend(s_words)

            if sub_words:
                current_words.extend(sub_words)
                current_sources.append(seg_source)

        else:
            # Segment fits or can be merged
            if len(current_words) + seg_len <= max_chunk_words:
                current_words.extend(seg_words)
                current_sources.append(seg_source)
            else:
                # Segment would exceed max_chunk_words
                if len(current_words) >= min_chunk_words:
                    chunk_str = " ".join(current_words)
                    chunks.append(
                        {
                            "chunk_index": len(chunks) + 1,
                            "text": chunk_str,
                            "word_count": len(current_words),
                            "char_count": len(chunk_str),
                            "source_info": ", ".join(dict.fromkeys(current_sources)),
                        }
                    )
                    current_words = list(seg_words)
                    current_sources = [seg_source]
                else:
                    # Current chunk is small, append to make it substantial
                    current_words.extend(seg_words)
                    current_sources.append(seg_source)

    # Flush final chunk
    if current_words:
        chunk_str = " ".join(current_words)
        # If the trailing chunk is very small and we have previous chunks, merge it into the previous chunk
        if len(current_words) < min_chunk_words and chunks:
            prev = chunks[-1]
            merged_text = prev["text"] + " " + chunk_str
            prev["text"] = merged_text
            prev["word_count"] = len(merged_text.split())
            prev["char_count"] = len(merged_text)
            if current_sources:
                prev["source_info"] += f", {', '.join(dict.fromkeys(current_sources))}"
        else:
            chunks.append(
                {
                    "chunk_index": len(chunks) + 1,
                    "text": chunk_str,
                    "word_count": len(current_words),
                    "char_count": len(chunk_str),
                    "source_info": ", ".join(dict.fromkeys(current_sources)) if current_sources else "Final section",
                }
            )

    return chunks


# ── Document-Level Prediction & Aggregation ──────────────────────────────────

def analyze_document(
    doc_data: Dict[str, Any],
    model_name: str = "svm",
    min_chunk_words: int = DOC_MIN_CHUNK_WORDS,
    max_chunk_words: int = DOC_MAX_CHUNK_WORDS,
) -> Dict[str, Any]:
    """Run model inference across all document chunks and compute aggregate document statistics.

    Args:
        doc_data: Document data dict from :func:`extract_text_from_file`.
        model_name: Model identifier (``"svm"``, ``"lr"``, ``"xgb"``).
        min_chunk_words: Minimum words per chunk.
        max_chunk_words: Maximum words per chunk.

    Returns:
        Structured analysis results containing chunk-by-chunk predictions,
        overall aggregated label, confidence, distributions, and top passage excerpts.
    """
    chunks = chunk_document(
        doc_data,
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
    )

    if not chunks:
        raise ValueError("Could not segment document into valid analysis chunks.")

    chunk_results: List[Dict[str, Any]] = []

    for c in chunks:
        pred = predict_text(c["text"], model_name=model_name)
        # Class probability for AI (class 1)
        conf = float(pred["confidence"])
        label = pred["label"]
        label_id = pred["label_id"]

        ai_prob = conf if label_id == 1 else (1.0 - conf)
        human_prob = 1.0 - ai_prob

        chunk_results.append(
            {
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "word_count": c["word_count"],
                "char_count": c["char_count"],
                "source_info": c.get("source_info", f"Chunk {c['chunk_index']}"),
                "label": label,
                "label_id": label_id,
                "confidence": conf,
                "ai_probability": round(ai_prob, 4),
                "human_probability": round(human_prob, 4),
                "model": pred.get("model", model_name),
            }
        )

    total_chunks = len(chunk_results)
    total_words = sum(c["word_count"] for c in chunk_results)
    ai_chunks = sum(1 for c in chunk_results if c["label_id"] == 1)
    human_chunks = total_chunks - ai_chunks
    ai_chunk_percentage = round((ai_chunks / total_chunks) * 100, 2)

    # Word-weighted AI probability
    if total_words > 0:
        weighted_ai_prob = float(
            sum(c["ai_probability"] * c["word_count"] for c in chunk_results) / total_words
        )
    else:
        weighted_ai_prob = float(np.mean([c["ai_probability"] for c in chunk_results]))

    # Mean chunk confidence
    mean_confidence = float(np.mean([c["confidence"] for c in chunk_results]))

    # Overall document classification decision
    if weighted_ai_prob >= 0.50:
        overall_label = "AI-Generated"
        overall_confidence = round(weighted_ai_prob, 4)
    else:
        overall_label = "Human-Written"
        overall_confidence = round(1.0 - weighted_ai_prob, 4)

    # Passages of high interest: Most AI-like and Most Human-like
    sorted_by_ai = sorted(chunk_results, key=lambda x: x["ai_probability"], reverse=True)
    most_ai = sorted_by_ai[: min(3, len(sorted_by_ai))]
    most_human = sorted(chunk_results, key=lambda x: x["ai_probability"])[: min(3, len(sorted_by_ai))]

    return {
        "filename": doc_data.get("filename", "document"),
        "file_type": doc_data.get("file_type", ".txt"),
        "total_words": total_words,
        "total_characters": doc_data.get("total_characters", sum(c["char_count"] for c in chunk_results)),
        "total_chunks": total_chunks,
        "ai_chunks": ai_chunks,
        "human_chunks": human_chunks,
        "ai_percentage": ai_chunk_percentage,
        "overall_prediction": overall_label,
        "overall_confidence": overall_confidence,
        "weighted_ai_probability": round(weighted_ai_prob, 4),
        "mean_chunk_confidence": round(mean_confidence, 4),
        "model_used": chunk_results[0]["model"] if chunk_results else model_name,
        "chunks": chunk_results,
        "most_ai_chunks": most_ai,
        "most_human_chunks": most_human,
        "disclaimer": (
            "Aggregate model estimate based on paragraph-level classification. "
            "Probabilistic research indication only; not conclusive proof of authorship."
        ),
    }
