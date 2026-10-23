"""
tests/test_document_parser.py — Automated Unit Tests for Document Extraction and Analysis.
PRG4206 | AI-Generated Text Detection System

Tests:
  - TXT extraction with UTF-8 and encoding fallback
  - PDF extraction
  - DOCX extraction
  - Empty and short document error handling
  - Unsupported file format rejection
  - Chunking boundaries and order preservation
  - Short vs long document chunking
  - Document-level prediction aggregation
  - Identification of top AI-like and Human-like passages
  - Cross-model document analysis (SVM, LR, XGBoost)
"""

import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import docx
import pytest

from src.document_parser import (
    extract_text_from_file,
    chunk_document,
    analyze_document,
)

SAMPLE_PDF_BYTES = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 120>>stream
BT
/F1 12 Tf
100 700 Td
(Artificial intelligence models have achieved remarkable success in automated text generation across domains.) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000224 00000 n 
0000000295 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
467
%%EOF"""


class TestDocumentExtraction:
    """Tests covering file text extraction and format validation."""

    def test_txt_extraction_utf8(self):
        text = (
            "Machine learning is an empirical branch of computational statistics.\n\n"
            "It develops algorithmic models capable of learning patterns from historical data."
        )
        f = io.BytesIO(text.encode("utf-8"))
        f.name = "sample.txt"
        data = extract_text_from_file(f)
        assert data["file_type"] == ".txt"
        assert data["total_words"] > 15
        assert data["segment_count"] == 2
        assert "computational statistics" in data["full_text"]

    def test_txt_extraction_encoding_fallback(self):
        # Text with latin-1 encoded accented characters
        text_latin1 = "Étude approfondie des modèles d'apprentissage automatique pour la détection de texte.".encode("latin-1")
        f = io.BytesIO(text_latin1)
        f.name = "french.txt"
        data = extract_text_from_file(f)
        assert data["total_words"] >= 10
        assert "modèles" in data["full_text"]

    def test_docx_extraction_valid(self):
        doc = docx.Document()
        doc.add_paragraph("First paragraph discussing natural language processing architectures.")
        doc.add_paragraph("Second paragraph addressing transformer self-attention mechanisms.")
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        buf.name = "essay.docx"

        data = extract_text_from_file(buf)
        assert data["file_type"] == ".docx"
        assert data["segment_count"] == 2
        assert data["total_words"] == 13
        assert "transformer" in data["full_text"]

    def test_pdf_extraction_valid(self):
        f = io.BytesIO(SAMPLE_PDF_BYTES)
        f.name = "paper.pdf"
        data = extract_text_from_file(f)
        assert data["file_type"] == ".pdf"
        assert data["total_words"] == 13
        assert "Artificial intelligence" in data["full_text"]

    def test_empty_document_raises_error(self):
        f = io.BytesIO(b"   \n  \t  ")
        f.name = "empty.txt"
        with pytest.raises(ValueError, match="too short"):
            extract_text_from_file(f)

    def test_short_document_below_min_words_raises(self):
        f = io.BytesIO(b"Hello world this is five words.")
        f.name = "short.txt"
        with pytest.raises(ValueError, match="too short"):
            extract_text_from_file(f, min_text_words=10)

    def test_unsupported_file_extension_raises(self):
        f = io.BytesIO(b"data,values,points")
        f.name = "data.csv"
        with pytest.raises(ValueError, match="Unsupported file format"):
            extract_text_from_file(f)


class TestDocumentChunking:
    """Tests covering document segmentation and chunking."""

    def test_chunking_short_document_produces_single_chunk(self):
        doc_data = {
            "filename": "short.txt",
            "file_type": ".txt",
            "full_text": "This is a concise paragraph with exactly twelve words written for unit testing.",
            "total_words": 12,
            "segments": [{"segment_id": 1, "type": "paragraph", "text": "This is a concise paragraph with exactly twelve words written for unit testing."}],
        }
        chunks = chunk_document(doc_data, min_chunk_words=50, max_chunk_words=250)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 1
        assert chunks[0]["word_count"] == 12

    def test_chunking_preserves_order_and_bounds(self):
        # Generate 10 paragraphs of 30 words each = 300 words total
        paras = []
        for i in range(10):
            paras.append(f"Paragraph {i+1}: " + " ".join([f"token_{j}" for j in range(28)]) + ".")
        full_text = "\n\n".join(paras)

        doc_data = {
            "filename": "multi_para.txt",
            "file_type": ".txt",
            "full_text": full_text,
            "total_words": len(full_text.split()),
            "segments": [
                {"segment_id": idx + 1, "type": "paragraph", "text": p}
                for idx, p in enumerate(paras)
            ],
        }

        chunks = chunk_document(doc_data, min_chunk_words=50, max_chunk_words=100)
        assert len(chunks) >= 3

        # Check order and chunk indices
        for idx, c in enumerate(chunks):
            assert c["chunk_index"] == idx + 1
            assert c["word_count"] <= 120  # bounded by max_chunk_words with small margin

        # Check total words preserved
        combined_words = sum(c["word_count"] for c in chunks)
        assert combined_words == doc_data["total_words"]


class TestDocumentAnalysis:
    """Tests covering end-to-end model inference and document-level aggregation."""

    @pytest.fixture
    def sample_doc(self):
        text = (
            "Natural language processing and machine learning algorithms have revolutionized modern digital computing. "
            "In this study, researchers evaluate how statistical classifiers separate synthetic text from human writing.\n\n"
            "Furthermore, empirical benchmarking across heterogeneous domains indicates that lexical vocabulary signals "
            "provide substantial discriminative power. In summary, rigorous cross-validation and explainability are essential."
        )
        f = io.BytesIO(text.encode("utf-8"))
        f.name = "sample_research.txt"
        return extract_text_from_file(f)

    def test_analyze_document_svm(self, sample_doc):
        res = analyze_document(sample_doc, model_name="svm", min_chunk_words=20, max_chunk_words=50)
        assert res["total_chunks"] >= 1
        assert res["overall_prediction"] in ["AI-Generated", "Human-Written"]
        assert 0.0 <= res["overall_confidence"] <= 1.0
        assert res["ai_chunks"] + res["human_chunks"] == res["total_chunks"]
        assert 0.0 <= res["ai_percentage"] <= 100.0
        assert len(res["chunks"]) == res["total_chunks"]
        assert len(res["most_ai_chunks"]) <= 3
        assert len(res["most_human_chunks"]) <= 3

    def test_analyze_document_all_models(self, sample_doc):
        for model in ["svm", "lr", "xgb"]:
            res = analyze_document(sample_doc, model_name=model, min_chunk_words=20, max_chunk_words=50)
            assert res["total_chunks"] >= 1
            assert res["overall_confidence"] >= 0.5
            assert "disclaimer" in res


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
