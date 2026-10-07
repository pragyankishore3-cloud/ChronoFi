import pytest
import os
from unittest.mock import patch, MagicMock
from engine.metadata_extractor import extract_metadata, DocumentMetadataExtraction
from models.database import DocumentType

def test_extract_metadata_success():
    """Exit Gate: Extracts metadata accurately with high confidence."""
    pages = [{"text": "Form 16\nYear: 2024-2025\nIssued by: HDFC Bank"}]
    
    mock_result = DocumentMetadataExtraction(
        document_type=DocumentType.FORM_16,
        financial_year="2024-2025",
        effective_date=None,
        publication_date=None,
        issuer="HDFC Bank",
        confidence=0.95
    )
    
    with patch("engine.metadata_extractor.ChatGoogleGenerativeAI") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        
        # We need the dummy key to bypass pydantic validation in tests
        os.environ["GOOGLE_API_KEY"] = "dummy"
        
        result = extract_metadata(pages, "form16.pdf")
        
        assert result["document_type"] == DocumentType.FORM_16
        assert result["financial_year"] == "2024-2025"
        assert result["issuer"] == "HDFC Bank"
        assert result["metadata_confidence"] == 0.95

def test_extract_metadata_low_confidence_fallback():
    """Exit Gate: Critical financial metadata must never silently become authoritative when extraction confidence is inadequate."""
    pages = [{"text": "Some vague document with a date 2025 but not sure what it is."}]
    
    # LLM guesses it's a loan but with low confidence
    mock_result = DocumentMetadataExtraction(
        document_type=DocumentType.LOAN,
        financial_year="2025",
        effective_date=None,
        publication_date=None,
        issuer=None,
        confidence=0.60 # Below 0.85 threshold
    )
    
    with patch("engine.metadata_extractor.ChatGoogleGenerativeAI") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        
        os.environ["GOOGLE_API_KEY"] = "dummy"
        
        result = extract_metadata(pages, "vague.pdf")
        
        # It must safely fallback to OTHER and nullify guesses
        assert result["document_type"] == DocumentType.OTHER
        assert result["financial_year"] is None
        assert result["metadata_confidence"] == 0.0

def test_extract_metadata_empty_or_error():
    """Exit Gate: Handled gracefully on error."""
    # Empty pages
    res1 = extract_metadata([], "empty.pdf")
    assert res1["document_type"] == DocumentType.OTHER
    
    # Exception during LLM call
    with patch("engine.metadata_extractor.ChatGoogleGenerativeAI", side_effect=Exception("API Down")):
        res2 = extract_metadata([{"text": "Data"}], "error.pdf")
        assert res2["document_type"] == DocumentType.OTHER
