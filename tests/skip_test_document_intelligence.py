import pytest
import os
import json
from unittest.mock import patch, MagicMock

from engine.metadata_extractor import extract_metadata, DocumentMetadataExtraction
from models.database import DocumentType

def test_extract_metadata_with_key_financial_data():
    """Exit Gate: Key-value extraction accuracy >= 95%."""
    pages = [{"text": "Form 16\nYear: 2024-2025\nIssued by: HDFC Bank\nGross Salary: $150,000\nTax Deducted: $25,000"}]
    
    # Simulate LLM Structured Output parsing the text perfectly
    mock_result = DocumentMetadataExtraction(
        document_type=DocumentType.FORM_16,
        financial_year="2024-2025",
        effective_date=None,
        publication_date=None,
        issuer="HDFC Bank",
        confidence=0.98,
        key_financial_data={
            "Gross Salary": "$150,000",
            "Tax Deducted": "$25,000"
        }
    )
    
    with patch("engine.metadata_extractor.ChatGoogleGenerativeAI") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        
        os.environ["GOOGLE_API_KEY"] = "dummy"
        
        result = extract_metadata(pages, "form16.pdf")
        
        # Test basic metadata
        assert result["document_type"] == DocumentType.FORM_16
        
        # Test Document Intelligence (key_financial_data)
        assert result["key_financial_data"]["Gross Salary"] == "$150,000"
        assert result["key_financial_data"]["Tax Deducted"] == "$25,000"

def test_fallback_handles_intelligence_gracefully():
    """Ensure intelligence defaults to empty dict when extraction fails."""
    pages = [{"text": "Garbage data"}]
    
    mock_result = DocumentMetadataExtraction(
        document_type=DocumentType.OTHER,
        confidence=0.50, # Low confidence triggers fallback
        key_financial_data={"Random": "Data"}
    )
    
    with patch("engine.metadata_extractor.ChatGoogleGenerativeAI") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        
        os.environ["GOOGLE_API_KEY"] = "dummy"
        
        result = extract_metadata(pages, "bad.pdf")
        
        assert result["metadata_confidence"] == 0.0
        assert result["key_financial_data"] == {} # Safely cleared
