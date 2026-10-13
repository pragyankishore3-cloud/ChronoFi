import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from models.database import DocumentType

logger = logging.getLogger(__name__)

class DocumentMetadataExtraction(BaseModel):
    document_type: DocumentType = Field(description="The type of financial document.")
    financial_year: str | None = Field(None, description="The financial year (e.g. 2024-2025) if applicable.")
    effective_date: str | None = Field(None, description="The effective date or start date of the document (YYYY-MM-DD).")
    publication_date: str | None = Field(None, description="The date the document was published or generated (YYYY-MM-DD).")
    issuer: str | None = Field(None, description="The bank, insurance company, or institution that issued the document.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 of the overall extraction accuracy.")
    key_financial_data: dict | None = Field(default_factory=dict, description="Extracted structured key-value pairs (e.g. Gross Salary, Total Tax).")

def extract_metadata(pages_text: list[dict], filename: str) -> dict:
    """
    Analyzes the first few pages of a document to extract structured metadata.
    Uses Gemini structured output. If confidence is low, safely falls back to OTHER/None.
    """
    if not pages_text:
        return _fallback_metadata()

    # Limit to first 3 pages to save tokens and latency, usually enough for metadata
    sample_text = "\n".join([p["text"] for p in pages_text[:3]])
    
    prompt = f"""
    Analyze the following extracted text from a financial document.
    Determine its type, dates, issuer, and financial year.
    Additionally, extract any critical financial key-value pairs (e.g. "Gross Salary": 150000, "Tax Deducted": 5000, "Account Balance": 200) into the `key_financial_data` dictionary.
    Filename: {filename}
    
    CRITICAL RULE:
    If you are not highly confident about a field, return null. 
    If you cannot definitively determine the document_type, use 'OTHER'.
    Score your confidence between 0.0 (complete guess) and 1.0 (certain).
    
    Document Text:
    {sample_text[:4000]}
    """
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        structured_llm = llm.with_structured_output(DocumentMetadataExtraction)
        result: DocumentMetadataExtraction = structured_llm.invoke(prompt)
        
        # Enforce confidence threshold
        if result.confidence < 0.85:
            logger.warning(f"Low confidence ({result.confidence}) metadata extraction for {filename}. Falling back to safe defaults.")
            return _fallback_metadata()
            
        return {
            "document_type": result.document_type,
            "financial_year": result.financial_year,
            "effective_date": result.effective_date,
            "publication_date": result.publication_date,
            "issuer": result.issuer,
            "metadata_confidence": result.confidence,
            "key_financial_data": result.key_financial_data
        }
        
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e!s}")
        return _fallback_metadata()

def _fallback_metadata() -> dict:
    """Returns safe null values for failed or low-confidence extractions."""
    return {
        "document_type": DocumentType.OTHER,
        "financial_year": None,
        "effective_date": None,
        "publication_date": None,
        "issuer": None,
        "metadata_confidence": 0.0,
        "key_financial_data": {}
    }
