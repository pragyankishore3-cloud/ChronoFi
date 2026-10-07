import os
import logging
import fitz  # PyMuPDF
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.database import Document, DocumentChunk, DocumentStatus

logger = logging.getLogger(__name__)

# Configurable chunking params for financial documents
# We want chunks large enough to retain context but small enough for precise retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

async def process_document_pipeline(document_id: int, db: AsyncSession):
    """
    Orchestrates the entire PDF processing pipeline:
    Extraction -> Cleaning -> Chunking -> Saving to DB.
    Updates the Document status deterministically.
    """
    # 1. Fetch Document
    result = await db.execute(select(Document).filter(Document.id == document_id))
    doc = result.scalars().first()
    
    if not doc:
        logger.error(f"Document {document_id} not found.")
        return

    try:
        # State: PROCESSING
        doc.status = DocumentStatus.PROCESSING
        await db.commit()
        
        # State: EXTRACTING
        doc.status = DocumentStatus.EXTRACTING
        await db.commit()
        
        import asyncio
        pages_text = await asyncio.to_thread(extract_text_from_pdf, doc.file_path)
        
        # State: CHUNKING
        doc.status = DocumentStatus.CHUNKING
        await db.commit()
        
        # Extract Metadata
        from engine.metadata_extractor import extract_metadata
        metadata = extract_metadata(pages_text, doc.filename)
        doc.document_type = metadata["document_type"]
        doc.financial_year = metadata["financial_year"]
        doc.effective_date = metadata["effective_date"]
        doc.publication_date = metadata["publication_date"]
        doc.issuer = metadata["issuer"]
        doc.metadata_confidence = metadata["metadata_confidence"]
        
        import json
        if metadata.get("key_financial_data"):
            doc.extracted_data = json.dumps(metadata["key_financial_data"])
        
        chunks = chunk_pages(pages_text, doc.id, doc.user_id)
        
        # Save chunks to DB
        db.add_all(chunks)
        await db.commit()
        
        # State: EMBEDDING
        doc.status = DocumentStatus.EMBEDDING
        await db.commit()
        
        from engine.vector_store import embed_chunks
        await embed_chunks(chunks, db)
        
        # State: INDEXED -> READY
        doc.status = DocumentStatus.INDEXED
        await db.commit()
        
        doc.status = DocumentStatus.READY
        await db.commit()
        logger.info(f"Document {document_id} successfully processed and embedded into {len(chunks)} chunks.")

    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {str(e)}")
        doc.status = DocumentStatus.FAILED
        await db.commit()
        raise e

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extracts text preserving page numbers.
    Utilizes PyMuPDF (fitz) to accurately extract text, tables, and financial numbers.
    Returns: [{"page_number": 1, "text": "..."}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = []
    # fitz handles malformed PDFs gracefully in many cases, or raises explicitly
    try:
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # extract_text(sort=True) attempts to read naturally left-to-right, top-to-bottom
                text = page.get_text("text", sort=True)
                
                # Basic cleaning: remove excessive newlines but preserve paragraph structure
                clean_text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
                
                if clean_text:
                    pages.append({
                        "page_number": page_num + 1,  # 1-indexed
                        "text": clean_text
                    })
    except Exception as e:
        raise RuntimeError(f"PyMuPDF failed to extract text: {str(e)}")
        
    return pages

def chunk_pages(pages_text: list[dict], document_id: int, user_id: int) -> list[DocumentChunk]:
    """
    Chunks extracted pages while strictly preserving document metadata and page mapping.
    Uses RecursiveCharacterTextSplitter optimized for paragraphs and financial statements.
    """
    # Financial docs often use tables and bullet points. 
    # Splitting by double newline first preserves paragraph boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    db_chunks = []
    global_chunk_index = 0
    
    for page in pages_text:
        page_num = page["page_number"]
        page_content = page["text"]
        
        # Split text for this specific page
        chunks = text_splitter.split_text(page_content)
        
        for chunk_text in chunks:
            # We don't guess sections in Phase 2 unless it's trivial, Phase 13 handles intelligence.
            # But we perfectly preserve Page mapping and numbers.
            db_chunks.append(DocumentChunk(
                document_id=document_id,
                user_id=user_id,
                page_number=page_num,
                chunk_index=global_chunk_index,
                content=chunk_text
            ))
            global_chunk_index += 1
            
    return db_chunks
