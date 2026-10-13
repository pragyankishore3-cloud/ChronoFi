import logging
from typing import Any

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.database import DocumentChunk

logger = logging.getLogger(__name__)

# Use Gemini embeddings optimized for retrieval
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

async def embed_chunks(chunks: list[DocumentChunk], db: AsyncSession):
    """
    Generates embeddings for a batch of DocumentChunks and persists them.
    Ensures 100% of successfully processed chunks receive embeddings.
    """
    if not chunks:
        return

    texts = [chunk.content for chunk in chunks]
    
    try:
        # Generate embeddings in batch
        embeddings = await embeddings_model.aembed_documents(texts)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            
        db.add_all(chunks)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e!s}")
        raise e

async def retrieve_documents(
    query: str, 
    user_id: int, 
    db: AsyncSession, 
    top_k: int = 5,
    similarity_threshold: float = 0.70,
    filters: dict[str, Any] = None
) -> list[DocumentChunk]:
    """
    CRITICAL SECURITY RULE: Never perform unrestricted vector search.
    Always enforces authorization by filtering rigidly on user_id.
    
    Supports:
    - Semantic Search
    - Metadata filtering (joined via Document)
    - Similarity Thresholding (rejects low confidence)
    """
    query_embedding = await embeddings_model.aembed_query(query)
    dialect = db.bind.dialect.name
    
    from sqlalchemy.orm import selectinload

    from models.database import Document
    
    # Base query joined with Document for metadata filtering
    # Use selectinload to eagerly fetch the document for citations (filename)
    stmt = select(DocumentChunk).join(Document).options(selectinload(DocumentChunk.document)).filter(DocumentChunk.user_id == user_id)
    
    # Apply metadata filters if provided
    if filters:
        if "document_type" in filters:
            stmt = stmt.filter(Document.document_type == filters["document_type"])
        if "financial_year" in filters:
            stmt = stmt.filter(Document.financial_year == filters["financial_year"])
        if "issuer" in filters:
            stmt = stmt.filter(Document.issuer == filters["issuer"])
    
    if dialect == "postgresql":
        # Native pgvector cosine distance (distance = 1 - similarity)
        # Cosine distance < 0.3 means similarity > 0.7
        distance_threshold = 1.0 - similarity_threshold
        
        stmt = stmt.filter(
            DocumentChunk.embedding.cosine_distance(query_embedding) <= distance_threshold
        ).order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        ).limit(top_k)
        
        result = await db.execute(stmt)
        return list(result.scalars().all())
    else:
        # SQLite Fallback for testing
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            return []
            
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        query_np = np.array(query_embedding)
        
        scored_chunks = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            chunk_np = np.array(chunk.embedding)
            score = cosine_sim(query_np, chunk_np)
            if score >= similarity_threshold:
                scored_chunks.append((score, chunk))
                
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in scored_chunks[:top_k]]

async def retrieve_trusted_knowledge(
    query: str, 
    db: AsyncSession, 
    top_k: int = 3,
    similarity_threshold: float = 0.70
) -> list[Any]:
    """
    Retrieves global trusted knowledge (tax codes, rules).
    Logically completely isolated from user data (queries KnowledgeBaseChunk).
    """
    from models.database import KnowledgeBaseChunk
    
    query_embedding = await embeddings_model.aembed_query(query)
    dialect = db.bind.dialect.name
    
    stmt = select(KnowledgeBaseChunk)
    
    if dialect == "postgresql":
        distance_threshold = 1.0 - similarity_threshold
        stmt = stmt.filter(
            KnowledgeBaseChunk.embedding.cosine_distance(query_embedding) <= distance_threshold
        ).order_by(
            KnowledgeBaseChunk.embedding.cosine_distance(query_embedding)
        ).limit(top_k)
        
        result = await db.execute(stmt)
        return list(result.scalars().all())
    else:
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        if not chunks:
            return []
            
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        query_np = np.array(query_embedding)
        scored_chunks = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            chunk_np = np.array(chunk.embedding)
            score = cosine_sim(query_np, chunk_np)
            if score >= similarity_threshold:
                scored_chunks.append((score, chunk))
                
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in scored_chunks[:top_k]]
