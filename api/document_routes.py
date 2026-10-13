"""
ChronoFi Document Management API.
Handles document upload, validation, background extraction, and listing.
"""

import os
from tempfile import NamedTemporaryFile
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api import deps
from engine.pdf_processor import process_document_pipeline
from models import database

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class DocumentResponse(BaseModel):
    id: int
    filename: str
    status: str
    size_bytes: int
    document_type: Optional[str] = None
    financial_year: Optional[str] = None
    issuer: Optional[str] = None


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Uploads a financial PDF document, validates size and MIME type,
    and schedules extraction in the background.
    """
    # 1. MIME / Extension validation
    if file.content_type != "application/pdf" and not (file.filename and file.filename.lower().endswith(".pdf")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # 2. Read contents and check size
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded. Please provide a valid file.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File exceeds maximum size of 10MB.",
        )

    # 3. Save to disk
    os.makedirs("./uploads", exist_ok=True)
    file_path = f"./uploads/{current_user.id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    # 4. Create Document record
    doc = database.Document(
        user_id=current_user.id,
        filename=file.filename or "uploaded.pdf",
        file_path=file_path,
        mime_type="application/pdf",
        size_bytes=len(contents),
        status=database.DocumentStatus.UPLOADED,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Schedule background processing pipeline
    # Note: process_document_pipeline opens its own AsyncSession or we pass db
    background_tasks.add_task(_run_pipeline_bg, doc.id)

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status="UPLOADED",
        size_bytes=doc.size_bytes,
        document_type=doc.document_type,
        financial_year=doc.financial_year,
        issuer=doc.issuer,
    )


async def _run_pipeline_bg(document_id: int):
    """Background helper to execute pipeline with a fresh session."""
    try:
        async with database.AsyncSessionLocal() as db:
            await process_document_pipeline(document_id, db)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error processing document {document_id} in background: {e}")


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """Lists all uploaded documents strictly belonging to the authenticated user."""
    res = await db.execute(
        select(database.Document)
        .where(database.Document.user_id == current_user.id)
        .order_by(database.Document.id.desc())
    )
    docs = res.scalars().all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            status=d.status.value if hasattr(d.status, "value") else str(d.status),
            size_bytes=d.size_bytes or 0,
            document_type=d.document_type,
            financial_year=d.financial_year,
            issuer=d.issuer,
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """Retrieves document metadata by ID, ensuring user isolation."""
    res = await db.execute(
        select(database.Document)
        .where(database.Document.id == document_id, database.Document.user_id == current_user.id)
    )
    doc = res.scalars().first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
        size_bytes=doc.size_bytes or 0,
        document_type=doc.document_type,
        financial_year=doc.financial_year,
        issuer=doc.issuer,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """Deletes document and cascades to its chunks."""
    res = await db.execute(
        select(database.Document)
        .where(database.Document.id == document_id, database.Document.user_id == current_user.id)
    )
    doc = res.scalars().first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted successfully"}
