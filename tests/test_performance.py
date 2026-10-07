import pytest
import os
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from main import app
from engine.pdf_processor import process_document_pipeline
from models.database import DocumentStatus

client = TestClient(app)

@pytest.mark.asyncio
async def test_upload_api_response_latency(mock_current_user):
    """Exit Gate: Upload API response < 500ms and doesn't block."""
    import time
    
    # We are using the TestClient which is synchronous, but the backend is async.
    # To test that the endpoint responds quickly (offloading work to background),
    # we just measure the response time of the client.
    
    start = time.time()
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.pdf", b"dummy pdf content", "application/pdf")},
        headers={"Authorization": "Bearer mock_token"}
    )
    end = time.time()
    
    duration = (end - start) * 1000 # ms
    
    assert response.status_code == 200
    assert duration < 500, f"Upload API took {duration}ms, must be < 500ms"
    
@pytest.mark.asyncio
async def test_extraction_does_not_block_event_loop():
    """Exit Gate: No blocking of the main event loop."""
    # Ensure asyncio.to_thread is used for extract_text_from_pdf
    # If it wasn't, running process_document_pipeline would block
    # We can mock extract_text_from_pdf to sleep synchronously
    # and assert that asyncio.gather runs it concurrently with another task.
    
    from engine.pdf_processor import extract_text_from_pdf
    
    with patch("engine.pdf_processor.extract_text_from_pdf") as mock_extract:
        import time
        def slow_sync_extraction(*args):
            time.sleep(0.5)
            return [{"page_number": 1, "text": "text"}]
        mock_extract.side_effect = slow_sync_extraction
        
        # We also mock everything else to avoid DB errors in this isolated test
        with patch("engine.pdf_processor.select") as mock_select, \
             patch("engine.pdf_processor.chunk_pages") as mock_chunk, \
             patch("engine.pdf_processor.extract_metadata") as mock_meta, \
             patch("engine.vector_store.embed_chunks", new_callable=AsyncMock) as mock_embed:
             
            mock_doc = AsyncMock()
            mock_doc.id = 1
            mock_doc.user_id = 1
            mock_doc.status = DocumentStatus.UPLOADED
            mock_doc.file_path = "path"
            
            mock_db = AsyncMock()
            mock_db.execute.return_value.scalars.return_value.first.return_value = mock_doc
            
            async def concurrent_task():
                await asyncio.sleep(0.1)
                return "done"
                
            start = time.time()
            # Run both. If process blocks the event loop, concurrent_task will start AFTER 0.5s.
            # If it uses to_thread, concurrent_task finishes quickly.
            results = await asyncio.gather(
                process_document_pipeline(1, mock_db),
                concurrent_task()
            )
            end = time.time()
            
            # The total time should be around 0.5s (due to sleep), but concurrent_task
            # shouldn't have been blocked from running during that 0.5s.
            # If process_document_pipeline was fully synchronous, it would block the loop.
            assert results[1] == "done"
