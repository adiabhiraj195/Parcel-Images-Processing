"""In-process asynchronous job queue with bounded concurrency for Apple Silicon M1."""

import asyncio
import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.worker.orchestrator import orchestrator
from app.models.image import Image
from app.config import settings

logger = logging.getLogger("worker_queue")


class LocalWorkerQueue:
    """
    In-process worker queue. Zero Redis, zero Celery.
    Runs bounded number of worker tasks in the background.
    """

    def __init__(self, concurrency: int = 1):
        self.concurrency = concurrency
        self._queue: Optional[asyncio.Queue] = None
        self._workers = []
        self._is_running = False

    async def start(self):
        """Start worker loop tasks on application startup after verifying OCR readiness."""
        if not settings.MOCK_INFERENCE and settings.OCR_ENGINE in ("paddleocr", "auto", "default"):
            from app.pipeline.ocr import get_ocr_engine
            engine = get_ocr_engine("paddleocr")
            if hasattr(engine, "check_readiness"):
                ready, err = await asyncio.to_thread(engine.check_readiness)
                if not ready:
                    logger.critical(
                        f"CRITICAL: PaddleOCR verification failed ({err}). "
                        "Worker image processing disabled to prevent silent unintended engine execution."
                    )
                    self._is_ready = False
                    return
        self._is_ready = True
        self._queue = asyncio.Queue()
        self._is_running = True
        self._workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.concurrency)
        ]
        logger.info(f"Local worker queue started with {self.concurrency} worker(s).")

    async def stop(self):
        """Cancel worker tasks on application shutdown."""
        self._is_running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("Local worker queue stopped.")

    async def enqueue_image(self, image_id: str, batch_id: str):
        """Enqueue an image for local processing."""
        if self._queue is not None:
            await self._queue.put((image_id, batch_id))
            logger.info(f"📥 [QUEUE] Enqueued image {image_id} (batch: {batch_id})")

    async def _worker_loop(self, worker_id: int):
        """Continuous worker loop processing images one by one."""
        logger.info(f"Worker {worker_id} active.")
        while self._is_running:
            try:
                item = await self._queue.get()
                if item is None:
                    break
                image_id, batch_id = item
                
                # Execute CPU-intensive processing in threadpool to prevent blocking event loop
                logger.info(f"⚙️  [WORKER {worker_id}] Dequeued image {image_id} -> starting pipeline execution")
                await asyncio.to_thread(self._process_single_image, image_id, batch_id)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {str(e)}", exc_info=True)

    def _process_single_image(self, image_id: str, batch_id: str):
        """Execute image processing inside a dedicated database session."""
        db: Session = SessionLocal()
        try:
            orchestrator.process_image(db, image_id)
            orchestrator.reconcile_batch(db, batch_id)
            logger.info(f"✨ [WORKER] Completed processing & batch reconciliation for image {image_id}")
        except Exception as e:
            logger.error(f"Failed processing image {image_id}: {str(e)}", exc_info=True)
        finally:
            db.close()


worker_queue = LocalWorkerQueue(concurrency=settings.WORKER_CONCURRENCY)
