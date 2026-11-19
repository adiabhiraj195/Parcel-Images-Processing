"""Real-time event broadcasting and stage tracking for parcel image pipeline.

Provides Server-Sent Events (SSE) streaming and live stage tracking across
worker threads and web clients without external message brokers (Zero Redis, Zero Celery).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any, AsyncGenerator

logger = logging.getLogger("broadcaster")


class PipelineStage:
    """Canonical stages for the parcel image processing pipeline."""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    INGESTING = "INGESTING"
    PREPROCESSING = "PREPROCESSING"
    SCENE_DETECTION = "SCENE_DETECTION"
    METADATA_EXTRACTION = "METADATA_EXTRACTION"
    LABEL_EXTRACTION = "LABEL_EXTRACTION"
    RECONCILIATION = "RECONCILIATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


TOTAL_PIPELINE_STEPS = 6

STAGE_METADATA: Dict[str, Dict[str, Any]] = {
    PipelineStage.PENDING: {
        "step_index": 0,
        "name": "Pending",
        "short_name": "Pending",
        "description": "Image registered, awaiting upload verification",
        "progress": 0,
    },
    PipelineStage.QUEUED: {
        "step_index": 0,
        "name": "Queued",
        "short_name": "Queued",
        "description": "Uploaded and waiting in processing queue",
        "progress": 5,
    },
    PipelineStage.INGESTING: {
        "step_index": 1,
        "name": "S3 Download & Ingest",
        "short_name": "Storage Ingest",
        "description": "Downloading original scan from private storage",
        "progress": 15,
    },
    PipelineStage.PREPROCESSING: {
        "step_index": 2,
        "name": "Validation & Blur Analysis",
        "short_name": "Preprocessing",
        "description": "Validating image resolution, EXIF orientation & sharpness",
        "progress": 30,
    },
    PipelineStage.SCENE_DETECTION: {
        "step_index": 3,
        "name": "Scene Validation (Branch C)",
        "short_name": "Scene Detection",
        "description": "Zero-shot parcel and label bounding box detection (OWLv2)",
        "progress": 50,
    },
    PipelineStage.METADATA_EXTRACTION: {
        "step_index": 4,
        "name": "Machine Overlay Extraction (Branch A)",
        "short_name": "Machine Overlay",
        "description": "Full-resolution text detection, line grouping & format profiles",
        "progress": 70,
    },
    PipelineStage.LABEL_EXTRACTION: {
        "step_index": 5,
        "name": "Physical Label & Barcodes (Branch B)",
        "short_name": "Label & Barcodes",
        "description": "Cropping shipping label, deskewing & decoding barcodes (ZXing)",
        "progress": 85,
    },
    PipelineStage.RECONCILIATION: {
        "step_index": 6,
        "name": "Evidence Reconciliation",
        "short_name": "Reconciliation",
        "description": "Reconciling multi-branch evidence & computing field confidence",
        "progress": 95,
    },
    PipelineStage.COMPLETED: {
        "step_index": 7,
        "name": "Completed",
        "short_name": "Completed",
        "description": "Extraction saved to database successfully",
        "progress": 100,
    },
    PipelineStage.FAILED: {
        "step_index": -1,
        "name": "Processing Failed",
        "short_name": "Failed",
        "description": "Pipeline failure during execution",
        "progress": 0,
    },
}


class PipelineEventBroadcaster:
    """
    In-memory async pub/sub broadcaster for live processing updates.
    Enables instant frontend stage transitions via Server-Sent Events (SSE).
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._active_stages: Dict[str, Dict[str, Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main application event loop reference."""
        self._loop = loop

    def get_active_stage(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve current in-memory active processing stage for a batch."""
        return self._active_stages.get(batch_id)

    def set_active_stage(self, batch_id: str, stage_data: Optional[Dict[str, Any]]):
        """Set or clear the active processing stage for a batch."""
        if stage_data is None:
            self._active_stages.pop(batch_id, None)
        else:
            self._active_stages[batch_id] = stage_data

    def _dispatch(self, batch_id: str, event: Dict[str, Any]):
        """Internal synchronous dispatch to all registered subscriber queues."""
        queues = self._subscribers.get(batch_id, set())
        for q in list(queues):
            try:
                q.put_nowait(event)
            except Exception as err:
                logger.debug(f"Failed to put event in subscriber queue: {err}")

    def broadcast(self, batch_id: str, event: Dict[str, Any]):
        """
        Thread-safe broadcast method. Can be called from any background worker thread
        or asyncio task.
        """
        if not event.get("timestamp"):
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Update cached active stage state if this is a stage_update
        if event.get("type") == "stage_update":
            stage = event.get("stage")
            if stage in (PipelineStage.COMPLETED, PipelineStage.FAILED):
                # If current image finished and matches cached active image, clear it or update status
                cached = self._active_stages.get(batch_id)
                if cached and cached.get("image_id") == event.get("image_id"):
                    self._active_stages.pop(batch_id, None)
            else:
                self._active_stages[batch_id] = {
                    "image_id": event.get("image_id"),
                    "filename": event.get("filename"),
                    "stage": event.get("stage"),
                    "stage_name": event.get("stage_name"),
                    "stage_message": event.get("stage_message"),
                    "stage_progress": event.get("stage_progress", 0),
                    "stage_index": event.get("stage_index", 0),
                    "total_stages": TOTAL_PIPELINE_STEPS,
                    "stage_started_at": event.get("timestamp"),
                    "preview_url": event.get("preview_url"),
                }
        elif event.get("type") == "batch_completed":
            self._active_stages.pop(batch_id, None)

        # Dispatch through event loop threadsafe
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._dispatch, batch_id, event)
        else:
            # Fallback direct call if already on running loop
            self._dispatch(batch_id, event)

    async def subscribe(self, batch_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Subscribe to live events for a specific batch.
        Yields event dictionaries as they occur.
        """
        queue: asyncio.Queue = asyncio.Queue()
        if batch_id not in self._subscribers:
            self._subscribers[batch_id] = set()
        self._subscribers[batch_id].add(queue)

        try:
            while True:
                # Wait for next event with a timeout for heartbeat/keep-alive
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield event
                except asyncio.TimeoutError:
                    # Yield heartbeat keep-alive comment
                    yield {"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()}
        finally:
            if batch_id in self._subscribers:
                self._subscribers[batch_id].discard(queue)
                if not self._subscribers[batch_id]:
                    del self._subscribers[batch_id]


# Global singleton broadcaster
pipeline_broadcaster = PipelineEventBroadcaster()
