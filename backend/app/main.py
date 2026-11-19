"""FastAPI application entrypoint, database initialization, background queue, and routing."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.batches import router as batches_router
from app.api.images import router as images_router
from app.api.export import router as export_router
from app.config import settings
from app.database import init_db
from app.worker.queue import worker_queue
from app.logging_config import setup_logging

# Initialize terminal logging on application module load
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    setup_logging()
    # Ensure temporary and cache directories exist
    os.makedirs(settings.TEMP_DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    # Initialize SQLite database schema
    init_db()

    # Configure event broadcaster event loop
    import asyncio
    from app.worker.broadcaster import pipeline_broadcaster
    pipeline_broadcaster.set_loop(asyncio.get_running_loop())

    # Start local in-process bounded worker as background task so port 8000 opens immediately
    worker_task = asyncio.create_task(worker_queue.start())

    yield

    # Shutdown local worker cleanly
    await worker_queue.stop()
    if not worker_task.done():
        worker_task.cancel()


app = FastAPI(
    title="Parcel Intelligence Backend",
    description="Local-first parcel-image processing prototype for warehouse shipment operations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS setup for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(batches_router)
app.include_router(images_router)
app.include_router(export_router)
