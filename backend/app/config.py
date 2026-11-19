"""Application configuration loaded from environment variables."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration settings for Parcel Intelligence backend."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: str = "*"

    DATABASE_URL: str = "sqlite:///./parcel_intelligence.db"
    WORKER_CONCURRENCY: int = 1

    TEMP_DOWNLOAD_DIR: str = "./temp_downloads"
    MODEL_CACHE_DIR: str = "./model_cache"

    @property
    def resolved_model_cache_dir(self) -> str:
        """Resolve model cache directory across execution working directories."""
        import os
        candidates = [
            os.path.abspath("./model_cache"),
            os.path.abspath("./backend/model_cache"),
            os.path.abspath("../model_cache"),
            os.path.abspath(self.MODEL_CACHE_DIR),
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.isdir(c):
                return c
        return os.path.abspath(self.MODEL_CACHE_DIR)

    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "your-private-parcel-intelligence-bucket"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_PROFILE: str = ""

    PRESIGNED_PUT_EXPIRY_SECONDS: int = 900
    PRESIGNED_GET_EXPIRY_SECONDS: int = 300

    LOG_LEVEL: str = "INFO"

    MOCK_INFERENCE: bool = False

    # OCR and Multi-Format DWS configuration
    OCR_ENGINE: str = "paddleocr"  # "paddleocr", "tesseract", "mock", "local_cpu"
    PADDLE_OCR_VERSION: str = "PP-OCRv5"  # "PP-OCRv5" or "PP-OCRv6"
    OVERLAY_UPSCALE_FACTOR: float = 2.0
    SCANNER_BED_ROI: str = ""  # Format: "xmin,ymin,xmax,ymax" normalized (0-1) or pixel
    SAVE_DEBUG_BUNDLE: bool = True

    # PaddleOCR-VL 1.6 Advanced Model Adapter configuration
    ENABLE_PADDLEOCR_VL: bool = False
    PADDLEOCR_VL_VERSION: str = "1.6"
    PADDLEOCR_VL_DEVICE: str = "cpu"

    def init_environment_variables(self) -> None:
        """Set up environment variables for local cache and headless Paddle / HuggingFace execution."""
        import os

        # Clean up empty AWS env variables so botocore does not fail with ProfileNotFound
        for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            if var in os.environ and not os.environ[var].strip():
                del os.environ[var]

        cache_dir = self.resolved_model_cache_dir
        paddlex_cache = os.path.join(cache_dir, "paddlex")
        os.makedirs(paddlex_cache, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", paddlex_cache)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_DISABLE_SIGNAL_HANDLER", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        if "HOME" not in os.environ or not os.environ["HOME"]:
            os.environ["HOME"] = "/tmp"

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
settings.init_environment_variables()
