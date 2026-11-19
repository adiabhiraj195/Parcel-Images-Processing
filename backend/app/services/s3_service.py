"""AWS S3 client and presigned URL management service."""

import os
import re
from typing import Optional, Dict, Tuple
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent S3 key injection and path traversal."""
    base = os.path.basename(filename)
    # Remove characters other than alphanumeric, dots, underscores, dashes
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return clean or "unnamed_image.jpg"


class S3Service:
    """Service wrapping AWS S3 operations."""

    def __init__(self):
        self.bucket = settings.AWS_S3_BUCKET
        self.region = settings.AWS_REGION
        self.client = self._create_client()

    def _create_client(self):
        """Initialize boto3 client with proper timeouts and region."""
        endpoint_url = None
        if self.region:
            endpoint_url = f"https://s3.{self.region}.amazonaws.com"

        kwargs = {
            "region_name": self.region,
            "endpoint_url": endpoint_url,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }
        # Ensure botocore does not choke on empty AWS environment variables
        for env_k in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            if env_k in os.environ and not os.environ[env_k].strip():
                del os.environ[env_k]

        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        elif settings.AWS_PROFILE and settings.AWS_PROFILE.strip():
            session = boto3.Session(profile_name=settings.AWS_PROFILE.strip())
            return session.client("s3", **kwargs)

        return boto3.client("s3", **kwargs)

    def generate_original_key(self, batch_id: str, image_id: str, filename: str) -> str:
        """Construct the standardized S3 key for original uploaded image."""
        safe_name = sanitize_filename(filename)
        return f"batches/{batch_id}/originals/{image_id}/{safe_name}"

    def generate_derived_key(self, batch_id: str, image_id: str, artifact_name: str) -> str:
        """Construct S3 key for derived artifacts (e.g. label-crop.jpg)."""
        safe_name = sanitize_filename(artifact_name)
        return f"batches/{batch_id}/derived/{image_id}/{safe_name}"

    def generate_export_key(self, batch_id: str, export_id: str) -> str:
        """Construct S3 key for exported CSV."""
        return f"batches/{batch_id}/exports/{export_id}.csv"

    def create_presigned_put_url(
        self,
        s3_key: str,
        content_type: str,
        expires_in: Optional[int] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate short-lived presigned PUT URL for direct browser-to-S3 upload.
        Never exposes AWS credentials to the frontend.
        """
        expiry = expires_in or settings.PRESIGNED_PUT_EXPIRY_SECONDS
        headers = {"Content-Type": content_type}

        try:
            url = self.client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": s3_key,
                    "ContentType": content_type,
                },
                ExpiresIn=expiry,
            )
            return url, headers
        except Exception as e:
            # Fallback for offline local dev mode if AWS credentials are not set
            if settings.MOCK_INFERENCE or not settings.AWS_ACCESS_KEY_ID:
                mock_url = f"http://localhost:{settings.PORT}/api/mock-s3-upload/{s3_key}"
                return mock_url, headers
            raise e

    def create_presigned_get_url(
        self,
        s3_key: str,
        expires_in: Optional[int] = None,
    ) -> str:
        """
        Generate short-lived presigned GET URL for image preview.
        Bucket remains private.
        """
        expiry = expires_in or settings.PRESIGNED_GET_EXPIRY_SECONDS
        try:
            return self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": s3_key,
                },
                ExpiresIn=expiry,
            )
        except Exception:
            return f"/api/mock-s3-preview/{s3_key}"

    def verify_object_exists(self, s3_key: str) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Check that an uploaded object exists in S3 and return (exists, size, content_type).
        """
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=s3_key)
            size = resp.get("ContentLength")
            content_type = resp.get("ContentType")
            return True, size, content_type
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False, None, None
            raise e
        except Exception:
            # If running in local mock mode without AWS
            if settings.MOCK_INFERENCE:
                return True, 1024, "image/jpeg"
            return False, None, None

    def download_to_file(self, s3_key: str, local_path: str):
        """Download an S3 object to a local temporary path."""
        self.client.download_file(self.bucket, s3_key, local_path)


s3_service = S3Service()
