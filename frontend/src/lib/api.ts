import {
  BatchSummary,
  BatchResponse,
  BatchResultsResponse,
  BatchProgressResponse,
  ExtractionResponse,
  HealthResponse,
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL !== undefined ? process.env.NEXT_PUBLIC_API_URL : 'http://localhost:8000';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch health status: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBatches(): Promise<BatchResponse[]> {
  const res = await fetch(`${API_BASE_URL}/api/batches`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch batches: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBatch(batchId: string): Promise<BatchResponse> {
  const res = await fetch(`${API_BASE_URL}/api/batches/${batchId}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch batch: ${res.statusText}`);
  }
  return res.json();
}

export interface CreateBatchFileItem {
  filename: string;
  content_type: string;
  file_size: number;
}

export interface PresignedUploadTarget {
  image_id: string;
  filename: string;
  content_type: string;
  file_size: number;
  upload_url: string;
  required_headers: Record<string, string>;
  s3_key: string;
}

export interface CreateBatchResponse {
  batch_id: string;
  name: string;
  total_images: number;
  uploads: PresignedUploadTarget[];
}

export async function createBatch(
  name: string,
  files: { filename: string; content_type: string; file_size: number }[]
): Promise<CreateBatchResponse> {
  const res = await fetch(`${API_BASE_URL}/api/batches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, files }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to create batch');
  }
  return res.json();
}

export async function completeBatchUploads(
  batchId: string,
  imageIds?: string[]
): Promise<BatchResponse> {
  const res = await fetch(`${API_BASE_URL}/api/batches/${batchId}/uploads/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(imageIds || null),
  });
  if (!res.ok) {
    throw new Error(`Failed to complete batch upload verification: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBatchResults(batchId: string): Promise<BatchResultsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/batches/${batchId}/results`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch batch results: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBatchProgress(batchId: string): Promise<BatchProgressResponse> {
  const res = await fetch(`${API_BASE_URL}/api/batches/${batchId}/progress`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch batch progress: ${res.statusText}`);
  }
  return res.json();
}

export function subscribeToBatchEvents(
  batchId: string,
  onEvent: (event: any) => void,
  onError?: (err: any) => void
): () => void {
  const url = `${API_BASE_URL}/api/batches/${batchId}/events`;
  let eventSource: EventSource | null = null;
  let isClosed = false;

  try {
    eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      if (isClosed) return;
      try {
        const data = JSON.parse(event.data);
        onEvent(data);
      } catch (e) {
        console.error('Failed to parse SSE event data', e);
      }
    };

    eventSource.onerror = (err) => {
      if (isClosed) return;
      if (onError) onError(err);
    };
  } catch (err) {
    if (onError) onError(err);
  }

  return () => {
    isClosed = true;
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  };
}

export async function fetchImageDetails(imageId: string): Promise<ExtractionResponse> {
  const res = await fetch(`${API_BASE_URL}/api/images/${imageId}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch image details: ${res.statusText}`);
  }
  return res.json();
}

export async function reviewImage(
  imageId: string,
  payload: Partial<ExtractionResponse>
): Promise<ExtractionResponse> {
  const res = await fetch(`${API_BASE_URL}/api/images/${imageId}/review`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to submit review');
  }
  return res.json();
}

export async function retryImage(imageId: string): Promise<ExtractionResponse> {
  const res = await fetch(`${API_BASE_URL}/api/images/${imageId}/retry`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(`Failed to retry image: ${res.statusText}`);
  }
  return res.json();
}

export function getBatchExportUrl(batchId: string): string {
  return `${API_BASE_URL}/api/batches/${batchId}/export.csv`;
}
