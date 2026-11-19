export type ExtractionStatus =
  | 'SUCCESS'
  | 'PARTIAL_EXTRACTION'
  | 'NO_PARCEL'
  | 'MULTIPLE_PARCELS'
  | 'PARTIAL_PARCEL'
  | 'LABEL_NOT_FOUND'
  | 'LABEL_OCCLUDED'
  | 'LABEL_UNREADABLE'
  | 'LOW_CONFIDENCE'
  | 'SOURCE_CONFLICT'
  | 'PROCESSING_FAILED';

export interface FieldProvenanceData {
  value?: any;
  source?: string;
  raw_text?: string;
  ocr_confidence?: number;
  bounding_box?: number[] | null;
  parser_rule?: string;
  unit?: string | null;
  normalized_value?: any;
  conflict_status?: string | null;
  raw_candidates?: string[];
}

export interface EvidenceData {
  tracking_provenance?: FieldProvenanceData;
  weight_provenance?: FieldProvenanceData;
  dimensions_provenance?: FieldProvenanceData;
  volumetric_weight_provenance?: FieldProvenanceData;
  scene_flags?: string[];
  overlay_profile?: string | null;
  reconciliation_time_ms?: number;
  actual_weight?: number | null;
  volumetric_weight?: number | null;
  parcel_count?: number | null;
  extraction_status?: string;
  location?: string | null;
  volume?: number | null;
  timestamp?: string | null;
  machine_id?: string | null;
  sorter_id?: string | null;
  debug_bundle?: Record<string, unknown>;
  error?: string;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  platform: string;
  python_version: string;
}

export interface BatchResponse {
  id: string;
  name: string;
  total_images: number;
  uploaded_images: number;
  processed_images: number;
  failed_images: number;
  created_at: string;
  updated_at: string;
}

export type BatchSummary = BatchResponse;

export interface ExtractionResponse {
  id: string;
  image_id: string;
  filename: string;
  tracking_number: string | null;
  weight_value: number | null;
  weight_unit: string | null;
  actual_weight?: number | null;
  volumetric_weight?: number | null;
  length: number | null;
  width: number | null;
  height: number | null;
  dimension_unit: string | null;
  parcel_count: number | null;
  status: ExtractionStatus;
  parcel_label_outcome?: string | null;
  scene_flags?: string[];
  raw_ocr_json?: string | null;
  barcode_json?: string | null;
  detection_json?: string | null;
  field_confidence_json?: string | null;
  evidence_json?: string | EvidenceData | null;
  reviewed: boolean;
  reviewed_at: string | null;
  processing_status?: 'PENDING' | 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | string;
  current_stage?: string | null;
  stage_message?: string | null;
  stage_progress?: number | null;
  stage_started_at?: string | null;
  created_at: string;
  updated_at: string;
  preview_url?: string | null;
}

export interface ActiveProcessingItem {
  image_id: string;
  filename: string;
  stage: string;
  stage_name: string;
  stage_message: string;
  stage_progress: number;
  stage_index: number;
  total_stages: number;
  stage_started_at?: string | null;
  preview_url?: string | null;
}

export interface BatchProgressItem {
  image_id: string;
  filename: string;
  processing_status: string;
  current_stage?: string | null;
  stage_message?: string | null;
  stage_progress?: number | null;
  stage_started_at?: string | null;
  preview_url?: string | null;
}

export interface BatchProgressResponse {
  batch: BatchResponse;
  is_processing: boolean;
  active_image?: ActiveProcessingItem | null;
  images: BatchProgressItem[];
}

export interface BatchResultsResponse {
  batch: BatchResponse;
  results: ExtractionResponse[];
  total_results: number;
  active_processing?: ActiveProcessingItem | null;
}
