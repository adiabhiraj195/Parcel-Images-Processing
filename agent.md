# Parcel Intelligence Prototype --- Agent Context

> This file is the source of truth for AI coding assistants working on
> this repository. Read it fully before changing code. If implementation
> decisions conflict with this file, explain the conflict and ask before
> making a major architectural change.

## 1. Product overview

Build a local-first parcel-image processing application for warehouse
shipment operations.

Warehouse operators upload batches of JPG/PNG photographs taken at
scanning stations. Images may contain: - exactly one clearly visible
parcel; - no parcel; - multiple parcels; - a parcel partly outside the
frame; - a shipping label obscured by a hand or another object; - a
label that is too small, blurry, tilted, damaged, or unreadable; - a
parcel with no visible shipping label; - labels with AWB/tracking
number, weight, dimensions, barcodes, QR codes, or only some of those
fields.

The application processes every accepted image and produces exactly one
final result record per uploaded image. It must prefer an explicit
exception or null value over an invented or guessed value.

The main deliverable is a batch CSV export, plus a review UI that lets
an operator inspect the original image, detected regions, OCR evidence,
extracted fields, and status.

This is a prototype, not a production logistics platform. Optimize for
correctness, explainability, low cost, and a reliable end-to-end demo.

## 2. Non-negotiable constraints

1.  The developer machine is an Apple Silicon Mac M1.
2.  Do not require an NVIDIA GPU or CUDA.
3.  Do not use paid OCR, object-detection, LLM, or hosted inference
    APIs.
4.  Run object detection, OCR, barcode decoding, and image processing
    locally on the Mac.
5.  Use AWS S3 for image and artifact storage.
6.  Keep the S3 bucket private. Never make uploaded images public.
7.  Never expose AWS access keys or secret keys to the browser.
8.  Use pre-trained models first. Do not train or fine-tune models
    unless the user explicitly requests it after evaluating the
    pretrained baseline.
9.  Do not make an LLM responsible for reading or inventing shipment
    values.
10. Preserve the one-result-row-per-uploaded-image invariant, including
    for processing failures.
11. Do not silently discard images or silently replace uncertain values
    with guesses.
12. Keep AWS usage conservative. S3 may incur charges; avoid unnecessary
    copies, downloads, polling, and long-lived test data.

## 3. Recommended technology stack

### Frontend

-   Next.js App Router
-   TypeScript
-   Tailwind CSS
-   A clean, responsive dashboard
-   Browser uploads directly to S3 using backend-issued presigned URLs
-   Backend APIs for batch creation, status, results, review, and CSV
    export

### Backend

-   Python 3.11 (or a compatible Python version supported by chosen
    model packages)
-   FastAPI
-   Pydantic schemas
-   SQLAlchemy
-   SQLite for the prototype
-   boto3 for AWS S3 access
-   Background processing using a simple local worker with bounded
    concurrency
-   Python `csv` module for export

### Computer vision

-   OpenCV and Pillow for image handling and preprocessing
-   Pretrained zero-shot detector as the first baseline:
    -   OWLv2 or Grounding DINO, selecting a model/runtime that works on
        Apple Silicon
    -   Start with CPU inference if MPS support is uncertain; optionally
        use PyTorch MPS only after verifying it works
-   PaddleOCR pretrained detection/recognition models for OCR
-   ZXing-C++ or pyzbar for barcode/QR decoding
-   Deterministic Python parsing and validation for AWB, weight, and
    dimensions

### Storage

-   AWS S3 private bucket for original images and optional derived
    artifacts
-   SQLite stores metadata, processing state, extracted fields, and
    evidence references
-   Do not store image binary data in SQLite
-   Use S3 object keys, not public URLs, in the database
-   Generate short-lived presigned GET URLs only when the UI needs to
    display an image

## 4. High-level architecture

1.  User creates a batch in the Next.js frontend.
2.  Frontend sends batch metadata and file metadata to FastAPI.
3.  Backend creates a batch and one image row per accepted file.
4.  Backend creates short-lived S3 presigned PUT URLs for each image.
5.  Browser uploads files directly to private S3 using those URLs.
6.  Frontend confirms upload completion to the backend.
7.  Backend marks each image `QUEUED` and schedules local processing.
8.  Local worker downloads one image from S3 to a temporary local
    workspace.
9.  Image validation and preprocessing run locally.
10. Pretrained object detector proposes parcel and shipping-label
    regions.
11. Detection and geometry rules classify parcel count and visibility.
12. Label crop is corrected/enhanced where appropriate.
13. PaddleOCR extracts text and geometry from the label crop.
14. Barcode/QR decoder attempts to extract barcode values.
15. Deterministic parser extracts AWB/tracking number, weight, and
    dimensions.
16. Validation and evidence reconciliation produce field-level
    confidence and status.
17. Backend writes exactly one extraction result per image.
18. Temporary local files are cleaned up.
19. Frontend polls or refreshes batch progress and displays results.
20. User can inspect and manually correct a result.
21. CSV export returns exactly one row per uploaded image.

Important: for the prototype, the API and worker may run in the same
local backend process. Do not add Redis, Celery, Kubernetes, or a
distributed queue unless explicitly requested.

## 5. S3 design

### Bucket

Use one private bucket, configured through environment variables: -
`AWS_REGION` - `AWS_S3_BUCKET` - `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` only if using local IAM-user credentials -
Prefer the AWS shared credentials/config profile or IAM role where
available. - `AWS_PROFILE` may be used for local development.

Never commit credentials or `.env` files.

### Suggested object key layout

``` text
batches/{batch_id}/originals/{image_id}/{safe_filename}
batches/{batch_id}/derived/{image_id}/label-crop.jpg
batches/{batch_id}/derived/{image_id}/annotated.jpg
batches/{batch_id}/exports/{export_id}.csv
```

Use UUIDs for batch, image, and export IDs. Sanitize filenames. Never
use an untrusted filename as the entire S3 key.

### Upload flow

-   Browser asks backend for upload URLs.
-   Backend validates allowed extension, declared MIME type, file size,
    batch ownership, and upload count.
-   Backend returns a short-lived presigned PUT URL and required
    headers.
-   Browser uploads directly to S3.
-   Browser calls a completion endpoint.
-   Backend verifies object existence and expected metadata/size before
    marking the image uploaded/queued.
-   Do not send AWS credentials to the browser.
-   For a small prototype, ordinary presigned PUT is sufficient. Do not
    implement multipart upload unless file-size requirements make it
    necessary.

### Download and display

-   Worker downloads the original from S3 to a temporary directory.
-   Use a short-lived presigned GET URL for browser preview.
-   Never expose a permanent public S3 URL.
-   Do not proxy large image bytes through FastAPI unless required.

### Cost and safety guardrails

-   Keep the bucket private and block public access.
-   Configure a lifecycle rule to expire prototype objects after a
    user-selected retention period, if appropriate.
-   Avoid storing every intermediate crop by default. Save derived
    images only when needed for debugging/review.
-   Do not repeatedly download an image if a valid local temporary copy
    is already being processed.
-   Do not enable expensive replication, versioning, or transfer
    acceleration for this prototype without a clear need.
-   Display a clear note that AWS S3 usage may incur charges.
-   Use least-privilege IAM permissions limited to the required bucket
    and object prefixes.
-   Never log secrets, presigned URLs, or full credentials.

## 6. Processing pipeline

Implement each stage behind a small interface so models can be swapped
without rewriting the application.

### 6.1 Image validation

-   Validate file type using actual file content where practical, not
    only the extension.
-   Reject corrupt or unsupported images with a clear per-image result.
-   Preserve the original image unchanged in S3.
-   Apply EXIF orientation before inference.
-   Record image dimensions and basic metadata.
-   Set safe maximum file size and pixel count to reduce
    memory-exhaustion risk.

### 6.2 Image quality checks

Use OpenCV for: - blur heuristic (e.g. Laplacian variance); -
brightness/contrast heuristics; - image dimensions; - optional resizing
for inference.

Quality heuristics must not be treated as definitive proof. Calibrate
thresholds against sample data. Do not classify a label as occluded
solely because blur is high.

### 6.3 Object detection

Use pretrained zero-shot detection initially. Evaluate OWLv2 and/or
Grounding DINO locally.

Prompt candidates may include: - `parcel` - `cardboard box` -
`shipping package` - `courier parcel` - `shipping label` -
`shipping sticker` - `waybill label`

Do not assume that prompt labels are mutually exclusive. Merge
duplicate/overlapping detections using NMS or an equivalent overlap
strategy. Preserve raw detections for debugging.

Detection output should be normalized into a project-owned schema:

``` json
{
  "model": "model-name-and-version",
  "detections": [
    {
      "class_name": "parcel",
      "score": 0.91,
      "bbox_xyxy": [100, 120, 600, 700]
    }
  ]
}
```

Use pixel coordinates in the normalized internal representation.
Validate that boxes are within image bounds.

### 6.4 Parcel count and visibility

-   Zero sufficiently credible parcel detections can produce
    `NO_PARCEL`, but use a conservative threshold.
-   More than one distinct parcel detection can produce
    `MULTIPLE_PARCELS`.
-   A box close to image edges is only a clue for `PARTIAL_PARCEL`, not
    conclusive proof.
-   Overlapping detections may represent the same parcel; deduplicate
    before counting.
-   If uncertain, use `LOW_CONFIDENCE` or send to manual review.
-   Do not assume the detector can infer whether a hand is covering a
    label. Use label-region visibility and OCR evidence; classify
    `LABEL_OCCLUDED` only when there is sufficient evidence.

### 6.5 Label crop

-   Use detected label bounding box with modest padding.
-   Clamp crop coordinates to image bounds.
-   Optionally deskew/perspective-correct when a reliable quadrilateral
    is available.
-   Do not fabricate missing corners or over-transform the label.
-   Try OCR on original crop and a limited number of enhanced variants.
-   Keep a reference to the crop/evidence if saved to S3.

### 6.6 OCR

Use pretrained PaddleOCR locally. - Return text, text-region geometry,
and OCR confidence where available. - Keep raw OCR output in structured
JSON for audit/debugging. - Do not discard low-confidence text
silently. - OCR confidence is not the same as final field confidence. -
Keep OCR calls bounded to avoid excessive memory use on an M1.

### 6.7 Barcode/QR

Use ZXing-C++ or pyzbar locally. - Attempt barcode decoding on the label
crop and, if useful, the parcel crop. - Normalize decoded strings
conservatively. - Preserve the raw decoded value and symbology where
available. - If barcode and printed OCR tracking values disagree, flag
for review. Do not silently pick one.

### 6.8 Field extraction

Extract: - tracking/AWB number; - weight value and unit; - dimensions
length, width, height and unit.

Use deterministic regexes and carrier-agnostic heuristics as a baseline.
Store raw text and evidence references. Do not assume one carrier
layout. Do not infer a missing weight or dimensions value. Keep absent
fields null.

Normalize units only when conversion is unambiguous. Preserve the
original extracted text.

### 6.9 Validation

-   Tracking number: validate allowed characters and plausible length,
    but do not claim validity with a carrier unless carrier-specific
    validation exists.
-   Weight: parse numeric value and explicit unit. Do not invent a unit
    if absent.
-   Dimensions: require three numeric components and an explicit or
    reliably inferred unit; otherwise retain raw text and mark
    normalized dimensions unavailable.
-   Compare barcode and OCR tracking number when both exist.
-   Use evidence and confidence thresholds calibrated on the case-study
    data.
-   Field confidence must account for OCR score, parsing quality, source
    agreement, and label association.

### 6.10 Status taxonomy

Use these exact initial status strings: - `SUCCESS` -
`PARTIAL_EXTRACTION` - `NO_PARCEL` - `MULTIPLE_PARCELS` -
`PARTIAL_PARCEL` - `LABEL_NOT_FOUND` - `LABEL_OCCLUDED` -
`LABEL_UNREADABLE` - `LOW_CONFIDENCE` - `PROCESSING_FAILED`

Status precedence must be documented in code. A processing exception
must still result in a row with `PROCESSING_FAILED`.

## 7. Data model

Use UUID primary keys.

### batches

-   `id`
-   `name`
-   `total_images`
-   `uploaded_images`
-   `processed_images`
-   `failed_images`
-   `created_at`
-   `updated_at`

### images

-   `id`
-   `batch_id`
-   `filename`
-   `content_type`
-   `file_size`
-   `s3_bucket`
-   `s3_key`
-   `upload_status`
-   `processing_status`
-   `created_at`
-   `updated_at`

### extractions

-   `id`
-   `image_id` (unique foreign key)
-   `tracking_number` nullable
-   `weight_value` nullable
-   `weight_unit` nullable
-   `length` nullable
-   `width` nullable
-   `height` nullable
-   `dimension_unit` nullable
-   `parcel_count` nullable
-   `status`
-   `raw_ocr_json`
-   `barcode_json`
-   `detection_json`
-   `field_confidence_json`
-   `evidence_json`
-   `reviewed`
-   `reviewed_at`
-   `created_at`
-   `updated_at`

Use a unique constraint on `extractions.image_id` to enforce at most one
result per image. Add a reconciliation step to ensure every
uploaded/accepted image eventually has one result. Use a
transaction/upsert to make processing idempotent.

## 8. API contract

Suggested endpoints:

### Batches

-   `POST /api/batches` --- create batch and register file metadata;
    return image IDs and presigned upload URLs.
-   `POST /api/batches/{batch_id}/uploads/complete` --- verify uploaded
    objects and queue processing.
-   `GET /api/batches/{batch_id}` --- batch status and progress.
-   `GET /api/batches/{batch_id}/results` --- paginated result rows.

### Images and review

-   `GET /api/images/{image_id}` --- result, evidence, and short-lived
    preview URL.
-   `PATCH /api/images/{image_id}/review` --- update corrected fields
    and mark reviewed.
-   `POST /api/images/{image_id}/retry` --- retry a failed image if
    safe/idempotent.

### Export

-   `GET /api/batches/{batch_id}/export.csv` --- CSV with exactly one
    row per accepted image.
-   Export should include original filename, tracking number, weight,
    dimensions, status, and reviewed flag.
-   Keep missing values empty, not guessed.

Use Pydantic request/response models. Keep API errors structured and do
not return stack traces to the browser.

## 9. Frontend requirements

Pages: 1. `/` or `/batches` --- batch list and create-batch action. 2.
`/batches/[batchId]` --- upload status, progress, summary counts,
results table, CSV download. 3. `/images/[imageId]` or a detail drawer
--- original image, detection overlays, label crop, OCR evidence,
extracted fields, confidence, status, review controls.

Upload UX: - multi-file drag/drop; - JPG and PNG only initially; - file
size and count validation; - per-file upload progress; - retry failed
uploads; - do not start processing until upload completion is confirmed.

Results UX: - filters by status; - search by filename or AWB; - show
missing values clearly; - status badges; - manual correction and review
action; - export CSV.

Never show fabricated demo values as real inference results. If mock
data is used during UI development, visibly label it as mock data.

## 10. CSV requirements

One row per accepted uploaded image, including failed/exception images.

Suggested columns: - `image_id` - `image_name` - `tracking_number` -
`weight_value` - `weight_unit` - `length` - `width` - `height` -
`dimension_unit` - `parcel_count` - `status` - `reviewed`

CSV export must be deterministic and stable. Escape values correctly
using Python's CSV library. Include UTF-8 support. Do not export S3
credentials or presigned URLs.

## 11. M1 performance guidance

-   Start with sequential processing or concurrency 1.
-   Keep image size and model input resolution modest.
-   Load model weights once per worker, not once per image.
-   Avoid loading multiple large vision models simultaneously if RAM is
    limited.
-   Prefer CPU inference as a reliable baseline; enable PyTorch MPS only
    after a tested compatibility check.
-   PaddleOCR should run locally; do not call a hosted OCR endpoint.
-   Clean temporary files after each image.
-   Log processing time and peak memory where practical.
-   Do not promise real-time throughput. Measure images/minute on the
    actual M1.

## 12. Security and reliability

-   Validate uploads and enforce file-size/image-pixel limits.
-   Use private S3 bucket and block public access.
-   Use short-lived presigned URLs.
-   Never put AWS secrets in frontend environment variables.
-   Restrict CORS to the local frontend origin in development.
-   Use least-privilege IAM permissions.
-   Sanitize filenames and never trust user-provided paths.
-   Avoid path traversal.
-   Do not log secrets or presigned URLs.
-   Use idempotent processing and database transactions.
-   Ensure per-image failure does not abort the entire batch.
-   Clean up temporary files in `finally` blocks.
-   Add structured logs with batch ID and image ID.

## 13. Testing strategy

Unit tests: - image validation; - crop coordinate clamping; - duplicate
detection merging; - tracking number parsing; - weight parsing; -
dimensions parsing; - barcode/OCR conflict logic; - status precedence; -
CSV row count and escaping.

Integration tests: - create batch; - presigned upload flow (mock S3 in
tests); - upload completion; - worker processing with mocked model
adapters; - one result per image; - CSV export.

Model evaluation: - create a small manually reviewed ground-truth set
from the actual case-study images; - evaluate parcel detection recall
and false positives; - evaluate label detection recall; - evaluate AWB
exact-match accuracy; - evaluate weight and dimensions exact/normalized
accuracy; - evaluate exception classification; - report the number of
images with no result (target: zero).

Do not claim model accuracy without measured results on held-out images.

## 14. Development workflow and coding-agent rules

1.  Inspect the existing repository before creating or changing files.
2.  Propose a short implementation plan before a large change.
3.  Implement one milestone at a time.
4.  Keep model integrations behind interfaces/adapters.
5.  Keep configuration in environment variables and provide
    `.env.example` with placeholders only.
6.  Add tests for business logic and API contracts.
7.  Run formatter, linter, and tests after each milestone.
8.  Do not silently change the architecture or add paid services.
9.  Do not train models unless explicitly asked.
10. Do not add LangChain, LangGraph, or an LLM; they are unnecessary for
    this deterministic CV workflow.
11. Do not add Redis/Celery unless explicitly requested.
12. If a model cannot run on M1, report the exact blocker and propose a
    local CPU-compatible alternative.
13. Do not fake successful OCR or detections. Use explicit mock mode
    only for frontend development.
14. Keep README setup instructions accurate for macOS Apple Silicon.
15. When changing S3 behavior, explain expected AWS permissions and
    possible charges.

## 15. Definition of done

The prototype is complete when: - user can create a batch and upload
multiple JPG/PNG files; - originals are stored in a private AWS S3
bucket; - browser never receives AWS credentials; - local M1 worker
processes images with pretrained local models; - parcel and label
detection evidence is available; - OCR and barcode extraction are
attempted; - AWB, weight, and dimensions are extracted only when
supported by evidence; - uncertain or unreadable cases receive an
explicit status; - every accepted uploaded image has exactly one result
row; - user can inspect and correct results; - CSV download contains
exactly one row per accepted image; - tests cover core parsing, status,
and one-row-per-image behavior; - README explains setup, AWS
configuration, model download, limitations, and cost controls.

------------------------------------------------------------------------

# Prompt sequence for the AI coding editor

Use these prompts in order. Do not ask the coding model to implement all
milestones in one shot.

## Prompt 1 --- Repository inspection and implementation plan

Read `agent.md` completely. Inspect the repository and report: -
existing framework and folder structure; - missing components; -
dependency and compatibility risks on Apple M1; - proposed
implementation plan split into small milestones.

Do not modify files yet. Do not introduce paid APIs, model training,
Redis, Celery, LangChain, or LangGraph. Confirm how S3 direct uploads
will work without exposing credentials to the browser.

## Prompt 2 --- Project scaffolding

Read `agent.md`. Create the monorepo structure for the Next.js frontend
and FastAPI backend. Add: - backend Python project and dependency
file; - frontend Next.js TypeScript project; - `.env.example` with
placeholders only; - `.gitignore`; - README with macOS Apple Silicon
setup; - basic health endpoint.

Do not implement model inference yet. Ensure frontend and backend run
independently. Run basic checks and report exact commands.

## Prompt 3 --- Database models and API contracts

Implement SQLAlchemy models and Pydantic schemas for batches, images,
and extractions according to `agent.md`. Add migrations or a clearly
documented SQLite schema initialization approach. Implement batch
creation and read endpoints. Add a unique constraint on
`extractions.image_id`. Write tests for the data model and API
contracts.

## Prompt 4 --- AWS S3 integration

Implement private S3 storage using boto3. Use backend-generated
presigned PUT URLs for direct browser-to-S3 uploads. Add endpoints to
create a batch with image metadata and presigned URLs, confirm upload
completion, and generate short-lived presigned GET URLs for previews.
Verify object existence and expected size/type on completion. Never
expose AWS credentials to the frontend. Add a mocked-S3 test suite and
document the minimum IAM permissions, bucket CORS configuration, and
cost considerations.

## Prompt 5 --- Frontend upload and batch dashboard

Implement the batch list, create-batch flow, multi-file JPG/PNG upload,
direct presigned S3 upload, per-file progress, upload retry, and batch
detail page. Add processing progress and results table using real
backend APIs. Do not use fake results. If temporary mock data is needed,
put it behind an explicit development-only mock flag and label it
visibly.

## Prompt 6 --- Local image preprocessing

Implement image validation and preprocessing adapters using OpenCV and
Pillow. Include EXIF orientation, corrupt-image handling, dimensions,
blur heuristic, resize limits, safe crop helper, and temporary-file
cleanup. Preserve original S3 objects. Write unit tests for invalid
files, oversized images, crop bounds, and orientation behavior.

## Prompt 7 --- Pretrained object detector adapter

Implement a detector interface and a local zero-shot detector adapter
using OWLv2 or Grounding DINO. First verify installation and inference
on Apple Silicon CPU. Use prompts for parcel, cardboard box, shipping
package, and shipping label. Normalize outputs to the project detection
schema with pixel-coordinate boxes. Add overlap deduplication and
confidence configuration. Do not train or fine-tune anything. If the
selected model is incompatible with M1, stop and report the issue before
substituting another model.

## Prompt 8 --- OCR and barcode adapters

Implement local PaddleOCR and barcode decoding adapters using ZXing-C++
or pyzbar. Return structured raw text, geometry, confidence, decoded
values, and symbology where available. Load models once and reuse them.
Add CPU-first configuration suitable for an M1. Add tests using mocked
OCR and barcode outputs. Do not call hosted inference APIs.

## Prompt 9 --- Field extraction and validation

Implement deterministic parsing for AWB/tracking number, weight, and
dimensions. Preserve raw OCR text. Add barcode/OCR comparison and
conflict handling. Keep fields null when evidence is missing or
ambiguous. Add field-level confidence metadata and unit normalization
only when unambiguous. Write extensive unit tests using realistic label
text variations.

## Prompt 10 --- Orchestrated per-image worker

Connect preprocessing, detection, label crop, OCR, barcode decoding,
extraction, validation, and status classification into one per-image
worker. Add bounded concurrency configurable with default 1. Make
processing idempotent. Ensure one failed image does not stop the batch.
Persist exactly one extraction row per accepted image, including
`PROCESSING_FAILED` rows. Add a reconciliation step for any accepted
image missing a result. Test the full flow with mocked model adapters.

## Prompt 11 --- Review UI and CSV export

Implement result filters, search, status badges, image detail view,
original image preview using presigned GET URLs, detection/label crop
evidence, raw OCR evidence, editable extracted fields, and reviewed
state. Implement CSV export with exactly one row per accepted uploaded
image, including exception rows. Test row count, escaping, missing
values, and export behavior.

## Prompt 12 --- End-to-end verification and documentation

Run the entire application locally on Apple Silicon. Verify S3 upload,
processing, review, and CSV export. Add an evaluation script that
reports parcel detection, label detection, AWB exact-match, weight,
dimensions, exception status, processing failure, and
one-result-per-image metrics against a manually labelled ground-truth
CSV. Update README with setup, AWS configuration, model download,
troubleshooting, limitations, and S3 cost controls. Do not claim
accuracy without measured results.

## Prompt 13 --- Code review and hardening

Review the whole codebase against `agent.md`. Find and fix: -
credentials or secrets exposed to frontend; - public S3 object access; -
unsafe upload validation; - path traversal; - unbounded concurrency or
memory use; - missing per-image failure handling; - duplicate extraction
rows; - incorrect status classification; - CSV rows missing for accepted
images; - fake model results accidentally enabled; - inaccurate README
commands; - incompatible Apple Silicon dependencies.

Run tests and report remaining known limitations. Do not add new product
features.
