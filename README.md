# Parcel Intelligence Prototype

A local-first parcel-image processing prototype for warehouse shipment operations, engineered for Apple Silicon Mac (M1).

Warehouse operators upload batches of scanner photos (JPG/PNG). The application stores originals securely in private AWS S3, runs zero-shot parcel and label detection, crops shipping labels, executes local OCR and barcode decoding, parses logistics tracking numbers, weights, and dimensions deterministically, enforces the one-result-record-per-image invariant, provides an interactive operator review dashboard, and exports deterministic CSVs.

---

## Non-Negotiable Constraints Adherence

1. **Local Apple Silicon M1 (arm64)**: Optimized for Apple Silicon CPU execution.
2. **No CUDA / No Hosted Inference APIs**: Zero external paid APIs (no cloud vision, no paid OCR, no LLM inference).
3. **AWS S3 Private Storage**: Bucket remains private. Zero AWS credentials exposed to the browser. Browser uploads directly using backend-issued short-lived presigned PUT URLs; previews use short-lived presigned GET URLs.
4. **Local Models Only**: Zero-shot parcel detection (OWLv2) on CPU, local OCR (TrOCR/PaddleOCR), and ZXing-C++ barcode/QR decoding.
5. **No Redis, No Celery, No LangChain, No LangGraph**: In-process bounded queue (`WORKER_CONCURRENCY=1`) embedded within the FastAPI application lifecycle.
6. **One Result Row Per Accepted Image**: Enforced by a database-level `UNIQUE(image_id)` constraint, failure handlers that persist `PROCESSING_FAILED` rows, and a batch reconciliation pass.

---

## Architecture Overview

```text
                                +---------------------------------------------+
                                |         Next.js App Router Frontend         |
                                |  - Multi-file drag/drop upload              |
                                |  - Presigned direct S3 upload & progress    |
                                |  - Results filtering, search & inspection   |
                                |  - Manual correction & review workflow      |
                                +--------------------+------------------------+
                                                     |
                         1. Presigned PUT Request    | 3. Direct Binary PUT
                         4. Upload Complete Callback |    (Zero AWS creds in browser)
                         5. Results & CSV Export     v
                                                     +------------------------+
                                                     |     Private AWS S3     |
                                                     |  - batches/{id}/...    |
                                                     |  - Short-lived GET URL |
                                                     +-----------+------------+
                                                                 |
                                       +-------------------------+
                                       | 6. Worker downloads original
                                       v    to temporary file
                        +--------------+--------------+
                        |       FastAPI Backend       |
                        |   - SQLite (WAL Mode)       |
                        |   - Local Bounded Worker    |
                        +--------------+--------------+
                                       |
                                       v
                             In-Process Pipeline:
                             1. Image Preprocessor (OpenCV/PIL EXIF & blur)
                             2. Object Detector (OWLv2 CPU)
                             3. Label Cropper & Clamping
                             4. Local OCR (TrOCR / PaddleOCR)
                             5. Barcode Decoder (ZXing-C++)
                             6. Deterministic Field Parser & Conflict Logic
                             7. Temp file cleanup in finally block
```

---

## Directory Layout

```text
parsel-intelegence/
├── agent.md                        # Project specification and context
├── README.md                       # Comprehensive guide and documentation
├── .gitignore                      # Git ignore patterns
├── .env.example                    # Environment variables template
├── backend/
│   ├── pyproject.toml              # Python dependencies & Ruff/Pytest config
│   ├── requirements.txt            # Core requirements
│   ├── requirements-dev.txt        # Dev & testing requirements
│   ├── app/
│   │   ├── main.py                 # FastAPI app, lifespan worker startup, CORS
│   │   ├── config.py               # Pydantic Settings
│   │   ├── database.py             # SQLite WAL engine & sessionmaker
│   │   ├── models/                 # SQLAlchemy ORM models (Batch, Image, Extraction)
│   │   ├── schemas/                # Pydantic v2 schemas
│   │   ├── api/                    # API route handlers (batches, images, export, health)
│   │   ├── services/               # S3 service (presigned URLs, HEAD verification)
│   │   ├── pipeline/               # Modular adapters (preprocessor, detector, OCR, barcode, parser)
│   │   └── worker/                 # In-process queue & orchestrator
│   ├── tests/                      # Pytest test suite (34 passing tests)
│   └── scripts/
│       └── evaluate_ground_truth.py# Benchmark evaluation script
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   └── src/
│       ├── app/
│       │   ├── page.tsx            # Batch list & operational dashboard
│       │   └── batches/[batchId]/  # Batch detail, progress, results & CSV download
│       ├── components/
│       │   ├── upload/             # Multi-file direct S3 uploader
│       │   ├── review/             # Side-by-side inspection & manual editing modal
│       │   └── ui/                 # Status badges & counters
│       └── lib/
│           ├── api.ts              # Backend API fetch client
│           └── s3-upload.ts        # Direct presigned PUT upload helper
└── evaluation/
    └── ground_truth_template.csv   # Template ground truth dataset
```

---

## Setup & Running Locally

### 1. Configure Environment

Copy the example environment file:
```bash
cp .env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

### 2. Backend Setup & Tests

From the repository root:
```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Run the complete test suite (34 tests)
pytest

# Start the backend server (runs on http://localhost:8000)
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

From a separate terminal window:
```bash
cd frontend

# Install node dependencies
npm install

# Build production bundle
npm run build

# Start Next.js development server (runs on http://localhost:3000)
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) in your browser.

---

## Docker & EC2 Deployment

The application is containerized with multi-stage builds and an Nginx reverse proxy. It can be deployed on any AWS EC2 instance (e.g., `c7g.2xlarge` for ARM64 Graviton or `t3.2xlarge` / `g4dn.xlarge` for x86).

### Architecture on EC2
```text
Port 80 (HTTP) ---> Nginx Reverse Proxy
                     ├── /api/batches/.../events ---> FastAPI (SSE streaming, no buffering)
                     ├── /api/*                 ---> FastAPI Backend (:8000)
                     └── /                      ---> Next.js Standalone Frontend (:3000)
```

### 1. Launching on an EC2 Instance
1. Launch an EC2 instance (Ubuntu 22.04/24.04 LTS or Amazon Linux 2023, recommended: `c7g.2xlarge` or `t3.2xlarge` with $\ge 8\text{ GB}$ RAM and $30\text{ GB}$ GP3 disk).
2. Attach an IAM Role with Amazon S3 permissions to the EC2 instance (recommended so you don't need access keys), or configure keys in `.env`.
3. Open Security Group inbound ports: **80** (HTTP), **443** (HTTPS), and **22** (SSH).

### 2. Deploy with Docker Compose
Clone the repository onto the instance and start with the automated script or manual commands:

```bash
# Clone repository
git clone <repo-url> parsel-intelegence
cd parsel-intelegence

# Configure environment variables
cp .env.example .env
nano .env   # Set AWS_S3_BUCKET and AWS_REGION

# Launch with one command:
./scripts/deploy_ec2.sh
```

Or run Docker Compose directly:
```bash
docker compose up -d --build
```

### 3. Verify Deployment
```bash
# Check container status
docker compose ps

# Check backend logs
docker compose logs -f backend

# Access dashboard in your browser
http://<YOUR-EC2-PUBLIC-IP>
```


---

## AWS S3 Configuration & Cost Controls

The application uses an Amazon S3 bucket for private storage.

### Minimum IAM Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:HeadObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::<your-bucket-name>",
        "arn:aws:s3:::<your-bucket-name>/batches/*"
      ]
    }
  ]
}
```

### Bucket CORS Policy
```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["PUT", "GET", "HEAD"],
    "AllowedOrigins": ["http://localhost:3000"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

### Cost Guardrails
- **Lifecycle Expiration**: Configure an S3 lifecycle rule to expire objects under `batches/*` after 7–14 days.
- **Single Download**: The worker downloads original images from S3 once to a local temporary directory and cleans them up immediately in `finally` blocks.
- **Zero S3 Costs in Testing**: Backend unit tests run with `moto[s3]` mocking S3 completely in memory.

---

## Ground Truth Evaluation

Run the evaluation script against ground truth data:
```bash
python backend/scripts/evaluate_ground_truth.py \
  --ground-truth evaluation/ground_truth_template.csv \
  --actual /path/to/exported_batch_results.csv
```

Reports:
- Total Ground Truth Images
- Total Actual Export Rows
- One-Result-Per-Image Invariant Verification
- Tracking AWB Exact Match Accuracy
- Weight Normalized Match Accuracy
- Dimensions Match Accuracy
- Status Taxonomy Match Accuracy
