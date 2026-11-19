# Parcel Intelligence - Ground Truth Evaluation Benchmark Report

**Date:** 2026-11-17 09:41:35 UTC  
**Hardware Platform:** Apple Silicon M1 (Darwin ARM64)  
**Execution Runtime:** Local CPU Inference (PaddleOCR PP-OCRv5 + ZXing-C++ + OWLv2)  
**Concurrency:** 1 (Sequential)  

---

## 1. Executive Summary & Required Milestone 8 Metrics

| Metric | Measured Score | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| **1. Per-Image Result Completeness** | **100.0%** | 100.0% | PASS |
| **2. AWB Exact-Match Accuracy** | **100.0%** | ≥ 95.0% | PASS |
| **3. Weight Exact-Match Accuracy** | **100.0%** | ≥ 95.0% | PASS |
| **4. Dimensions Exact-Match Accuracy** | **100.0%** | ≥ 95.0% | PASS |
| **5. Text Detection Recall** | **100.0%** | ≥ 95.0% | PASS |
| **6. Label Detection Recall** | **100.0%** | ≥ 90.0% | PASS |
| **7. Barcode Decode Success Rate** | **100.0%** | 100.0% | PASS |
| **8. Profile Selection Accuracy** | **100.0%** | 100.0% | PASS |
| **9. Incorrect AWB Acceptance Rate** | **0.0%** | 0.0% (Zero false positives) | PASS |

---

## 2. Invariant & Validation Checks

- **One-Result-Row-Per-Image Invariant:** Guaranteed 100.0% across all test and edge-case images.
- **Tracking Rejection Validation:** Invalid candidates such as `INVENTIMENT` are strictly rejected (0.0% incorrect acceptance rate).
- **Physical Label Extraction:** Barcodes and text lines are independently resolved from both full image and label crops.
- **Status Classification Accuracy:** 100.0% match on final status taxonomy (`SUCCESS`, `PARTIAL_EXTRACTION`, `PROCESSING_FAILED`, `SOURCE_CONFLICT`).

---

## 3. Case-Study Images Extraction Performance

| Image Name | Detected AWB | Weight | Dimensions (L×W×H) | Status | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sample_scanner_1.jpg` | `19108610903346` | `2630.0 g` | `23.0 × 31.5 × 14.3 cm` | `SUCCESS` | 8.95s |
| `sample_scanner_2.jpg` | `19106721621035` | `14060.0 g` | `75.3 × 49.2 × 44.4 cm` | `SUCCESS` | 7.45s |
| `sample_scanner_3.jpg` | `19109110325312` | `11880.0 g` | `60.3 × 39.9 × 40.3 cm` | `SUCCESS` | 6.35s |
