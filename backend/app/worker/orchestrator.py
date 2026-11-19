"""Image processing orchestrator coordinating the 3-branch multi-format DWS architecture.

Architecture:
- Branch A: Machine Metadata Extraction (candidate overlay detection, enhancement, profile OCR)
- Branch B: Physical Label Extraction (label detection, safe cropping, rotation correction, barcode & OCR)
- Branch C: Scene Validation (parcel detection, scanner-bed ROI filtering, edge proximity evidence)
- Evidence Reconciliation: Merges multi-source evidence into unified extraction record with provenance
- Debug Bundling: Auditable debug artifacts saved per processed image
"""

import torch  # Ensure PyTorch is imported before Paddle to prevent macOS tensor collision
import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.models.image import Image
from app.models.batch import Batch
from app.models.extraction import Extraction
from app.services.s3_service import s3_service
from app.pipeline.interfaces import (
    IImagePreprocessor,
    IObjectDetector,
    IOcrEngine,
    IBarcodeDecoder,
    IFieldParser,
    DetectionBox,
    DetectionResult,
    OcrResult,
    OcrTextBox,
    BarcodeResult,
    ExtractedLogisticsData,
    STATUS_PROCESSING_FAILED,
)
from app.pipeline.preprocessor import preprocessor as default_preprocessor, ImageValidationError, PreprocessedImage
from app.pipeline.detector import OwlV2ZeroShotDetector, MockObjectDetector, filter_parcels_in_roi, check_edge_proximity
from app.pipeline.ocr import get_ocr_engine, TesseractOcrEngine, MockOcrEngine
from app.pipeline.barcode import ZXingBarcodeDecoder, MockBarcodeDecoder
from app.pipeline.parser import field_parser as default_field_parser
from app.pipeline.overlay_extractor import (
    overlay_detector,
    CandidateOverlay,
    evaluate_image_quality,
    generate_controlled_variants,
)
from app.pipeline.spatial_clustering import (
    group_boxes_into_lines,
    cluster_lines_spatially,
    run_tiled_ocr,
    merge_text_boxes,
    get_box_bounds,
    SpatialCluster,
)
from app.pipeline.label_extractor import (
    PhysicalLabelExtractor,
    LabelCandidate,
    generate_label_crops,
)
from app.pipeline.paddleocr_vl_adapter import (
    PaddleOcrVlAdapter,
    MockPaddleOcrVlAdapter,
    VlmInterpretationResult,
)
from app.pipeline.parser_registry import parser_registry, ParsedOverlayMetadata, is_valid_tracking
from app.config import settings
from app.worker.broadcaster import (
    pipeline_broadcaster,
    PipelineStage,
    STAGE_METADATA,
    TOTAL_PIPELINE_STEPS,
)

logger = logging.getLogger("orchestrator")


class PipelineOrchestrator:
    """
    Coordinates the 3-branch CV/OCR pipeline for an individual image.
    Enforces the non-negotiable one-result-row-per-image invariant.
    """

    def __init__(
        self,
        preprocessor: Optional[IImagePreprocessor] = None,
        detector: Optional[IObjectDetector] = None,
        ocr_engine: Optional[IOcrEngine] = None,
        barcode_decoder: Optional[IBarcodeDecoder] = None,
        parser: Optional[IFieldParser] = None,
    ):
        self.preprocessor = preprocessor or default_preprocessor

        # Detector selection
        if detector:
            self.detector = detector
        elif settings.MOCK_INFERENCE:
            self.detector = MockObjectDetector()
        else:
            self.detector = OwlV2ZeroShotDetector()

        # OCR selection
        if ocr_engine:
            self.ocr_engine = ocr_engine
        else:
            self.ocr_engine = get_ocr_engine(settings.OCR_ENGINE)

        # Optional secondary high-contrast Tesseract adapter
        self.tesseract_engine = TesseractOcrEngine()

        # Barcode selection
        if barcode_decoder:
            self.barcode_decoder = barcode_decoder
        elif settings.MOCK_INFERENCE:
            self.barcode_decoder = MockBarcodeDecoder()
        else:
            self.barcode_decoder = ZXingBarcodeDecoder()

        self.parser = parser or default_field_parser
        self.label_extractor = PhysicalLabelExtractor(
            ocr_engine=self.ocr_engine,
            barcode_decoder=self.barcode_decoder,
        )
        self.vl_adapter = (
            MockPaddleOcrVlAdapter()
            if settings.MOCK_INFERENCE
            else PaddleOcrVlAdapter()
        )
        self._latest_full_ocr: Optional[OcrResult] = None
        self._latest_clusters: Optional[List[SpatialCluster]] = None

    # =========================================================================
    # Branch A: Machine Metadata Extraction
    # =========================================================================

    def _run_branch_a(
        self,
        preprocessed: PreprocessedImage,
    ) -> Tuple[Optional[ParsedOverlayMetadata], Optional[OcrResult], List[Dict[str, Any]]]:
        """
        Run full-resolution text detection, line grouping, and spatial clustering.
        Evaluate candidate overlay regions as an optimization, apply controlled variants
        when indicated by image illumination/quality, and parse via format profile registry.
        """
        bgr_image = preprocessed.bgr_image
        h, w = bgr_image.shape[:2]

        # 1. Image illumination and quality evaluation
        quality_info = evaluate_image_quality(bgr_image)
        logger.info(
            f"🏷️  [Branch A: Full-Res OCR & Metadata] Quality: brightness={quality_info['mean_brightness']:.1f}, "
            f"contrast={quality_info['contrast']:.1f}, dark_ratio={quality_info['dark_ratio']:.2f} (is_dark={quality_info['is_dark']})"
        )

        # 2. Primary text detection on original full-resolution image
        try:
            full_ocr = self.ocr_engine.extract_text(
                bgr_image,
                variant_label="original_full_res",
                source_crop="full_image",
            )
        except Exception as err:
            logger.error(f"Full-resolution OCR detection failed: {err}")
            full_ocr = OcrResult(
                engine=self.ocr_engine.__class__.__name__,
                model_version=getattr(self.ocr_engine, "MODEL_VERSION", "v1"),
                lines=[],
                full_text="",
                source_crop="full_image",
                preprocessing_variant="original_full_res",
            )

        # 3. For dark or low-contrast images, generate controlled variants (gamma / CLAHE)
        if quality_info["is_dark"] or quality_info["contrast"] < 35.0:
            controlled_variants = generate_controlled_variants(bgr_image, select_based_on_quality=True)
            for var_name, var_img in controlled_variants.items():
                if var_name == "original":
                    continue
                try:
                    var_ocr = self.ocr_engine.extract_text(
                        var_img,
                        variant_label=f"full_image_{var_name}",
                        source_crop="full_image",
                    )
                    # Compare outputs: merge new/better detections
                    if len(var_ocr.lines) > len(full_ocr.lines) or (
                        var_ocr.lines and not full_ocr.lines
                    ):
                        logger.info(f"   ├─ Controlled variant '{var_name}' yielded {len(var_ocr.lines)} line(s) vs {len(full_ocr.lines)} in original")
                        full_ocr.lines = merge_text_boxes(full_ocr.lines, var_ocr.lines)
                        full_ocr.full_text = "\n".join(b.text for b in full_ocr.lines)
                except Exception as var_err:
                    logger.debug(f"Variant OCR '{var_name}' failed: {var_err}")

        # 4. Tiled OCR for small text on high-resolution images
        if (w > 1200 or h > 1200) and full_ocr.lines:
            has_small_text = any((get_box_bounds(l)[3] - get_box_bounds(l)[1] < 18) for l in full_ocr.lines)
            if has_small_text or w > 2000 or h > 2000:
                tiled_boxes = run_tiled_ocr(self.ocr_engine, bgr_image, tile_w=960, tile_h=960, overlap_px=140)
                if tiled_boxes:
                    logger.info(f"   ├─ Tiled OCR detected {len(tiled_boxes)} text boxes for small text preservation")
                    full_ocr.lines = merge_text_boxes(full_ocr.lines, tiled_boxes)
                    full_ocr.full_text = "\n".join(b.text for b in full_ocr.lines)

        # 5. Group text boxes into lines and spatial clusters
        lines_grouped = group_boxes_into_lines(full_ocr.lines)
        clusters = cluster_lines_spatially(lines_grouped, w, h)
        logger.info(
            f"   ├─ Full-image OCR: {len(full_ocr.lines)} boxes -> {len(lines_grouped)} lines -> {len(clusters)} spatial cluster(s)"
        )
        for idx, cl in enumerate(clusters, 1):
            logger.info(f"   │  ├─ Cluster #{idx} [{cl.region_label}] bbox={cl.bbox_xyxy} ({len(cl.text_lines)} lines, conf={cl.confidence:.2f})")

        # 6. Candidate ROI detection as optimization / spatial cue
        candidate_regions = overlay_detector.get_candidate_regions(bgr_image)
        logger.info(f"   ├─ Candidate overlay detector identified {len(candidate_regions)} deduplicated candidate ROI(s)")
        rois_debug: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidate_regions, 1):
            logger.info(f"   │  ├─ Candidate ROI #{idx}: '{cand.format_hint}' bbox={cand.bbox_xyxy} score={cand.score:.2f}")
            rois_debug.append({
                "format_hint": cand.format_hint,
                "bbox_xyxy": cand.bbox_xyxy,
                "score": cand.score,
            })

        best_meta: Optional[ParsedOverlayMetadata] = None
        best_score = -1.0

        # 7. Evaluate spatial clusters against profile registry
        for cl in clusters:
            profile = parser_registry.select_profile(cl.full_text, format_hint=cl.region_label)
            parsed_meta = profile.parse(cl.full_text, cl.text_lines)

            if parsed_meta.is_populated():
                field_count = sum(1 for f in (parsed_meta.tracking_number, parsed_meta.weight_value, parsed_meta.length) if f is not None)
                pos_weight = 0.3 if cl.region_label in ("top_left", "bottom_panel") else 0.1
                cluster_score = parsed_meta.confidence + 0.35 * field_count + pos_weight

                if cluster_score > best_score:
                    best_score = cluster_score
                    best_meta = parsed_meta
                    logger.info(
                        f"   ├─ Match from Spatial Cluster [{cl.region_label}] -> Profile: '{profile.name}' (score={cluster_score:.2f}) | "
                        f"AWB='{parsed_meta.tracking_number}' Wt='{parsed_meta.weight_value}' Dims='{parsed_meta.length}x{parsed_meta.width}x{parsed_meta.height}'"
                    )

        # 8. Also evaluate candidate ROI crops (for specialized color segmentation e.g. green/yellow DWS)
        for candidate in candidate_regions:
            candidate_texts: List[Tuple[str, List[str], str]] = []
            for variant_name in ("original", "clahe", "upscaled"):
                variant_img = candidate.variants.get(variant_name)
                if variant_img is None or variant_img.size == 0:
                    continue
                try:
                    cand_ocr = self.ocr_engine.extract_text(
                        variant_img,
                        variant_label=variant_name,
                        source_crop=candidate.format_hint,
                    )
                    if cand_ocr.full_text.strip():
                        candidate_texts.append((cand_ocr.full_text, [l.text for l in cand_ocr.lines], variant_name))
                except Exception as cand_err:
                    logger.debug(f"Candidate crop OCR failed on {variant_name}: {cand_err}")

            for txt, lines_list, vname in candidate_texts:
                profile = parser_registry.select_profile(txt, candidate.format_hint)
                parsed_meta = profile.parse(txt, lines_list)
                if parsed_meta.is_populated():
                    field_count = sum(1 for f in (parsed_meta.tracking_number, parsed_meta.weight_value, parsed_meta.length) if f is not None)
                    cand_score = parsed_meta.confidence + 0.35 * field_count + 0.2 * candidate.score
                    if cand_score > best_score:
                        best_score = cand_score
                        best_meta = parsed_meta
                        logger.info(
                            f"   ├─ Match from Candidate ROI [{candidate.format_hint}:{vname}] -> Profile: '{profile.name}' (score={cand_score:.2f}) | "
                            f"AWB='{parsed_meta.tracking_number}' Wt='{parsed_meta.weight_value}' Dims='{parsed_meta.length}x{parsed_meta.width}x{parsed_meta.height}'"
                        )

        self._latest_full_ocr = full_ocr
        self._latest_clusters = clusters

        if best_meta:
            logger.info(f"   └─ Final Selected Overlay Profile: '{best_meta.profile_name}' (Score: {best_score:.2f})")
        else:
            logger.info("   └─ No readable machine metadata overlay matched")

        return best_meta, full_ocr, rois_debug

    # =========================================================================
    # Branch B: Physical Label Extraction
    # =========================================================================

    def _correct_perspective_or_rotation(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Correct subtle rotation if reliable orientation can be detected."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            # Find dominant line angles via HoughLinesP
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -45 < angle < 45:
                        angles.append(angle)
                if angles:
                    median_angle = float(np.median(angles))
                    if abs(median_angle) > 1.0 and abs(median_angle) < 30.0:
                        h, w = crop_bgr.shape[:2]
                        center = (w // 2, h // 2)
                        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        corrected = cv2.warpAffine(crop_bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        return corrected
        except Exception:
            pass
        return crop_bgr

    def _run_branch_b(
        self,
        preprocessed: PreprocessedImage,
        det_result: DetectionResult,
        full_ocr: Optional[OcrResult] = None,
        spatial_clusters: Optional[List[SpatialCluster]] = None,
        overlay_bboxes: Optional[List[List[int]]] = None,
    ) -> Tuple[Optional[OcrResult], List[BarcodeResult], Optional[List[int]], Optional[np.ndarray]]:
        """
        Extract physical shipping label: multi-cue candidate localization
        (spatial clustering, barcodes, and OWLv2), safe crop with padding,
        multi-angle rotations, contrast enhancement, independent barcode decoding,
        and full-fidelity label OCR.
        Explicitly excludes machine overlay boxes from physical-label candidates.
        """
        logger.info("📦 [Branch B: Physical Label] Processing shipping label & barcodes...")
        if full_ocr is None:
            full_ocr = getattr(self, "_latest_full_ocr", None)
        if spatial_clusters is None:
            spatial_clusters = getattr(self, "_latest_clusters", None)

        full_boxes = full_ocr.lines if full_ocr else None
        return self.label_extractor.extract_label_and_barcodes(
            bgr_image=preprocessed.bgr_image,
            det_result=det_result,
            full_ocr_lines=full_boxes,
            spatial_clusters=spatial_clusters,
            overlay_bboxes=overlay_bboxes,
        )

    # =========================================================================
    # Branch C: Scene Validation
    # =========================================================================

    def _run_branch_c(
        self,
        preprocessed: PreprocessedImage,
    ) -> DetectionResult:
        """
        Zero-shot parcel and label detection. Edge proximity is captured as evidence.
        """
        logger.info("🔍 [Branch C: Scene Validation] Running zero-shot detector...")
        try:
            det_res = self.detector.detect(preprocessed.pil_image)
            parcels = [d for d in det_res.detections if d.class_name == "parcel"]
            labels = [d for d in det_res.detections if d.class_name == "shipping_label"]
            logger.info(
                f"   └─ Detections: {len(parcels)} parcel(s), {len(labels)} label(s) (Detector: {det_res.model})"
            )
            for p in parcels:
                logger.info(f"      • Parcel detected at {p.bbox_xyxy} score={p.score:.2f}")
            for l in labels:
                logger.info(f"      • Label detected at {l.bbox_xyxy} score={l.score:.2f}")
            return det_res
        except Exception as e:
            logger.warning(f"   └─ Branch C detector inference failed: {e}")
            # Ensure detector exception never aborts metadata extraction
            return DetectionResult(model="detector-fallback", detections=[])

    # =========================================================================
    # Debug Bundle Builder
    # =========================================================================

    def _build_debug_bundle(
        self,
        image: Image,
        rois_debug: List[Dict[str, Any]],
        overlay_ocr: Optional[OcrResult],
        label_ocr: Optional[OcrResult],
        label_crop_bbox: Optional[List[int]],
        barcode_results: List[BarcodeResult],
        parsed_data: ExtractedLogisticsData,
        vlm_debug: Optional[Dict[str, Any]] = None,
        preprocessed: Optional[PreprocessedImage] = None,
        det_result: Optional[DetectionResult] = None,
        spatial_clusters: Optional[List[SpatialCluster]] = None,
    ) -> Dict[str, Any]:
        """Construct comprehensive auditable debug bundle for review and verification (Milestone 7)."""
        # 1. Original image metadata
        original_img_info: Dict[str, Any] = {
            "s3_key": image.s3_key,
            "filename": image.filename,
            "file_size": image.file_size,
        }
        if preprocessed:
            m = preprocessed.metrics
            original_img_info["dimensions"] = {"width": m.width, "height": m.height, "channels": m.channels}
            original_img_info["quality_metrics"] = {
                "blur_score": m.blur_score,
                "is_blurry": m.is_blurry,
                "mean_brightness": m.mean_brightness,
            }

        # 2. All candidate regions
        all_candidate_regions: List[Dict[str, Any]] = list(rois_debug)
        if det_result:
            for d in det_result.detections:
                all_candidate_regions.append({
                    "source": f"detector_{d.class_name}",
                    "bbox_xyxy": d.bbox_xyxy,
                    "score": d.score,
                })

        # 3. All OCR detections and bounding boxes
        all_ocr_lines = []
        if overlay_ocr:
            for l in overlay_ocr.lines:
                bx = get_box_bounds(l)
                all_ocr_lines.append({
                    "source": "overlay",
                    "text": l.text,
                    "score": l.score,
                    "quad_box": l.box,
                    "bbox_xyxy": list(bx),
                })
        if label_ocr:
            for l in label_ocr.lines:
                bx = get_box_bounds(l)
                all_ocr_lines.append({
                    "source": "label",
                    "text": l.text,
                    "score": l.score,
                    "quad_box": l.box,
                    "bbox_xyxy": list(bx),
                })

        # 4. Text from each region
        text_by_region: Dict[str, str] = {}
        if overlay_ocr and overlay_ocr.full_text:
            text_by_region["overlay_full_text"] = overlay_ocr.full_text
        if label_ocr and label_ocr.full_text:
            text_by_region["label_full_text"] = label_ocr.full_text
        if spatial_clusters:
            for idx, sc in enumerate(spatial_clusters, 1):
                text_by_region[f"cluster_{idx}_{sc.region_label}"] = sc.full_text

        # 5. Enhanced image variants
        enhanced_variants = ["original"]
        if overlay_ocr and overlay_ocr.preprocessing_variant:
            enhanced_variants.append(overlay_ocr.preprocessing_variant)
        if label_ocr and label_ocr.preprocessing_variant:
            enhanced_variants.append(label_ocr.preprocessing_variant)

        # 6. Label crops
        label_crop_info: Dict[str, Any] = {
            "label_crop_bbox": label_crop_bbox,
            "variants_evaluated": ["0_deg", "90_deg", "180_deg", "270_deg", "clahe", "deskewed"],
        }

        # 7. Barcode decoder attempts
        barcode_attempts = [b.model_dump() for b in barcode_results]

        # 8. Profile scores and matched anchors
        combined_text = (overlay_ocr.full_text if overlay_ocr else "") + "\n" + (label_ocr.full_text if label_ocr else "")
        profile_scores_and_anchors = parser_registry.audit_profiles(combined_text)

        # 9. Parsed values and validation failures
        validation_failures = []
        track_prov = parsed_data.provenance.get("tracking_number")
        if track_prov and getattr(track_prov, "raw_candidates", None):
            for cand in track_prov.raw_candidates:
                if not is_valid_tracking(cand):
                    validation_failures.append({
                        "field": "tracking_number",
                        "candidate": cand,
                        "reason": "Alphabetic-only tracking candidate rejected; requires >= 1 digit (e.g. INVENTIMENT)" if cand.isalpha() else "Invalid tracking length or format",
                    })

        # 10. Final field provenance and confidence
        bundle = {
            "original_image": original_img_info,
            "candidate_regions": all_candidate_regions,
            "all_ocr_detections": all_ocr_lines,
            "text_by_region": text_by_region,
            "enhanced_image_variants": list(set(enhanced_variants)),
            "label_crop_info": label_crop_info,
            "barcode_decoder_attempts": barcode_attempts,
            "profile_scores_and_matched_anchors": profile_scores_and_anchors,
            "parsed_values_and_validation_failures": {
                "parsed_values": {
                    "tracking_number": parsed_data.tracking_number,
                    "weight_value": parsed_data.weight_value,
                    "weight_unit": parsed_data.weight_unit,
                    "actual_weight": parsed_data.actual_weight,
                    "volumetric_weight": parsed_data.volumetric_weight,
                    "length": parsed_data.length,
                    "width": parsed_data.width,
                    "height": parsed_data.height,
                    "dimension_unit": parsed_data.dimension_unit,
                    "status": parsed_data.status,
                    "scene_flags": parsed_data.scene_flags,
                },
                "validation_failures": validation_failures,
            },
            "final_field_provenance_and_confidence": {
                "provenance": {k: v.model_dump() for k, v in parsed_data.provenance.items()},
                "field_confidence": parsed_data.field_confidence,
            },
            # Backward-compatible keys for existing tests
            "original_image_ref": image.s3_key,
            "detected_metadata_rois": rois_debug,
            "ocr_text_and_geometry": all_ocr_lines,
            "detected_label_crop": label_crop_bbox,
            "barcode_output": barcode_attempts,
            "parser_output": {
                "tracking_number": parsed_data.tracking_number,
                "weight_value": parsed_data.weight_value,
                "weight_unit": parsed_data.weight_unit,
                "actual_weight": parsed_data.actual_weight,
                "volumetric_weight": parsed_data.volumetric_weight,
                "length": parsed_data.length,
                "width": parsed_data.width,
                "height": parsed_data.height,
                "dimension_unit": parsed_data.dimension_unit,
                "status": parsed_data.status,
                "scene_flags": parsed_data.scene_flags,
            },
            "source_reconciliation_result": {
                k: v.model_dump() for k, v in parsed_data.provenance.items()
            },
        }
        if vlm_debug:
            bundle["vlm_interpretation"] = vlm_debug

        # Save debug bundle to local file for offline inspection and audit
        try:
            debug_dir = os.path.join("./debug_bundles", str(image.batch_id))
            os.makedirs(debug_dir, exist_ok=True)
            bundle_file = os.path.join(debug_dir, f"{image.id}_debug_bundle.json")
            with open(bundle_file, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2, default=str)
            logger.info(f"📁 [DEBUG BUNDLE] Saved comprehensive audit bundle to {bundle_file}")
        except Exception as err:
            logger.debug(f"Could not write debug bundle to disk: {err}")

        return bundle

    def _set_stage(
        self,
        db: Session,
        image: Image,
        stage: str,
        message: Optional[str] = None,
        progress: Optional[int] = None,
    ):
        """Update and commit image processing stage and broadcast real-time event."""
        meta = STAGE_METADATA.get(stage, {})
        step_idx = meta.get("step_index", 0)
        stage_name = meta.get("name", stage)
        stage_msg = message or meta.get("default_message", "")
        stage_prog = progress if progress is not None else meta.get("progress", 0)

        image.current_stage = stage
        image.stage_message = stage_msg
        image.stage_progress = stage_prog
        image.stage_started_at = datetime.now(timezone.utc)
        image.updated_at = datetime.now(timezone.utc)
        db.commit()

        preview_url = None
        try:
            preview_url = s3_service.create_presigned_get_url(image.s3_key)
        except Exception:
            pass

        pipeline_broadcaster.broadcast(
            image.batch_id,
            {
                "type": "stage_update",
                "batch_id": image.batch_id,
                "image_id": image.id,
                "filename": image.filename,
                "stage": stage,
                "stage_name": stage_name,
                "stage_message": stage_msg,
                "stage_progress": stage_prog,
                "stage_index": step_idx,
                "total_stages": TOTAL_PIPELINE_STEPS,
                "preview_url": preview_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # =========================================================================
    # Main Pipeline Execution
    # =========================================================================

    def process_image(self, db: Session, image_id: str) -> Extraction:
        """
        Process a single image through the 3-branch pipeline.
        Guaranteed to produce exactly one extraction record.
        """
        start_time = time.perf_counter()
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise ValueError(f"Image {image_id} not found in database.")

        image.processing_status = "PROCESSING"
        self._set_stage(
            db,
            image,
            PipelineStage.INGESTING,
            message="Downloading original scan from private storage...",
            progress=15,
        )

        logger.info("=" * 80)
        logger.info(f"🚀 [PIPELINE START] Processing Image: {image.filename}")
        logger.info(f"   Image ID:  {image.id}")
        logger.info(f"   Batch ID:  {image.batch_id}")
        logger.info(f"   S3 Key:    {image.s3_key}")
        logger.info(f"   File Size: {image.file_size} bytes ({image.content_type})")
        logger.info("-" * 80)

        local_temp_path = os.path.join(
            settings.TEMP_DOWNLOAD_DIR,
            f"{image.id}_{os.path.basename(image.filename)}",
        )

        try:
            # 1. Download original from S3 to temp directory
            t_dl = time.perf_counter()
            s3_service.download_to_file(image.s3_key, local_temp_path)
            dl_ms = (time.perf_counter() - t_dl) * 1000
            logger.info(f"📥 [STEP 1: S3 INGEST] Downloaded in {dl_ms:.1f}ms -> {local_temp_path}")

            # 2. Image validation and quality metrics
            self._set_stage(
                db,
                image,
                PipelineStage.PREPROCESSING,
                message="Evaluating image resolution, EXIF orientation & sharpness...",
                progress=30,
            )
            t_prep = time.perf_counter()
            preprocessed = self.preprocessor.validate_and_load(local_temp_path)
            prep_ms = (time.perf_counter() - t_prep) * 1000
            m = preprocessed.metrics
            img_dims = (m.width, m.height)
            logger.info(
                f"🖼️  [STEP 2: PREPROCESS] Validated in {prep_ms:.1f}ms: {m.width}x{m.height} ({m.channels}ch) | "
                f"Sharpness: {m.blur_score:.1f} (blurry={m.is_blurry}) | Brightness: {m.mean_brightness:.1f}"
            )

            # 3. Branch C: Scene Validation (detect parcel only when useful)
            self._set_stage(
                db,
                image,
                PipelineStage.SCENE_DETECTION,
                message="Branch C: Zero-shot parcel & scene detection (OWLv2)...",
                progress=50,
            )
            det_result = self._run_branch_c(preprocessed)

            # 4. Branch A: Machine Metadata Extraction (candidate overlay strategies + profile registry)
            self._set_stage(
                db,
                image,
                PipelineStage.METADATA_EXTRACTION,
                message="Branch A: Extracting machine overlay and format profiles (PaddleOCR)...",
                progress=70,
            )
            overlay_meta, overlay_ocr, rois_debug = self._run_branch_a(preprocessed)
            overlay_bboxes = [r["bbox_xyxy"] for r in rois_debug if "bbox_xyxy" in r]

            # 5. Branch B: Physical Label Extraction (crop, rotation, barcode decoding, label OCR)
            self._set_stage(
                db,
                image,
                PipelineStage.LABEL_EXTRACTION,
                message="Branch B: Cropping physical label, deskewing & decoding barcodes (ZXing)...",
                progress=85,
            )
            label_ocr, barcode_results, label_bbox, label_crop_bgr = self._run_branch_b(
                preprocessed,
                det_result,
                full_ocr=self._latest_full_ocr,
                spatial_clusters=self._latest_clusters,
                overlay_bboxes=overlay_bboxes,
            )

            # Check if physical label candidate was detected
            label_candidate_detected = (
                label_bbox is not None
                or bool(getattr(self.label_extractor, "last_candidates", []))
                or any(d.class_name == "shipping_label" for d in det_result.detections)
            )

            # Check if physical label produced readable text or barcodes
            label_readable = False
            if any(b.raw_value for b in barcode_results if b.source_crop and "label_crop" in b.source_crop):
                label_readable = True
            elif label_ocr and len(label_ocr.lines) >= 2:
                # Check for address/logistics keywords or high confidence text
                avg_score = np.mean([l.score for l in label_ocr.lines]) if label_ocr.lines else 0.0
                if avg_score >= 0.50 or any(len(l.text.strip()) >= 6 for l in label_ocr.lines):
                    label_readable = True

            # 6. Synthesize combined OCR result for audit logging
            combined_lines: List[OcrTextBox] = []
            combined_texts: List[str] = []

            if overlay_ocr:
                combined_lines.extend(overlay_ocr.lines)
                if overlay_ocr.full_text:
                    combined_texts.append(overlay_ocr.full_text)
            if label_ocr:
                combined_lines.extend(label_ocr.lines)
                if label_ocr.full_text:
                    combined_texts.append(label_ocr.full_text)

            combined_ocr_result = OcrResult(
                engine=self.ocr_engine.__class__.__name__,
                model_version=getattr(self.ocr_engine, "MODEL_VERSION", "v1"),
                lines=combined_lines,
                full_text="\n".join(combined_texts),
            )

            # 6b. Optional Second-stage Document/Metadata Interpretation (PaddleOCR-VL 1.6)
            vlm_debug = None
            if getattr(self, "vl_adapter", None) and self.vl_adapter.enabled and label_crop_bgr is not None:
                logger.info("🤖 [Stage 2: PaddleOCR-VL Interpreter] Interpreting physical label crop...")
                vlm_res = self.vl_adapter.interpret_region(
                    crop_bgr=label_crop_bgr,
                    region_type="shipping_label",
                    full_ocr_lines=combined_lines,
                    barcodes=barcode_results,
                )
                if vlm_res:
                    vlm_debug = {
                        "accepted_fields": vlm_res.accepted_fields,
                        "rejected_fields": vlm_res.rejected_fields,
                        "tracking_number": vlm_res.tracking_number,
                        "weight_value": vlm_res.weight_value,
                        "dimensions": vlm_res.dimensions,
                        "latency_ms": vlm_res.latency_ms,
                    }

            # Parse scanner bed ROI config if provided
            scanner_bed_roi = None
            if settings.SCANNER_BED_ROI:
                try:
                    parts = [int(p.strip()) for p in settings.SCANNER_BED_ROI.split(",")]
                    if len(parts) == 4:
                        scanner_bed_roi = parts
                except Exception:
                    pass

            # Filter OWLv2 label boxes that might overlap machine metadata overlays
            raw_label_boxes = [d for d in det_result.detections if d.class_name == "shipping_label"]
            clean_label_boxes = []
            for lb in raw_label_boxes:
                is_ov = False
                for obox in overlay_bboxes:
                    # If label box strongly overlaps machine overlay, reject it
                    x1 = max(lb.bbox_xyxy[0], obox[0])
                    y1 = max(lb.bbox_xyxy[1], obox[1])
                    x2 = min(lb.bbox_xyxy[2], obox[2])
                    y2 = min(lb.bbox_xyxy[3], obox[3])
                    if x2 > x1 and y2 > y1:
                        inter = (x2 - x1) * (y2 - y1)
                        b_area = max(1, (lb.bbox_xyxy[2] - lb.bbox_xyxy[0]) * (lb.bbox_xyxy[3] - lb.bbox_xyxy[1]))
                        if (inter / float(b_area)) > 0.35:
                            is_ov = True
                            break
                if not is_ov:
                    clean_label_boxes.append(lb)

            # If label_bbox was found by PhysicalLabelExtractor, add it as a label box if not present
            if label_bbox and not clean_label_boxes:
                clean_label_boxes.append(
                    DetectionBox(
                        class_name="shipping_label",
                        score=0.85,
                        bbox_xyxy=label_bbox,
                    )
                )

            # 7. Evidence Reconciliation across Branch A, Branch B, and Branch C
            self._set_stage(
                db,
                image,
                PipelineStage.RECONCILIATION,
                message="Reconciling multi-branch evidence & computing field confidence...",
                progress=95,
            )
            t_rec = time.perf_counter()
            parsed_data = self.parser.parse(
                ocr_result=combined_ocr_result,
                barcode_results=barcode_results,
                detection_result=det_result,
                overlay_meta=overlay_meta,
                image_dims=img_dims,
                scanner_bed_roi=scanner_bed_roi,
                image_metrics=preprocessed.metrics,
                label_candidate_detected=label_candidate_detected,
                label_readable=label_readable,
                label_boxes=clean_label_boxes,
            )
            rec_ms = (time.perf_counter() - t_rec) * 1000
            parsed_data.evidence["reconciliation_time_ms"] = round(rec_ms, 1)

            logger.info("-" * 80)
            logger.info(f"⚖️  [STEP 3: EVIDENCE RECONCILIATION] Reconciled in {rec_ms:.1f}ms:")
            track_prov = parsed_data.provenance.get("tracking_number")
            wt_prov = parsed_data.provenance.get("weight")
            dim_prov = parsed_data.provenance.get("dimensions")

            track_src = f"Source: {track_prov.source or 'None'}, Conf: {track_prov.ocr_confidence:.4f}" if track_prov else "None"
            wt_src = f"Source: {wt_prov.source or 'None'}" if wt_prov else "None"
            dim_src = f"Source: {dim_prov.source or 'None'}" if dim_prov else "None"

            logger.info(f"   ├─ Tracking Number:   {parsed_data.tracking_number or 'None'} ({track_src})")
            logger.info(f"   ├─ Weight:            {parsed_data.weight_value or 'None'} {parsed_data.weight_unit or ''} ({wt_src}) [Actual: {parsed_data.actual_weight or 'None'}, Volumetric: {parsed_data.volumetric_weight or 'None'}]")
            logger.info(f"   ├─ Dimensions:        {parsed_data.length or 'None'} x {parsed_data.width or 'None'} x {parsed_data.height or 'None'} {parsed_data.dimension_unit or ''} ({dim_src})")
            logger.info(f"   ├─ Parcel Count:      {parsed_data.parcel_count}")
            logger.info(f"   ├─ Extraction Status: {parsed_data.status}")
            logger.info(f"   └─ Scene Flags:       {parsed_data.scene_flags if parsed_data.scene_flags else 'None'}")

            if track_prov and track_prov.conflict_status:
                logger.warning(f"   ⚠️ CONFLICT STATUS: {track_prov.conflict_status}")

            # 8. Build comprehensive debug bundle (Milestone 7)
            if settings.SAVE_DEBUG_BUNDLE:
                debug_bundle = self._build_debug_bundle(
                    image=image,
                    rois_debug=rois_debug,
                    overlay_ocr=overlay_ocr,
                    label_ocr=label_ocr,
                    label_crop_bbox=label_bbox,
                    barcode_results=barcode_results,
                    parsed_data=parsed_data,
                    vlm_debug=vlm_debug,
                    preprocessed=preprocessed,
                    det_result=det_result,
                    spatial_clusters=self._latest_clusters,
                )
                parsed_data.evidence["debug_bundle"] = debug_bundle

            # 9. Save extraction record idempotently
            extraction = self._save_extraction(
                db=db,
                image=image,
                parsed_data=parsed_data,
                det_result=det_result,
                ocr_result=combined_ocr_result,
                barcode_results=barcode_results,
                status=parsed_data.status,
            )

            image.processing_status = "COMPLETED"
            self._set_stage(
                db,
                image,
                PipelineStage.COMPLETED,
                message="Image extraction completed successfully",
                progress=100,
            )

            # Broadcast image completed event
            pipeline_broadcaster.broadcast(
                image.batch_id,
                {
                    "type": "image_completed",
                    "batch_id": image.batch_id,
                    "image_id": image.id,
                    "filename": image.filename,
                    "status": extraction.status,
                    "tracking_number": extraction.tracking_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            total_sec = time.perf_counter() - start_time
            logger.info(f"💾 [STEP 4: DB SAVE] Saved extraction record {extraction.id} (Status: {extraction.status})")
            logger.info(f"✅ [PIPELINE FINISH] Completed {image.filename} in {total_sec:.2f}s | Result: {extraction.status}")
            logger.info("=" * 80)
            return extraction

        except ImageValidationError as e:
            logger.warning(f"⚠️ Image validation error for {image.filename} ({image_id}): {str(e)}")
            extraction = self._record_failure(db, image, reason=str(e))
            return extraction

        except Exception as e:
            logger.error(f"❌ Unexpected pipeline exception on {image.filename} ({image_id}): {str(e)}", exc_info=True)
            extraction = self._record_failure(db, image, reason=str(e))
            return extraction

        finally:
            # 10. Clean up temporary local file in finally block
            if os.path.exists(local_temp_path):
                try:
                    os.remove(local_temp_path)
                    logger.debug(f"🧹 Cleaned up temporary file {local_temp_path}")
                except OSError:
                    pass

    def _save_extraction(
        self,
        db: Session,
        image: Image,
        parsed_data: ExtractedLogisticsData,
        det_result: DetectionResult,
        ocr_result: OcrResult,
        barcode_results: List[BarcodeResult],
        status: str,
    ) -> Extraction:
        """Idempotently save or update extraction row."""
        ext = db.query(Extraction).filter(Extraction.image_id == image.id).first()
        if not ext:
            ext = Extraction(image_id=image.id)
            db.add(ext)

        ext.tracking_number = parsed_data.tracking_number
        ext.weight_value = parsed_data.weight_value
        ext.weight_unit = parsed_data.weight_unit
        ext.length = parsed_data.length
        ext.width = parsed_data.width
        ext.height = parsed_data.height
        ext.dimension_unit = parsed_data.dimension_unit
        ext.parcel_count = parsed_data.parcel_count
        ext.status = status
        ext.parcel_label_outcome = parsed_data.parcel_label_outcome

        ext.detection_json = det_result.model_dump_json() if det_result else None
        ext.raw_ocr_json = ocr_result.model_dump_json() if ocr_result else None
        ext.barcode_json = json.dumps([b.model_dump() for b in barcode_results])
        ext.field_confidence_json = json.dumps(parsed_data.field_confidence)
        ext.evidence_json = json.dumps(parsed_data.evidence)

        db.commit()
        db.refresh(ext)
        return ext

    def _record_failure(self, db: Session, image: Image, reason: str) -> Extraction:
        """Guarantee exactly one extraction record marked PROCESSING_FAILED."""
        logger.error(f"❌ [PIPELINE FAILED] Image: {image.filename} (ID: {image.id}) | Reason: {reason}")
        image.processing_status = "FAILED"
        self._set_stage(
            db,
            image,
            PipelineStage.FAILED,
            message=f"Processing failed: {reason}",
            progress=0,
        )

        pipeline_broadcaster.broadcast(
            image.batch_id,
            {
                "type": "image_failed",
                "batch_id": image.batch_id,
                "image_id": image.id,
                "filename": image.filename,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        ext = db.query(Extraction).filter(Extraction.image_id == image.id).first()
        if not ext:
            ext = Extraction(image_id=image.id)
            db.add(ext)

        ext.tracking_number = None
        ext.weight_value = None
        ext.weight_unit = None
        ext.length = None
        ext.width = None
        ext.height = None
        ext.dimension_unit = None
        ext.parcel_count = None
        ext.status = STATUS_PROCESSING_FAILED
        ext.parcel_label_outcome = "LOW_CONFIDENCE_REVIEW"
        ext.evidence_json = json.dumps({"error": reason})

        db.commit()
        db.refresh(ext)
        return ext

    def reconcile_batch(self, db: Session, batch_id: str):
        """
        Safety net: scan batch for any images missing an extraction row
        and insert a PROCESSING_FAILED row.
        """
        batch = db.query(Batch).filter(Batch.id == batch_id).first()
        if not batch:
            return

        images = db.query(Image).filter(Image.batch_id == batch_id).all()
        for img in images:
            if not img.extraction:
                self._record_failure(db, img, reason="Missing extraction reconciliation")

        # Update summary counts
        batch.processed_images = (
            db.query(Extraction)
            .join(Image)
            .filter(Image.batch_id == batch_id, Extraction.status != STATUS_PROCESSING_FAILED)
            .count()
        )
        batch.failed_images = (
            db.query(Extraction)
            .join(Image)
            .filter(Image.batch_id == batch_id, Extraction.status == STATUS_PROCESSING_FAILED)
            .count()
        )
        db.commit()

        # Check if all images in batch have reached terminal state
        remaining_unprocessed = (
            db.query(Image)
            .filter(
                Image.batch_id == batch_id,
                Image.processing_status.in_(["PENDING", "QUEUED", "PROCESSING"]),
            )
            .count()
        )
        if remaining_unprocessed == 0:
            logger.info(f"🎉 [BATCH COMPLETE] All {batch.total_images} image(s) processed for batch {batch_id}")
            pipeline_broadcaster.broadcast(
                batch_id,
                {
                    "type": "batch_completed",
                    "batch_id": batch_id,
                    "total_images": batch.total_images,
                    "processed_images": batch.processed_images,
                    "failed_images": batch.failed_images,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )


orchestrator = PipelineOrchestrator()
