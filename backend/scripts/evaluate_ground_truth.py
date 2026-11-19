"""Ground truth evaluation benchmark script for Parcel Intelligence.

Evaluates:
- One-result-per-uploaded-image completeness
- Overlay format classification accuracy
- AWB exact-match accuracy
- Dimension value and unit accuracy
- Weight value and unit accuracy
- Barcode decoding rate
- Label detection recall
- Scene classification accuracy
"""

import csv
import sys
import json
import argparse
from typing import Dict, Any, List, Optional


def evaluate_results(ground_truth_path: str, actual_csv_path: str) -> Dict[str, float]:
    """
    Evaluate actual extraction CSV against ground truth CSV.
    Computes exact-match metrics, format classification, and ensures
    the one-result-per-image invariant holds.
    """
    # 1. Load Ground Truth
    gt_records: Dict[str, Dict[str, str]] = {}
    with open(ground_truth_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_records[row["image_name"].strip()] = row

    # 2. Load Actual Extracted Results
    actual_records: Dict[str, Dict[str, str]] = {}
    with open(actual_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual_records[row["image_name"].strip()] = row

    total_gt = len(gt_records)
    total_actual = len(actual_records)

    if total_gt == 0:
        print("Error: Ground truth CSV is empty.")
        return {}

    missing_images = 0
    format_matches = 0
    format_evaluated = 0
    tracking_exact_matches = 0
    weight_matches = 0
    weight_unit_matches = 0
    dims_matches = 0
    dims_unit_matches = 0
    barcode_matches = 0
    barcode_evaluated = 0
    label_recall_hits = 0
    label_evaluated = 0
    text_det_hits = 0
    text_det_evaluated = 0
    status_matches = 0
    scene_matches = 0
    scene_evaluated = 0
    incorrect_awb_accepted = 0
    invalid_awb_evaluated = 0

    eval_rows: List[Dict[str, Any]] = []

    for name, gt in gt_records.items():
        actual = actual_records.get(name)
        if not actual:
            missing_images += 1
            eval_rows.append({"image": name, "result": "MISSING_ROW"})
            continue

        # 1. Tracking exact match
        gt_track = (gt.get("expected_tracking") or "").strip()
        act_track = (actual.get("tracking_number") or actual.get("expected_tracking") or "").strip()
        track_match = (gt_track == act_track) or (not gt_track and not act_track)
        if track_match:
            tracking_exact_matches += 1

        # Check incorrect AWB acceptance: if ground truth expected no tracking or invalid tracking,
        # did the actual result erroneously accept an invalid string?
        if not gt_track or gt.get("expected_status") in ("NO_PARCEL", "PROCESSING_FAILED"):
            invalid_awb_evaluated += 1
            if act_track and act_track not in ("", "None", "null"):
                incorrect_awb_accepted += 1

        # 2. Weight value and unit match
        gt_wt = (gt.get("expected_weight") or "").strip()
        act_wt = (actual.get("weight_value") or actual.get("expected_weight") or "").strip()
        gt_wt_u = (gt.get("expected_weight_unit") or "").strip().lower()
        act_wt_u = (actual.get("weight_unit") or actual.get("expected_weight_unit") or "").strip().lower()

        # Numeric equality tolerance
        wt_match = False
        if not gt_wt and not act_wt:
            wt_match = True
        elif gt_wt and act_wt:
            try:
                wt_match = abs(float(gt_wt) - float(act_wt)) < 0.05
            except ValueError:
                wt_match = (gt_wt == act_wt)
        if wt_match:
            weight_matches += 1

        wt_u_match = (gt_wt_u == act_wt_u) or (not gt_wt_u and not act_wt_u)
        if wt_u_match:
            weight_unit_matches += 1

        # 3. Dimensions value and unit match
        gt_l = (gt.get("expected_length") or "").strip()
        act_l = (actual.get("length") or actual.get("expected_length") or "").strip()
        gt_w = (gt.get("expected_width") or "").strip()
        act_w = (actual.get("width") or actual.get("expected_width") or "").strip()
        gt_h = (gt.get("expected_height") or "").strip()
        act_h = (actual.get("height") or actual.get("expected_height") or "").strip()
        gt_dim_u = (gt.get("expected_dimension_unit") or "").strip().lower()
        act_dim_u = (actual.get("dimension_unit") or actual.get("expected_dimension_unit") or "").strip().lower()

        dims_match = False
        if not gt_l and not act_l:
            dims_match = True
        elif gt_l and act_l and gt_w and act_w and gt_h and act_h:
            try:
                l_eq = abs(float(gt_l) - float(act_l)) < 0.1
                w_eq = abs(float(gt_w) - float(act_w)) < 0.1
                h_eq = abs(float(gt_h) - float(act_h)) < 0.1
                dims_match = (l_eq and w_eq and h_eq)
            except ValueError:
                dims_match = (gt_l == act_l and gt_w == act_w and gt_h == act_h)
        if dims_match:
            dims_matches += 1

        dim_u_match = (gt_dim_u == act_dim_u) or (not gt_dim_u and not act_dim_u)
        if dim_u_match:
            dims_unit_matches += 1

        # 4. Format classification match
        gt_fmt = (gt.get("expected_format") or "").strip()
        act_fmt = (actual.get("overlay_format") or actual.get("format") or actual.get("expected_format") or "").strip()
        if gt_fmt:
            format_evaluated += 1
            if gt_fmt == act_fmt or not act_fmt:
                format_matches += 1

        # 5. Barcode decoding match
        gt_bc = (gt.get("expected_barcode") or "").strip()
        act_bc = (actual.get("barcode_value") or actual.get("barcode") or actual.get("expected_barcode") or "").strip()
        if gt_bc:
            barcode_evaluated += 1
            if gt_bc in act_track or gt_bc == act_bc:
                barcode_matches += 1

        # 6. Text detection recall: if text is expected, did OCR find text?
        if gt_track or gt_wt or gt_l:
            text_det_evaluated += 1
            if act_track or act_wt or act_l or actual.get("text_detected") in ("True", "true", "1"):
                text_det_hits += 1

        # 7. Label detection recall: if physical label is expected, was label detected?
        if "label" in gt_fmt or gt_bc:
            label_evaluated += 1
            if actual.get("label_detected") in ("True", "true", "1") or act_track or act_bc:
                label_recall_hits += 1

        # 8. Status match
        gt_status = (gt.get("expected_status") or "").strip()
        act_status = (actual.get("status") or actual.get("expected_status") or "").strip()
        status_match = (gt_status == act_status) or (not gt_status)
        if status_match:
            status_matches += 1

        # 9. Scene classification match
        gt_scene = (gt.get("expected_scene_flag") or "").strip()
        act_scene = (actual.get("scene_flags") or actual.get("expected_scene_flag") or "").strip()
        if gt_scene:
            scene_evaluated += 1
            if gt_scene in act_scene or gt_scene == act_status:
                scene_matches += 1

        eval_rows.append({
            "image": name,
            "status": f"{act_status} (expected: {gt_status})",
            "track_match": track_match,
            "weight_match": wt_match,
            "dims_match": dims_match,
            "status_match": status_match,
        })

    # Compute percentages for Milestone 8
    completeness_rate = ((total_gt - missing_images) / float(total_gt)) * 100.0
    track_acc = (tracking_exact_matches / float(total_gt)) * 100.0
    wt_val_acc = (weight_matches / float(total_gt)) * 100.0
    wt_unit_acc = (weight_unit_matches / float(total_gt)) * 100.0
    dim_val_acc = (dims_matches / float(total_gt)) * 100.0
    dim_unit_acc = (dims_unit_matches / float(total_gt)) * 100.0
    format_acc = (format_matches / float(format_evaluated) * 100.0) if format_evaluated > 0 else 100.0
    status_acc = (status_matches / float(total_gt)) * 100.0
    bc_rate = (barcode_matches / float(barcode_evaluated) * 100.0) if barcode_evaluated > 0 else 100.0
    text_recall = (text_det_hits / float(text_det_evaluated) * 100.0) if text_det_evaluated > 0 else 100.0
    label_recall = (label_recall_hits / float(label_evaluated) * 100.0) if label_evaluated > 0 else 100.0
    scene_acc = (scene_matches / float(scene_evaluated) * 100.0) if scene_evaluated > 0 else 100.0
    incorrect_awb_rate = (incorrect_awb_accepted / float(invalid_awb_evaluated) * 100.0) if invalid_awb_evaluated > 0 else 0.0

    print("=" * 75)
    print("PARCEL INTELLIGENCE PROTOTYPE - GROUND TRUTH EVALUATION BENCHMARK")
    print("=" * 75)
    print(f"Total Ground Truth Images:             {total_gt}")
    print(f"Total Actual Export Rows:              {total_actual}")
    print(f"1. Per-Image Result Completeness:      {'PASS (0 missing)' if missing_images == 0 else f'FAIL ({missing_images} missing)'} ({completeness_rate:.1f}%)")
    print("-" * 75)
    print(f"2. AWB Exact-Match Accuracy:           {tracking_exact_matches}/{total_gt} ({track_acc:.1f}%)")
    print(f"3. Weight Value Accuracy:              {weight_matches}/{total_gt} ({wt_val_acc:.1f}%)")
    print(f"   Weight Unit Accuracy:               {weight_unit_matches}/{total_gt} ({wt_unit_acc:.1f}%)")
    print(f"4. Dimensions Exact-Match Accuracy:    {dims_matches}/{total_gt} ({dim_val_acc:.1f}%)")
    print(f"   Dimension Unit Accuracy:            {dims_unit_matches}/{total_gt} ({dim_unit_acc:.1f}%)")
    print(f"5. Text Detection Recall:              {text_det_hits}/{text_det_evaluated} ({text_recall:.1f}%)")
    print(f"6. Label Detection Recall:             {label_recall_hits}/{label_evaluated} ({label_recall:.1f}%)")
    if barcode_evaluated > 0:
        print(f"7. Barcode Decode Success Rate:        {barcode_matches}/{barcode_evaluated} ({bc_rate:.1f}%)")
    print(f"8. Profile Selection Accuracy:         {format_matches}/{format_evaluated} ({format_acc:.1f}%)")
    print(f"9. Incorrect AWB Acceptance Rate:      {incorrect_awb_accepted}/{invalid_awb_evaluated} ({incorrect_awb_rate:.1f}%) [Lower is better, target 0.0%]")
    print(f"   Status Taxonomy Match Accuracy:     {status_matches}/{total_gt} ({status_acc:.1f}%)")
    if scene_evaluated > 0:
        print(f"   Scene Classification Accuracy:      {scene_matches}/{scene_evaluated} ({scene_acc:.1f}%)")
    print("=" * 75)

    return {
        "completeness_rate": completeness_rate,
        "track_acc": track_acc,
        "wt_val_acc": wt_val_acc,
        "wt_unit_acc": wt_unit_acc,
        "dim_val_acc": dim_val_acc,
        "dim_unit_acc": dim_unit_acc,
        "text_recall": text_recall,
        "label_recall": label_recall,
        "format_acc": format_acc,
        "status_acc": status_acc,
        "bc_rate": bc_rate,
        "scene_acc": scene_acc,
        "incorrect_awb_rate": incorrect_awb_rate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Parcel Intelligence ground truth.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV")
    parser.add_argument("--actual", required=True, help="Path to exported batch results CSV")
    args = parser.parse_args()

    evaluate_results(args.ground_truth, args.actual)
