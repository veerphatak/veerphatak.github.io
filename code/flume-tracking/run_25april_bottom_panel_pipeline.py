"""
Pipeline runner: run_25april_bottom_panel_pipeline.py

Purpose
-------
Runs the complete bottom-view flume sphere workflow, including tracking, calibration, overlays and physics reporting.

Inputs
------
Bottom-view flume videos, case start information and flow-speed/FPS settings.

Outputs
-------
Bottom-view trajectories, overlays and physics-report tables/figures.

Methodological notes
--------------------
Bottom-view coordinate calibration is generated consistently so lateral and streamwise trends can be compared between trials.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

from __future__ import annotations
import csv
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parent
for extra in [WORKSPACE / '.vendor_video_libs', WORKSPACE / '.tmp_video_libs']:
    if extra.exists():
        sys.path.append(str(extra))
import cv2
import numpy as np
TRACKER_PATH = WORKSPACE / 'Kalman_Tracker_Spheres_binarized.py'
TRIM_TRACK_PATH = WORKSPACE / 'trim_and_track_pxl_spheres.py'
OVERLAY_PATH = WORKSPACE / 'trajectory_validation_overlay_full_video_editable_fixed_v3.py'
OVERLAY_SUMMARY_PATH = WORKSPACE / 'trajectory_validation_master_summary_editable_fixed_v2.py'
PHYSICS_PATH = WORKSPACE / 'trajectory_physics_bottom_panel_report.py'
RAW_ROOT = Path('C:\\Users\\User\\Downloads\\25 April viewing panel runs')
OUTPUT_ROOT = Path(os.environ.get('BOTTOM_PANEL_OUTPUT_ROOT', 'C:\\IP Work\\10mm sphere bottom view runs'))
CALIBRATION_VIDEO = RAW_ROOT / 'Calibration 1 + 10mm sphere 1.mp4'
CALIBRATION_FRAME_CANDIDATES = [60, 120, 180, 221, 260, 320]
GRID_SPACING_MM = 10.0
OVERLAY_FRAME_STEP = 5
OVERLAY_TARGET_POSITIONS = 8
OVERLAY_BACKGROUND_MARGIN_FRAMES = 8
VIDEO_PATHS = [RAW_ROOT / 'Calibration 1 + 10mm sphere 1.mp4', RAW_ROOT / 'Calibration 2 + 10mm sphere 2.mp4', RAW_ROOT / '10mm sphere 3.mp4', RAW_ROOT / '10mm sphere 4.mp4', RAW_ROOT / '10mm sphere 5.mp4', RAW_ROOT / '10mm sphere 6.mp4', RAW_ROOT / '10mm sphere 7.mp4']
SCAN_START_HINTS = {'calibration 1 + 10mm sphere 1.mp4': 320, 'calibration 2 + 10mm sphere 2.mp4': 420}

def load_module(script_path: Path, module_name: str):
    """Load the module used by this script."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def infer_sphere_diameter_mm(name: str) -> float:
    """Handle the infer sphere diameter mm step used by this script."""
    lower = name.lower()
    if '10mm' in lower:
        return 10.0
    if '14mm' in lower:
        return 14.0
    raise ValueError(f'Could not infer sphere diameter from: {name}')

def normalize_case_name(video_name: str) -> str:
    """Normalise the case name used by this script."""
    stem = Path(video_name).stem.strip().lower()
    return stem.replace(' ', '_').replace('+', 'plus')

def initial_scan_start_frame(video_path: Path) -> int:
    """Handle the initial scan start frame step used by this script."""
    return int(SCAN_START_HINTS.get(video_path.name.lower(), 0))

def should_retry_scan(video_path: Path, scan: dict) -> bool:
    """Handle the should retry scan step used by this script."""
    if 'calibration' not in video_path.name.lower():
        return False
    expected_diameter_mm = infer_sphere_diameter_mm(video_path.name)
    estimated_diameter_mm = float(scan['fine_track'].get('diameter_mm_estimate', 0.0) or 0.0)
    first_found_frame = int(scan['fine_track'].get('first_found_frame', 0))
    last_found_frame = int(scan['fine_track'].get('last_found_frame', first_found_frame))
    requested_start = int(scan.get('scan_start_frame_requested', 0))
    obviously_too_large = estimated_diameter_mm > expected_diameter_mm * 1.8
    starts_suspiciously_early = first_found_frame <= max(requested_start + 30, 120)
    covers_only_the_calibration_section = last_found_frame <= max(requested_start + 360, 360)
    return obviously_too_large or (starts_suspiciously_early and covers_only_the_calibration_section)

def scan_video_with_retry(trim_mod, video_path: Path, base_cal: dict, tracker_module) -> dict:
    """Scan the video with retry used by this script."""
    scan_start = initial_scan_start_frame(video_path)
    scan = trim_mod.scan_video(video_path, base_cal, tracker_module, scan_start_frame=scan_start)
    if should_retry_scan(video_path, scan):
        retry_start = max(scan_start + 120, int(scan['fine_track']['last_found_frame']) + 30)
        scan = trim_mod.scan_video(video_path, base_cal, tracker_module, scan_start_frame=retry_start)
    return scan

def find_peaks(arr: np.ndarray, min_dist: int, quantile: float) -> np.ndarray:
    """Handle the find peaks step used by this script."""
    threshold = float(np.quantile(arr, quantile))
    peaks: list[int] = []
    for i in range(1, len(arr) - 1):
        if arr[i] >= threshold and arr[i] >= arr[i - 1] and (arr[i] >= arr[i + 1]):
            if peaks and i - peaks[-1] < min_dist:
                if arr[i] > arr[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.asarray(peaks, dtype=int)

def estimate_bottomview_grid(frame_bgr: np.ndarray) -> dict:
    """Handle the estimate bottomview grid step used by this script."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crop_y0 = 0
    crop_y1 = min(gray.shape[0], 900)
    crop_x0 = 0
    crop_x1 = min(gray.shape[1], 1100)
    crop = gray[crop_y0:crop_y1, crop_x0:crop_x1]
    vert = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 151)))
    horz = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (151, 21)))
    vert_proj = vert.mean(axis=0).astype(np.float32)
    horz_proj = horz.mean(axis=1).astype(np.float32)
    vertical_peaks = find_peaks(vert_proj, min_dist=40, quantile=0.9) + crop_x0
    horizontal_peaks = find_peaks(horz_proj, min_dist=40, quantile=0.9) + crop_y0
    if len(vertical_peaks) < 4 or len(horizontal_peaks) < 4:
        raise RuntimeError('Could not estimate bottom-panel calibration grid spacing.')
    vertical_spacing_px = float(np.median(np.diff(vertical_peaks)))
    horizontal_spacing_px = float(np.median(np.diff(horizontal_peaks)))
    px_per_mm = float(np.mean([vertical_spacing_px, horizontal_spacing_px]) / GRID_SPACING_MM)
    return {'vertical_peaks_px': vertical_peaks.tolist(), 'horizontal_peaks_px': horizontal_peaks.tolist(), 'vertical_spacing_px': vertical_spacing_px, 'horizontal_spacing_px': horizontal_spacing_px, 'estimated_px_per_mm': px_per_mm}

def estimate_score(est: dict) -> float:
    """Handle the estimate score step used by this script."""
    v = float(est['vertical_spacing_px'])
    h = float(est['horizontal_spacing_px'])
    ratio_penalty = abs(np.log(max(v, 1e-06) / max(h, 1e-06)))
    return len(est['vertical_peaks_px']) + len(est['horizontal_peaks_px']) - 12.0 * ratio_penalty

def create_identity_calibration(width: int, height: int, px_per_mm: float, notes: str, frame_index: int) -> dict:
    """Create the identity calibration used by this script."""
    src = [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]]
    H = np.eye(3, dtype=np.float64)
    return {'calibration_source_type': 'video_frame_bottomview_identity', 'calibration_video_path': str(CALIBRATION_VIDEO), 'calibration_frame_index': int(frame_index), 'calibration_image_width_px': int(width), 'calibration_image_height_px': int(height), 'rect_width_mm': float(width / px_per_mm), 'rect_height_mm': float(height / px_per_mm), 'rectified_px_per_mm': float(px_per_mm), 'rectified_width_px': int(width), 'rectified_height_px': int(height), 'src_points_px': src, 'raw_click_points_px': src, 'src_points_normalized': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 'dst_points_px': src, 'homography': H.tolist(), 'homography_inverse': H.tolist(), 'grid_spacing_mm': float(GRID_SPACING_MM), 'point_order': ['TL', 'TR', 'BR', 'BL'], 'click_snap_to_grid_intersection': False, 'notes': notes}

def build_bottomview_calibration(out_dir: Path) -> Path:
    """Build the bottomview calibration used by this script."""
    cap = cv2.VideoCapture(str(CALIBRATION_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open calibration video: {CALIBRATION_VIDEO}')
    candidates = []
    try:
        for frame_idx in CALIBRATION_FRAME_CANDIDATES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                estimate = estimate_bottomview_grid(frame)
            except Exception:
                continue
            candidates.append((estimate_score(estimate), frame_idx, frame.copy(), estimate))
    finally:
        cap.release()
    if not candidates:
        raise RuntimeError('Could not estimate bottom-panel calibration from the calibration clip.')
    _, chosen_frame_idx, frame, estimate = max(candidates, key=lambda item: item[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / 'calibration_frame.png'), frame)
    vis = frame.copy()
    for x in estimate['vertical_peaks_px']:
        cv2.line(vis, (int(x), 0), (int(x), vis.shape[0] - 1), (0, 255, 255), 2)
    for y in estimate['horizontal_peaks_px']:
        cv2.line(vis, (0, int(y)), (vis.shape[1] - 1, int(y)), (255, 0, 255), 2)
    cv2.imwrite(str(out_dir / 'calibration_grid_peaks.png'), vis)
    (out_dir / 'calibration_grid_estimate.json').write_text(json.dumps(estimate, indent=2), encoding='utf-8')
    (out_dir / 'calibration_grid_candidates.json').write_text(json.dumps([{'score': float(score), 'frame_index': int(frame_idx), **est} for score, frame_idx, _frame, est in candidates], indent=2), encoding='utf-8')
    calibration = create_identity_calibration(width=frame.shape[1], height=frame.shape[0], px_per_mm=estimate['estimated_px_per_mm'], frame_index=chosen_frame_idx, notes='Bottom-panel identity calibration for the 25 April viewing-panel runs using the measured grid spacing from the calibration clip. The underside view is treated as an in-plane view with raw image coordinates preserved and scaled by the measured average grid spacing.')
    out_path = out_dir / 'bottom_panel_calibration.json'
    out_path.write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    return out_path

def tune_tracker_for_bottomview(module) -> None:
    """Handle the tune tracker for bottomview step used by this script."""
    module.DEFAULT_SPHERE_BACKTRACK_MIN_PROGRESS_MM = 1000000000.0
    module.DEFAULT_SPHERE_TAIL_MISS_STOP = 40
    red_acquire = module.PARAMS['sphere_red']['acquire']
    red_track = module.PARAMS['sphere_red']['track']
    white_acquire = module.PARAMS['sphere_white']['acquire']
    white_track = module.PARAMS['sphere_white']['track']
    red_acquire.update({'roi_half_size_px': 280, 'binary_blur_k': 7, 'binary_bg_blur_k': 45, 'binary_min_threshold': 6, 'red_excess_blur_k': 5, 'open_k': 3, 'close_k': 9, 'close_iter': 2, 'min_area_px': 120, 'max_area_px': 30000, 'min_circularity': 0.3, 'min_solidity': 0.35, 'aspect_tol': 1.8, 'min_diameter_factor': 0.4, 'max_diameter_factor': 2.5, 'max_assoc_dist_px': 180.0, 'max_misses': 36, 'border_relax_px': 36, 'relaxed_min_area_px': 60, 'relaxed_min_circularity': 0.16, 'relaxed_min_solidity': 0.18, 'relaxed_aspect_tol': 2.4, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.8, 'roi_expand_per_miss_px': 110, 'assoc_expand_per_miss_px': 150, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 70, 'exit_misses': 14})
    red_track.update({'roi_half_size_px': 220, 'binary_blur_k': 7, 'binary_bg_blur_k': 41, 'binary_min_threshold': 8, 'red_excess_blur_k': 5, 'open_k': 3, 'close_k': 9, 'close_iter': 2, 'min_area_px': 140, 'max_area_px': 26000, 'min_circularity': 0.34, 'min_solidity': 0.38, 'aspect_tol': 1.65, 'min_diameter_factor': 0.45, 'max_diameter_factor': 2.4, 'max_assoc_dist_px': 170.0, 'max_misses': 34, 'border_relax_px': 36, 'relaxed_min_area_px': 70, 'relaxed_min_circularity': 0.16, 'relaxed_min_solidity': 0.18, 'relaxed_aspect_tol': 2.4, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.8, 'roi_expand_per_miss_px': 130, 'assoc_expand_per_miss_px': 170, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 75, 'exit_misses': 14})
    white_acquire.update({'roi_half_size_px': 280, 'binary_blur_k': 7, 'binary_bg_blur_k': 45, 'binary_min_threshold': 6, 'gray_min_threshold': 6, 'use_gray_branch': True, 'open_k': 3, 'close_k': 9, 'close_iter': 1, 'min_component_area_px': 120, 'min_area_px': 140, 'max_area_px': 30000, 'min_circularity': 0.38, 'min_solidity': 0.42, 'aspect_tol': 1.6, 'min_diameter_factor': 0.4, 'max_diameter_factor': 2.5, 'max_assoc_dist_px': 180.0, 'max_misses': 36, 'border_relax_px': 36, 'relaxed_min_area_px': 70, 'relaxed_min_circularity': 0.18, 'relaxed_min_solidity': 0.22, 'relaxed_aspect_tol': 2.0, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.8, 'roi_expand_per_miss_px': 110, 'assoc_expand_per_miss_px': 150, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 70, 'exit_misses': 14})
    white_track.update({'roi_half_size_px': 220, 'binary_blur_k': 7, 'binary_bg_blur_k': 41, 'binary_min_threshold': 8, 'gray_min_threshold': 8, 'use_gray_branch': True, 'open_k': 3, 'close_k': 9, 'close_iter': 1, 'min_component_area_px': 140, 'min_area_px': 160, 'max_area_px': 26000, 'min_circularity': 0.44, 'min_solidity': 0.48, 'aspect_tol': 1.5, 'min_diameter_factor': 0.45, 'max_diameter_factor': 2.4, 'max_assoc_dist_px': 170.0, 'max_misses': 34, 'border_relax_px': 36, 'relaxed_min_area_px': 80, 'relaxed_min_circularity': 0.2, 'relaxed_min_solidity': 0.24, 'relaxed_aspect_tol': 1.9, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.8, 'roi_expand_per_miss_px': 130, 'assoc_expand_per_miss_px': 170, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 75, 'exit_misses': 14})

def detect_video_fps(video_path: Path) -> float:
    """Detect the video fps used by this script."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video to read fps: {video_path}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0.0 else 30.0
    finally:
        cap.release()

def summarize_tracker_output(out_dir: Path) -> dict:
    """Handle the summarize tracker output step used by this script."""
    raw_csv = out_dir / 'tracks_raw.csv'
    proc_csv = out_dir / 'tracks_processed.csv'
    if not raw_csv.exists() or not proc_csv.exists():
        return {}
    with raw_csv.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    found_values = [int(float(row.get('found_this_frame', 0) or 0)) for row in rows]
    found = int(sum(found_values))
    total = int(len(rows))
    x_vals: list[float] = []
    y_vals: list[float] = []
    disp_vals: list[float] = []
    with proc_csv.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            try:
                x_vals.append(float(row['x_rel_mm']))
                y_vals.append(float(row['y_rel_mm']))
                disp_vals.append(float(row['disp_mm']))
            except Exception:
                continue
    summary = {'rows': total, 'found': found, 'found_ratio': found / total if total else None}
    if x_vals:
        summary['x_rel_mm_range'] = [float(min(x_vals)), float(max(x_vals))]
        summary['y_rel_mm_range'] = [float(min(y_vals)), float(max(y_vals))]
        summary['disp_mm_max'] = float(max(disp_vals))
        summary['lateral_span_mm'] = float(max(x_vals) - min(x_vals))
    return summary

def _tracker_attempts(seed_kind: str | None, expected_diameter_mm: float) -> list[tuple[str, float]]:
    """Handle the  tracker attempts step used by this script."""
    seed_kind = (seed_kind or '').strip().lower()
    if seed_kind == 'red':
        return [('red', expected_diameter_mm), ('white', expected_diameter_mm), ('red', 14.0)]
    return [('white', expected_diameter_mm), ('red', expected_diameter_mm), ('white', 14.0)]

def clear_output_dir(out_dir: Path) -> None:
    """Handle the clear output dir step used by this script."""
    if out_dir.exists():
        for path in sorted(out_dir.rglob('*'), key=lambda p: (len(p.parts), str(p)), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)

def run_tracker_case(module, video_path: Path, calibration_json: Path, out_dir: Path, expected_diameter_mm: float, start_frame: int, click_xy: list[float], seed_kind: str | None) -> dict:
    """Run the tracker case used by this script."""
    tune_tracker_for_bottomview(module)
    module.cv2.destroyAllWindows = lambda: None
    real_fps = detect_video_fps(video_path)
    last_error = None
    for sphere_color_mode, sphere_diameter_mm in _tracker_attempts(seed_kind, expected_diameter_mm):
        try:
            module.USE_RUNTIME_PROMPTS = False
            module.ENABLE_IMSHOW = False
            module.DEFAULT_VIDEO_PATH = str(video_path)
            module.DEFAULT_CALIBRATION_JSON = str(calibration_json)
            module.DEFAULT_OUT_DIR = str(out_dir)
            module.DEFAULT_REAL_FPS = float(real_fps)
            module.DEFAULT_FRAME_STRIDE = 1
            module.DEFAULT_START_FRAME = int(start_frame)
            module.DEFAULT_OBJECT_MODE = 'sphere'
            module.DEFAULT_SPHERE_COLOR_MODE = sphere_color_mode
            module.DEFAULT_SPHERE_DIAMETER_MM = float(sphere_diameter_mm)
            module.DEFAULT_USE_PRESET_CLICK = True
            module.DEFAULT_PRESET_CLICK = (int(round(click_xy[0])), int(round(click_xy[1])))
            module.DEFAULT_START_SEARCH_FRAMES = 25
            module.DEFAULT_SAVE_DEBUG_IMAGES = True
            module.DEFAULT_NUM_DEBUG_IMAGES = 12
            module.DEFAULT_ENABLE_SPHERE_BACKGROUND_SUBTRACTION = True
            module.DEFAULT_SPHERE_BACKGROUND_SECONDS = 2.0
            module.DEFAULT_SPHERE_BACKGROUND_SAMPLES = 20
            module.DEFAULT_SPHERE_BACKGROUND_MARGIN_FRAMES = 6
            module.DEFAULT_SPHERE_BACKGROUND_KEEP_FRACTION = 0.75
            clear_output_dir(out_dir)
            module.main()
            return {'status': 'ok', 'real_fps': float(real_fps), 'sphere_color_mode': sphere_color_mode, 'sphere_diameter_mm': float(sphere_diameter_mm), 'summary': summarize_tracker_output(out_dir)}
        except Exception as exc:
            last_error = {'status': 'error', 'real_fps': float(real_fps), 'sphere_color_mode': sphere_color_mode, 'sphere_diameter_mm': float(sphere_diameter_mm), 'error': str(exc), 'traceback': traceback.format_exc()}
    assert last_error is not None
    return last_error

def make_case_summary(video_path: Path, scan: dict, tracker_result: dict, tracker_dir: Path) -> dict:
    """Handle the make case summary step used by this script."""
    fine = scan['fine_track']
    return {'video_name': video_path.name, 'video_path': str(video_path), 'status': tracker_result['status'], 'kind': scan.get('kind'), 'frame_count_total': int(scan.get('frame_count', 0)), 'fps': float(scan.get('fps', tracker_result.get('real_fps', 0.0))), 'start_frame': int(fine['first_found_frame']), 'seed_last_found_frame': int(fine['last_found_frame']), 'click_xy': fine['click_xy'], 'diameter_mm_estimate': float(fine['diameter_mm_estimate']), 'tracker_output_dir': str(tracker_dir), 'scan': {'coarse_track': scan['coarse_track'], 'fine_track': scan['fine_track']}, 'tracker_result': tracker_result}

def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Handle the read csv rows step used by this script."""
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return (list(reader.fieldnames or []), rows)

def parse_int(value: str | None) -> int | None:
    """Handle the parse int step used by this script."""
    if value is None or value == '':
        return None
    try:
        return int(float(value))
    except Exception:
        return None

def choose_key_col(fieldnames: list[str]) -> str:
    """Handle the choose key col step used by this script."""
    for cand in ['frame_idx', 'frame', 'frame_index']:
        if cand in fieldnames:
            return cand
    raise ValueError(f'Could not find frame key column in {fieldnames}')

def build_found_segments(fieldnames: list[str], rows: list[dict[str, str]]) -> tuple[str, list[list[dict[str, str]]]]:
    """Build the found segments used by this script."""
    key_col = choose_key_col(fieldnames)
    rows_sorted = sorted(rows, key=lambda row: parse_int(row.get(key_col)) if parse_int(row.get(key_col)) is not None else 10 ** 12)
    segments: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    prev_key: int | None = None
    has_found_col = 'found_this_frame' in fieldnames
    for row in rows_sorted:
        key = parse_int(row.get(key_col))
        if key is None:
            continue
        found = 1
        if has_found_col:
            found = parse_int(row.get('found_this_frame'))
            found = 0 if found is None else found
        if found != 1:
            if current:
                segments.append(current)
                current = []
            prev_key = None
            continue
        if prev_key is not None and key - prev_key > 1:
            if current:
                segments.append(current)
            current = []
        current.append(row)
        prev_key = key
    if current:
        segments.append(current)
    return (key_col, [seg for seg in segments if seg])

def write_csv_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the csv rows used by this script."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def prepare_primary_segment_csv(tracker_dir: Path, out_dir: Path) -> tuple[Path, dict]:
    """Handle the prepare primary segment csv step used by this script."""
    csv_path = tracker_dir / 'tracks_processed.csv'
    fieldnames, rows = read_csv_rows(csv_path)
    key_col, segments = build_found_segments(fieldnames, rows)
    if segments:
        primary = max(segments, key=len)
        selected_rows = primary
        primary_start = parse_int(primary[0].get(key_col))
        primary_end = parse_int(primary[-1].get(key_col))
    else:
        selected_rows = rows
        primary_start = None
        primary_end = None
    overlay_fieldnames = [name for name in fieldnames if name != 'out_idx']
    overlay_rows = [{k: v for k, v in row.items() if k != 'out_idx'} for row in selected_rows]
    filtered_csv = out_dir / 'tracks_processed_primary_found_segment.csv'
    write_csv_rows(filtered_csv, overlay_fieldnames, overlay_rows)
    summary = {'source_tracks_processed_csv': str(csv_path), 'filtered_tracks_processed_csv': str(filtered_csv), 'frame_key_used': key_col, 'total_rows': int(len(rows)), 'found_segments': int(len(segments)), 'primary_segment_length': int(len(selected_rows)), 'primary_segment_start': primary_start, 'primary_segment_end': primary_end}
    return (filtered_csv, summary)

def build_no_particle_background_frames(frame_count: int, segment_summary: dict, margin_frames: int=OVERLAY_BACKGROUND_MARGIN_FRAMES) -> list[int]:
    """Build the no particle background frames used by this script."""
    if frame_count <= 0:
        return []
    start = segment_summary.get('primary_segment_start')
    end = segment_summary.get('primary_segment_end')
    if start is None or end is None:
        return list(range(frame_count))
    start = int(start)
    end = int(end)
    pre_end = max(0, start - int(margin_frames))
    post_start = min(frame_count, end + int(margin_frames) + 1)
    background_frames = list(range(0, pre_end)) + list(range(post_start, frame_count))
    if background_frames:
        return background_frames
    return [idx for idx in range(frame_count) if idx < start or idx > end]

def configure_overlay_module(overlay_mod) -> None:
    """Handle the configure overlay module step used by this script."""
    overlay_mod.USE_RAW = False
    overlay_mod.FRAME_STEP = OVERLAY_FRAME_STEP
    overlay_mod.MIN_VISIBLE_POSITIONS = 5
    overlay_mod.MAX_VISIBLE_POSITIONS = 10
    overlay_mod.TARGET_VISIBLE_POSITIONS = OVERLAY_TARGET_POSITIONS
    overlay_mod.PREFERRED_SOURCE_VIDEO = 'annotated_binary.mp4'
    overlay_mod.BACKGROUND_FROM_GRAYSCALE = False
    overlay_mod.MOTION_THRESHOLD = 10
    overlay_mod.TRAJECTORY_LINE_COLOUR_BGR = (0, 140, 255)
    overlay_mod.TRAJECTORY_POINT_COLOUR_BGR = (255, 0, 255)

def build_overlay_batch(overlay_module, summary_module, tracker_root: Path, overlay_root: Path, case_lookup: dict[str, dict]) -> dict:
    """Build the overlay batch used by this script."""
    overlay_root.mkdir(parents=True, exist_ok=True)
    configure_overlay_module(overlay_module)
    results = []
    tracker_dirs = sorted((p for p in tracker_root.iterdir() if p.is_dir() and (p / 'tracks_processed.csv').exists()))
    for tracker_dir in tracker_dirs:
        out_dir = overlay_root / tracker_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        case_summary = case_lookup.get(tracker_dir.name, {})
        video_path = case_summary.get('video_path')
        frame_count = int(case_summary.get('frame_count_total', 0) or 0)
        filtered_csv, segment_summary = prepare_primary_segment_csv(tracker_dir, out_dir)
        background_frames = build_no_particle_background_frames(frame_count, segment_summary)
        if video_path:
            (out_dir / 'source_case_summary.json').write_text(json.dumps(case_summary, indent=2), encoding='utf-8')
        _, overlay_summary = overlay_module.make_validation_visualisation_full_video(str(tracker_dir), out_dir=str(out_dir), use_raw=False, frame_step=OVERLAY_FRAME_STEP, tracks_folder=str(tracker_dir), source_folder=str(tracker_dir), source_video_path=str(video_path) if video_path else None, csv_path=str(filtered_csv), background_frame_indices=background_frames, background_use_grayscale=True, background_reducer='mean', binary_background_only=True, particle_render_mode='binary', sample_from_detected_positions_only=True, save_layout_black_background=True)
        overlay_summary['source_case_summary'] = str(out_dir / 'source_case_summary.json') if video_path else None
        overlay_summary['background_frame_count_planned'] = int(len(background_frames))
        results.append(overlay_summary)
    summary_module.BATCH_OUTPUT_DIR = str(overlay_root)
    summary_module.MASTER_OUT_DIR = str(overlay_root / 'master_summary')
    summary_module.main()
    return {'overlay_count': len(results), 'overlay_root': str(overlay_root), 'master_summary_dir': str(overlay_root / 'master_summary')}

def build_physics_batch(physics_module, tracker_root: Path, physics_root: Path) -> None:
    """Build the physics batch used by this script."""
    physics_module.OUTPUT_ROOT = physics_root
    physics_module.BATCHES = [{'label': '25_April_Bottom_Panel_10mm_Spheres', 'source_dirs': [tracker_root]}]
    physics_module.main()

def main() -> int:
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    calibration_dir = OUTPUT_ROOT / 'calibration'
    calibration_json = build_bottomview_calibration(calibration_dir)
    base_cal = json.loads(calibration_json.read_text(encoding='utf-8'))
    trim_mod = load_module(TRIM_TRACK_PATH, 'trim_track_bottomview_25april')
    tracker_module = trim_mod.load_tracker_module()
    overlay_module = load_module(OVERLAY_PATH, 'overlay_bottomview_25april')
    overlay_summary_module = load_module(OVERLAY_SUMMARY_PATH, 'overlay_summary_bottomview_25april')
    physics_module = load_module(PHYSICS_PATH, 'physics_bottomview_25april')
    tune_tracker_for_bottomview(tracker_module)
    trim_mod.rotation_candidates = lambda width, height: ['none']
    tracker_root = OUTPUT_ROOT / 'tracker_outputs'
    overlay_root = OUTPUT_ROOT / 'trajectory_overlays'
    physics_root = OUTPUT_ROOT / 'full_physics_reports'
    tracker_root.mkdir(parents=True, exist_ok=True)
    case_results = []
    case_lookup: dict[str, dict] = {}
    targets = {arg.lower() for arg in sys.argv[1:]}
    for video_path in VIDEO_PATHS:
        if targets and video_path.name.lower() not in targets and (video_path.stem.lower() not in targets):
            continue
        case_name = normalize_case_name(video_path.name)
        tracker_dir = tracker_root / f'{case_name}_tracker_output'
        case_summary_path = tracker_root / f'{case_name}_case_summary.json'
        expected_diameter_mm = infer_sphere_diameter_mm(video_path.name)
        try:
            scan = scan_video_with_retry(trim_mod, video_path, base_cal, tracker_module)
            tracker_result = run_tracker_case(module=tracker_module, video_path=video_path, calibration_json=calibration_json, out_dir=tracker_dir, expected_diameter_mm=expected_diameter_mm, start_frame=int(scan['fine_track']['first_found_frame']), click_xy=scan['fine_track']['click_xy'], seed_kind=str(scan.get('kind', '')))
            case_result = make_case_summary(video_path, scan, tracker_result, tracker_dir)
        except Exception as exc:
            case_result = {'video_name': video_path.name, 'video_path': str(video_path), 'status': 'error', 'tracker_output_dir': str(tracker_dir), 'error': str(exc), 'traceback': traceback.format_exc()}
        case_results.append(case_result)
        case_lookup[tracker_dir.name] = case_result
        case_summary_path.write_text(json.dumps(case_result, indent=2), encoding='utf-8')
        print(f"{video_path.name}: {case_result['status']}")
    ok_dirs = [Path(r['tracker_output_dir']) for r in case_results if r.get('status') == 'ok']
    if not ok_dirs:
        raise RuntimeError('No April 25 bottom-panel runs completed successfully.')
    overlay_info = build_overlay_batch(overlay_module, overlay_summary_module, tracker_root, overlay_root, case_lookup)
    build_physics_batch(physics_module, tracker_root, physics_root)
    manifest = {'raw_root': str(RAW_ROOT), 'output_root': str(OUTPUT_ROOT), 'calibration_json': str(calibration_json), 'tracker_outputs': str(tracker_root), 'trajectory_overlays': str(overlay_root), 'full_physics_reports': str(physics_root), 'overlay_info': overlay_info, 'cases': case_results}
    (OUTPUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUTPUT_ROOT / 'batch_summary.json').write_text(json.dumps(case_results, indent=2), encoding='utf-8')
    for script_path in [Path(__file__), TRACKER_PATH, TRIM_TRACK_PATH, OVERLAY_PATH, OVERLAY_SUMMARY_PATH, PHYSICS_PATH]:
        shutil.copy2(script_path, OUTPUT_ROOT / script_path.name)
    print(f'Saved refreshed bottom-panel outputs to: {OUTPUT_ROOT}')
    return 0

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    raise SystemExit(main())
