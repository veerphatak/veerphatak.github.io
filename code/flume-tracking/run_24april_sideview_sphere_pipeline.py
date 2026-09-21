"""
Pipeline runner: run_24april_sideview_sphere_pipeline.py

Purpose
-------
Runs the complete side-view flume sphere workflow, including tracking, overlays and physics reporting.

Inputs
------
Side-view flume videos, flow-speed settings and paths to the tracker/report scripts.

Outputs
-------
Side-view processed trajectories, validation overlays and physics summary outputs.

Methodological notes
--------------------
The flow-speed setting is used as the hydraulic reference for later non-dimensional and slip analyses.

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
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
WORKSPACE = Path(__file__).resolve().parent
TRACKER_PATH = WORKSPACE / 'Kalman_Tracker_Spheres_binarized.py'
OVERLAY_PATH = WORKSPACE / 'trajectory_validation_overlay_full_video_editable_fixed_v3.py'
OVERLAY_SUMMARY_PATH = WORKSPACE / 'trajectory_validation_master_summary_editable_fixed_v2.py'
PHYSICS_PATH = WORKSPACE / 'trajectory_physics_open_channel_full_report.py'
RAW_ROOT = Path('C:\\IP Work\\24 April flume side view raw videos')
OUTPUT_ROOT = Path(os.environ.get('SIDEVIEW_OUTPUT_ROOT', 'C:\\IP Work\\24 April Flume Side Views Processed'))
CALIBRATION_VIDEO = RAW_ROOT / 'Calibration + 14mm sphere drop 1.mp4'
CALIBRATION_FRAME_INDEX = 30
GRID_SPACING_MM = 10.0
REAL_FPS = 240.0
SIDEVIEW_FLOW_SPEED_M_S = float(os.environ.get('SIDEVIEW_FLOW_SPEED_M_S', '0.25'))
SIDEVIEW_FLOW_SPEED_IN_S = SIDEVIEW_FLOW_SPEED_M_S / 0.0254
SCAN_BACKGROUND_SAMPLES = 48
SCAN_TOP_MARGIN_PX = 35
SCAN_BOTTOM_MARGIN_PX = 120
SCAN_SIDE_MARGIN_PX = 12
SCAN_THRESHOLD = 18
SCAN_MIN_AREA_PX = 700.0
SCAN_MAX_AREA_PX = 85000.0
SCAN_MIN_CIRCULARITY = 0.45
SCAN_MAX_GAP_FRAMES = 10
SCAN_MAX_STEP_PX = 240.0
SCAN_DIAM_FACTOR_MIN = 0.55
SCAN_DIAM_FACTOR_MAX = 2.2
MANUAL_SEED_OVERRIDES = {'14mm sphere drop 5.mp4': {'start_frame': 74, 'click_xy': [1737.0, 322.0], 'notes': 'Late visible sphere selected manually to avoid early false moving object.'}}

@dataclass
class Candidate:
    """Store the data needed by the Candidate structure."""
    frame_idx: int
    cx: float
    cy: float
    radius_px: float
    diameter_mm: float
    area_px2: float
    circularity: float
    compactness: float
    score: float

def infer_sphere_diameter_mm(name: str) -> float:
    """Handle the infer sphere diameter mm step used by this script."""
    lower = name.lower()
    if '10mm' in lower:
        return 10.0
    if '14mm' in lower:
        return 14.0
    raise ValueError(f'Could not infer sphere diameter from: {name}')

def load_module(path: Path, name: str):
    """Load the module used by this script."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def evenly_spaced_indices(count: int, samples: int) -> list[int]:
    """Handle the evenly spaced indices step used by this script."""
    if count <= 0:
        return []
    samples = max(1, min(int(samples), int(count)))
    if samples == 1:
        return [0]
    return [int(round(v)) for v in np.linspace(0, count - 1, num=samples)]

def build_video_background(cap: cv2.VideoCapture, frame_count: int, sample_count: int) -> np.ndarray:
    """Build the video background used by this script."""
    frames: list[np.ndarray] = []
    for idx in evenly_spaced_indices(frame_count, sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError('Could not build background from sampled video frames.')
    return np.median(np.stack(frames, axis=0).astype(np.float32), axis=0).astype(np.uint8)

def contour_circularity(contour: np.ndarray) -> float:
    """Handle the contour circularity step used by this script."""
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    if area <= 0.0 or per <= 1e-06:
        return 0.0
    return float(4.0 * math.pi * area / (per * per))

def diameter_mm_from_radius(radius_px: float, px_per_mm: float) -> float:
    """Handle the diameter mm from radius step used by this script."""
    return float(2.0 * radius_px / px_per_mm)

def build_motion_mask(frame_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    """Build the motion mask used by this script."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, bw = cv2.threshold(diff, SCAN_THRESHOLD, 255, cv2.THRESH_BINARY)
    h, w = bw.shape[:2]
    roi_mask = np.zeros_like(bw)
    roi_mask[SCAN_TOP_MARGIN_PX:max(SCAN_TOP_MARGIN_PX + 1, h - SCAN_BOTTOM_MARGIN_PX), SCAN_SIDE_MARGIN_PX:max(SCAN_SIDE_MARGIN_PX + 1, w - SCAN_SIDE_MARGIN_PX)] = 255
    bw = cv2.bitwise_and(bw, roi_mask)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    return bw

def extract_candidates(frame_bgr: np.ndarray, background_bgr: np.ndarray, px_per_mm: float, expected_diameter_mm: float, frame_idx: int) -> list[Candidate]:
    """Extract the candidates used by this script."""
    bw = build_motion_mask(frame_bgr, background_bgr)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []
    h, w = bw.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < SCAN_MIN_AREA_PX or area > SCAN_MAX_AREA_PX:
            continue
        circularity = contour_circularity(contour)
        if circularity < SCAN_MIN_CIRCULARITY:
            continue
        (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
        if radius_px <= 1.0:
            continue
        diameter_mm = diameter_mm_from_radius(radius_px, px_per_mm)
        if not SCAN_DIAM_FACTOR_MIN * expected_diameter_mm <= diameter_mm <= SCAN_DIAM_FACTOR_MAX * expected_diameter_mm:
            continue
        x, y, ww, hh = cv2.boundingRect(contour)
        compactness = area / float(max(ww * hh, 1))
        aspect = max(ww, hh) / float(max(min(ww, hh), 1))
        border_penalty = 0.65 if x <= 3 or y <= 3 or x + ww >= w - 3 or (y + hh >= h - 3) else 1.0
        score = area * max(circularity, 0.25) * (0.55 + compactness) * border_penalty / max(aspect, 1.0) ** 0.3
        candidates.append(Candidate(frame_idx=int(frame_idx), cx=float(cx), cy=float(cy), radius_px=float(radius_px), diameter_mm=float(diameter_mm), area_px2=float(area), circularity=float(circularity), compactness=float(compactness), score=float(score)))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:6]

def link_tracks(candidates_by_frame: dict[int, list[Candidate]]) -> list[dict]:
    """Handle the link tracks step used by this script."""
    active: list[dict] = []
    finished: list[dict] = []
    for frame_idx in sorted(candidates_by_frame):
        frame_candidates = candidates_by_frame[frame_idx]
        used_ids: set[int] = set()
        for cand in frame_candidates:
            best_track = None
            best_score = None
            for track in active:
                if id(track) in used_ids:
                    continue
                gap = frame_idx - track['last_frame']
                if gap < 1 or gap > SCAN_MAX_GAP_FRAMES:
                    continue
                dist = math.hypot(cand.cx - track['last_pos'][0], cand.cy - track['last_pos'][1])
                if dist > SCAN_MAX_STEP_PX * max(1.0, gap / 2.0):
                    continue
                diam_ratio = cand.diameter_mm / max(track['diameter_mm'], 1e-06)
                if diam_ratio < 0.7 or diam_ratio > 1.45:
                    continue
                match_score = dist + abs(cand.diameter_mm - track['diameter_mm']) * 30.0
                if best_track is None or match_score < best_score:
                    best_track = track
                    best_score = match_score
            if best_track is None:
                active.append({'frames': [cand.frame_idx], 'centers': [[cand.cx, cand.cy]], 'scores': [cand.score], 'diameters_mm': [cand.diameter_mm], 'last_frame': cand.frame_idx, 'last_pos': (cand.cx, cand.cy), 'diameter_mm': cand.diameter_mm})
            else:
                best_track['frames'].append(cand.frame_idx)
                best_track['centers'].append([cand.cx, cand.cy])
                best_track['scores'].append(cand.score)
                best_track['diameters_mm'].append(cand.diameter_mm)
                best_track['last_frame'] = cand.frame_idx
                best_track['last_pos'] = (cand.cx, cand.cy)
                best_track['diameter_mm'] = 0.5 * best_track['diameter_mm'] + 0.5 * cand.diameter_mm
                used_ids.add(id(best_track))
        still_active: list[dict] = []
        for track in active:
            if frame_idx - track['last_frame'] > SCAN_MAX_GAP_FRAMES:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active
    finished.extend(active)
    return [t for t in finished if len(t['frames']) >= 4]

def rank_track_for_expected(track: dict, expected_diameter_mm: float) -> tuple[float, float, float]:
    """Handle the rank track for expected step used by this script."""
    diameters = np.asarray(track['diameters_mm'], dtype=float)
    median_d = float(np.median(diameters))
    size_rel_error = abs(median_d - expected_diameter_mm) / max(expected_diameter_mm, 1e-06)
    score_sum = float(sum(track['scores']))
    track_len = float(len(track['frames']))
    return (track_len * 100000.0 - size_rel_error * 80000.0, score_sum - size_rel_error * 120000.0, -float(np.std(diameters)))

def summarize_tracks_for_seed(tracks: list[dict], expected_diameter_mm: float) -> dict | None:
    """Handle the summarize tracks for seed step used by this script."""
    if not tracks:
        return None
    ranked = sorted(tracks, key=lambda t: rank_track_for_expected(t, expected_diameter_mm), reverse=True)
    best = ranked[0]
    centers = np.asarray(best['centers'], dtype=float)
    start_click = np.mean(centers[:min(3, len(centers))], axis=0)
    return {'start_frame': int(best['frames'][0]), 'end_frame': int(best['frames'][-1]), 'click_xy': [float(start_click[0]), float(start_click[1])], 'frame_count': int(len(best['frames'])), 'diameter_mm_median': float(np.median(np.asarray(best['diameters_mm'], dtype=float))), 'diameter_mm_std': float(np.std(np.asarray(best['diameters_mm'], dtype=float))), 'score_sum': float(sum(best['scores'])), 'diameter_match_error_mm': float(abs(np.median(np.asarray(best['diameters_mm'], dtype=float)) - expected_diameter_mm)), 'frames': [int(v) for v in best['frames']]}

def estimate_sideview_grid(frame_bgr: np.ndarray) -> dict:
    """Handle the estimate sideview grid step used by this script."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray[60:860, :]
    vert = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 151)))
    horz = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (151, 21)))
    vert_proj = vert.mean(axis=0).astype(np.float32)
    horz_proj = horz.mean(axis=1).astype(np.float32)

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
    vertical_peaks = find_peaks(vert_proj, min_dist=80, quantile=0.82)
    horizontal_peaks = find_peaks(horz_proj, min_dist=60, quantile=0.82) + 60
    if len(vertical_peaks) < 3 or len(horizontal_peaks) < 3:
        raise RuntimeError('Could not estimate side-view calibration grid spacing.')
    vertical_spacing_px = float(np.median(np.diff(vertical_peaks)))
    horizontal_spacing_px = float(np.median(np.diff(horizontal_peaks)))
    px_per_mm = float(np.mean([vertical_spacing_px, horizontal_spacing_px]) / GRID_SPACING_MM)
    return {'vertical_peaks_px': vertical_peaks.tolist(), 'horizontal_peaks_px': horizontal_peaks.tolist(), 'vertical_spacing_px': vertical_spacing_px, 'horizontal_spacing_px': horizontal_spacing_px, 'estimated_px_per_mm': px_per_mm}

def create_identity_calibration(width: int, height: int, px_per_mm: float, notes: str) -> dict:
    """Create the identity calibration used by this script."""
    src = [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]]
    H = np.eye(3, dtype=np.float64)
    return {'calibration_source_type': 'video_frame_sideview_identity', 'calibration_video_path': str(CALIBRATION_VIDEO), 'calibration_frame_index': int(CALIBRATION_FRAME_INDEX), 'calibration_image_width_px': int(width), 'calibration_image_height_px': int(height), 'rect_width_mm': float(width / px_per_mm), 'rect_height_mm': float(height / px_per_mm), 'rectified_px_per_mm': float(px_per_mm), 'rectified_width_px': int(width), 'rectified_height_px': int(height), 'src_points_px': src, 'raw_click_points_px': src, 'src_points_normalized': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 'dst_points_px': src, 'homography': H.tolist(), 'homography_inverse': H.tolist(), 'grid_spacing_mm': float(GRID_SPACING_MM), 'point_order': ['TL', 'TR', 'BR', 'BL'], 'click_snap_to_grid_intersection': False, 'notes': notes}

def build_sideview_calibration(out_dir: Path) -> Path:
    """Build the sideview calibration used by this script."""
    cap = cv2.VideoCapture(str(CALIBRATION_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open calibration video: {CALIBRATION_VIDEO}')
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME_INDEX)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError('Could not read the selected calibration frame.')
    finally:
        cap.release()
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / 'calibration_frame.png'), frame)
    estimate = estimate_sideview_grid(frame)
    vis = frame.copy()
    for x in estimate['vertical_peaks_px']:
        cv2.line(vis, (int(x), 0), (int(x), vis.shape[0] - 1), (0, 255, 255), 2)
    for y in estimate['horizontal_peaks_px']:
        cv2.line(vis, (0, int(y)), (vis.shape[1] - 1, int(y)), (255, 0, 255), 2)
    cv2.imwrite(str(out_dir / 'calibration_grid_peaks.png'), vis)
    (out_dir / 'calibration_grid_estimate.json').write_text(json.dumps(estimate, indent=2), encoding='utf-8')
    calibration = create_identity_calibration(width=frame.shape[1], height=frame.shape[0], px_per_mm=estimate['estimated_px_per_mm'], notes='Side-view identity calibration using measured grid spacing from the April 24 calibration clip. Perspective distortion in this view is small, so raw pixel coordinates are preserved and scaled using the measured average grid spacing.')
    out_path = out_dir / 'sideview_calibration.json'
    out_path.write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    return out_path

def tune_tracker_for_sideview(module) -> None:
    """Handle the tune tracker for sideview step used by this script."""
    acquire = module.PARAMS['sphere_white']['acquire']
    track = module.PARAMS['sphere_white']['track']
    acquire.update({'roi_half_size_px': 260, 'binary_blur_k': 7, 'binary_bg_blur_k': 51, 'binary_min_threshold': 6, 'gray_min_threshold': 6, 'use_gray_branch': True, 'open_k': 5, 'close_k': 13, 'close_iter': 2, 'min_component_area_px': 500, 'min_area_px': 3000, 'max_area_px': 90000, 'min_circularity': 0.35, 'min_solidity': 0.45, 'aspect_tol': 1.65, 'min_diameter_factor': 0.55, 'max_diameter_factor': 2.1, 'max_assoc_dist_px': 170.0, 'max_misses': 18, 'border_relax_px': 28, 'relaxed_min_area_px': 1200, 'relaxed_min_circularity': 0.22, 'relaxed_min_solidity': 0.25, 'relaxed_aspect_tol': 2.2, 'relaxed_min_diameter_factor': 0.45, 'relaxed_max_diameter_factor': 2.3, 'roi_expand_per_miss_px': 80, 'assoc_expand_per_miss_px': 120, 'reacquire_fullframe_after_misses': 2, 'exit_margin_px': 50, 'exit_misses': 7})
    track.update({'roi_half_size_px': 210, 'binary_blur_k': 7, 'binary_bg_blur_k': 45, 'binary_min_threshold': 8, 'gray_min_threshold': 8, 'use_gray_branch': True, 'open_k': 5, 'close_k': 11, 'close_iter': 2, 'min_component_area_px': 600, 'min_area_px': 3800, 'max_area_px': 90000, 'min_circularity': 0.42, 'min_solidity': 0.5, 'aspect_tol': 1.5, 'min_diameter_factor': 0.6, 'max_diameter_factor': 2.1, 'max_assoc_dist_px': 160.0, 'max_misses': 18, 'border_relax_px': 28, 'relaxed_min_area_px': 1400, 'relaxed_min_circularity': 0.2, 'relaxed_min_solidity': 0.24, 'relaxed_aspect_tol': 2.3, 'relaxed_min_diameter_factor': 0.45, 'relaxed_max_diameter_factor': 2.3, 'roi_expand_per_miss_px': 90, 'assoc_expand_per_miss_px': 130, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 55, 'exit_misses': 7})

def scan_video_for_seed(video_path: Path, px_per_mm: float, expected_diameter_mm: float) -> dict:
    """Scan the video for seed used by this script."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video for scanning: {video_path}')
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise RuntimeError('Video reports zero frames.')
        background = build_video_background(cap, frame_count, SCAN_BACKGROUND_SAMPLES)
        candidates_by_frame: dict[int, list[Candidate]] = {}
        for frame_idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            candidates = extract_candidates(frame, background, px_per_mm, expected_diameter_mm, frame_idx)
            if candidates:
                candidates_by_frame[frame_idx] = candidates
        tracks = link_tracks(candidates_by_frame)
        seed = summarize_tracks_for_seed(tracks, expected_diameter_mm)
        if seed is None:
            raise RuntimeError('No coherent sphere track found during side-view scan.')
        seed['frame_count_total'] = frame_count
        return seed
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
    with proc_csv.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            try:
                x_vals.append(float(row['x_rel_mm']))
                y_vals.append(float(row['y_rel_mm']))
            except Exception:
                continue
    return {'rows': total, 'found': found, 'found_ratio': found / total if total else None, 'x_rel_mm_range': [float(min(x_vals)), float(max(x_vals))] if x_vals else None, 'y_rel_mm_range': [float(min(y_vals)), float(max(y_vals))] if y_vals else None}

def run_tracker_case(module, video_path: Path, calibration_json: Path, out_dir: Path, diameter_mm: float, start_frame: int, click_xy: list[float]) -> dict:
    """Run the tracker case used by this script."""
    tune_tracker_for_sideview(module)
    module.cv2.destroyAllWindows = lambda: None
    module.USE_RUNTIME_PROMPTS = False
    module.ENABLE_IMSHOW = False
    module.DEFAULT_VIDEO_PATH = str(video_path)
    module.DEFAULT_CALIBRATION_JSON = str(calibration_json)
    module.DEFAULT_OUT_DIR = str(out_dir)
    module.DEFAULT_REAL_FPS = REAL_FPS
    module.DEFAULT_FRAME_STRIDE = 1
    module.DEFAULT_START_FRAME = int(start_frame)
    module.DEFAULT_OBJECT_MODE = 'sphere'
    module.DEFAULT_SPHERE_COLOR_MODE = 'white'
    module.DEFAULT_SPHERE_DIAMETER_MM = float(diameter_mm)
    module.DEFAULT_USE_PRESET_CLICK = True
    module.DEFAULT_PRESET_CLICK = (int(round(click_xy[0])), int(round(click_xy[1])))
    module.DEFAULT_START_SEARCH_FRAMES = 20
    module.DEFAULT_SAVE_DEBUG_IMAGES = True
    module.DEFAULT_NUM_DEBUG_IMAGES = 12
    module.DEFAULT_ENABLE_SPHERE_BACKGROUND_SUBTRACTION = True
    module.DEFAULT_SPHERE_BACKGROUND_SECONDS = 2.0
    module.DEFAULT_SPHERE_BACKGROUND_SAMPLES = 20
    module.DEFAULT_SPHERE_BACKGROUND_MARGIN_FRAMES = 6
    module.DEFAULT_SPHERE_BACKGROUND_KEEP_FRACTION = 0.75
    out_dir.mkdir(parents=True, exist_ok=True)
    module.main()
    return summarize_tracker_output(out_dir)

def normalize_case_name(video_name: str) -> str:
    """Normalise the case name used by this script."""
    stem = Path(video_name).stem.strip().lower()
    return stem.replace(' ', '_').replace('+', 'plus')

def make_tracker_case_summary(video_path: Path, seed: dict, tracker_summary: dict, tracker_dir: Path) -> dict:
    """Handle the make tracker case summary step used by this script."""
    return {'video_name': video_path.name, 'video_path': str(video_path), 'tracker_output_dir': str(tracker_dir), 'seed': seed, 'tracker_summary': tracker_summary}

def build_overlay_batch(overlay_module, summary_module, tracker_root: Path, overlay_root: Path) -> dict:
    """Build the overlay batch used by this script."""
    overlay_root.mkdir(parents=True, exist_ok=True)
    results = []
    for tracker_dir in sorted((p for p in tracker_root.iterdir() if p.is_dir() and (p / 'tracks_processed.csv').exists())):
        out_dir = overlay_root / tracker_dir.name
        _, summary = overlay_module.make_validation_visualisation_full_video(str(tracker_dir), out_dir=str(out_dir), use_raw=False, frame_step=5, tracks_folder=str(tracker_dir), source_folder=str(tracker_dir))
        results.append(summary)
    summary_module.BATCH_OUTPUT_DIR = str(overlay_root)
    summary_module.MASTER_OUT_DIR = str(overlay_root / 'master_summary')
    summary_module.main()
    return {'overlay_count': len(results), 'overlay_root': str(overlay_root), 'master_summary_dir': str(overlay_root / 'master_summary')}

def build_physics_batch(physics_module, tracker_root: Path, physics_root: Path) -> None:
    """Build the physics batch used by this script."""
    physics_module.OUTPUT_ROOT = Path(physics_root)
    physics_module.BATCHES = [{'label': '24_April_Side_Views', 'flow_speed_in_s': float(SIDEVIEW_FLOW_SPEED_IN_S), 'source_dirs': [Path(tracker_root)]}]
    physics_module.main()

def main() -> None:
    """Run the main entry point for this script."""
    tracker_module = load_module(TRACKER_PATH, 'tracker_sideview_24april')
    overlay_module = load_module(OVERLAY_PATH, 'overlay_sideview_24april')
    overlay_summary_module = load_module(OVERLAY_SUMMARY_PATH, 'overlay_summary_sideview_24april')
    physics_module = load_module(PHYSICS_PATH, 'physics_sideview_24april')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    calibration_dir = OUTPUT_ROOT / 'calibration'
    calibration_json = build_sideview_calibration(calibration_dir)
    calibration = json.loads(calibration_json.read_text(encoding='utf-8'))
    px_per_mm = float(calibration['rectified_px_per_mm'])
    tracker_root = OUTPUT_ROOT / 'tracker_outputs'
    overlay_root = OUTPUT_ROOT / 'trajectory_overlays'
    physics_root = OUTPUT_ROOT / 'full_physics_reports'
    tracker_root.mkdir(parents=True, exist_ok=True)
    case_results = []
    targets = {arg.lower() for arg in sys.argv[1:]}
    for video_path in sorted(RAW_ROOT.glob('*.mp4')):
        if 'calibration' in video_path.name.lower():
            continue
        if targets and video_path.name.lower() not in targets and (video_path.stem.lower() not in targets):
            continue
        case_name = normalize_case_name(video_path.name)
        tracker_dir = tracker_root / f'{case_name}_tracker_output'
        expected_diameter_mm = infer_sphere_diameter_mm(video_path.name)
        try:
            if video_path.name in MANUAL_SEED_OVERRIDES:
                seed = dict(MANUAL_SEED_OVERRIDES[video_path.name])
                cap_seed = cv2.VideoCapture(str(video_path))
                try:
                    seed['frame_count_total'] = int(cap_seed.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                finally:
                    cap_seed.release()
            else:
                seed = scan_video_for_seed(video_path, px_per_mm, expected_diameter_mm)
            tracker_summary = run_tracker_case(module=tracker_module, video_path=video_path, calibration_json=calibration_json, out_dir=tracker_dir, diameter_mm=expected_diameter_mm, start_frame=int(seed['start_frame']), click_xy=seed['click_xy'])
            case_result = {'status': 'ok', **make_tracker_case_summary(video_path, seed, tracker_summary, tracker_dir)}
        except Exception as exc:
            case_result = {'status': 'error', 'video_name': video_path.name, 'video_path': str(video_path), 'tracker_output_dir': str(tracker_dir), 'error': str(exc), 'traceback': traceback.format_exc()}
        case_results.append(case_result)
        (tracker_root / f'{case_name}_case_summary.json').write_text(json.dumps(case_result, indent=2), encoding='utf-8')
        print(f"{video_path.name}: {case_result['status']}")
    ok_dirs = [Path(r['tracker_output_dir']) for r in case_results if r.get('status') == 'ok']
    if not ok_dirs:
        raise RuntimeError('No side-view sphere runs completed successfully.')
    overlay_info = build_overlay_batch(overlay_module, overlay_summary_module, tracker_root, overlay_root)
    build_physics_batch(physics_module, tracker_root, physics_root)
    manifest = {'raw_root': str(RAW_ROOT), 'output_root': str(OUTPUT_ROOT), 'flow_speed_m_s': float(SIDEVIEW_FLOW_SPEED_M_S), 'flow_speed_in_s_equivalent': float(SIDEVIEW_FLOW_SPEED_IN_S), 'calibration_json': str(calibration_json), 'tracker_outputs': str(tracker_root), 'trajectory_overlays': str(overlay_root), 'full_physics_reports': str(physics_root), 'overlay_info': overlay_info, 'cases': case_results}
    (OUTPUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUTPUT_ROOT / 'batch_summary.json').write_text(json.dumps(case_results, indent=2), encoding='utf-8')
    for script_path in [Path(__file__), TRACKER_PATH, OVERLAY_PATH, OVERLAY_SUMMARY_PATH, PHYSICS_PATH]:
        target = OUTPUT_ROOT / script_path.name
        target.write_text(script_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Saved processed side-view outputs to: {OUTPUT_ROOT}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    sys.exit(main())
