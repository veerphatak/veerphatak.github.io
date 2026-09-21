"""
Validation/overlay builder: build_straight_view_slip_overlays_from_raw.py

Purpose
-------
Rebuilds straight-view sphere slip overlays from raw videos using backgrounds, ROI masks and particle detections.

Inputs
------
Raw straight-view sphere videos and ROI/type configuration dictionaries.

Outputs
-------
Trajectory overlays, slip summaries and diagnostic outputs.

Methodological notes
--------------------
The static background and ROI masks restrict detection to the physical test section and reduce false positives from reflections.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
RAW_VIDEOS = {'10mm': ['C:\\Users\\User\\Downloads\\Straight view 10mm sphere drop 1.mp4', 'C:\\Users\\User\\Downloads\\Straight view 10mm sphere drop 2.mp4', 'C:\\Users\\User\\Downloads\\Straight view 10mm sphere drop 3.mp4', 'C:\\Users\\User\\Downloads\\Straight view 10mm sphere drop 4.mp4', 'C:\\Users\\User\\Downloads\\Straight view 10mm sphere drop 5.mp4', 'C:\\Users\\User\\Downloads\\Straight view 10mm sphere drop 6.mp4'], '14mm': ['C:\\Users\\User\\Downloads\\Straight view 14mm sphere drop 1.mp4', 'C:\\Users\\User\\Downloads\\Straight view 14mm sphere drop 2.mp4', 'C:\\Users\\User\\Downloads\\Straight view 14mm sphere drop 3.mp4', 'C:\\Users\\User\\Downloads\\Straight view 14mm sphere drop 4.mp4', 'C:\\Users\\User\\Downloads\\Straight view 14mm sphere drop 5.mp4', 'C:\\Users\\User\\Downloads\\Straight view 14mm sphere drop 6.mp4']}
OLD_TRACKER_ROOT = 'C:\\IP Work\\Straight View Sphere Drops Processed\\tracker_outputs'
OUTPUT_ROOT = 'C:\\IP Work\\Straight View Sphere Slip Overlays Rebuilt'
PRESERVE_CAPTURE_ORIENTATION = True
BASE_ROI_POLYGON = [(370, 430), (735, 430), (1020, 1840), (240, 1840)]
BASE_VALID_CENTER_POLYGON = [(392, 455), (720, 455), (930, 1815), (290, 1815)]
TYPE_CONFIG = {'10mm': {'slug': '10mm', 'use_hint_track': True, 'particle_fill': (220, 38, 38, 235), 'marker_radius': 8, 'window_px': 27, 'search_radius_px': 85, 'search_expand_px': 16, 'min_score': 10500.0, 'seed_min_score': 12500.0, 'max_misses': 10, 'coarse_step': 5, 'coarse_scan_fallback': True, 'seed_pad_before': 420, 'seed_pad_after': 260}, '14mm': {'slug': '14mm', 'use_hint_track': True, 'particle_fill': (255, 255, 255, 235), 'marker_radius': 10, 'window_px': 39, 'search_radius_px': 110, 'search_expand_px': 18, 'min_score': 9000.0, 'seed_min_score': 12000.0, 'max_misses': 12, 'coarse_step': 4, 'coarse_scan_fallback': True, 'seed_pad_before': 220, 'seed_pad_after': 220}}
RUN_COLORS = [(16, 78, 139, 255), (0, 140, 114, 255), (199, 80, 0, 255), (166, 0, 81, 255), (123, 88, 0, 255), (82, 64, 196, 255)]

@dataclass
class HintTrack:
    """Store the data needed by the HintTrack structure."""
    folder_name: str
    first_frame: int
    last_frame: int
    seed_frame: int
    frame_to_xy: Dict[int, Tuple[float, float]]
    seed_xy: Tuple[float, float]

@dataclass
class TrackPoint:
    """Store the data needed by the TrackPoint structure."""
    frame_idx: int
    x_px: float
    y_px: float
    score: float
    x_smooth_px: float = 0.0
    y_smooth_px: float = 0.0
    is_interpolated: bool = False

def rotate_frame_clockwise(frame: np.ndarray) -> np.ndarray:
    """Rotate the frame clockwise used by this script."""
    return np.rot90(frame, k=-1)

def rotate_points_clockwise(points: Sequence[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    """Rotate the points clockwise used by this script."""
    return [(float(height - 1 - y), float(x)) for x, y in points]

def get_active_polygons(width: int, height: int, needs_rotate: bool) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Handle the get active polygons step used by this script."""
    if not needs_rotate:
        return (list(BASE_ROI_POLYGON), list(BASE_VALID_CENTER_POLYGON))
    return (rotate_points_clockwise(BASE_ROI_POLYGON, width, height), rotate_points_clockwise(BASE_VALID_CENTER_POLYGON, width, height))

def polygon_bbox(polygon: Sequence[Tuple[float, float]], width: int, height: int, pad: int=20) -> Tuple[int, int, int, int]:
    """Handle the polygon bbox step used by this script."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = max(0, int(math.floor(min(xs))) - pad)
    y0 = max(0, int(math.floor(min(ys))) - pad)
    x1 = min(width, int(math.ceil(max(xs))) + pad)
    y1 = min(height, int(math.ceil(max(ys))) + pad)
    return (x0, y0, x1, y1)

def slugify_video_path(path: str) -> str:
    """Convert the video path used by this script."""
    stem = os.path.splitext(os.path.basename(path))[0]
    slug = stem.lower().replace('+', ' plus ')
    slug = '_'.join(slug.split())
    return slug

def ensure_dir(path: str) -> None:
    """Ensure the dir used by this script."""
    os.makedirs(path, exist_ok=True)

def build_roi_mask(size: Tuple[int, int], polygon: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Build the roi mask used by this script."""
    width, height = size
    mask_image = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(list(polygon), fill=255)
    return np.array(mask_image, dtype=np.uint8) > 0

def build_polygon_mask(size: Tuple[int, int], polygon: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Build the polygon mask used by this script."""
    width, height = size
    mask_image = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(list(polygon), fill=255)
    return np.array(mask_image, dtype=np.uint8) > 0

def box_sum(arr: np.ndarray, k: int) -> np.ndarray:
    """Handle the box sum step used by this script."""
    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode='constant')
    ii = np.pad(padded, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]

def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Handle the moving average step used by this script."""
    if len(values) == 0:
        return values
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')

def load_hint_track(video_path: str) -> Optional[HintTrack]:
    """Load the hint track used by this script."""
    slug = slugify_video_path(video_path)
    preferred = os.path.join(OLD_TRACKER_ROOT, f'{slug}_tracker_output', 'tracks_raw.csv')
    candidates: List[Tuple[int, str, str]] = []
    if os.path.exists(preferred):
        candidates.append((1, os.path.dirname(preferred), preferred))
    for name in os.listdir(OLD_TRACKER_ROOT):
        if not name.startswith(slug):
            continue
        csv_path = os.path.join(OLD_TRACKER_ROOT, name, 'tracks_raw.csv')
        if os.path.exists(csv_path):
            priority = 0 if name.endswith('_tracker_output') else -1
            candidates.append((priority, os.path.join(OLD_TRACKER_ROOT, name), csv_path))
    best: Optional[HintTrack] = None
    best_score = (-1, -1.0)
    for priority, folder, csv_path in candidates:
        frame_to_xy: Dict[int, Tuple[float, float]] = {}
        areas: List[Tuple[float, int, float, float]] = []
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
        except OSError:
            continue
        for row in rows:
            if row.get('found_this_frame') != '1':
                continue
            try:
                frame_idx = int(float(row['frame_idx']))
                x_px = float(row['cx_raw_px'])
                y_px = float(row['cy_raw_px'])
                area = float(row.get('area_px2') or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            frame_to_xy[frame_idx] = (x_px, y_px)
            areas.append((area, frame_idx, x_px, y_px))
        if not areas:
            continue
        areas.sort(reverse=True)
        found_count = len(areas)
        peak_area = float(areas[0][0])
        score = (priority, found_count + peak_area / 100000.0)
        if score > best_score:
            frames = sorted(frame_to_xy.keys())
            best = HintTrack(folder_name=os.path.basename(folder), first_frame=frames[0], last_frame=frames[-1], seed_frame=int(areas[0][1]), frame_to_xy=frame_to_xy, seed_xy=(float(areas[0][2]), float(areas[0][3])))
            best_score = score
    return best

def normalize_frame(frame: np.ndarray, meta: Dict[str, object]) -> Tuple[np.ndarray, bool]:
    """Normalise the frame used by this script."""
    if PRESERVE_CAPTURE_ORIENTATION:
        if frame.shape[1] > frame.shape[0]:
            return (rotate_frame_clockwise(frame), False)
        return (frame, False)
    source_size = meta.get('source_size')
    if not source_size or len(source_size) != 2:
        return (frame, False)
    source_w, source_h = (int(source_size[0]), int(source_size[1]))
    frame_h, frame_w = (int(frame.shape[0]), int(frame.shape[1]))
    needs_rotate = (frame_w, frame_h) != (source_w, source_h) and (frame_h, frame_w) == (source_w, source_h)
    if needs_rotate:
        return (rotate_frame_clockwise(frame), True)
    return (frame, False)

def compute_background(video_path: str, sample_count: int=50) -> Tuple[np.ndarray, Dict[str, object]]:
    """Compute the background used by this script."""
    reader = imageio.get_reader(video_path)
    total_frames = int(reader.count_frames())
    meta = reader.get_meta_data()
    sample_indices = np.linspace(0, max(0, total_frames - 1), sample_count, dtype=int)
    first_frame = reader.get_data(int(sample_indices[0]))
    normalized_first, needs_rotate = normalize_frame(first_frame, meta)
    frames = [normalized_first.astype(np.float32)]
    for idx in sample_indices[1:]:
        frame = reader.get_data(int(idx))
        normalized, _ = normalize_frame(frame, meta)
        frames.append(normalized.astype(np.float32))
    reader.close()
    bg = np.median(np.stack(frames, axis=0), axis=0)
    return (bg, {'total_frames': total_frames, 'fps': float(meta.get('fps', 30.0)), 'width': int(bg.shape[1]), 'height': int(bg.shape[0]), 'needs_rotate': bool(needs_rotate), 'read_width': int(first_frame.shape[1]), 'read_height': int(first_frame.shape[0])})

def detect_red_candidate(frame: np.ndarray, bg: np.ndarray, roi_mask: np.ndarray, roi_bbox: Tuple[int, int, int, int], guess_xy: Optional[Tuple[float, float]], window_px: int, search_radius_px: int) -> Optional[Tuple[float, float, float]]:
    """Detect the red candidate used by this script."""
    x0, y0, x1, y1 = roi_bbox
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    crop_bg = bg[y0:y1, x0:x1]
    crop_mask = roi_mask[y0:y1, x0:x1]
    red_excess = crop[..., 0] - 0.55 * crop[..., 1] - 0.55 * crop[..., 2]
    bg_delta = crop[..., 0] - crop_bg[..., 0]
    score = np.maximum(red_excess - 10.0, 0.0) + 0.3 * np.maximum(bg_delta - 8.0, 0.0)
    score *= crop_mask
    score[score < 5.0] = 0.0
    if guess_xy is not None:
        gx = float(guess_xy[0]) - x0
        gy = float(guess_xy[1]) - y0
        yy, xx = np.indices(score.shape)
        keep = (xx - gx) ** 2 + (yy - gy) ** 2 <= float(search_radius_px * search_radius_px)
        score *= keep
    if float(score.max()) <= 0.0:
        return None
    local = box_sum(score, window_px)
    peak = float(local.max())
    if peak <= 0.0:
        return None
    y_peak, x_peak = np.unravel_index(np.argmax(local), local.shape)
    yy0 = max(0, y_peak - window_px // 2)
    yy1 = min(score.shape[0], y_peak + window_px // 2 + 1)
    xx0 = max(0, x_peak - window_px // 2)
    xx1 = min(score.shape[1], x_peak + window_px // 2 + 1)
    patch = score[yy0:yy1, xx0:xx1]
    if float(patch.sum()) <= 0.0:
        return None
    yy, xx = np.indices(patch.shape)
    cx = float((patch * (xx + xx0)).sum() / patch.sum()) + x0
    cy = float((patch * (yy + yy0)).sum() / patch.sum()) + y0
    return (cx, cy, peak)

def detect_white_candidate(frame: np.ndarray, bg: np.ndarray, roi_mask: np.ndarray, roi_bbox: Tuple[int, int, int, int], guess_xy: Optional[Tuple[float, float]], window_px: int, search_radius_px: int) -> Optional[Tuple[float, float, float]]:
    """Detect the white candidate used by this script."""
    x0, y0, x1, y1 = roi_bbox
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    crop_bg = bg[y0:y1, x0:x1]
    crop_mask = roi_mask[y0:y1, x0:x1]
    gray = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    bg_gray = 0.299 * crop_bg[..., 0] + 0.587 * crop_bg[..., 1] + 0.114 * crop_bg[..., 2]
    sat = np.max(crop, axis=2) - np.min(crop, axis=2)
    bright = np.maximum(gray - bg_gray - 2.0, 0.0)
    contrast = np.maximum(gray - np.median(gray), 0.0)
    low_sat = np.clip((65.0 - sat) / 65.0, 0.0, 1.0)
    edge_softener = np.clip((gray - 40.0) / 60.0, 0.0, 1.0)
    score = (0.9 * bright + 0.18 * contrast) * (0.55 + 0.45 * low_sat) * edge_softener
    score *= crop_mask
    score[score < 3.0] = 0.0
    if guess_xy is not None:
        gx = float(guess_xy[0]) - x0
        gy = float(guess_xy[1]) - y0
        yy, xx = np.indices(score.shape)
        keep = (xx - gx) ** 2 + (yy - gy) ** 2 <= float(search_radius_px * search_radius_px)
        score *= keep
    if float(score.max()) <= 0.0:
        return None
    local = box_sum(score, window_px)
    peak = float(local.max())
    if peak <= 0.0:
        return None
    y_peak, x_peak = np.unravel_index(np.argmax(local), local.shape)
    yy0 = max(0, y_peak - window_px // 2)
    yy1 = min(score.shape[0], y_peak + window_px // 2 + 1)
    xx0 = max(0, x_peak - window_px // 2)
    xx1 = min(score.shape[1], x_peak + window_px // 2 + 1)
    patch = score[yy0:yy1, xx0:xx1]
    if float(patch.sum()) <= 0.0:
        return None
    yy, xx = np.indices(patch.shape)
    cx = float((patch * (xx + xx0)).sum() / patch.sum()) + x0
    cy = float((patch * (yy + yy0)).sum() / patch.sum()) + y0
    return (cx, cy, peak)

def detect_candidate(particle_type: str, frame: np.ndarray, bg: np.ndarray, roi_mask: np.ndarray, roi_bbox: Tuple[int, int, int, int], guess_xy: Optional[Tuple[float, float]], window_px: int, search_radius_px: int) -> Optional[Tuple[float, float, float]]:
    """Detect the candidate used by this script."""
    if particle_type == '10mm':
        return detect_red_candidate(frame, bg, roi_mask, roi_bbox, guess_xy, window_px, search_radius_px)
    return detect_white_candidate(frame, bg, roi_mask, roi_bbox, guess_xy, window_px, search_radius_px)

def coarse_find_seed(video_path: str, particle_type: str, bg: np.ndarray, roi_mask: np.ndarray, roi_bbox: Tuple[int, int, int, int], total_frames: int, window_px: int, coarse_step: int, needs_rotate: bool, scan_first: int, scan_last: int) -> Optional[Tuple[int, float, float, float]]:
    """Handle the coarse find seed step used by this script."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    best: Optional[Tuple[int, float, float, float]] = None
    scan_first = max(0, scan_first)
    scan_last = min(total_frames - 1, scan_last)
    for frame_idx in range(scan_first, scan_last + 1, coarse_step):
        frame = reader.get_data(frame_idx)
        if needs_rotate:
            frame, _ = normalize_frame(frame, meta)
        found = detect_candidate(particle_type=particle_type, frame=frame, bg=bg, roi_mask=roi_mask, roi_bbox=roi_bbox, guess_xy=None, window_px=window_px, search_radius_px=10000)
        if found is None:
            continue
        x_px, y_px, score = found
        if best is None or score > best[3]:
            best = (frame_idx, x_px, y_px, score)
    reader.close()
    return best

def track_video(video_path: str, particle_type: str, out_dir: str) -> Dict[str, object]:
    """Track the video used by this script."""
    config = TYPE_CONFIG[particle_type]
    hint_track = load_hint_track(video_path) if config.get('use_hint_track', True) else None
    bg, meta = compute_background(video_path)
    roi_polygon, valid_center_polygon = get_active_polygons(int(meta['read_width']), int(meta['read_height']), bool(meta['needs_rotate']))
    roi_mask = build_roi_mask((int(meta['width']), int(meta['height'])), roi_polygon)
    valid_center_mask = build_polygon_mask((int(meta['width']), int(meta['height'])), valid_center_polygon)
    roi_bbox = polygon_bbox(roi_polygon, int(meta['width']), int(meta['height']))
    total_frames = int(meta['total_frames'])
    fps = float(meta['fps'])
    if hint_track is not None and bool(meta['needs_rotate']):
        frame_to_xy = {frame_idx: rotate_points_clockwise([(x_px, y_px)], int(meta['read_width']), int(meta['read_height']))[0] for frame_idx, (x_px, y_px) in hint_track.frame_to_xy.items()}
        hint_track = HintTrack(folder_name=hint_track.folder_name, first_frame=hint_track.first_frame, last_frame=hint_track.last_frame, seed_frame=hint_track.seed_frame, frame_to_xy=frame_to_xy, seed_xy=rotate_points_clockwise([hint_track.seed_xy], int(meta['read_width']), int(meta['read_height']))[0])
    if hint_track is not None:
        scan_first = max(0, hint_track.first_frame - int(config['seed_pad_before']))
        scan_last = min(total_frames - 1, hint_track.last_frame + int(config['seed_pad_after']))
    else:
        scan_first = 0
        scan_last = total_frames - 1
    coarse = coarse_find_seed(video_path=video_path, particle_type=particle_type, bg=bg, roi_mask=roi_mask, roi_bbox=roi_bbox, total_frames=total_frames, window_px=int(config['window_px']), coarse_step=int(config['coarse_step']), needs_rotate=bool(meta['needs_rotate']), scan_first=scan_first, scan_last=scan_last)
    seed_frame: int
    guess_xy: Tuple[float, float]
    if hint_track is not None and coarse is None:
        seed_frame = hint_track.seed_frame
        guess_xy = hint_track.seed_xy
        first_frame = max(0, hint_track.first_frame - int(config['seed_pad_before']))
        last_frame = min(total_frames - 1, hint_track.last_frame + int(config['seed_pad_after']))
    elif coarse is not None:
        seed_frame, guess_x, guess_y, _ = coarse
        guess_xy = (guess_x, guess_y)
        if hint_track is not None:
            first_frame = max(0, min(hint_track.first_frame, seed_frame) - int(config['seed_pad_before']))
            last_frame = min(total_frames - 1, max(hint_track.last_frame, seed_frame) + int(config['seed_pad_after']))
        else:
            first_frame = max(0, seed_frame - int(config['seed_pad_before']))
            last_frame = min(total_frames - 1, seed_frame + int(config['seed_pad_after']))
    elif hint_track is not None:
        seed_frame = hint_track.seed_frame
        guess_xy = hint_track.seed_xy
        first_frame = max(0, hint_track.first_frame - int(config['seed_pad_before']))
        last_frame = min(total_frames - 1, hint_track.last_frame + int(config['seed_pad_after']))
    else:
        raise RuntimeError(f'No seed found for {video_path}')
    reader = imageio.get_reader(video_path)
    reader_meta = reader.get_meta_data()
    particle_overlay_layer = Image.new('RGBA', (int(meta['width']), int(meta['height'])), (0, 0, 0, 0))
    seed_detect = detect_candidate(particle_type=particle_type, frame=normalize_frame(reader.get_data(seed_frame), reader_meta)[0], bg=bg, roi_mask=roi_mask, roi_bbox=roi_bbox, guess_xy=guess_xy, window_px=int(config['window_px']), search_radius_px=int(config['search_radius_px']))
    if seed_detect is None:
        reader.close()
        raise RuntimeError(f'Seed refinement failed for {video_path}')
    seed_x, seed_y, seed_score = seed_detect
    if seed_score < float(config['seed_min_score']):
        reader.close()
        raise RuntimeError(f'Seed score too low for {video_path}: {seed_score:.1f}')
    points: Dict[int, TrackPoint] = {seed_frame: TrackPoint(frame_idx=seed_frame, x_px=seed_x, y_px=seed_y, score=seed_score)}
    prev_x, prev_y = (seed_x, seed_y)
    misses = 0
    for frame_idx in range(seed_frame + 1, last_frame + 1):
        frame = normalize_frame(reader.get_data(frame_idx), reader_meta)[0]
        guess = (prev_x, prev_y)
        if hint_track is not None and frame_idx in hint_track.frame_to_xy:
            hx, hy = hint_track.frame_to_xy[frame_idx]
            guess = ((guess[0] + hx) * 0.5, (guess[1] + hy) * 0.5)
        found = detect_candidate(particle_type=particle_type, frame=frame, bg=bg, roi_mask=roi_mask, roi_bbox=roi_bbox, guess_xy=guess, window_px=int(config['window_px']), search_radius_px=int(config['search_radius_px'] + misses * config['search_expand_px']))
        if found is None or found[2] < float(config['min_score']):
            misses += 1
            if misses >= int(config['max_misses']):
                break
            continue
        prev_x, prev_y, score = found
        points[frame_idx] = TrackPoint(frame_idx=frame_idx, x_px=prev_x, y_px=prev_y, score=score)
        misses = 0
    prev_x, prev_y = (seed_x, seed_y)
    misses = 0
    for frame_idx in range(seed_frame - 1, first_frame - 1, -1):
        frame = normalize_frame(reader.get_data(frame_idx), reader_meta)[0]
        guess = (prev_x, prev_y)
        if hint_track is not None and frame_idx in hint_track.frame_to_xy:
            hx, hy = hint_track.frame_to_xy[frame_idx]
            guess = ((guess[0] + hx) * 0.5, (guess[1] + hy) * 0.5)
        found = detect_candidate(particle_type=particle_type, frame=frame, bg=bg, roi_mask=roi_mask, roi_bbox=roi_bbox, guess_xy=guess, window_px=int(config['window_px']), search_radius_px=int(config['search_radius_px'] + misses * config['search_expand_px']))
        if found is None or found[2] < float(config['min_score']):
            misses += 1
            if misses >= int(config['max_misses']):
                break
            continue
        prev_x, prev_y, score = found
        points[frame_idx] = TrackPoint(frame_idx=frame_idx, x_px=prev_x, y_px=prev_y, score=score)
        misses = 0
    detected_frames = sorted(points.keys())
    if not detected_frames:
        reader.close()
        raise RuntimeError(f'No accepted detections for {video_path}')
    first_detected = detected_frames[0]
    last_detected = detected_frames[-1]
    dense_points: Dict[int, TrackPoint] = {}
    prev_x = points[first_detected].x_px
    prev_y = points[first_detected].y_px
    misses = 0
    for frame_idx in range(first_detected, last_detected + 1):
        frame = normalize_frame(reader.get_data(frame_idx), reader_meta)[0]
        if frame_idx in points:
            base_point = points[frame_idx]
            dense_points[frame_idx] = TrackPoint(frame_idx=frame_idx, x_px=base_point.x_px, y_px=base_point.y_px, score=base_point.score, is_interpolated=False)
            prev_x, prev_y = (base_point.x_px, base_point.y_px)
            misses = 0
            stamp, xy = extract_particle_stamp(frame=frame, bg=bg, particle_type=particle_type, cx=base_point.x_px, cy=base_point.y_px)
            if stamp is not None and xy is not None:
                particle_overlay_layer.paste(stamp, xy, stamp)
        else:
            guess = (prev_x, prev_y)
            if hint_track is not None and frame_idx in hint_track.frame_to_xy:
                hx, hy = hint_track.frame_to_xy[frame_idx]
                guess = ((guess[0] + hx) * 0.5, (guess[1] + hy) * 0.5)
            found = detect_candidate(particle_type=particle_type, frame=frame, bg=bg, roi_mask=roi_mask, roi_bbox=roi_bbox, guess_xy=guess, window_px=int(config['window_px']), search_radius_px=int(config['search_radius_px'] + misses * config['search_expand_px']))
            if found is None or found[2] < float(config['min_score']) * 0.65:
                misses += 1
                continue
            prev_x, prev_y, score = found
            dense_points[frame_idx] = TrackPoint(frame_idx=frame_idx, x_px=prev_x, y_px=prev_y, score=score, is_interpolated=False)
            stamp, xy = extract_particle_stamp(frame=frame, bg=bg, particle_type=particle_type, cx=prev_x, cy=prev_y)
            if stamp is not None and xy is not None:
                particle_overlay_layer.paste(stamp, xy, stamp)
            misses = 0
    reader.close()
    dense_frames = list(range(first_detected, last_detected + 1))
    detected_dense_frames = sorted(dense_points.keys())
    if not detected_dense_frames:
        raise RuntimeError(f'No dense detections for {video_path}')
    completed_points: List[TrackPoint] = []
    for frame_idx in dense_frames:
        if frame_idx in dense_points:
            completed_points.append(dense_points[frame_idx])
            continue
        prev_candidates = [f for f in detected_dense_frames if f < frame_idx]
        next_candidates = [f for f in detected_dense_frames if f > frame_idx]
        if not prev_candidates or not next_candidates:
            continue
        prev_frame = prev_candidates[-1]
        next_frame = next_candidates[0]
        p0 = dense_points[prev_frame]
        p1 = dense_points[next_frame]
        alpha = (frame_idx - prev_frame) / float(next_frame - prev_frame)
        completed_points.append(TrackPoint(frame_idx=frame_idx, x_px=float((1.0 - alpha) * p0.x_px + alpha * p1.x_px), y_px=float((1.0 - alpha) * p0.y_px + alpha * p1.y_px), score=0.0, is_interpolated=True))
    completed_points.sort(key=lambda p: p.frame_idx)
    xs = np.array([point.x_px for point in completed_points], dtype=np.float32)
    ys = np.array([point.y_px for point in completed_points], dtype=np.float32)
    xs_smooth = moving_average(xs, 7)
    ys_smooth = moving_average(ys, 7)
    ordered_points: List[TrackPoint] = []
    for idx, point in enumerate(completed_points):
        point.x_smooth_px = float(xs_smooth[idx])
        point.y_smooth_px = float(ys_smooth[idx])
        ix = int(round(point.x_smooth_px))
        iy = int(round(point.y_smooth_px))
        if 0 <= ix < valid_center_mask.shape[1] and 0 <= iy < valid_center_mask.shape[0]:
            if valid_center_mask[iy, ix]:
                ordered_points.append(point)
    if not ordered_points:
        raise RuntimeError(f'No valid centres survived trimming for {video_path}')
    run_name = f'{slugify_video_path(video_path)}_raw_retracked'
    run_dir = os.path.join(out_dir, particle_type, run_name)
    ensure_dir(run_dir)
    bg_uint8 = np.clip(bg, 0, 255).astype(np.uint8)
    Image.fromarray(bg_uint8).save(os.path.join(run_dir, 'average_background.png'))
    save_per_run_overlay(background=bg_uint8, points=ordered_points, particle_type=particle_type, output_path=os.path.join(run_dir, 'journey_overlay.png'), title=os.path.basename(video_path))
    particle_base = enhance_background(bg_uint8).convert('RGBA')
    particle_composed = Image.alpha_composite(particle_base, particle_overlay_layer).convert('RGB')
    particle_draw = ImageDraw.Draw(particle_composed)
    draw_text(particle_draw, (24, 24), f'{os.path.basename(video_path)} isolated particle positions', (255, 255, 255))
    particle_composed.save(os.path.join(run_dir, 'isolated_particle_positions_overlay.png'))
    write_points_csv(points=ordered_points, fps=fps, output_path=os.path.join(run_dir, 'trajectory_points.csv'))
    return {'video_path': video_path, 'particle_type': particle_type, 'fps': fps, 'run_name': run_name, 'run_dir': run_dir, 'background': bg_uint8, 'background_float': bg, 'particle_overlay_layer': particle_overlay_layer, 'points': ordered_points, 'hint_folder': None if hint_track is None else hint_track.folder_name, 'seed_frame': seed_frame, 'first_frame': ordered_points[0].frame_idx, 'last_frame': ordered_points[-1].frame_idx}

def write_points_csv(points: Sequence[TrackPoint], fps: float, output_path: str) -> None:
    """Write the points csv used by this script."""
    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['frame_idx', 'time_s', 'x_px', 'y_px', 'x_smooth_px', 'y_smooth_px', 'score', 'is_interpolated'])
        for point in points:
            writer.writerow([point.frame_idx, point.frame_idx / fps, f'{point.x_px:.4f}', f'{point.y_px:.4f}', f'{point.x_smooth_px:.4f}', f'{point.y_smooth_px:.4f}', f'{point.score:.4f}', int(point.is_interpolated)])

def enhance_background(background: np.ndarray) -> Image.Image:
    """Enhance the background used by this script."""
    image = Image.fromarray(background).convert('RGB')
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.45)
    image = ImageEnhance.Brightness(image).enhance(0.96)
    return image

def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    """Handle the draw text step used by this script."""
    font = ImageFont.load_default()
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=fill, font=font)

def save_per_run_overlay(background: np.ndarray, points: Sequence[TrackPoint], particle_type: str, output_path: str, title: str) -> None:
    """Save the per run overlay used by this script."""
    image = enhance_background(background).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = RUN_COLORS[0]
    coords = [(p.x_smooth_px, p.y_smooth_px) for p in points]
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=7)
        draw.line(coords, fill=(255, 255, 255, 180), width=3)
    marker_fill = TYPE_CONFIG[particle_type]['particle_fill']
    marker_radius = int(TYPE_CONFIG[particle_type]['marker_radius'])
    for idx, point in enumerate(points):
        if idx % 10 != 0 and idx not in (0, len(points) - 1):
            continue
        x = point.x_smooth_px
        y = point.y_smooth_px
        draw.ellipse((x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius), fill=marker_fill, outline=color, width=3)
    composed = Image.alpha_composite(image, overlay).convert('RGB')
    draw_out = ImageDraw.Draw(composed)
    draw_text(draw_out, (24, 24), title, (255, 255, 255))
    composed.save(output_path)

def extract_particle_stamp(frame: np.ndarray, bg: np.ndarray, particle_type: str, cx: float, cy: float) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int]]]:
    """Extract the particle stamp used by this script."""
    patch_radius = 26 if particle_type == '10mm' else 34
    x0 = max(0, int(math.floor(cx)) - patch_radius)
    y0 = max(0, int(math.floor(cy)) - patch_radius)
    x1 = min(frame.shape[1], int(math.floor(cx)) + patch_radius + 1)
    y1 = min(frame.shape[0], int(math.floor(cy)) + patch_radius + 1)
    if x1 <= x0 or y1 <= y0:
        return (None, None)
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    crop_bg = bg[y0:y1, x0:x1]
    local_cx = float(cx - x0)
    local_cy = float(cy - y0)
    yy, xx = np.indices((y1 - y0, x1 - x0))
    radial_keep = (xx - local_cx) ** 2 + (yy - local_cy) ** 2 <= float((patch_radius + 2) ** 2)
    if particle_type == '10mm':
        red_excess = crop[..., 0] - 0.55 * crop[..., 1] - 0.55 * crop[..., 2]
        bg_delta = crop[..., 0] - crop_bg[..., 0]
        score = np.maximum(red_excess - 8.0, 0.0) + 0.3 * np.maximum(bg_delta - 6.0, 0.0)
        threshold = max(10.0, float(score.max()) * 0.34)
        rgb = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        rgb[..., 0] = 235
        rgb[..., 1] = 48
        rgb[..., 2] = 48
    else:
        gray = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
        bg_gray = 0.299 * crop_bg[..., 0] + 0.587 * crop_bg[..., 1] + 0.114 * crop_bg[..., 2]
        sat = np.max(crop, axis=2) - np.min(crop, axis=2)
        bright = np.maximum(gray - bg_gray - 2.0, 0.0)
        score = bright * np.clip((70.0 - sat) / 70.0, 0.0, 1.0)
        threshold = max(8.0, float(score.max()) * 0.42)
        rgb = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        rgb[..., 1] = 255
        rgb[..., 2] = 255
    mask = (score >= threshold) & radial_keep
    if not np.any(mask):
        return (None, None)
    alpha = np.zeros_like(score, dtype=np.uint8)
    max_score = max(float(score.max()), threshold + 1.0)
    alpha_float = np.clip((score - threshold) / (max_score - threshold + 1e-06), 0.0, 1.0)
    alpha[mask] = np.clip(110 + 145 * alpha_float[mask], 0, 255).astype(np.uint8)
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return (Image.fromarray(rgba, mode='RGBA'), (x0, y0))

def save_type_overlay(particle_type: str, results: Sequence[Dict[str, object]], output_dir: str) -> None:
    """Save the type overlay used by this script."""
    backgrounds = [np.array(result['background'], dtype=np.float32) for result in results]
    mean_bg = np.mean(np.stack(backgrounds, axis=0), axis=0)
    background = np.clip(mean_bg, 0, 255).astype(np.uint8)
    image = enhance_background(background).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for idx, result in enumerate(results):
        points = result['points']
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        coords = [(p.x_smooth_px, p.y_smooth_px) for p in points]
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=7)
            draw.line(coords, fill=(255, 255, 255, 150), width=3)
        marker_fill = TYPE_CONFIG[particle_type]['particle_fill']
        marker_radius = int(TYPE_CONFIG[particle_type]['marker_radius'])
        for jdx, point in enumerate(points):
            if jdx % 14 != 0 and jdx not in (0, len(points) - 1):
                continue
            x = point.x_smooth_px
            y = point.y_smooth_px
            draw.ellipse((x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius), fill=marker_fill, outline=color, width=3)
        end = coords[-1]
        draw_text(draw, (int(end[0]) + 10, int(end[1]) - 10), str(idx + 1), (255, 255, 255))
    composed = Image.alpha_composite(image, overlay).convert('RGB')
    draw_out = ImageDraw.Draw(composed)
    title = f'Straight-view {particle_type} sphere journeys (all 6 runs)'
    subtitle = 'High-contrast average background with full re-tracked journeys'
    draw_text(draw_out, (24, 24), title, (255, 255, 255))
    draw_text(draw_out, (24, 52), subtitle, (255, 255, 255))
    overlay_path = os.path.join(output_dir, f'{particle_type}_all_6_journeys_overlay.png')
    composed.save(overlay_path)
    Image.fromarray(background).save(os.path.join(output_dir, f'{particle_type}_average_background.png'))
    particle_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    for result in results:
        particle_overlay.alpha_composite(result['particle_overlay_layer'])
    particle_composed = Image.alpha_composite(image, particle_overlay).convert('RGB')
    draw_particle = ImageDraw.Draw(particle_composed)
    draw_text(draw_particle, (24, 24), title, (255, 255, 255))
    draw_text(draw_particle, (24, 52), 'All isolated particle positions on pooled average background', (255, 255, 255))
    particle_composed.save(os.path.join(output_dir, f'{particle_type}_all_6_particle_positions_overlay.png'))
    write_slip_summary(particle_type=particle_type, results=results, output_dir=output_dir)

def write_slip_summary(particle_type: str, results: Sequence[Dict[str, object]], output_dir: str) -> None:
    """Write the slip summary used by this script."""
    summary_path = os.path.join(output_dir, f'{particle_type}_journey_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['run_name', 'video_path', 'hint_folder', 'seed_frame', 'first_frame', 'last_frame', 'n_points', 'path_length_px', 'start_x_px', 'start_y_px', 'end_x_px', 'end_y_px'])
        for result in results:
            points = result['points']
            path = 0.0
            for p0, p1 in zip(points[:-1], points[1:]):
                path += math.hypot(p1.x_smooth_px - p0.x_smooth_px, p1.y_smooth_px - p0.y_smooth_px)
            writer.writerow([result['run_name'], result['video_path'], result['hint_folder'] or '', result['seed_frame'], result['first_frame'], result['last_frame'], len(points), f'{path:.4f}', f'{points[0].x_smooth_px:.4f}', f'{points[0].y_smooth_px:.4f}', f'{points[-1].x_smooth_px:.4f}', f'{points[-1].y_smooth_px:.4f}'])
    lateral_summary = os.path.join(output_dir, f'{particle_type}_lateral_slip_summary.csv')
    with open(lateral_summary, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['run_name', 'start_x_px', 'end_x_px', 'start_y_px', 'end_y_px', 'net_lateral_drift_px', 'max_lateral_span_px'])
        for result in results:
            points = result['points']
            xs = np.array([p.x_smooth_px for p in points], dtype=np.float32)
            ys = np.array([p.y_smooth_px for p in points], dtype=np.float32)
            writer.writerow([result['run_name'], f'{xs[0]:.4f}', f'{xs[-1]:.4f}', f'{ys[0]:.4f}', f'{ys[-1]:.4f}', f'{xs[-1] - xs[0]:.4f}', f'{xs.max() - xs.min():.4f}'])
    progress_csv = os.path.join(output_dir, f'{particle_type}_progress_aligned_spread.csv')
    lateral_progress_csv = os.path.join(output_dir, f'{particle_type}_lateral_slip_by_progress.csv')
    grid = np.linspace(0.0, 1.0, 101)
    interpolated_x = []
    interpolated_y = []
    run_names = []
    for result in results:
        points = result['points']
        coords = np.array([(p.x_smooth_px, p.y_smooth_px) for p in points], dtype=np.float32)
        step = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(step)])
        if float(s[-1]) <= 0.0:
            continue
        s_norm = s / s[-1]
        interpolated_x.append(np.interp(grid, s_norm, coords[:, 0]))
        interpolated_y.append(np.interp(grid, s_norm, coords[:, 1]))
        run_names.append(result['run_name'])
    if interpolated_x and interpolated_y:
        xs = np.stack(interpolated_x, axis=0)
        ys = np.stack(interpolated_y, axis=0)
        mean_x = xs.mean(axis=0)
        mean_y = ys.mean(axis=0)
        spread = np.sqrt(((xs - mean_x) ** 2 + (ys - mean_y) ** 2).mean(axis=0))
        with open(progress_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['progress', 'mean_x_px', 'mean_y_px', 'rms_spread_px'])
            for idx, g in enumerate(grid):
                writer.writerow([f'{g:.4f}', f'{mean_x[idx]:.4f}', f'{mean_y[idx]:.4f}', f'{spread[idx]:.4f}'])
        with open(lateral_progress_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            header = ['progress', 'mean_lateral_px'] + [f'{name}_lateral_px' for name in run_names]
            writer.writerow(header)
            for idx, g in enumerate(grid):
                row = [f'{g:.4f}', '0.0000']
                row.extend((f'{xs[ridx, idx] - mean_x[idx]:.4f}' for ridx in range(xs.shape[0])))
                writer.writerow(row)

def main() -> None:
    """Run the main entry point for this script."""
    ensure_dir(OUTPUT_ROOT)
    manifest = {'output_root': OUTPUT_ROOT, 'runs': [], 'particle_types': {}}
    for particle_type, videos in RAW_VIDEOS.items():
        type_dir = os.path.join(OUTPUT_ROOT, particle_type)
        ensure_dir(type_dir)
        results = []
        for video_path in videos:
            print(f'Tracking {video_path}')
            result = track_video(video_path=video_path, particle_type=particle_type, out_dir=OUTPUT_ROOT)
            results.append(result)
            manifest['runs'].append({'particle_type': particle_type, 'video_path': video_path, 'run_name': result['run_name'], 'run_dir': result['run_dir'], 'seed_frame': result['seed_frame'], 'first_frame': result['first_frame'], 'last_frame': result['last_frame'], 'n_points': len(result['points']), 'hint_folder': result['hint_folder']})
        save_type_overlay(particle_type=particle_type, results=results, output_dir=type_dir)
        manifest['particle_types'][particle_type] = {'n_runs': len(results), 'overlay_png': os.path.join(type_dir, f'{particle_type}_all_6_journeys_overlay.png'), 'summary_csv': os.path.join(type_dir, f'{particle_type}_journey_summary.csv'), 'spread_csv': os.path.join(type_dir, f'{particle_type}_progress_aligned_spread.csv')}
    with open(os.path.join(OUTPUT_ROOT, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
