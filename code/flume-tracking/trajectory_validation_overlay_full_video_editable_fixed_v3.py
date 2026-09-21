"""
Validation script: trajectory_validation_overlay_full_video_editable_fixed_v3.py

Purpose
-------
Builds full-video validation overlays by compositing selected particle positions and drawing the tracked trajectory on the still background.

Inputs
------
Tracker output CSV files and corresponding raw/binary videos.

Outputs
-------
Validation overlays and visual comparison figures.

Methodological notes
--------------------
The overlay verifies whether the numerical track follows the visible particle path rather than reflections, bubbles or hand motion.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import os
import cv2
import numpy as np
import pandas as pd
FOLDER_TO_ANALYSE = 'C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\10mm_sphere_release_2_tracker_output'
OUTPUT_DIR = 'C:\\Users\\User\\Downloads\\trajectory_validation_manual_outputs\\10mm_sphere_release_2_tracker_output'
USE_RAW = False
FRAME_STEP = 5
MIN_VISIBLE_POSITIONS = 5
MAX_VISIBLE_POSITIONS = 10
TARGET_VISIBLE_POSITIONS = 8
BACKGROUND_SAMPLES = 60
BACKGROUND_FROM_GRAYSCALE = False
MOTION_THRESHOLD = 25
PARTICLE_COLOUR_BGR = (255, 255, 255)
TRAJECTORY_LINE_COLOUR_BGR = (0, 0, 255)
TRAJECTORY_POINT_COLOUR_BGR = (0, 255, 255)
SAMPLE_BOX_COLOUR_BGR = (0, 255, 0)
PREFERRED_SOURCE_VIDEO = 'annotated_binary.mp4'

def _find_col(df, candidates):
    """Handle the  find col step used by this script."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def _prepare_tracks(df, use_raw=False):
    """Handle the  prepare tracks step used by this script."""
    frame_col = _find_col(df, ['frame', 'frame_idx', 'frame_index'])
    out_idx_col = _find_col(df, ['out_idx', 'output_idx', 'clip_frame', 'frame_in_clip'])
    if frame_col is None:
        raise ValueError(f'Could not find frame column. Columns were: {list(df.columns)}')
    if use_raw:
        x_candidates = ['cx_raw_px', 'raw_x', 'measured_x', 'detected_x', 'x_rectified_px', 'x_mm', 'x_kalman_mm']
        y_candidates = ['cy_raw_px', 'raw_y', 'measured_y', 'detected_y', 'y_rectified_px', 'y_mm', 'y_kalman_mm']
    else:
        x_candidates = ['cx_kalman_raw_px', 'kalman_x', 'x_kalman', 'tracked_x', 'cx_raw_px', 'x_rectified_px', 'x_kalman_mm', 'x_mm']
        y_candidates = ['cy_kalman_raw_px', 'kalman_y', 'y_kalman', 'tracked_y', 'cy_raw_px', 'y_rectified_px', 'y_kalman_mm', 'y_mm']
    x_col = _find_col(df, x_candidates)
    y_col = _find_col(df, y_candidates)
    if x_col is None or y_col is None:
        raise ValueError('Could not find usable x/y columns.\nColumns were: ' + str(list(df.columns)))
    keep = [frame_col, x_col, y_col]
    if out_idx_col:
        keep.append(out_idx_col)
    poly_cols = ['p0x_raw', 'p0y_raw', 'p1x_raw', 'p1y_raw', 'p2x_raw', 'p2y_raw', 'p3x_raw', 'p3y_raw']
    for c in poly_cols:
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy().rename(columns={frame_col: 'frame', x_col: 'track_x', y_col: 'track_y'})
    if out_idx_col:
        out = out.rename(columns={out_idx_col: 'out_idx'})
    out['frame'] = pd.to_numeric(out['frame'], errors='coerce')
    out['track_x'] = pd.to_numeric(out['track_x'], errors='coerce')
    out['track_y'] = pd.to_numeric(out['track_y'], errors='coerce')
    if 'out_idx' in out.columns:
        out['out_idx'] = pd.to_numeric(out['out_idx'], errors='coerce')
    for c in poly_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['frame', 'track_x', 'track_y']).copy()
    out['frame'] = out['frame'].astype(int)
    if 'out_idx' in out.columns:
        out = out.dropna(subset=['out_idx']).copy()
        out['out_idx'] = out['out_idx'].astype(int)
    return out

def _build_background(video_path, max_samples=60, frame_indices=None, grayscale=False, reducer='median'):
    """Handle the  build background step used by this script."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        raise RuntimeError('Video appears to have zero frames')
    if frame_indices is None:
        idxs = np.linspace(0, max(n_frames - 1, 0), min(max_samples, n_frames), dtype=int)
    else:
        unique_idxs = sorted({int(idx) for idx in frame_indices if 0 <= int(idx) < n_frames})
        if not unique_idxs:
            raise RuntimeError('No valid background frame indices were provided')
        if len(unique_idxs) > max_samples:
            pick = np.linspace(0, len(unique_idxs) - 1, max_samples, dtype=int)
            idxs = np.asarray([unique_idxs[i] for i in pick], dtype=int)
        else:
            idxs = np.asarray(unique_idxs, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError('Could not sample frames to build background')
    stack = np.stack(frames).astype(np.float32)
    if str(reducer).lower() == 'mean':
        background = np.mean(stack, axis=0)
    else:
        background = np.median(stack, axis=0)
    return np.clip(background, 0, 255).astype(np.uint8)

def _to_gray_bgr(frame_bgr):
    """Handle the  to gray bgr step used by this script."""
    if len(frame_bgr.shape) == 2:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _safe_box(row, width, height, pad=10):
    """Handle the  safe box step used by this script."""
    poly_x = [c for c in ['p0x_raw', 'p1x_raw', 'p2x_raw', 'p3x_raw'] if c in row.index and pd.notna(row[c])]
    poly_y = [c for c in ['p0y_raw', 'p1y_raw', 'p2y_raw', 'p3y_raw'] if c in row.index and pd.notna(row[c])]
    if len(poly_x) == 4 and len(poly_y) == 4:
        xs = [row[c] for c in poly_x]
        ys = [row[c] for c in poly_y]
        x1 = int(max(0, np.floor(min(xs) - pad)))
        y1 = int(max(0, np.floor(min(ys) - pad)))
        x2 = int(min(width, np.ceil(max(xs) + pad)))
        y2 = int(min(height, np.ceil(max(ys) + pad)))
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    x = int(round(row['track_x']))
    y = int(round(row['track_y']))
    return (max(0, x - 25), max(0, y - 25), min(width, x + 25), min(height, y + 25))

def _fit_min_area_box_global(contour, x1, y1):
    """Handle the  fit min area box global step used by this script."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += float(x1)
    box[:, 1] += float(y1)
    return box

def _order_box_points(pts):
    """Handle the  order box points step used by this script."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _box_axes(box):
    """Handle the  box axes step used by this script."""
    pts = np.asarray(box, dtype=np.float32)
    center = pts.mean(axis=0).astype(np.float32)
    edges = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        vec = p1 - p0
        length = float(np.hypot(vec[0], vec[1]))
        edges.append((length, vec))
    edges_sorted = sorted(edges, key=lambda item: item[0], reverse=True)
    long_vec = edges_sorted[0][1].astype(np.float32)
    short_vec = edges_sorted[-1][1].astype(np.float32)
    long_len = max(float(np.hypot(long_vec[0], long_vec[1])), 1e-09)
    short_len = max(float(np.hypot(short_vec[0], short_vec[1])), 1e-09)
    u_long = long_vec / long_len
    u_short = short_vec / short_len
    if np.cross(u_long, u_short) < 0:
        u_short = -u_short
    return (center, u_long, u_short, long_len, short_len)

def _box_from_axes(center, u_long, u_short, long_len, short_len):
    """Handle the  box from axes step used by this script."""
    hl = max(float(long_len) / 2.0, 1e-09)
    hs = max(float(short_len) / 2.0, 1e-09)
    pts = np.array([center - hl * u_long - hs * u_short, center + hl * u_long - hs * u_short, center + hl * u_long + hs * u_short, center - hl * u_long + hs * u_short], dtype=np.float32)
    return _order_box_points(pts)

def _regularize_sphere_box_global(contour, x1, y1, max_aspect_ratio=1.35, min_short_factor=0.88, max_long_factor=1.18, max_short_factor=1.08):
    """Handle the  regularize sphere box global step used by this script."""
    rect = cv2.minAreaRect(contour)
    local_box = _order_box_points(cv2.boxPoints(rect))
    center, u_long, u_short, long_len_raw, short_len_raw = _box_axes(local_box)
    (ccx, ccy), circle_radius = cv2.minEnclosingCircle(contour)
    area = float(cv2.contourArea(contour))
    circle_d = max(2.0 * float(circle_radius), 1e-06)
    eq_d = max(2.0 * np.sqrt(max(area, 1e-09) / np.pi), 1e-06)
    d_lo = min(eq_d, circle_d)
    d_hi = max(eq_d, circle_d)
    base_d = float(np.clip(0.55 * eq_d + 0.45 * circle_d, 0.9 * d_lo, 1.08 * d_hi))
    long_len = max(float(long_len_raw), 0.95 * base_d)
    long_len = min(long_len, max_long_factor * base_d)
    short_len = max(float(short_len_raw), min_short_factor * base_d)
    short_len = min(short_len, max_short_factor * base_d)
    short_len = max(short_len, long_len / max(max_aspect_ratio, 1.0))
    if short_len > max_short_factor * base_d:
        short_len = max_short_factor * base_d
        long_len = min(long_len, short_len * max_aspect_ratio)
    if long_len / max(short_len, 1e-09) > max_aspect_ratio:
        long_len = short_len * max_aspect_ratio
    reg_local = _box_from_axes(center, u_long, u_short, long_len, short_len)
    reg_global = reg_local.copy()
    reg_global[:, 0] += float(x1)
    reg_global[:, 1] += float(y1)
    return (reg_global, float(ccx + x1), float(ccy + y1))

def _extract_particle_mask(frame, background, row, motion_threshold=25, background_is_grayscale=False):
    """Handle the  extract particle mask step used by this script."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _safe_box(row, w, h, pad=10)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return (None, None)
    if background_is_grayscale:
        bg_roi = background[y1:y2, x1:x2]
        if bg_roi.size == 0:
            return (None, None)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.absdiff(roi_gray, bg_roi)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        bg_roi = background[y1:y2, x1:x2]
        if bg_roi.size == 0:
            return (None, None)
        diff = cv2.absdiff(roi, bg_roi)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, motion_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, None)
    tx = row['track_x'] - x1
    ty = row['track_y'] - y1
    best = None
    best_score = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 1e-09:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        solidity = float(area / hull_area) if hull_area > 1e-09 else 0.0
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = max(float(max(bw, bh)) / max(float(min(bw, bh)), 1.0), 1.0)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist2 = (cx - tx) ** 2 + (cy - ty) ** 2
        shape_penalty = 180.0 * max(0.0, 0.72 - circularity) ** 2 + 80.0 * max(0.0, 0.84 - solidity) ** 2 + 18.0 * max(0.0, aspect_ratio - 1.35) ** 2
        score = dist2 + shape_penalty - 0.08 * area
        if best is None or score < best_score:
            best = cnt
            best_score = score
    if best is None:
        return (None, None)
    particle_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(particle_mask, [best], -1, 255, thickness=-1)
    box_global, cx, cy = _regularize_sphere_box_global(best, x1, y1)
    return (particle_mask, {'cx': float(cx), 'cy': float(cy), 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'box_global': box_global})

def _make_binary_particle_pixels(mask, colour_bgr):
    """Handle the  make binary particle pixels step used by this script."""
    particle_pixels = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    particle_pixels[mask > 0] = np.asarray(colour_bgr, dtype=np.uint8)
    return particle_pixels

def _masked_source_pixels(frame_roi, mask, fallback_colour):
    """Handle the  masked source pixels step used by this script."""
    particle_pixels = np.zeros_like(frame_roi)
    if frame_roi.size == 0:
        return particle_pixels
    particle_pixels[mask > 0] = frame_roi[mask > 0]
    if not np.any(particle_pixels[mask > 0]):
        particle_pixels = _make_binary_particle_pixels(mask, fallback_colour)
    return particle_pixels

def _layout_bounds(points, sample_records, pad=48):
    """Handle the  layout bounds step used by this script."""
    xs = []
    ys = []
    for x, y in points:
        xs.append(float(x))
        ys.append(float(y))
    for rec in sample_records:
        box = rec.get('box_global')
        if box is not None and len(box) > 0:
            xs.extend((float(v) for v in box[:, 0]))
            ys.extend((float(v) for v in box[:, 1]))
        else:
            xs.extend([float(rec['cx']), float(rec['cx'])])
            ys.extend([float(rec['cy']), float(rec['cy'])])
    if not xs or not ys:
        return (0, 0, 512, 512)
    min_x = int(np.floor(min(xs))) - pad
    min_y = int(np.floor(min(ys))) - pad
    max_x = int(np.ceil(max(xs))) + pad
    max_y = int(np.ceil(max(ys))) + pad
    width = max(256, max_x - min_x + 1)
    height = max(256, max_y - min_y + 1)
    return (min_x, min_y, width, height)

def _paste_max(canvas, patch, x, y):
    """Handle the  paste max step used by this script."""
    if patch is None or patch.size == 0:
        return
    h, w = patch.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(canvas.shape[1], int(x + w))
    y2 = min(canvas.shape[0], int(y + h))
    if x2 <= x1 or y2 <= y1:
        return
    src_x1 = x1 - int(x)
    src_y1 = y1 - int(y)
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)
    roi = canvas[y1:y2, x1:x2]
    np.maximum(roi, patch[src_y1:src_y2, src_x1:src_x2], out=roi)

def _draw_layout_black_background(path_points, sample_records, line_colour, point_colour):
    """Handle the  draw layout black background step used by this script."""
    min_x, min_y, width, height = _layout_bounds(path_points, sample_records)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if path_points:
        pts = np.array([[int(round(x - min_x)), int(round(y - min_y))] for x, y in path_points], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, line_colour, 2, lineType=cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(canvas, (int(x), int(y)), 2, point_colour, thickness=-1, lineType=cv2.LINE_AA)
    for idx, rec in enumerate(sample_records, start=1):
        patch = rec.get('particle_pixels')
        x1 = int(rec.get('x1', 0) - min_x)
        y1 = int(rec.get('y1', 0) - min_y)
        _paste_max(canvas, patch, x1, y1)
        box = rec.get('box_global')
        if box is not None:
            box_shifted = np.round(box - np.array([[min_x, min_y]], dtype=np.float32)).astype(np.int32)
            cv2.polylines(canvas, [box_shifted], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cx = int(round(float(rec['cx']) - min_x))
        cy = int(round(float(rec['cy']) - min_y))
        cv2.circle(canvas, (cx, cy), 3, point_colour, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, lineType=cv2.LINE_AA)
    return canvas

def _choose_source_video(folder):
    """Handle the  choose source video step used by this script."""
    preferred = os.path.join(folder, PREFERRED_SOURCE_VIDEO)
    fallback = os.path.join(folder, 'annotated_raw.mp4')
    if os.path.isfile(preferred):
        return (preferred, os.path.basename(preferred))
    if os.path.isfile(fallback):
        return (fallback, os.path.basename(fallback))
    raise FileNotFoundError(f'Folder must contain {PREFERRED_SOURCE_VIDEO} or annotated_raw.mp4: {folder}')

def _select_sample_keys(tracks_sorted, key_col, min_count=5, max_count=10, preferred_count=8):
    """Handle the  select sample keys step used by this script."""
    if tracks_sorted.empty:
        return ([], 0)
    n_points = int(len(tracks_sorted))
    if n_points <= min_count:
        sampled = tracks_sorted[key_col].astype(int).tolist()
        return (sampled, len(sampled))
    auto_count = int(round(n_points / 60.0))
    target_count = max(min_count, min(max_count, auto_count if auto_count > 0 else preferred_count))
    target_count = min(target_count, max_count, n_points)
    target_count = max(min_count if n_points >= min_count else n_points, target_count)
    if preferred_count:
        target_count = min(target_count, preferred_count) if n_points < preferred_count else max(target_count, preferred_count)
        target_count = min(target_count, max_count, n_points)
    sample_idxs = np.linspace(0, n_points - 1, target_count, dtype=int)
    sample_idxs = np.unique(sample_idxs)
    sampled = tracks_sorted.iloc[sample_idxs][key_col].astype(int).tolist()
    return (sampled, int(len(sampled)))

def _draw_trajectory_bgra(height, width, points, frame_step, line_colour, point_colour):
    """Handle the  draw trajectory bgra step used by this script."""
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    if not points:
        return overlay
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    bgr = np.ascontiguousarray(overlay[:, :, :3])
    alpha = np.ascontiguousarray(overlay[:, :, 3])
    if len(pts) >= 2:
        cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], False, line_colour, 2)
        cv2.polylines(alpha, [pts.reshape(-1, 1, 2)], False, 255, 2)
    sampled = pts[::max(1, frame_step)]
    for x, y in sampled:
        cv2.circle(bgr, (int(x), int(y)), 2, point_colour, -1)
        cv2.circle(alpha, (int(x), int(y)), 2, 255, -1)
    overlay[:, :, :3] = bgr
    overlay[:, :, 3] = alpha
    return overlay

def _alpha_blend_bgra(base_bgr, overlay_bgra):
    """Handle the  alpha blend bgra step used by this script."""
    if overlay_bgra.shape[:2] != base_bgr.shape[:2]:
        raise ValueError('Base image and overlay must have the same height and width')
    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay_bgra[:, :, :3].astype(np.float32)
    base = base_bgr.astype(np.float32)
    blended = overlay_bgr * alpha + base * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

def make_validation_visualisation_full_video(folder, out_dir=None, use_raw=False, frame_step=5, tracks_folder=None, source_folder=None, source_video_path=None, csv_path=None, background_frame_indices=None, background_use_grayscale=False, background_reducer='median', binary_background_only=False, particle_render_mode='source_pixels', sample_from_detected_positions_only=False, save_layout_black_background=True):
    """Handle the make validation visualisation full video step used by this script."""
    trajectory_folder = tracks_folder or folder
    binary_source_folder = source_folder or folder
    if source_video_path is None:
        video_path, source_video_name = _choose_source_video(binary_source_folder)
    else:
        video_path = source_video_path
        source_video_name = os.path.basename(source_video_path)
    if csv_path is None:
        csv_path = os.path.join(trajectory_folder, 'tracks_processed.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Folder must contain tracks_processed.csv: {trajectory_folder}')
    if out_dir is None:
        out_dir = os.path.join(folder, 'trajectory_validation_output_full_video')
    os.makedirs(out_dir, exist_ok=True)
    tracks = _prepare_tracks(pd.read_csv(csv_path), use_raw=use_raw)
    background_raw = _build_background(video_path, max_samples=BACKGROUND_SAMPLES, frame_indices=background_frame_indices, grayscale=background_use_grayscale, reducer=background_reducer)
    if background_use_grayscale:
        background_display = _to_gray_bgr(background_raw)
    else:
        background_display = _to_gray_bgr(background_raw) if BACKGROUND_FROM_GRAYSCALE else background_raw.copy()
    binary_composite = np.zeros_like(background_display) if binary_background_only else background_display.copy()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    use_out_idx = False
    if 'out_idx' in tracks.columns:
        frame_in_range = ((tracks['frame'] >= 0) & (tracks['frame'] < n_frames)).sum()
        outidx_in_range = ((tracks['out_idx'] >= 0) & (tracks['out_idx'] < n_frames)).sum()
        use_out_idx = outidx_in_range >= frame_in_range
    key_col = 'out_idx' if use_out_idx else 'frame'
    tracks_sorted = tracks.sort_values(key_col).drop_duplicates(subset=[key_col], keep='first').copy()
    frame_lookup = {int(f): g.iloc[0] for f, g in tracks_sorted.groupby(key_col)}
    all_track_points = [(float(r['track_x']), float(r['track_y'])) for _, r in tracks_sorted.iterrows()]
    sample_keys, target_sample_count = _select_sample_keys(tracks_sorted, key_col, min_count=MIN_VISIBLE_POSITIONS, max_count=MAX_VISIBLE_POSITIONS, preferred_count=TARGET_VISIBLE_POSITIONS)
    sample_strategy = 'adaptive_even_spacing_5_to_10_positions'
    candidate_detected_frames_total = np.nan
    if sample_from_detected_positions_only:
        candidate_records = []
        cap_candidates = cv2.VideoCapture(video_path)
        if not cap_candidates.isOpened():
            raise RuntimeError(f'Could not open video: {video_path}')
        candidate_keys = set((int(k) for k in tracks_sorted[key_col].astype(int).tolist()))
        try:
            for i in range(n_frames):
                ok, frame = cap_candidates.read()
                if not ok:
                    break
                if i not in candidate_keys:
                    continue
                row = frame_lookup.get(i)
                if row is None:
                    continue
                particle_mask, info = _extract_particle_mask(frame, background_raw, row, motion_threshold=MOTION_THRESHOLD, background_is_grayscale=background_use_grayscale)
                if particle_mask is None:
                    continue
                candidate_records.append({'sample_key': int(i), 'track_x': float(row['track_x']), 'track_y': float(row['track_y']), 'detected_x': float(info['cx']), 'detected_y': float(info['cy']), 'pixel_error': float(np.hypot(float(info['cx']) - row['track_x'], float(info['cy']) - row['track_y']))})
        finally:
            cap_candidates.release()
        candidate_detected_frames_total = int(len(candidate_records))
        if candidate_records:
            candidate_df = pd.DataFrame(candidate_records)
            sample_keys, target_sample_count = _select_sample_keys(candidate_df, 'sample_key', min_count=MIN_VISIBLE_POSITIONS, max_count=MAX_VISIBLE_POSITIONS, preferred_count=TARGET_VISIBLE_POSITIONS)
            sample_strategy = 'adaptive_even_spacing_from_successful_background_subtracted_detections'
        else:
            sample_keys = []
            target_sample_count = 0
            sample_strategy = 'adaptive_even_spacing_from_successful_background_subtracted_detections'
    sample_key_set = set(sample_keys)
    errors = []
    sampled_track_points = []
    visible_frames = 0
    detected_frames = 0
    sample_render_records = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i not in frame_lookup:
            continue
        if i not in sample_key_set:
            continue
        row = frame_lookup[i]
        visible_frames += 1
        sampled_track_points.append((float(row['track_x']), float(row['track_y'])))
        particle_mask, info = _extract_particle_mask(frame, background_raw, row, motion_threshold=MOTION_THRESHOLD, background_is_grayscale=background_use_grayscale)
        if particle_mask is None:
            errors.append({'frame': i, 'track_x': float(row['track_x']), 'track_y': float(row['track_y']), 'detected_x': np.nan, 'detected_y': np.nan, 'pixel_error': np.nan, 'detected': 0})
            continue
        detected_frames += 1
        cx = float(info['cx'])
        cy = float(info['cy'])
        x1 = int(info['x1'])
        y1 = int(info['y1'])
        x2 = int(info['x2'])
        y2 = int(info['y2'])
        frame_roi = frame[y1:y2, x1:x2]
        if str(particle_render_mode).lower() == 'binary':
            particle_pixels = _make_binary_particle_pixels(particle_mask, PARTICLE_COLOUR_BGR)
        else:
            particle_pixels = _masked_source_pixels(frame_roi, particle_mask, PARTICLE_COLOUR_BGR)
        roi_comp = binary_composite[y1:y2, x1:x2]
        roi_comp[particle_mask > 0] = particle_pixels[particle_mask > 0]
        sample_render_records.append({'frame': int(i), 'cx': cx, 'cy': cy, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'box_global': info.get('box_global'), 'particle_pixels': particle_pixels.copy()})
        box_global = info.get('box_global')
        if box_global is not None:
            box_i = np.round(box_global).astype(np.int32)
            cv2.polylines(binary_composite, [box_i], True, SAMPLE_BOX_COLOUR_BGR, 2, lineType=cv2.LINE_AA)
        mark_x = int(round(cx))
        mark_y = int(round(cy))
        sample_number = len(sample_render_records)
        cv2.circle(binary_composite, (mark_x, mark_y), 3, TRAJECTORY_POINT_COLOUR_BGR, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(binary_composite, str(sample_number), (mark_x + 6, mark_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, lineType=cv2.LINE_AA)
        errors.append({'frame': i, 'track_x': float(row['track_x']), 'track_y': float(row['track_y']), 'detected_x': cx, 'detected_y': cy, 'pixel_error': float(np.hypot(cx - row['track_x'], cy - row['track_y'])), 'detected': 1})
    cap.release()
    trajectory_overlay_bgra = _draw_trajectory_bgra(height, width, all_track_points, frame_step=max(1, int(frame_step)), line_colour=TRAJECTORY_LINE_COLOUR_BGR, point_colour=TRAJECTORY_POINT_COLOUR_BGR)
    combined = _alpha_blend_bgra(binary_composite, trajectory_overlay_bgra)
    if errors:
        errors_df = pd.DataFrame(errors)
    else:
        errors_df = pd.DataFrame(columns=['frame', 'track_x', 'track_y', 'detected_x', 'detected_y', 'pixel_error', 'detected'])
    errors_path = os.path.join(out_dir, 'trajectory_validation_errors_full_video.csv')
    errors_df.to_csv(errors_path, index=False)
    has_valid_error = len(errors_df) > 0 and 'pixel_error' in errors_df.columns and errors_df['pixel_error'].notna().any()
    summary = {'trial_name': os.path.basename(trajectory_folder), 'video_frames': int(n_frames), 'visible_frames_used': int(visible_frames), 'detected_frames': int(detected_frames), 'detection_coverage_pct': float(100.0 * detected_frames / visible_frames) if visible_frames else np.nan, 'mean_pixel_error': float(errors_df['pixel_error'].dropna().mean()) if has_valid_error else np.nan, 'median_pixel_error': float(errors_df['pixel_error'].dropna().median()) if has_valid_error else np.nan, 'max_pixel_error': float(errors_df['pixel_error'].dropna().max()) if has_valid_error else np.nan, 'frame_key_used': key_col, 'frame_step': int(frame_step), 'foreground_mode': 'sampled_binary_source_pixels_on_binary_background' if str(particle_render_mode).lower() != 'binary' else 'background_subtracted_binary_particles_on_clean_binary_background', 'source_video': source_video_name, 'trajectory_folder': trajectory_folder, 'binary_source_folder': binary_source_folder, 'sample_strategy': sample_strategy, 'sample_positions_target': int(target_sample_count), 'sampled_positions_used': int(visible_frames), 'trajectory_points_available': int(len(all_track_points)), 'trajectory_points_sampled': int(len(sampled_track_points)), 'background_mode': 'mean_grayscale_background_subtraction_from_explicit_no_particle_frames' if background_use_grayscale and str(background_reducer).lower() == 'mean' else f'{background_reducer}_background', 'background_frame_count_used': int(len(set((int(idx) for idx in background_frame_indices)))) if background_frame_indices is not None else np.nan, 'binary_background_only': int(bool(binary_background_only)), 'particle_render_mode': str(particle_render_mode), 'candidate_detected_frames_total': candidate_detected_frames_total, 'trajectory_point_source': 'tracker_positions_primary_found_segment'}
    summary_path = os.path.join(out_dir, 'trajectory_validation_summary_full_video.csv')
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    binary_composite_path = os.path.join(out_dir, 'true_visibility_composite_full_video.png')
    binary_composite_step_path = os.path.join(out_dir, f'true_visibility_binary_composite_{int(visible_frames)}_sampled_positions.png')
    trajectory_overlay_path = os.path.join(out_dir, 'tracked_trajectory_only_full_video.png')
    trajectory_overlay_step_path = os.path.join(out_dir, f'tracked_trajectory_overlay_{int(visible_frames)}_sampled_positions.png')
    combined_path = os.path.join(out_dir, 'true_vs_tracked_overlay_full_video.png')
    combined_step_path = os.path.join(out_dir, f'true_vs_tracked_overlay_{int(visible_frames)}_sampled_positions.png')
    cv2.imwrite(binary_composite_path, binary_composite)
    cv2.imwrite(binary_composite_step_path, binary_composite)
    cv2.imwrite(trajectory_overlay_path, trajectory_overlay_bgra)
    cv2.imwrite(trajectory_overlay_step_path, trajectory_overlay_bgra)
    cv2.imwrite(combined_path, combined)
    cv2.imwrite(combined_step_path, combined)
    layout_path = None
    layout_step_path = None
    if save_layout_black_background:
        layout = _draw_layout_black_background(path_points=all_track_points, sample_records=sample_render_records, line_colour=TRAJECTORY_LINE_COLOUR_BGR, point_colour=TRAJECTORY_POINT_COLOUR_BGR)
        layout_path = os.path.join(out_dir, 'trajectory_layout_black_background.png')
        layout_step_path = os.path.join(out_dir, f'trajectory_layout_black_background_{int(len(sample_render_records))}_sampled_positions.png')
        cv2.imwrite(layout_path, layout)
        cv2.imwrite(layout_step_path, layout)
    if layout_path is not None:
        summary['trajectory_layout_black_background'] = layout_path
    if layout_step_path is not None:
        summary['trajectory_layout_black_background_step'] = layout_step_path
    return (out_dir, summary)

def main():
    """Run the main entry point for this script."""
    if not FOLDER_TO_ANALYSE:
        raise ValueError('Edit FOLDER_TO_ANALYSE at the top of the script before running.')
    out_dir, summary = make_validation_visualisation_full_video(FOLDER_TO_ANALYSE, out_dir=OUTPUT_DIR, use_raw=USE_RAW, frame_step=FRAME_STEP)
    print(f'Saved outputs to: {out_dir}')
    print(pd.DataFrame([summary]).to_string(index=False))

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
