"""
Core tracking script: Kalman_Tracker_Spheres_binarized.py

Purpose
-------
Tracks red and white sphere trajectories using colour/grayscale segmentation and Kalman smoothing.

Inputs
------
Sphere videos (.mp4), calibration JSON, particle colour mode, seed frame and initial click/centre.

Outputs
-------
Per-frame trajectory CSV files, annotated tracking videos, binary masks and diagnostic plots.

Methodological notes
--------------------
Sphere centres are represented by fitted circular/contour centroids; motion is treated as planar in the calibrated camera view; high-frequency centroid noise is smoothed using a constant-velocity Kalman model.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import os
import json
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
USE_RUNTIME_PROMPTS = True
ENABLE_IMSHOW = True
DEFAULT_VIDEO_PATH = 'C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\14mm sphere release 1.mp4'
DEFAULT_CALIBRATION_JSON = 'C:\\Users\\User\\Downloads\\rod_calibration_snap_fit\\calibration.json'
DEFAULT_OUT_DIR = 'C:\\IP Work\\4.5_Inchps_Spheres_Binarised\\14mm_sphere_release_1_binarized_tracker_output'
DEFAULT_REAL_FPS = 240.0
DEFAULT_FRAME_STRIDE = 1
DEFAULT_START_FRAME = 0
DEFAULT_OBJECT_MODE = 'sphere'
DEFAULT_SPHERE_COLOR_MODE = 'red'
DEFAULT_ROD_DIAMETER_MM = None
DEFAULT_DISC_DIAMETER_MM = 20.0
DEFAULT_SPHERE_DIAMETER_MM = 10.0
DEFAULT_USE_PRESET_CLICK = False
DEFAULT_PRESET_CLICK = (500, 300)
DEFAULT_START_SEARCH_FRAMES = 25
DEFAULT_SAVE_DEBUG_IMAGES = True
DEFAULT_NUM_DEBUG_IMAGES = 12
DEFAULT_SMOOTH_WINDOW = 7
DEFAULT_ENABLE_SPHERE_BACKGROUND_SUBTRACTION = True
DEFAULT_SPHERE_BACKGROUND_SECONDS = 2.5
DEFAULT_SPHERE_BACKGROUND_SAMPLES = 24
DEFAULT_SPHERE_BACKGROUND_MARGIN_FRAMES = 24
DEFAULT_SPHERE_BACKGROUND_KEEP_FRACTION = 0.7
DEFAULT_SPHERE_BACKTRACK_MIN_PROGRESS_MM = 20.0
DEFAULT_SPHERE_BACKTRACK_TOL_X_MM = 6.0
DEFAULT_SPHERE_SEVERE_BACKTRACK_X_MM = 12.0
DEFAULT_SPHERE_TAIL_MISS_STOP = 8
# Centralised processing parameters. These values were tuned for the final reported videos.
PARAMS = {'rod': {'acquire': {'roi_half_size_px': 150, 'edge_low': 20, 'edge_high': 100, 'close_k': 5, 'close_iter': 1, 'min_area_px': 6, 'max_area_px': 40000, 'min_solidity': 0.02, 'min_aspect_ratio': 1.05, 'max_aspect_ratio': 50.0, 'min_length_mm': 3.0, 'max_length_mm': 50.0, 'max_assoc_dist_px': 95.0, 'max_misses': 20}, 'track': {'roi_half_size_px': 150, 'edge_low': 30, 'edge_high': 110, 'close_k': 3, 'close_iter': 1, 'min_area_px': 8, 'max_area_px': 30000, 'min_solidity': 0.04, 'min_aspect_ratio': 1.15, 'max_aspect_ratio': 35.0, 'min_length_mm': 4.0, 'max_length_mm': 60.0, 'max_assoc_dist_px': 85.0, 'max_misses': 15}}, 'disc': {'acquire': {'roi_half_size_px': 150, 'edge_low': 20, 'edge_high': 90, 'close_k': 9, 'close_iter': 3, 'min_area_px': 10, 'max_area_px': 60000, 'min_solidity': 0.1, 'min_circularity': 0.22, 'aspect_tol': 3.5, 'min_diameter_factor': 0.45, 'max_diameter_factor': 2.0, 'max_assoc_dist_px': 95.0, 'max_misses': 18}, 'track': {'roi_half_size_px': 120, 'edge_low': 25, 'edge_high': 100, 'close_k': 7, 'close_iter': 2, 'min_area_px': 14, 'max_area_px': 50000, 'min_solidity': 0.15, 'min_circularity': 0.3, 'aspect_tol': 2.8, 'min_diameter_factor': 0.55, 'max_diameter_factor': 1.7, 'max_assoc_dist_px': 75.0, 'max_misses': 15}}, 'sphere_red': {'acquire': {'roi_half_size_px': 180, 'hsv_lower1': [0, 70, 35], 'hsv_upper1': [16, 255, 255], 'hsv_lower2': [165, 70, 35], 'hsv_upper2': [180, 255, 255], 'binary_blur_k': 5, 'binary_bg_blur_k': 31, 'binary_min_threshold': 8, 'red_excess_blur_k': 5, 'red_excess_thresh': 28, 'use_otsu': True, 'open_k': 3, 'close_k': 7, 'close_iter': 2, 'min_area_px': 25, 'max_area_px': 12000, 'min_circularity': 0.45, 'min_solidity': 0.6, 'aspect_tol': 1.45, 'min_diameter_factor': 0.45, 'max_diameter_factor': 2.2, 'max_assoc_dist_px': 120.0, 'max_misses': 20, 'border_relax_px': 24, 'relaxed_min_area_px': 16, 'relaxed_min_circularity': 0.22, 'relaxed_min_solidity': 0.3, 'relaxed_aspect_tol': 2.25, 'relaxed_min_diameter_factor': 0.25, 'relaxed_max_diameter_factor': 2.4, 'roi_expand_per_miss_px': 40, 'assoc_expand_per_miss_px': 60, 'reacquire_fullframe_after_misses': 2, 'exit_margin_px': 40, 'exit_misses': 6}, 'track': {'roi_half_size_px': 120, 'hsv_lower1': [0, 75, 40], 'hsv_upper1': [15, 255, 255], 'hsv_lower2': [166, 75, 40], 'hsv_upper2': [180, 255, 255], 'binary_blur_k': 5, 'binary_bg_blur_k': 31, 'binary_min_threshold': 10, 'red_excess_blur_k': 5, 'red_excess_thresh': 32, 'use_otsu': True, 'open_k': 3, 'close_k': 7, 'close_iter': 2, 'min_area_px': 30, 'max_area_px': 9000, 'min_circularity': 0.52, 'min_solidity': 0.55, 'aspect_tol': 1.35, 'min_diameter_factor': 0.55, 'max_diameter_factor': 2.2, 'max_assoc_dist_px': 120.0, 'max_misses': 20, 'border_relax_px': 24, 'relaxed_min_area_px': 12, 'relaxed_min_circularity': 0.16, 'relaxed_min_solidity': 0.22, 'relaxed_aspect_tol': 2.6, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.6, 'roi_expand_per_miss_px': 60, 'assoc_expand_per_miss_px': 90, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 40, 'exit_misses': 6}}, 'sphere_white': {'acquire': {'roi_half_size_px': 140, 'binary_blur_k': 5, 'binary_bg_blur_k': 35, 'binary_min_threshold': 8, 'gray_min_threshold': 8, 'use_gray_branch': True, 'open_k': 3, 'close_k': 11, 'close_iter': 2, 'min_component_area_px': 24, 'min_area_px': 90, 'max_area_px': 12000, 'min_circularity': 0.45, 'min_solidity': 0.45, 'aspect_tol': 1.9, 'min_diameter_factor': 0.5, 'max_diameter_factor': 2.2, 'max_assoc_dist_px': 120.0, 'max_misses': 20, 'border_relax_px': 24, 'relaxed_min_area_px': 40, 'relaxed_min_circularity': 0.18, 'relaxed_min_solidity': 0.18, 'relaxed_aspect_tol': 2.6, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.5, 'roi_expand_per_miss_px': 50, 'assoc_expand_per_miss_px': 70, 'reacquire_fullframe_after_misses': 2, 'exit_margin_px': 40, 'exit_misses': 6}, 'track': {'roi_half_size_px': 110, 'binary_blur_k': 5, 'binary_bg_blur_k': 35, 'binary_min_threshold': 10, 'gray_min_threshold': 10, 'use_gray_branch': True, 'open_k': 3, 'close_k': 9, 'close_iter': 2, 'min_component_area_px': 28, 'min_area_px': 100, 'max_area_px': 10000, 'min_circularity': 0.5, 'min_solidity': 0.5, 'aspect_tol': 1.7, 'min_diameter_factor': 0.55, 'max_diameter_factor': 2.0, 'max_assoc_dist_px': 110.0, 'max_misses': 20, 'border_relax_px': 24, 'relaxed_min_area_px': 40, 'relaxed_min_circularity': 0.18, 'relaxed_min_solidity': 0.18, 'relaxed_aspect_tol': 2.6, 'relaxed_min_diameter_factor': 0.2, 'relaxed_max_diameter_factor': 2.5, 'roi_expand_per_miss_px': 70, 'assoc_expand_per_miss_px': 100, 'reacquire_fullframe_after_misses': 1, 'exit_margin_px': 40, 'exit_misses': 6}}}

def smooth_series(values, window=5):
    """Handle the smooth series step used by this script."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    if window < 2:
        return arr
    if window % 2 == 0:
        window += 1
    s = pd.Series(arr, dtype=float)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)

def centred_gradient(values, times):
    """Handle the centred gradient step used by this script."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(values) < 2:
        return np.full(len(values), np.nan, dtype=float)
    return np.gradient(values, times)

def order_box_points(pts):
    """Handle the order box points step used by this script."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def wrap_angle_deg_180(angle_deg):
    """Handle the wrap angle deg 180 step used by this script."""
    a = angle_deg % 180.0
    if a < 0:
        a += 180.0
    return a

def unwrap_angle_deg(angle_deg_series):
    """Handle the unwrap angle deg step used by this script."""
    ang = np.asarray(angle_deg_series, dtype=float)
    ang_rad = np.deg2rad(ang)
    ang_unwrapped = np.unwrap(ang_rad)
    return np.rad2deg(ang_unwrapped)

def load_calibration(json_path):
    """Load the calibration used by this script."""
    with open(json_path, 'r', encoding='utf-8') as f:
        cal = json.load(f)
    H = np.array(cal['homography'], dtype=np.float64)
    px_per_mm = float(cal['rectified_px_per_mm'])
    rect_w_px = int(cal['rectified_width_px'])
    rect_h_px = int(cal['rectified_height_px'])
    return (cal, H, px_per_mm, rect_w_px, rect_h_px)

def transform_points_homography(points_xy, H):
    """Handle the transform points homography step used by this script."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)

def px_to_mm_rectified(px, px_per_mm):
    """Handle the px to mm rectified step used by this script."""
    return float(px) / float(px_per_mm)

def gray_to_bgr(image):
    """Handle the gray to bgr step used by this script."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()

def pick_initial_click_raw(frame_bgr, mode_label):
    """Handle the pick initial click raw step used by this script."""
    click_pt = {'xy': None}

    def cb(event, x, y, flags, param):
        """Handle the cb step used by this script."""
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt['xy'] = (x, y)
    title = f'Click the {mode_label} centre on RAW frame, then press any key'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, cb)
    while True:
        disp = frame_bgr.copy()
        if click_pt['xy'] is not None:
            cv2.drawMarker(disp, click_pt['xy'], (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(disp, f'Click the {mode_label} centre on RAW frame, then press any key. ESC to quit.', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(title, disp)
        key = cv2.waitKey(20) & 255
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError('User cancelled object selection.')
        if click_pt['xy'] is not None and key != 255:
            break
    cv2.destroyWindow(title)
    return click_pt['xy']

def pick_initial_click_binary(binary_frame, mode_label):
    """Handle the pick initial click binary step used by this script."""
    click_pt = {'xy': None}

    def cb(event, x, y, flags, param):
        """Handle the cb step used by this script."""
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt['xy'] = (x, y)
    title = f'Click the {mode_label} centre on BINARISED frame, then press any key'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, cb)
    while True:
        disp = gray_to_bgr(binary_frame)
        if click_pt['xy'] is not None:
            cv2.drawMarker(disp, click_pt['xy'], (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(disp, f'Click the {mode_label} centre on BINARISED frame, then press any key. ESC to quit.', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(title, disp)
        key = cv2.waitKey(20) & 255
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError('User cancelled object selection.')
        if click_pt['xy'] is not None and key != 255:
            break
    cv2.destroyWindow(title)
    return click_pt['xy']

def create_kalman(initial_x, initial_y):
    """Create the kalman used by this script."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.array([[0.001, 0, 0, 0], [0, 0.001, 0, 0], [0, 0, 0.008, 0], [0, 0, 0, 0.008]], dtype=np.float32)
    kf.measurementNoiseCov = np.array([[1.2, 0], [0, 1.2]], dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
    kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
    return kf

def contour_metrics(contour):
    """Handle the contour metrics step used by this script."""
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-09:
        return None
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = area / hull_area if hull_area > 1e-09 else 0.0
    x, y, w, h = cv2.boundingRect(contour)
    if min(w, h) <= 0:
        return None
    aspect_ratio = max(w, h) / min(w, h)
    M = cv2.moments(contour)
    if abs(M['m00']) > 1e-12:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        rect = cv2.minAreaRect(contour)
        (cx, cy), _, _ = rect
    rect = cv2.minAreaRect(contour)
    (cx_rect, cy_rect), (rw, rh), _ = rect
    box = order_box_points(cv2.boxPoints(rect))
    (ecx, ecy), radius = cv2.minEnclosingCircle(contour)
    return {'area': area, 'perimeter': perimeter, 'circularity': circularity, 'solidity': solidity, 'aspect_ratio': aspect_ratio, 'cx': float(cx), 'cy': float(cy), 'cx_rect': float(cx_rect), 'cy_rect': float(cy_rect), 'rect_w': float(rw), 'rect_h': float(rh), 'box': box, 'circle_cx': float(ecx), 'circle_cy': float(ecy), 'circle_r': float(radius)}

def primary_found_segment_df(df):
    """Handle the primary found segment df step used by this script."""
    work = df.copy().reset_index(drop=True)
    if len(work) == 0 or 'found_this_frame' not in work.columns:
        return work
    found = pd.to_numeric(work['found_this_frame'], errors='coerce').fillna(0).astype(int).to_numpy()
    if 'frame_idx' in work.columns:
        keys = pd.to_numeric(work['frame_idx'], errors='coerce').ffill().fillna(0).to_numpy(dtype=int)
    elif 'frame' in work.columns:
        keys = pd.to_numeric(work['frame'], errors='coerce').ffill().fillna(0).to_numpy(dtype=int)
    elif 'out_idx' in work.columns:
        keys = pd.to_numeric(work['out_idx'], errors='coerce').ffill().fillna(0).to_numpy(dtype=int)
    else:
        keys = np.arange(len(work), dtype=int)
    segments = []
    start = None
    prev_key = None
    for idx, is_found in enumerate(found):
        key = int(keys[idx])
        if is_found != 1:
            if start is not None:
                segments.append(work.iloc[start:idx].copy())
                start = None
            prev_key = None
            continue
        if start is None:
            start = idx
        elif prev_key is not None and key - prev_key > 1:
            segments.append(work.iloc[start:idx].copy())
            start = idx
        prev_key = key
    if start is not None:
        segments.append(work.iloc[start:].copy())
    if segments:
        return max(segments, key=len).reset_index(drop=True)
    found_only = work.loc[found == 1].copy()
    if len(found_only) > 0:
        return found_only.reset_index(drop=True)
    return work

def rod_angle_from_box(box):
    """Handle the rod angle from box step used by this script."""
    pts = np.asarray(box, dtype=float)
    edges = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        vec = p1 - p0
        length = np.hypot(vec[0], vec[1])
        edges.append((length, vec))
    edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
    v1 = edges_sorted[0][1]
    v2 = edges_sorted[1][1]
    if np.dot(v1, v2) < 0:
        v2 = -v2
    v = 0.5 * (v1 + v2)
    angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    return wrap_angle_deg_180(angle_deg)

def rod_box_axes(box):
    """Handle the rod box axes step used by this script."""
    pts = np.asarray(box, dtype=float)
    center = pts.mean(axis=0)
    edges = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        vec = p1 - p0
        length = np.hypot(vec[0], vec[1])
        edges.append((length, vec))
    edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
    long_vec = edges_sorted[0][1].astype(float)
    short_vec = edges_sorted[-1][1].astype(float)
    long_len = max(np.hypot(long_vec[0], long_vec[1]), 1e-09)
    short_len = max(np.hypot(short_vec[0], short_vec[1]), 1e-09)
    u_long = long_vec / long_len
    u_short = short_vec / short_len
    if np.cross(u_long, u_short) < 0:
        u_short = -u_short
    return (center, u_long, u_short, long_len, short_len)

def box_from_axes(center, u_long, u_short, long_len, short_len):
    """Handle the box from axes step used by this script."""
    hl = max(float(long_len) / 2.0, 1e-09)
    hs = max(float(short_len) / 2.0, 1e-09)
    pts = np.array([center - hl * u_long - hs * u_short, center + hl * u_long - hs * u_short, center + hl * u_long + hs * u_short, center - hl * u_long + hs * u_short], dtype=np.float32)
    return order_box_points(pts)

def regularize_sphere_box(box, area_px2, circle_radius_px, max_aspect_ratio=1.35, min_short_factor=0.88, max_long_factor=1.18, max_short_factor=1.08):
    """Handle the regularize sphere box step used by this script."""
    center, u_long, u_short, long_len_raw, short_len_raw = rod_box_axes(box)
    circle_d = max(2.0 * float(circle_radius_px), 1e-06)
    eq_d = max(2.0 * math.sqrt(max(float(area_px2), 1e-09) / math.pi), 1e-06)
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
    long_len = max(long_len, short_len)
    reg_box = box_from_axes(center, u_long, u_short, long_len, short_len)
    reg_diameter_px = float(math.sqrt(max(long_len * short_len, 1e-09)))
    return (reg_box, long_len, short_len, reg_diameter_px)

def rod_anchor_uv_from_click(click_xy, box):
    """Handle the rod anchor uv from click step used by this script."""
    center, u_long, u_short, long_len, short_len = rod_box_axes(box)
    rel = np.asarray(click_xy, dtype=float) - center
    a = float(np.dot(rel, u_long) / max(long_len / 2.0, 1e-09))
    b = float(np.dot(rel, u_short) / max(short_len / 2.0, 1e-09))
    a = float(np.clip(a, -1.0, 1.0))
    b = float(np.clip(b, -1.0, 1.0))
    return np.array([a, b], dtype=float)

def rod_point_from_uv(box, uv):
    """Handle the rod point from uv step used by this script."""
    center, u_long, u_short, long_len, short_len = rod_box_axes(box)
    a, b = (float(uv[0]), float(uv[1]))
    pt = center + a * (long_len / 2.0) * u_long + b * (short_len / 2.0) * u_short
    return np.asarray(pt, dtype=float)

def get_param_block(object_mode, sphere_color_mode, stage):
    """Handle the get param block step used by this script."""
    if object_mode == 'sphere':
        key = f'sphere_{sphere_color_mode}'
    else:
        key = object_mode
    return PARAMS[key][stage].copy()

def safe_odd(value, minimum=1):
    """Handle the safe odd step used by this script."""
    k = int(round(value))
    if k < minimum:
        k = minimum
    if k % 2 == 0:
        k += 1
    return k

def evenly_spaced_frame_indices(start_frame, end_frame, count):
    """Handle the evenly spaced frame indices step used by this script."""
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    count = max(1, int(count))
    if end_frame < start_frame:
        return []
    if count == 1 or end_frame == start_frame:
        return [start_frame]
    return [int(round(v)) for v in np.linspace(start_frame, end_frame, num=count)]

def otsu_threshold_with_floor(gray_image, min_threshold):
    """Handle the otsu threshold with floor step used by this script."""
    min_threshold = float(min_threshold)
    otsu_thr, bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_used = max(float(otsu_thr), min_threshold)
    if threshold_used > float(otsu_thr):
        _, bw = cv2.threshold(gray_image, int(round(threshold_used)), 255, cv2.THRESH_BINARY)
    return (threshold_used, bw)

def build_average_background_gray(cap, reference_start_frame, window_frames, sample_frames, margin_frames=0, keep_fraction=0.7):
    """Build the average background gray used by this script."""
    if cap is None or not cap.isOpened():
        return (None, {'status': 'video_unavailable'})
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    current_pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))
    end_frame = int(reference_start_frame) - int(margin_frames)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    end_frame = max(-1, end_frame)
    if end_frame < 0:
        return (None, {'status': 'no_prestart_frames', 'reference_start_frame': int(reference_start_frame), 'margin_frames': int(margin_frames)})
    start_frame = max(0, end_frame - max(0, int(window_frames)) + 1)
    frame_indices = evenly_spaced_frame_indices(start_frame, end_frame, max(1, int(sample_frames)))
    if not frame_indices:
        return (None, {'status': 'no_sample_indices', 'start_frame': int(start_frame), 'end_frame': int(end_frame)})
    stack = []
    used_indices = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        stack.append(gray)
        used_indices.append(int(frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    if not stack:
        return (None, {'status': 'sample_read_failed', 'requested_indices': frame_indices})
    stack_arr = np.stack(stack, axis=0)
    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))
    if keep_fraction <= 0.0:
        keep_fraction = 1.0
    if keep_fraction < 0.999:
        pixel_std = np.std(stack_arr, axis=0)
        threshold = float(np.quantile(pixel_std, keep_fraction))
        keep_mask = pixel_std <= threshold
        mean_frame = np.mean(stack_arr, axis=0)
        median_frame = np.median(stack_arr, axis=0)
        background = np.where(keep_mask, mean_frame, median_frame)
    else:
        background = np.mean(stack_arr, axis=0)
    info = {'status': 'ok', 'reference_start_frame': int(reference_start_frame), 'window_frames': int(window_frames), 'sample_frames_requested': int(sample_frames), 'sample_frames_used': int(len(used_indices)), 'margin_frames': int(margin_frames), 'keep_fraction': float(keep_fraction), 'start_frame': int(start_frame), 'end_frame': int(end_frame), 'used_indices': used_indices}
    return (background.astype(np.float32), info)

def preprocess_edge_roi(gray_roi, params):
    """Handle the preprocess edge roi step used by this script."""
    g = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(g, params['edge_low'], params['edge_high'])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['close_k'], params['close_k']))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=params['close_iter'])
    bw = np.zeros_like(edges_closed)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bw, contours, -1, 255, thickness=cv2.FILLED)
    return (g, edges, edges_closed, bw)

def preprocess_bright_binary_roi(gray_roi, params, background_gray_roi=None):
    """Handle the preprocess bright binary roi step used by this script."""
    blur_k = safe_odd(params.get('binary_blur_k', 5), minimum=1)
    bg_k = safe_odd(params.get('binary_bg_blur_k', 31), minimum=3)
    open_k = safe_odd(params.get('open_k', 3), minimum=1)
    close_k = safe_odd(params.get('close_k', 7), minimum=1)
    background_roi_u8 = None
    deviation_map = None
    work_roi = gray_roi
    if background_gray_roi is not None and background_gray_roi.shape[:2] == gray_roi.shape[:2]:
        background_roi_u8 = np.clip(background_gray_roi, 0, 255).astype(np.uint8)
        deviation_map = cv2.absdiff(gray_roi, background_roi_u8)
        work_roi = deviation_map
    blur = cv2.GaussianBlur(work_roi, (blur_k, blur_k), 0)
    bg = cv2.GaussianBlur(blur, (bg_k, bg_k), 0)
    bright_map = cv2.subtract(blur, bg)
    threshold_used, bw_bright = otsu_threshold_with_floor(bright_map, params.get('binary_min_threshold', 0))
    use_gray_branch = bool(params.get('use_gray_branch', True))
    if use_gray_branch:
        gray_threshold_used, bw_gray = otsu_threshold_with_floor(blur, params.get('gray_min_threshold', 0))
        bw_raw = cv2.bitwise_or(bw_bright, bw_gray)
    else:
        gray_threshold_used = np.nan
        bw_gray = np.zeros_like(bw_bright)
        bw_raw = bw_bright.copy()
    bw = bw_raw.copy()
    if open_k > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=1)
    if close_k > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=int(params.get('close_iter', 1)))
    min_component_area = float(params.get('min_component_area_px', 1.0))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bw_filled = np.zeros_like(bw)
    kept_contours = [c for c in contours if cv2.contourArea(c) >= min_component_area]
    hull_contours = []
    for contour in kept_contours:
        hull = cv2.convexHull(contour)
        if hull is None or cv2.contourArea(hull) < min_component_area:
            continue
        hull_contours.append(hull)
    if hull_contours:
        cv2.drawContours(bw_filled, hull_contours, -1, 255, thickness=cv2.FILLED)
    return {'background_roi': background_roi_u8, 'deviation_map': deviation_map, 'blur': blur, 'bg': bg, 'bright_map': bright_map, 'threshold_used': float(threshold_used), 'gray_threshold_used': float(gray_threshold_used) if np.isfinite(gray_threshold_used) else np.nan, 'bw_bright': bw_bright, 'bw_gray': bw_gray, 'bw_raw': bw_raw, 'bw_clean': bw, 'bw': bw_filled}

def preprocess_red_roi(bgr_roi, params, background_gray_roi=None):
    """Handle the preprocess red roi step used by this script."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    l1 = np.array(params['hsv_lower1'], dtype=np.uint8)
    u1 = np.array(params['hsv_upper1'], dtype=np.uint8)
    l2 = np.array(params['hsv_lower2'], dtype=np.uint8)
    u2 = np.array(params['hsv_upper2'], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, l1, u1)
    mask2 = cv2.inRange(hsv, l2, u2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    b, g, r = cv2.split(bgr_roi)
    gb_max = cv2.max(g, b)
    red_excess = cv2.subtract(r, gb_max)
    k = int(params.get('red_excess_blur_k', 5))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    red_excess_blur = cv2.GaussianBlur(red_excess, (k, k), 0)
    if params.get('use_otsu', True):
        _, rex_mask = cv2.threshold(red_excess_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, rex_mask = cv2.threshold(red_excess_blur, int(params.get('red_excess_thresh', 30)), 255, cv2.THRESH_BINARY)
    motion_background = motion_deviation = motion_blur = motion_bg = motion_map = motion_mask = None
    motion_threshold_used = np.nan
    motion_gray_threshold_used = np.nan
    motion_bw_bright = motion_bw_gray = motion_bw_raw = motion_bw_clean = None
    bw = cv2.bitwise_and(hsv_mask, rex_mask)
    if background_gray_roi is not None and background_gray_roi.shape[:2] == bgr_roi.shape[:2]:
        gray_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        motion_dbg = preprocess_bright_binary_roi(gray_roi, params, background_gray_roi=background_gray_roi)
        motion_background = motion_dbg['background_roi']
        motion_deviation = motion_dbg['deviation_map']
        motion_blur = motion_dbg['blur']
        motion_bg = motion_dbg['bg']
        motion_map = motion_dbg['bright_map']
        motion_threshold_used = motion_dbg['threshold_used']
        motion_gray_threshold_used = motion_dbg['gray_threshold_used']
        motion_bw_bright = motion_dbg['bw_bright']
        motion_bw_gray = motion_dbg['bw_gray']
        motion_bw_raw = motion_dbg['bw_raw']
        motion_bw_clean = motion_dbg['bw_clean']
        motion_mask = motion_dbg['bw']
        bw = cv2.bitwise_and(bw, motion_mask)
    ok = int(params.get('open_k', 3))
    ck = int(params.get('close_k', 7))
    close_iter = int(params.get('close_iter', 2))
    if ok > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=1)
    if ck > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    return {'hsv': hsv, 'hsv_mask': hsv_mask, 'red_excess': red_excess, 'red_excess_blur': red_excess_blur, 'rex_mask': rex_mask, 'background_roi': motion_background, 'deviation_map': motion_deviation, 'blur': motion_blur, 'bg': motion_bg, 'bright_map': motion_map, 'threshold_used': float(motion_threshold_used) if np.isfinite(motion_threshold_used) else np.nan, 'gray_threshold_used': float(motion_gray_threshold_used) if np.isfinite(motion_gray_threshold_used) else np.nan, 'bw_bright': motion_bw_bright, 'bw_gray': motion_bw_gray, 'bw_raw': motion_bw_raw, 'bw_clean': motion_bw_clean, 'motion_mask': motion_mask, 'bw': bw}

def detect_in_raw_roi(frame_bgr, pred_x, pred_y, H, px_per_mm, object_mode, sphere_color_mode, disc_diameter_mm, sphere_diameter_mm, stage_params, rod_anchor_uv=None, sphere_background_gray=None):
    """Detect the in raw roi used by this script."""
    h, w = frame_bgr.shape[:2]
    roi_half = stage_params['roi_half_size_px']
    x0 = max(0, int(round(pred_x - roi_half)))
    x1 = min(w, int(round(pred_x + roi_half)))
    y0 = max(0, int(round(pred_y - roi_half)))
    y1 = min(h, int(round(pred_y + roi_half)))
    if x1 <= x0 or y1 <= y0:
        return ([], {'roi_box': (x0, y0, x1, y1), 'bw': np.zeros((max(1, h), max(1, w)), dtype=np.uint8), 'gray_roi': np.zeros((1, 1), dtype=np.uint8)})
    roi = frame_bgr[y0:y1, x0:x1]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sphere_background_roi = None
    if sphere_background_gray is not None:
        sphere_background_roi = sphere_background_gray[y0:y1, x0:x1]
    aux_a = aux_b = aux_c = None
    binary_dbg = None
    red_dbg = None
    if object_mode == 'sphere' and sphere_color_mode == 'red':
        red_dbg = preprocess_red_roi(roi, stage_params, background_gray_roi=sphere_background_roi)
        bw = red_dbg['bw']
    elif object_mode == 'sphere' and sphere_color_mode == 'white':
        binary_dbg = preprocess_bright_binary_roi(gray_roi, stage_params, background_gray_roi=sphere_background_roi)
        bw = binary_dbg['bw']
    else:
        aux_a, aux_b, aux_c, bw = preprocess_edge_roi(gray_roi, stage_params)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in contours:
        m = contour_metrics(c)
        if m is None:
            continue
        bx, by, bw_rect, bh_rect = cv2.boundingRect(c)
        border_relax_px = int(stage_params.get('border_relax_px', 0))
        touches_frame_border = x0 + bx <= border_relax_px or y0 + by <= border_relax_px or x0 + bx + bw_rect >= w - border_relax_px or (y0 + by + bh_rect >= h - border_relax_px)
        touches_roi_border = bx <= 1 or by <= 1 or bx + bw_rect >= roi.shape[1] - 1 or (by + bh_rect >= roi.shape[0] - 1)
        box_global = m['box'].copy()
        box_global[:, 0] += float(x0)
        box_global[:, 1] += float(y0)
        cx_global = float(m['cx'] + x0)
        cy_global = float(m['cy'] + y0)
        if object_mode == 'rod' and rod_anchor_uv is not None:
            anchor_raw = rod_point_from_uv(box_global, rod_anchor_uv)
            cx_track = float(anchor_raw[0])
            cy_track = float(anchor_raw[1])
        else:
            cx_track = cx_global
            cy_track = cy_global
        centre_rectified = transform_points_homography(np.array([[cx_track, cy_track]]), H)[0]
        circle_diameter_mm = px_to_mm_rectified(2.0 * m['circle_r'], px_per_mm)
        keep = False
        obj_diameter_mm = None
        if object_mode == 'rod':
            keep = stage_params['min_area_px'] <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params['min_solidity'] and (stage_params['min_aspect_ratio'] <= m['aspect_ratio'] <= stage_params['max_aspect_ratio']) and (stage_params['min_length_mm'] <= length_mm <= stage_params['max_length_mm'])
        elif object_mode == 'disc':
            d_true = float(disc_diameter_mm)
            keep = stage_params['min_area_px'] <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params['min_solidity'] and (m['circularity'] >= stage_params['min_circularity']) and (m['aspect_ratio'] <= stage_params['aspect_tol']) and (stage_params['min_diameter_factor'] * d_true <= circle_diameter_mm <= stage_params['max_diameter_factor'] * d_true)
            obj_diameter_mm = circle_diameter_mm
        elif object_mode == 'sphere':
            d_true = float(sphere_diameter_mm)
            strict_keep = stage_params['min_area_px'] <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params['min_solidity'] and (m['circularity'] >= stage_params['min_circularity']) and (m['aspect_ratio'] <= stage_params['aspect_tol']) and (stage_params['min_diameter_factor'] * d_true <= circle_diameter_mm <= stage_params['max_diameter_factor'] * d_true)
            relaxed_keep = False
            if not strict_keep and (touches_frame_border or touches_roi_border):
                relaxed_keep = stage_params.get('relaxed_min_area_px', stage_params['min_area_px']) <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params.get('relaxed_min_solidity', stage_params['min_solidity']) and (m['circularity'] >= stage_params.get('relaxed_min_circularity', stage_params['min_circularity'])) and (m['aspect_ratio'] <= stage_params.get('relaxed_aspect_tol', stage_params['aspect_tol'])) and (stage_params.get('relaxed_min_diameter_factor', stage_params['min_diameter_factor']) * d_true <= circle_diameter_mm <= stage_params.get('relaxed_max_diameter_factor', stage_params['max_diameter_factor']) * d_true)
            keep = strict_keep or relaxed_keep
            obj_diameter_mm = circle_diameter_mm
        if not keep:
            continue
        if object_mode == 'sphere':
            box_global, _, _, reg_diameter_px = regularize_sphere_box(box_global, m['area'], m['circle_r'])
            obj_diameter_mm = px_to_mm_rectified(reg_diameter_px, px_per_mm)
        box_rectified = transform_points_homography(box_global, H)
        rect_edges = [np.hypot(*box_rectified[(i + 1) % 4] - box_rectified[i]) for i in range(4)]
        length_rect_px = max(rect_edges)
        width_rect_px = min(rect_edges)
        length_mm = px_to_mm_rectified(length_rect_px, px_per_mm)
        width_mm = px_to_mm_rectified(width_rect_px, px_per_mm)
        angle_deg = rod_angle_from_box(box_rectified)
        dets.append({'cx_raw': cx_track, 'cy_raw': cy_track, 'cx_box_raw': cx_global, 'cy_box_raw': cy_global, 'cx_rect_raw': float(m['cx_rect'] + x0), 'cy_rect_raw': float(m['cy_rect'] + y0), 'box_raw': box_global, 'area_px2': float(m['area']), 'solidity': float(m['solidity']), 'aspect_ratio': float(m['aspect_ratio']), 'circularity': float(m['circularity']), 'centre_rectified_px': centre_rectified, 'box_rectified_px': box_rectified, 'length_mm': float(length_mm), 'width_mm': float(width_mm), 'angle_deg': float(angle_deg), 'diameter_mm': obj_diameter_mm, 'touches_frame_border': bool(touches_frame_border), 'touches_roi_border': bool(touches_roi_border)})
    dets.sort(key=lambda d: math.hypot(d['cx_raw'] - pred_x, d['cy_raw'] - pred_y))
    debug = {'roi_box': (x0, y0, x1, y1), 'bw': bw, 'gray_roi': gray_roi}
    if object_mode == 'sphere' and sphere_color_mode == 'red':
        debug['roi_hsv'] = red_dbg['hsv']
        debug['roi_hsv_mask'] = red_dbg['hsv_mask']
        debug['roi_red_excess'] = red_dbg['red_excess']
        debug['roi_red_excess_blur'] = red_dbg['red_excess_blur']
        debug['roi_rex_mask'] = red_dbg['rex_mask']
        debug['roi_binary_background'] = red_dbg['background_roi']
        debug['roi_binary_deviation'] = red_dbg['deviation_map']
        debug['roi_binary_blur'] = red_dbg['blur']
        debug['roi_binary_bg'] = red_dbg['bg']
        debug['roi_binary_map'] = red_dbg['bright_map']
        debug['roi_binary_bright'] = red_dbg['bw_bright']
        debug['roi_binary_gray'] = red_dbg['bw_gray']
        debug['roi_binary_raw'] = red_dbg['bw_raw']
        debug['roi_binary_clean'] = red_dbg['bw_clean']
        debug['roi_binary_filled'] = red_dbg['motion_mask']
        debug['binary_threshold'] = red_dbg['threshold_used']
    elif object_mode == 'sphere' and sphere_color_mode == 'white':
        debug['roi_binary_background'] = binary_dbg['background_roi']
        debug['roi_binary_deviation'] = binary_dbg['deviation_map']
        debug['roi_binary_blur'] = binary_dbg['blur']
        debug['roi_binary_bg'] = binary_dbg['bg']
        debug['roi_binary_map'] = binary_dbg['bright_map']
        debug['roi_binary_bright'] = binary_dbg['bw_bright']
        debug['roi_binary_gray'] = binary_dbg['bw_gray']
        debug['roi_binary_raw'] = binary_dbg['bw_raw']
        debug['roi_binary_clean'] = binary_dbg['bw_clean']
        debug['roi_binary_filled'] = binary_dbg['bw']
        debug['binary_threshold'] = binary_dbg['threshold_used']
    else:
        debug['roi_blur'] = aux_a
        debug['roi_edges'] = aux_b
        debug['roi_close'] = aux_c
    return (dets, debug)

def search_start_detection(cap, start_frame, click_xy, H, px_per_mm, object_mode, sphere_color_mode, disc_diameter_mm, sphere_diameter_mm, stage_params, search_radius, sphere_background_gray=None):
    """Handle the search start detection step used by this script."""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_attempt = None
    offsets = [0]
    for delta in range(1, int(search_radius) + 1):
        offsets.extend([delta, -delta])
    for offset in offsets:
        frame_idx = int(start_frame + offset)
        if frame_idx < 0:
            continue
        if frame_count > 0 and frame_idx >= frame_count:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        dets, debug = detect_in_raw_roi(frame, click_xy[0], click_xy[1], H, px_per_mm, object_mode, sphere_color_mode, disc_diameter_mm, sphere_diameter_mm, stage_params, rod_anchor_uv=None, sphere_background_gray=sphere_background_gray)
        last_attempt = (frame_idx, frame, dets, debug)
        if dets:
            return (frame_idx, frame, dets, debug)
    if last_attempt is None:
        raise RuntimeError('Could not read any candidate start frames from the video.')
    return last_attempt

def build_binary_frame_for_display(frame_bgr, object_mode, sphere_color_mode, stage_params, sphere_background_gray=None):
    """Build the binary frame for display used by this script."""
    if object_mode == 'sphere' and sphere_color_mode == 'red':
        return preprocess_red_roi(frame_bgr, stage_params, background_gray_roi=sphere_background_gray)['bw']
    if object_mode == 'sphere' and sphere_color_mode == 'white':
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return preprocess_bright_binary_roi(gray, stage_params, background_gray_roi=sphere_background_gray)['bw']
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, _, _, bw = preprocess_edge_roi(gray, stage_params)
    return bw

def score_detection(det, pred_x, pred_y, object_mode, track_angle_deg=None, track_length_mm=None, expected_disc_diameter_mm=None, expected_sphere_diameter_mm=None):
    """Handle the score detection step used by this script."""
    dist = math.hypot(pred_x - det['cx_raw'], pred_y - det['cy_raw'])
    if object_mode == 'rod':
        dtheta = abs(det['angle_deg'] - float(track_angle_deg))
        dtheta = min(dtheta, 180.0 - dtheta)
        dlen = abs(det['length_mm'] - float(track_length_mm))
        return dist + 0.25 * dtheta + 1.5 * dlen
    if object_mode == 'sphere':
        expected_d = float(expected_sphere_diameter_mm)
        d_diam = abs(float(det['diameter_mm']) - expected_d)
        circularity_penalty = 25.0 * max(0.0, 0.75 - float(det['circularity']))
        aspect_penalty = 16.0 * max(0.0, float(det['aspect_ratio']) - 1.35)
        solidity_penalty = 20.0 * max(0.0, 0.82 - float(det['solidity']))
        return dist + 6.0 * d_diam + circularity_penalty + aspect_penalty + solidity_penalty
    if object_mode == 'disc':
        expected_d = float(expected_disc_diameter_mm)
        d_diam = abs(float(det['diameter_mm']) - expected_d)
        circularity_penalty = 18.0 * max(0.0, 0.7 - float(det['circularity']))
        return dist + 4.0 * d_diam + circularity_penalty
    return dist

def sphere_relative_x_mm(first_centre_rectified_px, centre_rectified_px, px_per_mm):
    """Handle the sphere relative x mm step used by this script."""
    x0_mm = px_to_mm_rectified(float(first_centre_rectified_px[0]), px_per_mm)
    x_mm = px_to_mm_rectified(float(centre_rectified_px[0]), px_per_mm)
    return x_mm - x0_mm

def sphere_detection_backtracks(track, det, px_per_mm):
    """Handle the sphere detection backtracks step used by this script."""
    if track.object_mode != 'sphere':
        return False
    if track.max_abs_x_progress_mm < DEFAULT_SPHERE_BACKTRACK_MIN_PROGRESS_MM:
        return False
    cand_x_rel_mm = sphere_relative_x_mm(track.first_centre_rectified_px, det['centre_rectified_px'], px_per_mm)
    dominant_positive = abs(track.max_x_rel_mm) >= abs(track.min_x_rel_mm)
    if dominant_positive:
        back_x_mm = track.max_x_rel_mm - cand_x_rel_mm
    else:
        back_x_mm = cand_x_rel_mm - track.min_x_rel_mm
    if back_x_mm > DEFAULT_SPHERE_SEVERE_BACKTRACK_X_MM:
        return True
    if track.misses >= 1 and back_x_mm > DEFAULT_SPHERE_BACKTRACK_TOL_X_MM:
        return True
    return False

class ObjectTrack:
    """Store the data needed by the ObjectTrack structure."""

    def __init__(self, det, time_s, object_mode):
        """Handle the   init   step used by this script."""
        self.object_mode = object_mode
        self.cx_raw = float(det['cx_raw'])
        self.cy_raw = float(det['cy_raw'])
        self.cx_box_raw = float(det.get('cx_box_raw', det['cx_raw']))
        self.cy_box_raw = float(det.get('cy_box_raw', det['cy_raw']))
        self.box_raw = det['box_raw']
        self.rod_anchor_uv = None
        self.area_px2 = float(det['area_px2'])
        self.solidity = float(det['solidity'])
        self.aspect_ratio = float(det['aspect_ratio'])
        self.circularity = float(det['circularity'])
        self.centre_rectified_px = np.array(det['centre_rectified_px'], dtype=float)
        self.box_rectified_px = np.array(det['box_rectified_px'], dtype=float)
        self.length_mm = float(det['length_mm'])
        self.width_mm = float(det['width_mm'])
        self.angle_deg = float(det['angle_deg'])
        self.diameter_mm = det['diameter_mm']
        self.first_centre_rectified_px = self.centre_rectified_px.copy()
        self.first_time_s = float(time_s)
        self.misses = 0
        self.kf = create_kalman(self.cx_raw, self.cy_raw)
        self.kalman_x_raw = float(det['cx_raw'])
        self.kalman_y_raw = float(det['cy_raw'])
        self.max_x_rel_mm = 0.0
        self.min_x_rel_mm = 0.0
        self.max_abs_x_progress_mm = 0.0

    def predict(self):
        """Handle the predict step used by this script."""
        pred = self.kf.predict()
        self.kalman_x_raw = float(pred[0, 0])
        self.kalman_y_raw = float(pred[1, 0])
        return (self.kalman_x_raw, self.kalman_y_raw)

    def correct(self, mx, my):
        """Handle the correct step used by this script."""
        measurement = np.array([[np.float32(mx)], [np.float32(my)]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        self.kalman_x_raw = float(corrected[0, 0])
        self.kalman_y_raw = float(corrected[1, 0])

    def set_rod_anchor_from_click(self, click_xy, H):
        """Handle the set rod anchor from click step used by this script."""
        if self.object_mode != 'rod':
            return
        self.rod_anchor_uv = rod_anchor_uv_from_click(click_xy, self.box_raw)
        anchor_raw = rod_point_from_uv(self.box_raw, self.rod_anchor_uv)
        self.cx_raw = float(anchor_raw[0])
        self.cy_raw = float(anchor_raw[1])
        self.centre_rectified_px = transform_points_homography(np.array([[self.cx_raw, self.cy_raw]]), H)[0]
        self.first_centre_rectified_px = self.centre_rectified_px.copy()
        self.kf = create_kalman(self.cx_raw, self.cy_raw)
        self.kalman_x_raw = float(self.cx_raw)
        self.kalman_y_raw = float(self.cy_raw)

    def update(self, det):
        """Handle the update step used by this script."""
        self.cx_raw = 0.7 * self.cx_raw + 0.3 * float(det['cx_raw'])
        self.cy_raw = 0.7 * self.cy_raw + 0.3 * float(det['cy_raw'])
        self.cx_box_raw = float(det.get('cx_box_raw', det['cx_raw']))
        self.cy_box_raw = float(det.get('cy_box_raw', det['cy_raw']))
        self.box_raw = det['box_raw']
        self.area_px2 = float(det['area_px2'])
        self.solidity = float(det['solidity'])
        self.aspect_ratio = float(det['aspect_ratio'])
        self.circularity = float(det['circularity'])
        self.centre_rectified_px = np.array(det['centre_rectified_px'], dtype=float)
        self.box_rectified_px = np.array(det['box_rectified_px'], dtype=float)
        self.length_mm = float(det['length_mm'])
        self.width_mm = float(det['width_mm'])
        self.angle_deg = float(det['angle_deg'])
        self.diameter_mm = det['diameter_mm']
        self.correct(self.cx_raw, self.cy_raw)
        self.misses = 0

    def update_progress_extrema(self, px_per_mm):
        """Handle the update progress extrema step used by this script."""
        x_rel_mm = sphere_relative_x_mm(self.first_centre_rectified_px, self.centre_rectified_px, px_per_mm)
        self.max_x_rel_mm = max(self.max_x_rel_mm, float(x_rel_mm))
        self.min_x_rel_mm = min(self.min_x_rel_mm, float(x_rel_mm))
        self.max_abs_x_progress_mm = max(self.max_abs_x_progress_mm, abs(float(x_rel_mm)))

def add_kinematics(df, object_mode):
    """Handle the add kinematics step used by this script."""
    df = df.copy().sort_values('time_s').reset_index(drop=True)
    if len(df) == 0:
        return df
    use_kalman = {'x_kalman_mm', 'y_kalman_mm'}.issubset(df.columns)
    if use_kalman:
        x_base = df['x_kalman_mm'].to_numpy(dtype=float)
        y_base = df['y_kalman_mm'].to_numpy(dtype=float)
    else:
        x_base = df['x_mm'].to_numpy(dtype=float)
        y_base = df['y_mm'].to_numpy(dtype=float)
    x0 = float(x_base[0])
    y0 = float(y_base[0])
    t0 = float(df.loc[0, 'time_s'])
    df['t_rel_s'] = df['time_s'] - t0
    df['x_rel_mm'] = x_base - x0
    df['y_rel_mm'] = y_base - y0
    df['disp_mm'] = np.sqrt(df['x_rel_mm'] ** 2 + df['y_rel_mm'] ** 2)
    dx_step = pd.Series(df['x_rel_mm']).diff().fillna(0.0)
    dy_step = pd.Series(df['y_rel_mm']).diff().fillna(0.0)
    df['step_dist_mm'] = np.sqrt(dx_step ** 2 + dy_step ** 2)
    df['path_length_mm'] = df['step_dist_mm'].cumsum()
    x_s = smooth_series(df['x_rel_mm'].to_numpy(dtype=float), DEFAULT_SMOOTH_WINDOW)
    y_s = smooth_series(df['y_rel_mm'].to_numpy(dtype=float), DEFAULT_SMOOTH_WINDOW)
    t_s = df['t_rel_s'].to_numpy(dtype=float)
    df['x_rel_smooth_mm'] = x_s
    df['y_rel_smooth_mm'] = y_s
    df['vx_mm_s'] = centred_gradient(x_s, t_s)
    df['vy_mm_s'] = centred_gradient(y_s, t_s)
    df['speed_mm_s'] = np.sqrt(df['vx_mm_s'] ** 2 + df['vy_mm_s'] ** 2)
    df['ax_mm_s2'] = centred_gradient(df['vx_mm_s'].to_numpy(dtype=float), t_s)
    df['ay_mm_s2'] = centred_gradient(df['vy_mm_s'].to_numpy(dtype=float), t_s)
    df['accel_mm_s2'] = np.sqrt(df['ax_mm_s2'] ** 2 + df['ay_mm_s2'] ** 2)
    if object_mode == 'rod':
        angle_unwrapped = unwrap_angle_deg(df['angle_deg'].to_numpy(dtype=float))
        angle_smooth = smooth_series(angle_unwrapped, DEFAULT_SMOOTH_WINDOW)
        df['angle_unwrapped_deg'] = angle_unwrapped
        df['angle_smooth_deg'] = angle_smooth
        df['omega_deg_s'] = centred_gradient(angle_smooth, t_s)
    return df
    x0 = float(df.loc[0, 'x_mm'])
    y0 = float(df.loc[0, 'y_mm'])
    t0 = float(df.loc[0, 'time_s'])
    df['t_rel_s'] = df['time_s'] - t0
    df['x_rel_mm'] = df['x_mm'] - x0
    df['y_rel_mm'] = df['y_mm'] - y0
    df['disp_mm'] = np.sqrt(df['x_rel_mm'] ** 2 + df['y_rel_mm'] ** 2)
    dx_step = df['x_rel_mm'].diff().fillna(0.0)
    dy_step = df['y_rel_mm'].diff().fillna(0.0)
    df['step_dist_mm'] = np.sqrt(dx_step ** 2 + dy_step ** 2)
    df['path_length_mm'] = df['step_dist_mm'].cumsum()
    x_s = smooth_series(df['x_rel_mm'].to_numpy(dtype=float), DEFAULT_SMOOTH_WINDOW)
    y_s = smooth_series(df['y_rel_mm'].to_numpy(dtype=float), DEFAULT_SMOOTH_WINDOW)
    t_s = df['t_rel_s'].to_numpy(dtype=float)
    df['x_rel_smooth_mm'] = x_s
    df['y_rel_smooth_mm'] = y_s
    df['vx_mm_s'] = centred_gradient(x_s, t_s)
    df['vy_mm_s'] = centred_gradient(y_s, t_s)
    df['speed_mm_s'] = np.sqrt(df['vx_mm_s'] ** 2 + df['vy_mm_s'] ** 2)
    df['ax_mm_s2'] = centred_gradient(df['vx_mm_s'].to_numpy(dtype=float), t_s)
    df['ay_mm_s2'] = centred_gradient(df['vy_mm_s'].to_numpy(dtype=float), t_s)
    df['accel_mm_s2'] = np.sqrt(df['ax_mm_s2'] ** 2 + df['ay_mm_s2'] ** 2)
    if object_mode == 'rod':
        angle_unwrapped = unwrap_angle_deg(df['angle_deg'].to_numpy(dtype=float))
        angle_smooth = smooth_series(angle_unwrapped, DEFAULT_SMOOTH_WINDOW)
        df['angle_unwrapped_deg'] = angle_unwrapped
        df['angle_smooth_deg'] = angle_smooth
        df['omega_deg_s'] = centred_gradient(angle_smooth, t_s)
    return df

def save_plot(x, y, xlabel, ylabel, title, path, legend=None):
    """Save the plot used by this script."""
    plt.figure()
    if isinstance(y, (list, tuple)):
        for idx, yy in enumerate(y):
            if legend:
                plt.plot(x[idx] if isinstance(x, (list, tuple)) else x, yy, label=legend[idx])
            else:
                plt.plot(x[idx] if isinstance(x, (list, tuple)) else x, yy)
        if legend:
            plt.legend()
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def save_outputs(dfp, out_dir, object_mode):
    """Save the outputs used by this script."""
    pngs = []

    def reg(name):
        """Handle the reg step used by this script."""
        p = os.path.join(out_dir, name)
        pngs.append(p)
        return p
    plot_df = primary_found_segment_df(dfp)
    if len(plot_df) == 0:
        plot_df = dfp
    else:
        plot_df = plot_df.copy().reset_index(drop=True)
        plot_df['t_rel_s'] = plot_df['t_rel_s'] - float(plot_df.loc[0, 't_rel_s'])
        plot_df['x_rel_mm'] = plot_df['x_rel_mm'] - float(plot_df.loc[0, 'x_rel_mm'])
        plot_df['y_rel_mm'] = plot_df['y_rel_mm'] - float(plot_df.loc[0, 'y_rel_mm'])
        plot_df['disp_mm'] = np.sqrt(plot_df['x_rel_mm'] ** 2 + plot_df['y_rel_mm'] ** 2)
        dx_step = pd.Series(plot_df['x_rel_mm']).diff().fillna(0.0)
        dy_step = pd.Series(plot_df['y_rel_mm']).diff().fillna(0.0)
        plot_df['step_dist_mm'] = np.sqrt(dx_step ** 2 + dy_step ** 2)
        plot_df['path_length_mm'] = plot_df['step_dist_mm'].cumsum()
        if 'x_rel_smooth_mm' in plot_df.columns:
            plot_df['x_rel_smooth_mm'] = plot_df['x_rel_smooth_mm'] - float(plot_df.loc[0, 'x_rel_smooth_mm'])
        if 'y_rel_smooth_mm' in plot_df.columns:
            plot_df['y_rel_smooth_mm'] = plot_df['y_rel_smooth_mm'] - float(plot_df.loc[0, 'y_rel_smooth_mm'])
    save_plot(plot_df['t_rel_s'], plot_df['x_rel_mm'], 'time from first detection (s)', 'x displacement (mm)', f'{object_mode} x displacement vs time', reg('x_vs_time.png'))
    save_plot(plot_df['t_rel_s'], plot_df['y_rel_mm'], 'time from first detection (s)', 'y displacement (mm)', f'{object_mode} y displacement vs time', reg('y_vs_time.png'))
    save_plot(plot_df['t_rel_s'], plot_df['disp_mm'], 'time from first detection (s)', 'displacement (mm)', f'{object_mode} displacement vs time', reg('displacement_vs_time.png'))
    save_plot(plot_df['t_rel_s'], plot_df['speed_mm_s'], 'time from first detection (s)', 'speed (mm/s)', f'{object_mode} speed vs time', reg('speed_vs_time.png'))
    plt.figure()
    plt.plot(plot_df['x_rel_mm'], plot_df['y_rel_mm'])
    plt.xlabel('x displacement (mm)')
    if object_mode == 'sphere':
        plt.ylabel('y displacement (mm, plot follows video motion)')
    else:
        plt.ylabel('y displacement (mm, down positive)')
    plt.title(f'{object_mode} trajectory')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.grid(True)
    tp = reg('trajectory.png')
    plt.savefig(tp, dpi=200, bbox_inches='tight')
    plt.close()
    save_plot([plot_df['t_rel_s'], plot_df['t_rel_s']], [plot_df['vx_mm_s'], plot_df['vy_mm_s']], 'time from first detection (s)', 'velocity (mm/s)', f'{object_mode} velocity components vs time', reg('vx_vy_vs_time.png'), legend=['vx', 'vy'])
    save_plot([plot_df['t_rel_s'], plot_df['t_rel_s']], [plot_df['ax_mm_s2'], plot_df['ay_mm_s2']], 'time from first detection (s)', 'acceleration (mm/s^2)', f'{object_mode} acceleration components vs time', reg('ax_ay_vs_time.png'), legend=['ax', 'ay'])
    if object_mode == 'rod':
        save_plot(plot_df['t_rel_s'], plot_df['angle_smooth_deg'], 'time from first detection (s)', 'rod angle (deg)', 'rod angle vs time', reg('angle_vs_time.png'))
        save_plot(plot_df['t_rel_s'], plot_df['omega_deg_s'], 'time from first detection (s)', 'angular velocity (deg/s)', 'rod angular velocity vs time', reg('omega_vs_time.png'))
        plt.figure()
        plt.plot(plot_df['y_rel_mm'], plot_df['angle_smooth_deg'], '.-')
        plt.xlabel('y displacement (mm)')
        plt.ylabel('rod angle (deg)')
        plt.title('rod angle vs depth')
        plt.grid(True)
        p = reg('angle_vs_depth.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
    xlsx = os.path.join(out_dir, 'processed_with_plots.xlsx')
    wb = Workbook()
    ws = wb.active
    ws.title = 'processed'
    ws.append(list(dfp.columns))
    for _, r in dfp.iterrows():
        ws.append(list(r.values))
    img_ws = wb.create_sheet('plots')
    row = 1
    for p in pngs:
        if os.path.exists(p):
            img = XLImage(p)
            img.anchor = f'A{row}'
            img_ws.add_image(img)
            row += 24
    wb.save(xlsx)
    return (pngs, xlsx)

def prompt_yes_no(msg, default=False):
    """Handle the prompt yes no step used by this script."""
    s = input(msg).strip().lower()
    if not s:
        return default
    return s in {'y', 'yes', '1', 'true'}

def get_runtime_config():
    """Handle the get runtime config step used by this script."""
    if not USE_RUNTIME_PROMPTS:
        return {'video_path': DEFAULT_VIDEO_PATH, 'calibration_json': DEFAULT_CALIBRATION_JSON, 'out_dir': DEFAULT_OUT_DIR, 'real_fps': DEFAULT_REAL_FPS, 'frame_stride': DEFAULT_FRAME_STRIDE, 'start_frame': DEFAULT_START_FRAME, 'object_mode': DEFAULT_OBJECT_MODE, 'sphere_color_mode': DEFAULT_SPHERE_COLOR_MODE, 'disc_diameter_mm': DEFAULT_DISC_DIAMETER_MM, 'sphere_diameter_mm': DEFAULT_SPHERE_DIAMETER_MM, 'use_preset_click': DEFAULT_USE_PRESET_CLICK, 'preset_click': DEFAULT_PRESET_CLICK, 'start_search_frames': DEFAULT_START_SEARCH_FRAMES, 'save_debug_images': DEFAULT_SAVE_DEBUG_IMAGES, 'num_debug_images': DEFAULT_NUM_DEBUG_IMAGES, 'enable_sphere_background_subtraction': DEFAULT_ENABLE_SPHERE_BACKGROUND_SUBTRACTION, 'sphere_background_seconds': DEFAULT_SPHERE_BACKGROUND_SECONDS, 'sphere_background_samples': DEFAULT_SPHERE_BACKGROUND_SAMPLES, 'sphere_background_margin_frames': DEFAULT_SPHERE_BACKGROUND_MARGIN_FRAMES, 'sphere_background_keep_fraction': DEFAULT_SPHERE_BACKGROUND_KEEP_FRACTION, 'enable_imshow': ENABLE_IMSHOW}
    object_mode = input('Choose object mode [rod/disc/sphere]: ').strip().lower() or DEFAULT_OBJECT_MODE
    sphere_color_mode = DEFAULT_SPHERE_COLOR_MODE
    if object_mode == 'sphere':
        sphere_color_mode = input('Choose sphere type [red/white]: ').strip().lower() or DEFAULT_SPHERE_COLOR_MODE
    video_path = input(f'Video path [{DEFAULT_VIDEO_PATH}]: ').strip() or DEFAULT_VIDEO_PATH
    calibration_json = input(f'Calibration json path [{DEFAULT_CALIBRATION_JSON}]: ').strip() or DEFAULT_CALIBRATION_JSON
    out_dir = input(f'Output folder [{DEFAULT_OUT_DIR}]: ').strip() or DEFAULT_OUT_DIR
    real_fps = float(input(f'Real capture FPS [{DEFAULT_REAL_FPS}]: ').strip() or DEFAULT_REAL_FPS)
    frame_stride = int(input(f'Frame stride [{DEFAULT_FRAME_STRIDE}]: ').strip() or DEFAULT_FRAME_STRIDE)
    start_frame = int(input(f'Start frame [{DEFAULT_START_FRAME}]: ').strip() or DEFAULT_START_FRAME)
    disc_diameter_mm = DEFAULT_DISC_DIAMETER_MM
    sphere_diameter_mm = DEFAULT_SPHERE_DIAMETER_MM
    if object_mode == 'disc':
        disc_diameter_mm = float(input(f'Disc diameter in mm [{DEFAULT_DISC_DIAMETER_MM}]: ').strip() or DEFAULT_DISC_DIAMETER_MM)
    if object_mode == 'sphere':
        sphere_diameter_mm = float(input(f'Sphere diameter in mm [{DEFAULT_SPHERE_DIAMETER_MM}]: ').strip() or DEFAULT_SPHERE_DIAMETER_MM)
    use_preset_click = prompt_yes_no(f'Use preset startup click? [y/N]: ', default=DEFAULT_USE_PRESET_CLICK)
    preset_click = DEFAULT_PRESET_CLICK
    if use_preset_click:
        s = input(f'Preset click as x,y [{DEFAULT_PRESET_CLICK[0]},{DEFAULT_PRESET_CLICK[1]}]: ').strip()
        if s:
            x, y = s.split(',')
            preset_click = (int(x), int(y))
    save_debug_images = prompt_yes_no(f'Save debug images? [Y/n]: ', default=DEFAULT_SAVE_DEBUG_IMAGES)
    num_debug_images = int(input(f'Number of debug images [{DEFAULT_NUM_DEBUG_IMAGES}]: ').strip() or DEFAULT_NUM_DEBUG_IMAGES)
    return {'video_path': video_path, 'calibration_json': calibration_json, 'out_dir': out_dir, 'real_fps': real_fps, 'frame_stride': frame_stride, 'start_frame': start_frame, 'object_mode': object_mode, 'sphere_color_mode': sphere_color_mode, 'disc_diameter_mm': disc_diameter_mm, 'sphere_diameter_mm': sphere_diameter_mm, 'use_preset_click': use_preset_click, 'preset_click': preset_click, 'start_search_frames': DEFAULT_START_SEARCH_FRAMES, 'save_debug_images': save_debug_images, 'num_debug_images': num_debug_images, 'enable_sphere_background_subtraction': DEFAULT_ENABLE_SPHERE_BACKGROUND_SUBTRACTION, 'sphere_background_seconds': DEFAULT_SPHERE_BACKGROUND_SECONDS, 'sphere_background_samples': DEFAULT_SPHERE_BACKGROUND_SAMPLES, 'sphere_background_margin_frames': DEFAULT_SPHERE_BACKGROUND_MARGIN_FRAMES, 'sphere_background_keep_fraction': DEFAULT_SPHERE_BACKGROUND_KEEP_FRACTION, 'enable_imshow': ENABLE_IMSHOW}

def main():
    """Run the main entry point for this script."""
    cfg = get_runtime_config()
    os.makedirs(cfg['out_dir'], exist_ok=True)
    _, H, px_per_mm, _, _ = load_calibration(cfg['calibration_json'])
    cap = cv2.VideoCapture(cfg['video_path'])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg['video_path']}")
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(file_fps) or file_fps <= 1e-06:
        file_fps = 30.0
    fps_used = float(cfg['real_fps']) if cfg['real_fps'] is not None else float(file_fps)
    sphere_background_gray = None
    sphere_background_info = None
    if cfg['object_mode'] == 'sphere' and bool(cfg.get('enable_sphere_background_subtraction', DEFAULT_ENABLE_SPHERE_BACKGROUND_SUBTRACTION)):
        background_seconds = float(cfg.get('sphere_background_seconds', DEFAULT_SPHERE_BACKGROUND_SECONDS))
        if background_seconds > 0.0:
            background_window_frames = max(1, int(round(background_seconds * fps_used)))
        else:
            background_window_frames = 0
        if background_window_frames > 0:
            sphere_background_gray, sphere_background_info = build_average_background_gray(cap, cfg['start_frame'], background_window_frames, int(cfg.get('sphere_background_samples', DEFAULT_SPHERE_BACKGROUND_SAMPLES)), margin_frames=int(cfg.get('sphere_background_margin_frames', DEFAULT_SPHERE_BACKGROUND_MARGIN_FRAMES)), keep_fraction=float(cfg.get('sphere_background_keep_fraction', DEFAULT_SPHERE_BACKGROUND_KEEP_FRACTION)))
        else:
            sphere_background_info = {'status': 'disabled', 'reason': 'sphere_background_seconds<=0'}
        if sphere_background_gray is not None:
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_background_average.png'), np.clip(sphere_background_gray, 0, 255).astype(np.uint8))
        if sphere_background_info is not None:
            with open(os.path.join(cfg['out_dir'], 'debug_background_info.json'), 'w', encoding='utf-8') as f:
                json.dump(sphere_background_info, f, indent=2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg['start_frame'])
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Could not read start frame from video.')
    acquire_params = get_param_block(cfg['object_mode'], cfg['sphere_color_mode'], 'acquire')
    if cfg['use_preset_click']:
        init_click = cfg['preset_click']
    else:
        mode_label = cfg['object_mode'] if cfg['object_mode'] != 'sphere' else f"{cfg['sphere_color_mode']} sphere"
        if cfg['object_mode'] == 'sphere':
            start_binary = build_binary_frame_for_display(frame, cfg['object_mode'], cfg['sphere_color_mode'], acquire_params, sphere_background_gray=sphere_background_gray)
            init_click = pick_initial_click_binary(start_binary, mode_label)
        else:
            init_click = pick_initial_click_raw(frame, mode_label)
    dets, debug = detect_in_raw_roi(frame, init_click[0], init_click[1], H, px_per_mm, cfg['object_mode'], cfg['sphere_color_mode'], cfg['disc_diameter_mm'], cfg['sphere_diameter_mm'], acquire_params, rod_anchor_uv=None, sphere_background_gray=sphere_background_gray)
    if len(dets) == 0 and int(cfg.get('start_search_frames', 0)) > 0:
        found_frame_idx, found_frame, dets, debug = search_start_detection(cap, cfg['start_frame'], init_click, H, px_per_mm, cfg['object_mode'], cfg['sphere_color_mode'], cfg['disc_diameter_mm'], cfg['sphere_diameter_mm'], acquire_params, cfg['start_search_frames'], sphere_background_gray=sphere_background_gray)
        if len(dets) > 0:
            cfg['start_frame'] = int(found_frame_idx)
            frame = found_frame
            print(f"Adjusted start frame from seed to detected frame {cfg['start_frame']}.")
    if len(dets) == 0:
        cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_raw.png'), frame)
        if 'roi_binary_map' in debug:
            if debug.get('roi_binary_background') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_background.png'), debug['roi_binary_background'])
            if debug.get('roi_binary_deviation') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_deviation.png'), debug['roi_binary_deviation'])
            if debug.get('roi_binary_blur') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_blur.png'), debug['roi_binary_blur'])
            if debug.get('roi_binary_bg') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_bg.png'), debug['roi_binary_bg'])
            if debug.get('roi_binary_map') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_map.png'), debug['roi_binary_map'])
            if debug.get('roi_binary_bright') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_bright.png'), debug['roi_binary_bright'])
            if debug.get('roi_binary_gray') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_gray.png'), debug['roi_binary_gray'])
            if debug.get('roi_binary_raw') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_raw.png'), debug['roi_binary_raw'])
            if debug.get('roi_binary_clean') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_clean.png'), debug['roi_binary_clean'])
            if debug.get('roi_binary_filled') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_filled.png'), debug['roi_binary_filled'])
        elif 'roi_blur' in debug:
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_blur.png'), debug['roi_blur'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_edges.png'), debug['roi_edges'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_close.png'), debug['roi_close'])
        cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_bw.png'), debug['bw'])
        raise RuntimeError('No valid contour found near the selected start point on the binarised frame. Adjust start frame or detection settings.')
    track = ObjectTrack(dets[0], time_s=0.0, object_mode=cfg['object_mode'])
    if cfg['object_mode'] == 'sphere':
        track.update_progress_extrema(px_per_mm)
    if cfg['object_mode'] == 'rod':
        track.set_rod_anchor_from_click(init_click, H)
    h_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_mp4 = os.path.join(cfg['out_dir'], 'annotated_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_mp4, fourcc, float(file_fps / cfg['frame_stride']), (w_raw, h_raw), True)
    out_binary_mp4 = os.path.join(cfg['out_dir'], 'annotated_binary.mp4')
    binary_vw = cv2.VideoWriter(out_binary_mp4, fourcc, float(file_fps / cfg['frame_stride']), (w_raw, h_raw), True) if cfg['object_mode'] == 'sphere' else None
    rows = []
    frame_idx = cfg['start_frame']
    out_idx = 0
    debug_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg['start_frame'])
    print(f"Mode = {cfg['object_mode']}")
    if cfg['object_mode'] == 'sphere':
        print(f"Sphere subtype = {cfg['sphere_color_mode']}")
    print(f'Using FPS for calculations = {fps_used:.3f}')
    print(f'Using FPS for output video = {file_fps:.3f}')
    print(f'Rectified scale = {1.0 / px_per_mm:.4f} mm/px')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg['frame_stride'] > 1 and (frame_idx - cfg['start_frame']) % cfg['frame_stride'] != 0:
            frame_idx += 1
            continue
        t_s = out_idx / float(fps_used / cfg['frame_stride'])
        pred_x, pred_y = track.predict()
        stage = 'acquire' if out_idx < 8 else 'track'
        stage_params = get_param_block(cfg['object_mode'], cfg['sphere_color_mode'], stage)
        search_params = stage_params.copy()
        if cfg['object_mode'] == 'sphere':
            expand_per_miss = int(search_params.get('roi_expand_per_miss_px', 0))
            if expand_per_miss > 0 and track.misses > 0:
                search_params['roi_half_size_px'] = min(max(h_raw, w_raw), int(search_params['roi_half_size_px'] + track.misses * expand_per_miss))
        dets, debug = detect_in_raw_roi(frame, pred_x, pred_y, H, px_per_mm, cfg['object_mode'], cfg['sphere_color_mode'], cfg['disc_diameter_mm'], cfg['sphere_diameter_mm'], search_params, rod_anchor_uv=track.rod_anchor_uv if cfg['object_mode'] == 'rod' else None, sphere_background_gray=sphere_background_gray)
        if cfg['object_mode'] == 'sphere' and track.misses >= int(search_params.get('reacquire_fullframe_after_misses', 99)):
            reacquire_params = stage_params.copy()
            reacquire_params['roi_half_size_px'] = max(h_raw, w_raw)
            reacquire_dets, reacquire_debug = detect_in_raw_roi(frame, pred_x, pred_y, H, px_per_mm, cfg['object_mode'], cfg['sphere_color_mode'], cfg['disc_diameter_mm'], cfg['sphere_diameter_mm'], reacquire_params, rod_anchor_uv=None, sphere_background_gray=sphere_background_gray)
            if reacquire_dets:
                dets = reacquire_dets
                debug = reacquire_debug
        ann_binary = None
        if cfg['object_mode'] == 'sphere':
            full_binary = build_binary_frame_for_display(frame, cfg['object_mode'], cfg['sphere_color_mode'], stage_params, sphere_background_gray=sphere_background_gray)
            ann_binary = gray_to_bgr(full_binary)
        chosen_det = None
        best_score = np.inf
        for d in dets:
            score = score_detection(d, pred_x, pred_y, cfg['object_mode'], track_angle_deg=track.angle_deg, track_length_mm=track.length_mm, expected_disc_diameter_mm=cfg['disc_diameter_mm'], expected_sphere_diameter_mm=cfg['sphere_diameter_mm'])
            if score < best_score:
                best_score = score
                chosen_det = d
        found_this_frame = False
        max_assoc_dist = float(stage_params['max_assoc_dist_px'])
        if cfg['object_mode'] == 'sphere' and track.misses > 0:
            max_assoc_dist = min(float(max(h_raw, w_raw)), max_assoc_dist + track.misses * float(stage_params.get('assoc_expand_per_miss_px', 0.0)))
        if cfg['object_mode'] == 'sphere' and chosen_det is not None and sphere_detection_backtracks(track, chosen_det, px_per_mm):
            chosen_det = None
        if chosen_det is not None and math.hypot(pred_x - chosen_det['cx_raw'], pred_y - chosen_det['cy_raw']) <= max_assoc_dist:
            track.update(chosen_det)
            if cfg['object_mode'] == 'sphere':
                track.update_progress_extrema(px_per_mm)
            found_this_frame = True
        else:
            track.misses += 1
            track.cx_raw = float(track.kalman_x_raw)
            track.cy_raw = float(track.kalman_y_raw)
        if cfg['object_mode'] == 'sphere':
            exit_margin_px = float(stage_params.get('exit_margin_px', 0.0))
            exit_misses = int(stage_params.get('exit_misses', stage_params['max_misses']))
            if track.misses >= exit_misses and (track.kalman_x_raw < -exit_margin_px or track.kalman_x_raw > w_raw + exit_margin_px or track.kalman_y_raw < -exit_margin_px or (track.kalman_y_raw > h_raw + exit_margin_px)):
                print('Tracking ended after the sphere exited the frame.')
                break
            if track.max_abs_x_progress_mm >= DEFAULT_SPHERE_BACKTRACK_MIN_PROGRESS_MM and track.misses >= DEFAULT_SPHERE_TAIL_MISS_STOP:
                print('Tracking ended after sustained loss near the sphere trajectory frontier.')
                break
        if track.misses > stage_params['max_misses']:
            print('Tracking lost: maximum missed frames exceeded.')
            break
        ann = frame.copy()
        x0r, y0r, x1r, y1r = debug['roi_box']
        cv2.rectangle(ann, (x0r, y0r), (x1r, y1r), (255, 255, 0), 1)
        cv2.drawMarker(ann, (int(round(pred_x)), int(round(pred_y))), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        if ann_binary is not None:
            cv2.rectangle(ann_binary, (x0r, y0r), (x1r, y1r), (255, 255, 0), 1)
            cv2.drawMarker(ann_binary, (int(round(pred_x)), int(round(pred_y))), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        meas_cx = track.cx_raw
        meas_cy = track.cy_raw
        kf_cx = track.kalman_x_raw
        kf_cy = track.kalman_y_raw
        centre_for_metrics_raw = np.array([[kf_cx, kf_cy]], dtype=np.float64)
        kf_rect = transform_points_homography(centre_for_metrics_raw, H)[0]
        x_kalman_mm = px_to_mm_rectified(kf_rect[0], px_per_mm)
        y_kalman_mm = px_to_mm_rectified(kf_rect[1], px_per_mm)
        x0_mm = px_to_mm_rectified(track.first_centre_rectified_px[0], px_per_mm)
        y0_mm = px_to_mm_rectified(track.first_centre_rectified_px[1], px_per_mm)
        x_rel_mm = x_kalman_mm - x0_mm
        y_rel_mm = y_kalman_mm - y0_mm
        disp_mm = math.hypot(x_rel_mm, y_rel_mm)
        if found_this_frame:
            box_raw = track.box_raw.astype(np.int32)
            x_rect_px = track.centre_rectified_px[0]
            y_rect_px = track.centre_rectified_px[1]
            x_mm = px_to_mm_rectified(x_rect_px, px_per_mm)
            y_mm = px_to_mm_rectified(y_rect_px, px_per_mm)
            cv2.polylines(ann, [box_raw], True, (0, 255, 0), 2)
            cv2.circle(ann, (int(round(meas_cx)), int(round(meas_cy))), 4, (0, 255, 0), -1)
            cv2.circle(ann, (int(round(kf_cx)), int(round(kf_cy))), 4, (255, 0, 255), -1)
            if ann_binary is not None:
                cv2.polylines(ann_binary, [box_raw], True, (0, 255, 0), 2)
                cv2.circle(ann_binary, (int(round(meas_cx)), int(round(meas_cy))), 4, (0, 255, 0), -1)
                cv2.circle(ann_binary, (int(round(kf_cx)), int(round(kf_cy))), 4, (255, 0, 255), -1)
            cv2.putText(ann, f'dx={x_rel_mm:.1f} dy={y_rel_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if ann_binary is not None:
                cv2.putText(ann_binary, f'dx={x_rel_mm:.1f} dy={y_rel_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if cfg['object_mode'] == 'rod':
                cv2.putText(ann, f'theta={track.angle_deg:.1f} deg', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(ann, f'L={track.length_mm:.1f} W={track.width_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(ann, f'D={track.diameter_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if ann_binary is not None:
                    cv2.putText(ann_binary, f'D={track.diameter_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(ann, f's={disp_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if ann_binary is not None:
                cv2.putText(ann_binary, f's={disp_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        else:
            x_rect_px = np.nan
            y_rect_px = np.nan
            x_mm = np.nan
            y_mm = np.nan
            cv2.circle(ann, (int(round(kf_cx)), int(round(kf_cy))), 4, (0, 0, 255), -1)
            cv2.putText(ann, f"{cfg['object_mode']} predicted only", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if ann_binary is not None:
                cv2.circle(ann_binary, (int(round(kf_cx)), int(round(kf_cy))), 4, (0, 0, 255), -1)
                cv2.putText(ann_binary, f"{cfg['object_mode']} predicted only", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        row = {'frame_idx': frame_idx, 'out_idx': out_idx, 'time_s': t_s, 'found_this_frame': int(found_this_frame), 'cx_raw_px': meas_cx, 'cy_raw_px': meas_cy, 'cx_kalman_raw_px': kf_cx, 'cy_kalman_raw_px': kf_cy, 'x_rectified_px': x_rect_px, 'y_rectified_px': y_rect_px, 'x_mm': x_mm, 'y_mm': y_mm, 'x_kalman_mm': x_kalman_mm, 'y_kalman_mm': y_kalman_mm, 'x0_mm': x0_mm, 'y0_mm': y0_mm, 'x_rel_mm': x_rel_mm, 'y_rel_mm': y_rel_mm, 'disp_mm': disp_mm, 'area_px2': track.area_px2, 'solidity': track.solidity, 'aspect_ratio': track.aspect_ratio, 'circularity': track.circularity, 'misses': track.misses, 'p0x_raw': track.box_raw[0, 0], 'p0y_raw': track.box_raw[0, 1], 'p1x_raw': track.box_raw[1, 0], 'p1y_raw': track.box_raw[1, 1], 'p2x_raw': track.box_raw[2, 0], 'p2y_raw': track.box_raw[2, 1], 'p3x_raw': track.box_raw[3, 0], 'p3y_raw': track.box_raw[3, 1]}
        if cfg['object_mode'] == 'rod':
            row.update({'length_mm': track.length_mm, 'width_mm': track.width_mm, 'angle_deg': track.angle_deg})
        else:
            row.update({'diameter_mm': track.diameter_mm})
        rows.append(row)
        vw.write(ann)
        if binary_vw is not None and ann_binary is not None:
            binary_vw.write(ann_binary)
        if cfg['save_debug_images'] and debug_count < cfg['num_debug_images']:
            cv2.imwrite(os.path.join(cfg['out_dir'], f'raw_{out_idx:04d}.png'), frame)
            if 'roi_binary_map' in debug:
                if debug.get('roi_binary_background') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_background_{out_idx:04d}.png'), debug['roi_binary_background'])
                if debug.get('roi_binary_deviation') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_deviation_{out_idx:04d}.png'), debug['roi_binary_deviation'])
                if debug.get('roi_binary_blur') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_blur_{out_idx:04d}.png'), debug['roi_binary_blur'])
                if debug.get('roi_binary_bg') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_bg_{out_idx:04d}.png'), debug['roi_binary_bg'])
                if debug.get('roi_binary_map') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_map_{out_idx:04d}.png'), debug['roi_binary_map'])
                if debug.get('roi_binary_bright') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_bright_{out_idx:04d}.png'), debug['roi_binary_bright'])
                if debug.get('roi_binary_gray') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_gray_{out_idx:04d}.png'), debug['roi_binary_gray'])
                if debug.get('roi_binary_raw') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_raw_{out_idx:04d}.png'), debug['roi_binary_raw'])
                if debug.get('roi_binary_clean') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_clean_{out_idx:04d}.png'), debug['roi_binary_clean'])
                if debug.get('roi_binary_filled') is not None:
                    cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_filled_{out_idx:04d}.png'), debug['roi_binary_filled'])
            elif 'roi_blur' in debug:
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_blur_{out_idx:04d}.png'), debug['roi_blur'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_edges_{out_idx:04d}.png'), debug['roi_edges'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_close_{out_idx:04d}.png'), debug['roi_close'])
            if 'roi_hsv_mask' in debug:
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_hsv_mask_{out_idx:04d}.png'), debug['roi_hsv_mask'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_red_excess_{out_idx:04d}.png'), debug['roi_red_excess'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_red_excess_blur_{out_idx:04d}.png'), debug['roi_red_excess_blur'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_rex_mask_{out_idx:04d}.png'), debug['roi_rex_mask'])
            cv2.imwrite(os.path.join(cfg['out_dir'], f'bw_{out_idx:04d}.png'), debug['bw'])
            cv2.imwrite(os.path.join(cfg['out_dir'], f'ann_{out_idx:04d}.png'), ann)
            if ann_binary is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], f'ann_binary_{out_idx:04d}.png'), ann_binary)
            debug_count += 1
        if cfg.get('enable_imshow', True):
            cv2.imshow('annotated raw', ann)
            if ann_binary is not None:
                cv2.imshow('annotated binary', ann_binary)
            if cv2.waitKey(1) & 255 == 27:
                break
        frame_idx += 1
        out_idx += 1
    cap.release()
    vw.release()
    if binary_vw is not None:
        binary_vw.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    df = pd.DataFrame(rows)
    raw_csv = os.path.join(cfg['out_dir'], 'tracks_raw.csv')
    df.to_csv(raw_csv, index=False)
    print('Saved:', raw_csv)
    print('Saved:', out_mp4)
    if binary_vw is not None:
        print('Saved:', out_binary_mp4)
    if len(df) == 0:
        raise RuntimeError('No detections were recorded.')
    dfp = add_kinematics(df, cfg['object_mode'])
    proc_csv = os.path.join(cfg['out_dir'], 'tracks_processed.csv')
    dfp.to_csv(proc_csv, index=False)
    print('Saved:', proc_csv)
    _, xlsx = save_outputs(dfp, cfg['out_dir'], cfg['object_mode'])
    print('Saved:', xlsx)
    print('Plots saved to:', cfg['out_dir'])

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
