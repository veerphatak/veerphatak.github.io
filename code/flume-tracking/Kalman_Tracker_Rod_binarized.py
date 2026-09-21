"""
Core tracking script: Kalman_Tracker_Rod_binarized.py

Purpose
-------
Tracks rod trajectories from binarised video frames using contour extraction, fitted quadrilaterals and Kalman smoothing.

Inputs
------
Rod videos (.mp4), calibration JSON, start frame and initial particle centre.

Outputs
-------
Per-frame CSV trajectory data, annotated videos, binary tracking outputs and diagnostic plots.

Methodological notes
--------------------
Rod orientation is inferred from the fitted particle silhouette; motion is analysed in the calibrated image plane; short detection gaps are bridged only when consistent with the predicted trajectory.

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
DEFAULT_VIDEO_PATH = 'C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\rod drop 2.mp4'
DEFAULT_CALIBRATION_JSON = 'C:\\Users\\User\\Downloads\\rod_calibration_snap_fit\\calibration.json'
DEFAULT_OUT_DIR = 'C:\\IP Work\\4.5_Inchps_rod_binarised_widthlocked_5mm_fullbatch_v2\\rod_drop_2_binarized_tracker_output_widthlocked_5mm'
DEFAULT_REAL_FPS = 240.0
DEFAULT_FRAME_STRIDE = 1
DEFAULT_START_FRAME = 0
DEFAULT_OBJECT_MODE = 'rod'
DEFAULT_SPHERE_COLOR_MODE = 'red'
DEFAULT_ROD_DIAMETER_MM = 5.0
DEFAULT_DISC_DIAMETER_MM = 20.0
DEFAULT_SPHERE_DIAMETER_MM = 10.0
DEFAULT_USE_PRESET_CLICK = False
DEFAULT_PRESET_CLICK = (500, 300)
DEFAULT_SAVE_DEBUG_IMAGES = True
DEFAULT_NUM_DEBUG_IMAGES = 12
DEFAULT_SMOOTH_WINDOW = 7
DEFAULT_ROD_LENGTH_MM = 25.0
DEFAULT_ENABLE_ROD_BACKGROUND_SUBTRACTION = True
DEFAULT_ROD_BACKGROUND_SECONDS = 2.5
DEFAULT_ROD_BACKGROUND_SAMPLES = 24
DEFAULT_ROD_BACKGROUND_MARGIN_FRAMES = 24
DEFAULT_ROD_BACKGROUND_KEEP_FRACTION = 0.7
DEFAULT_ROD_BACKTRACK_MIN_PROGRESS_MM = 20.0
DEFAULT_ROD_BACKTRACK_TOL_X_MM = 5.0
DEFAULT_ROD_BACKTRACK_TOL_Y_MM = 5.0
DEFAULT_ROD_SEVERE_BACKTRACK_X_MM = 12.0
DEFAULT_ROD_SEVERE_BACKTRACK_Y_MM = 12.0
DEFAULT_ROD_TAIL_MISS_STOP = 8
# Centralised processing parameters. These values were tuned for the final reported videos.
PARAMS = {'rod': {'acquire': {'roi_half_size_px': 150, 'binary_blur_k': 5, 'binary_bg_blur_k': 31, 'binary_min_threshold': 8, 'open_k': 3, 'close_k': 5, 'close_iter': 1, 'min_area_px': 6, 'max_area_px': 40000, 'min_solidity': 0.02, 'min_fill_ratio': 0.15, 'min_aspect_ratio': 1.05, 'max_aspect_ratio': 50.0, 'rod_length_target_mm': DEFAULT_ROD_LENGTH_MM, 'rod_short_length_penalty_mm': 10.0, 'rod_binary_short_penalty': 7.0, 'rod_pair_bonus': 6.0, 'min_length_mm': 3.0, 'max_length_mm': 50.0, 'rod_width_target_factor': 1.0, 'rod_width_min_factor': 0.75, 'rod_width_max_factor': 1.85, 'rod_width_lock_min_factor': 0.88, 'rod_width_lock_max_factor': 1.12, 'rod_regularize_min_aspect_ratio': 2.0, 'rod_regularize_max_circularity': 0.72, 'circular_noise_min_circularity': 0.65, 'circular_noise_max_aspect_ratio': 2.4, 'circular_noise_max_diameter_factor': 1.35, 'min_component_area_px': 4, 'max_pair_components': 6, 'pair_max_gap_px': 120.0, 'pair_min_axis_alignment': 0.55, 'edge_sliver_border_margin_px': 1, 'edge_sliver_max_minor_px': 18, 'edge_sliver_min_aspect_ratio': 4.0, 'edge_sliver_max_rectified_fill_ratio': 0.45, 'miss_roi_growth_px': 10, 'miss_assoc_growth_px': 8.0, 'miss_max_roi_half_size_px': 210, 'max_assoc_dist_px': 95.0, 'max_misses': 20}, 'track': {'roi_half_size_px': 150, 'binary_blur_k': 5, 'binary_bg_blur_k': 31, 'binary_min_threshold': 10, 'open_k': 3, 'close_k': 3, 'close_iter': 1, 'min_area_px': 8, 'max_area_px': 30000, 'min_solidity': 0.04, 'min_fill_ratio': 0.18, 'min_aspect_ratio': 1.15, 'max_aspect_ratio': 35.0, 'rod_length_target_mm': DEFAULT_ROD_LENGTH_MM, 'rod_short_length_penalty_mm': 9.0, 'rod_binary_short_penalty': 8.0, 'rod_pair_bonus': 7.0, 'min_length_mm': 4.0, 'max_length_mm': 60.0, 'rod_width_target_factor': 1.0, 'rod_width_min_factor': 0.75, 'rod_width_max_factor': 1.85, 'rod_width_lock_min_factor': 0.88, 'rod_width_lock_max_factor': 1.12, 'rod_regularize_min_aspect_ratio': 2.2, 'rod_regularize_max_circularity': 0.68, 'circular_noise_min_circularity': 0.68, 'circular_noise_max_aspect_ratio': 2.2, 'circular_noise_max_diameter_factor': 1.3, 'min_component_area_px': 4, 'max_pair_components': 6, 'pair_max_gap_px': 110.0, 'pair_min_axis_alignment': 0.55, 'edge_sliver_border_margin_px': 1, 'edge_sliver_max_minor_px': 18, 'edge_sliver_min_aspect_ratio': 4.0, 'edge_sliver_max_rectified_fill_ratio': 0.45, 'miss_roi_growth_px': 12, 'miss_assoc_growth_px': 10.0, 'miss_max_roi_half_size_px': 230, 'max_assoc_dist_px': 85.0, 'max_misses': 15}}, 'disc': {'acquire': {'roi_half_size_px': 150, 'edge_low': 20, 'edge_high': 90, 'close_k': 9, 'close_iter': 3, 'min_area_px': 10, 'max_area_px': 60000, 'min_solidity': 0.1, 'min_circularity': 0.22, 'aspect_tol': 3.5, 'min_diameter_factor': 0.45, 'max_diameter_factor': 2.0, 'max_assoc_dist_px': 95.0, 'max_misses': 18}, 'track': {'roi_half_size_px': 120, 'edge_low': 25, 'edge_high': 100, 'close_k': 7, 'close_iter': 2, 'min_area_px': 14, 'max_area_px': 50000, 'min_solidity': 0.15, 'min_circularity': 0.3, 'aspect_tol': 2.8, 'min_diameter_factor': 0.55, 'max_diameter_factor': 1.7, 'max_assoc_dist_px': 75.0, 'max_misses': 15}}, 'sphere_red': {'acquire': {'roi_half_size_px': 180, 'hsv_lower1': [0, 70, 35], 'hsv_upper1': [16, 255, 255], 'hsv_lower2': [165, 70, 35], 'hsv_upper2': [180, 255, 255], 'red_excess_blur_k': 5, 'red_excess_thresh': 28, 'use_otsu': True, 'open_k': 3, 'close_k': 7, 'close_iter': 2, 'min_area_px': 25, 'max_area_px': 12000, 'min_circularity': 0.45, 'min_solidity': 0.6, 'aspect_tol': 1.45, 'min_diameter_factor': 0.45, 'max_diameter_factor': 2.2, 'max_assoc_dist_px': 120.0, 'max_misses': 20}, 'track': {'roi_half_size_px': 120, 'hsv_lower1': [0, 75, 40], 'hsv_upper1': [15, 255, 255], 'hsv_lower2': [166, 75, 40], 'hsv_upper2': [180, 255, 255], 'red_excess_blur_k': 5, 'red_excess_thresh': 32, 'use_otsu': True, 'open_k': 3, 'close_k': 7, 'close_iter': 2, 'min_area_px': 30, 'max_area_px': 9000, 'min_circularity': 0.52, 'min_solidity': 0.55, 'aspect_tol': 1.35, 'min_diameter_factor': 0.55, 'max_diameter_factor': 2.2, 'max_assoc_dist_px': 120.0, 'max_misses': 20}}, 'sphere_white': {'acquire': {'roi_half_size_px': 130, 'edge_low': 15, 'edge_high': 75, 'close_k': 7, 'close_iter': 2, 'min_area_px': 8, 'max_area_px': 50000, 'min_circularity': 0.38, 'min_solidity': 0.1, 'aspect_tol': 2.0, 'min_diameter_factor': 0.6, 'max_diameter_factor': 1.7, 'max_assoc_dist_px': 90.0, 'max_misses': 15}, 'track': {'roi_half_size_px': 110, 'edge_low': 20, 'edge_high': 85, 'close_k': 7, 'close_iter': 2, 'min_area_px': 10, 'max_area_px': 40000, 'min_circularity': 0.45, 'min_solidity': 0.15, 'aspect_tol': 1.8, 'min_diameter_factor': 0.7, 'max_diameter_factor': 1.45, 'max_assoc_dist_px': 75.0, 'max_misses': 12}}}

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

def unwrap_angle_deg(angle_deg_series, period_deg=180.0):
    """Handle the unwrap angle deg step used by this script."""
    ang = np.asarray(angle_deg_series, dtype=float)
    if len(ang) == 0:
        return ang
    scale = 360.0 / float(period_deg)
    ang_rad = np.deg2rad(ang * scale)
    ang_unwrapped = np.unwrap(ang_rad)
    return np.rad2deg(ang_unwrapped) / scale

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

def safe_odd(value, minimum=1):
    """Handle the safe odd step used by this script."""
    k = max(int(round(value)), int(minimum))
    if k % 2 == 0:
        k += 1
    return k

def gray_to_bgr(image):
    """Handle the gray to bgr step used by this script."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()

def otsu_threshold_with_floor(gray_image, floor_value):
    """Handle the otsu threshold with floor step used by this script."""
    otsu_thr, bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_used = max(float(floor_value), float(otsu_thr))
    if threshold_used > otsu_thr + 1e-09:
        _, bw = cv2.threshold(gray_image, int(round(threshold_used)), 255, cv2.THRESH_BINARY)
    return (float(threshold_used), bw)

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
    center = np.asarray(center, dtype=float)
    p0 = center - 0.5 * long_len * u_long - 0.5 * short_len * u_short
    p1 = center + 0.5 * long_len * u_long - 0.5 * short_len * u_short
    p2 = center + 0.5 * long_len * u_long + 0.5 * short_len * u_short
    p3 = center - 0.5 * long_len * u_long + 0.5 * short_len * u_short
    return order_box_points(np.vstack([p0, p1, p2, p3]))

def regularize_rod_box_rectified(box_rectified, rod_diameter_mm, px_per_mm, params, mask_source, aspect_ratio, circularity):
    """Handle the regularize rod box rectified step used by this script."""
    if rod_diameter_mm is None or not np.isfinite(rod_diameter_mm) or rod_diameter_mm <= 0.0:
        return (np.asarray(box_rectified, dtype=float), False)
    center, u_long, u_short, long_len, short_len = rod_box_axes(box_rectified)
    target_short_len = float(rod_diameter_mm) * float(px_per_mm) * float(params.get('rod_width_target_factor', 1.0))
    lock_min_short_len = target_short_len * float(params.get('rod_width_lock_min_factor', 0.88))
    lock_max_short_len = target_short_len * float(params.get('rod_width_lock_max_factor', 1.12))
    rod_like = aspect_ratio >= float(params.get('rod_regularize_min_aspect_ratio', 2.0)) and circularity <= float(params.get('rod_regularize_max_circularity', 0.72))
    width_out_of_band = short_len < lock_min_short_len or short_len > lock_max_short_len
    if mask_source == 'paired_faces':
        should_regularize = width_out_of_band
    else:
        should_regularize = rod_like and width_out_of_band
    if not should_regularize:
        return (np.asarray(box_rectified, dtype=float), False)
    box_regularized = box_from_axes(center, u_long, u_short, long_len, target_short_len)
    return (box_regularized.astype(float), True)

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

def evenly_spaced_frame_indices(start_frame, end_frame, count):
    """Handle the evenly spaced frame indices step used by this script."""
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame < start_frame:
        return []
    available = end_frame - start_frame + 1
    count = max(1, min(int(count), available))
    if count == 1:
        return [int(round((start_frame + end_frame) / 2.0))]
    return sorted({int(round(v)) for v in np.linspace(start_frame, end_frame, num=count)})

def build_rod_average_background_gray(cap, reference_start_frame, window_frames, sample_frames, margin_frames=0, keep_fraction=0.7):
    """Build the rod average background gray used by this script."""
    if cap is None or not cap.isOpened():
        return (None, {'status': 'video_unavailable'})
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    current_pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))
    end_frame = int(reference_start_frame) - int(margin_frames)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    end_frame = max(-1, end_frame)
    if end_frame < 0:
        end_frame = max(0, int(reference_start_frame) - 1)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)
    if end_frame < 0:
        return (None, {'status': 'no_background_frames'})
    start_frame = max(0, end_frame - max(int(window_frames), 1) + 1)
    frame_indices = evenly_spaced_frame_indices(start_frame, end_frame, sample_frames)
    gray_frames = []
    valid_indices = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray.astype(np.float32))
        valid_indices.append(int(frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    if not gray_frames:
        return (None, {'status': 'no_background_samples_read', 'requested_frame_indices': frame_indices})
    stack = np.stack(gray_frames, axis=0)
    provisional_mean = stack.mean(axis=0)
    kept_order = np.arange(len(gray_frames))
    keep_fraction = float(np.clip(keep_fraction, 0.25, 1.0))
    if len(gray_frames) >= 6 and keep_fraction < 0.999:
        deviation_scores = np.mean(np.abs(stack - provisional_mean[None, :, :]), axis=(1, 2))
        keep_count = max(4, int(round(len(gray_frames) * keep_fraction)))
        keep_count = min(len(gray_frames), keep_count)
        kept_order = np.sort(np.argsort(deviation_scores)[:keep_count])
        stack = stack[kept_order]
    else:
        deviation_scores = np.zeros(len(gray_frames), dtype=float)
    background_mean = stack.mean(axis=0).astype(np.float32)
    kept_frame_indices = [valid_indices[int(i)] for i in np.asarray(kept_order, dtype=int)]
    background_info = {'status': 'ok', 'reference_start_frame': int(reference_start_frame), 'window_start_frame': int(start_frame), 'window_end_frame': int(end_frame), 'requested_sample_count': int(sample_frames), 'requested_frame_indices': frame_indices, 'valid_frame_indices': valid_indices, 'kept_frame_indices': kept_frame_indices, 'keep_fraction': keep_fraction, 'deviation_scores': [float(v) for v in np.asarray(deviation_scores, dtype=float)]}
    return (background_mean, background_info)

def preprocess_rod_binary_roi(gray_roi, params, background_gray_roi=None):
    """Handle the preprocess rod binary roi step used by this script."""
    blur_k = safe_odd(params.get('binary_blur_k', 5), minimum=1)
    bg_k = safe_odd(params.get('binary_bg_blur_k', 31), minimum=3)
    open_k = safe_odd(params.get('open_k', 3), minimum=1)
    close_k = safe_odd(params.get('close_k', 5), minimum=1)
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
    threshold_used, bw_raw = otsu_threshold_with_floor(bright_map, params.get('binary_min_threshold', 0))
    bw = bw_raw.copy()
    if open_k > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=1)
    if close_k > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=int(params.get('close_iter', 1)))
    return {'background_roi': background_roi_u8, 'deviation_map': deviation_map, 'blur': blur, 'bg': bg, 'bright_map': bright_map, 'threshold_used': float(threshold_used), 'bw_raw': bw_raw, 'bw': bw}

def contour_center_xy(contour):
    """Handle the contour center xy step used by this script."""
    M = cv2.moments(contour)
    if abs(M['m00']) > 1e-12:
        return np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=float)
    rect = cv2.minAreaRect(contour)
    return np.array(rect[0], dtype=float)

def rod_edge_sliver_reject(bbox_xywh, roi_shape, rectified_fill_ratio, params):
    """Handle the rod edge sliver reject step used by this script."""
    roi_h, roi_w = roi_shape[:2]
    x, y, w_box, h_box = [int(v) for v in bbox_xywh]
    border_margin = int(params.get('edge_sliver_border_margin_px', 1))
    max_minor_px = float(params.get('edge_sliver_max_minor_px', 18))
    min_aspect = float(params.get('edge_sliver_min_aspect_ratio', 4.0))
    max_rectified_fill_ratio = float(params.get('edge_sliver_max_rectified_fill_ratio', 0.45))
    touches_top = y <= border_margin
    touches_bottom = y + h_box >= roi_h - border_margin
    touches_left = x <= border_margin
    touches_right = x + w_box >= roi_w - border_margin
    horizontal_sliver = (touches_top or touches_bottom) and h_box <= max_minor_px and (w_box >= max(h_box, 1) * min_aspect)
    vertical_sliver = (touches_left or touches_right) and w_box <= max_minor_px and (h_box >= max(w_box, 1) * min_aspect)
    return (horizontal_sliver or vertical_sliver) and rectified_fill_ratio <= max_rectified_fill_ratio

def build_rod_binary_candidate_contours(bw, params):
    """Build the rod binary candidate contours used by this script."""
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_component_area = float(params.get('min_component_area_px', 1.0))
    contours = [c for c in contours if cv2.contourArea(c) >= min_component_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    candidates = [{'contour': c, 'mask_source': 'binary_component'} for c in contours]
    filled_mask = bw.copy()
    pair_limit = max(0, int(params.get('max_pair_components', 6)))
    pair_gap = float(params.get('pair_max_gap_px', 120.0))
    pair_contours = contours[:pair_limit]
    pair_centres = [contour_center_xy(c) for c in pair_contours]
    for i in range(len(pair_contours)):
        for j in range(i + 1, len(pair_contours)):
            centre_gap = float(np.linalg.norm(pair_centres[i] - pair_centres[j]))
            if centre_gap > pair_gap:
                continue
            merged = np.vstack([pair_contours[i], pair_contours[j]])
            hull = cv2.convexHull(merged)
            if hull is None or cv2.contourArea(hull) <= 0.0:
                continue
            candidates.append({'contour': hull, 'mask_source': 'paired_faces', 'pair_contours': [pair_contours[i].copy(), pair_contours[j].copy()]})
    return (candidates, filled_mask)

def contour_long_axis_unit(points_xy):
    """Handle the contour long axis unit step used by this script."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    box = order_box_points(cv2.boxPoints(rect))
    _, u_long, _, long_len, _ = rod_box_axes(box)
    if not np.isfinite(long_len) or long_len <= 1e-09:
        return None
    return u_long

def robust_endpoint_mean(points_xy, proj_long, high_end):
    """Handle the robust endpoint mean step used by this script."""
    if len(points_xy) == 0:
        return (None, np.nan)
    q = 0.85 if high_end else 0.15
    thresh = float(np.quantile(proj_long, q))
    if high_end:
        mask = proj_long >= thresh
    else:
        mask = proj_long <= thresh
    selected = points_xy[mask]
    if len(selected) == 0:
        selected = points_xy
    mean_pt = selected.mean(axis=0)
    mean_proj = float(np.mean(proj_long[mask])) if np.any(mask) else float(np.mean(proj_long))
    return (mean_pt.astype(float), mean_proj)

def build_paired_face_geometry(contour_a_global, contour_b_global, H, H_inv, px_per_mm, rod_diameter_mm, params):
    """Build the paired face geometry used by this script."""
    if rod_diameter_mm is None or not np.isfinite(rod_diameter_mm) or rod_diameter_mm <= 0.0:
        return None
    pts_a_rect = transform_points_homography(contour_a_global, H)
    pts_b_rect = transform_points_homography(contour_b_global, H)
    u_a = contour_long_axis_unit(pts_a_rect)
    u_b = contour_long_axis_unit(pts_b_rect)
    if u_a is None or u_b is None:
        return None
    if np.dot(u_a, u_b) < 0:
        u_b = -u_b
    axis_alignment = float(np.clip(np.dot(u_a, u_b), -1.0, 1.0))
    if axis_alignment < float(params.get('pair_min_axis_alignment', 0.55)):
        return None
    u_long = u_a + u_b
    u_long_norm = float(np.hypot(u_long[0], u_long[1]))
    if u_long_norm <= 1e-09:
        u_long = u_a
        u_long_norm = float(np.hypot(u_long[0], u_long[1]))
    if u_long_norm <= 1e-09:
        return None
    u_long = u_long / u_long_norm
    u_short = np.array([-u_long[1], u_long[0]], dtype=float)
    proj_a = pts_a_rect @ u_long
    proj_b = pts_b_rect @ u_long
    short_a = pts_a_rect @ u_short
    short_b = pts_b_rect @ u_short
    short_center_a = float(np.median(short_a))
    short_center_b = float(np.median(short_b))
    if short_center_a > short_center_b:
        pts_a_rect, pts_b_rect = (pts_b_rect, pts_a_rect)
        proj_a, proj_b = (proj_b, proj_a)
        short_a, short_b = (short_b, short_a)
        short_center_a, short_center_b = (short_center_b, short_center_a)
    low_a_pt, low_a = robust_endpoint_mean(pts_a_rect, proj_a, high_end=False)
    high_a_pt, high_a = robust_endpoint_mean(pts_a_rect, proj_a, high_end=True)
    low_b_pt, low_b = robust_endpoint_mean(pts_b_rect, proj_b, high_end=False)
    high_b_pt, high_b = robust_endpoint_mean(pts_b_rect, proj_b, high_end=True)
    if any((pt is None for pt in [low_a_pt, high_a_pt, low_b_pt, high_b_pt])):
        return None
    long_min = min(low_a, low_b)
    long_max = max(high_a, high_b)
    long_len = float(long_max - long_min)
    if not np.isfinite(long_len) or long_len <= 1e-09:
        return None
    short_len = float(rod_diameter_mm) * float(px_per_mm) * float(params.get('rod_width_target_factor', 1.0))
    center_long = 0.5 * (long_min + long_max)
    center_short = 0.5 * (short_center_a + short_center_b)
    center = center_long * u_long + center_short * u_short
    box_rectified = box_from_axes(center, u_long, u_short, long_len, short_len)
    metric_poly_rectified = np.vstack([low_a_pt, high_a_pt, high_b_pt, low_b_pt]).astype(float)
    metric_poly_global = transform_points_homography(metric_poly_rectified, H_inv).astype(float)
    fill_poly_global = transform_points_homography(box_rectified, H_inv).astype(float)
    return {'box_rectified': box_rectified.astype(float), 'fill_poly_global': fill_poly_global.astype(float), 'metric_poly_global': metric_poly_global.astype(float), 'metric_poly_rectified': metric_poly_rectified.astype(float), 'axis_alignment': axis_alignment}

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

def preprocess_red_roi(bgr_roi, params):
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
    bw = cv2.bitwise_and(hsv_mask, rex_mask)
    ok = int(params.get('open_k', 3))
    ck = int(params.get('close_k', 7))
    close_iter = int(params.get('close_iter', 2))
    if ok > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=1)
    if ck > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    return {'hsv': hsv, 'hsv_mask': hsv_mask, 'red_excess': red_excess, 'red_excess_blur': red_excess_blur, 'rex_mask': rex_mask, 'bw': bw}

def detect_in_raw_roi(frame_bgr, pred_x, pred_y, H, px_per_mm, object_mode, sphere_color_mode, disc_diameter_mm, sphere_diameter_mm, stage_params, rod_anchor_uv=None, rod_diameter_mm=None, rod_background_gray=None):
    """Detect the in raw roi used by this script."""
    h, w = frame_bgr.shape[:2]
    roi_half = stage_params['roi_half_size_px']
    x0 = max(0, int(round(pred_x - roi_half)))
    x1 = min(w, int(round(pred_x + roi_half)))
    y0 = max(0, int(round(pred_y - roi_half)))
    y1 = min(h, int(round(pred_y + roi_half)))
    roi = frame_bgr[y0:y1, x0:x1]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rod_background_roi = None
    if object_mode == 'rod' and rod_background_gray is not None:
        if rod_background_gray.shape[:2] == frame_bgr.shape[:2]:
            rod_background_roi = rod_background_gray[y0:y1, x0:x1]
    aux_a = aux_b = aux_c = None
    red_dbg = None
    rod_dbg = None
    rod_candidates = None
    if object_mode == 'rod':
        rod_dbg = preprocess_rod_binary_roi(gray_roi, stage_params, background_gray_roi=rod_background_roi)
        bw = rod_dbg['bw']
        rod_candidates, bw_filled = build_rod_binary_candidate_contours(bw, stage_params)
    elif object_mode == 'sphere' and sphere_color_mode == 'red':
        red_dbg = preprocess_red_roi(roi, stage_params)
        bw = red_dbg['bw']
    else:
        aux_a, aux_b, aux_c, bw = preprocess_edge_roi(gray_roi, stage_params)
    H_inv = np.linalg.inv(H) if object_mode == 'rod' else None
    if object_mode == 'rod':
        contour_entries = rod_candidates
    else:
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_entries = [{'contour': c, 'mask_source': 'raw_contour'} for c in contours]
    dets = []
    for entry in contour_entries:
        c = entry['contour']
        contour_local = np.asarray(c, dtype=np.float32).reshape(-1, 2)
        contour_global = contour_local.copy()
        contour_global[:, 0] += float(x0)
        contour_global[:, 1] += float(y0)
        pair_geometry = None
        pair_fill_local = None
        if object_mode == 'rod' and 'pair_contours' in entry:
            contour_a_global = np.asarray(entry['pair_contours'][0], dtype=np.float32).reshape(-1, 2)
            contour_b_global = np.asarray(entry['pair_contours'][1], dtype=np.float32).reshape(-1, 2)
            contour_a_global[:, 0] += float(x0)
            contour_a_global[:, 1] += float(y0)
            contour_b_global[:, 0] += float(x0)
            contour_b_global[:, 1] += float(y0)
            pair_geometry = build_paired_face_geometry(contour_a_global, contour_b_global, H, H_inv, px_per_mm, rod_diameter_mm, stage_params)
            if pair_geometry is not None:
                contour_global = np.asarray(pair_geometry['metric_poly_global'], dtype=np.float32).reshape(-1, 2)
                contour_local = contour_global.copy()
                contour_local[:, 0] -= float(x0)
                contour_local[:, 1] -= float(y0)
                pair_fill_local = np.asarray(pair_geometry['fill_poly_global'], dtype=np.float32).reshape(-1, 2)
                pair_fill_local[:, 0] -= float(x0)
                pair_fill_local[:, 1] -= float(y0)
        metric_contour = contour_local.astype(np.float32).reshape(-1, 1, 2)
        m = contour_metrics(metric_contour)
        if m is None:
            continue
        bbox_local = cv2.boundingRect(metric_contour)
        box_global = m['box'].copy()
        box_global[:, 0] += float(x0)
        box_global[:, 1] += float(y0)
        box_regularized = False
        if object_mode == 'rod':
            contour_rectified = transform_points_homography(contour_global, H).astype(np.float32)
            if pair_geometry is not None:
                box_rectified = pair_geometry['box_rectified'].copy()
                box_regularized = True
            else:
                rectified_rect = cv2.minAreaRect(contour_rectified.reshape(-1, 1, 2))
                box_rectified = order_box_points(cv2.boxPoints(rectified_rect))
            rect_long_px_raw = max((np.hypot(*box_rectified[(i + 1) % 4] - box_rectified[i]) for i in range(4)))
            rect_short_px_raw = min((np.hypot(*box_rectified[(i + 1) % 4] - box_rectified[i]) for i in range(4)))
            width_mm_raw = px_to_mm_rectified(rect_short_px_raw, px_per_mm)
            if rod_diameter_mm is not None and np.isfinite(rod_diameter_mm):
                box_rectified, width_regularized = regularize_rod_box_rectified(box_rectified, rod_diameter_mm, px_per_mm, stage_params, entry.get('mask_source', ''), m['aspect_ratio'], m['circularity'])
                box_regularized = box_regularized or width_regularized
            box_global = transform_points_homography(box_rectified, H_inv)
            box_center_raw = box_global.mean(axis=0)
            cx_global = float(box_center_raw[0])
            cy_global = float(box_center_raw[1])
        else:
            cx_global = float(m['cx'] + x0)
            cy_global = float(m['cy'] + y0)
            box_rectified = transform_points_homography(box_global, H)
            width_mm_raw = np.nan
        cx_track = cx_global
        cy_track = cy_global
        centre_rectified = transform_points_homography(np.array([[cx_track, cy_track]]), H)[0]
        rect_edges = [np.hypot(*box_rectified[(i + 1) % 4] - box_rectified[i]) for i in range(4)]
        length_rect_px = max(rect_edges)
        width_rect_px = min(rect_edges)
        box_area_rectified_px2 = max(length_rect_px * width_rect_px, 1e-09)
        contour_area_rectified_px2 = abs(float(cv2.contourArea(contour_rectified.reshape(-1, 1, 2)))) if object_mode == 'rod' else np.nan
        rectified_fill_ratio = contour_area_rectified_px2 / box_area_rectified_px2 if object_mode == 'rod' else np.nan
        length_mm = px_to_mm_rectified(length_rect_px, px_per_mm)
        width_mm = px_to_mm_rectified(width_rect_px, px_per_mm)
        circle_diameter_mm = px_to_mm_rectified(2.0 * m['circle_r'], px_per_mm)
        angle_deg = rod_angle_from_box(box_rectified)
        rect_long_px = max(float(m['rect_w']), float(m['rect_h']))
        rect_short_px = max(1e-09, min(float(m['rect_w']), float(m['rect_h'])))
        fill_ratio = float(m['area'] / max(rect_long_px * rect_short_px, 1e-09))
        keep = False
        obj_diameter_mm = None
        if object_mode == 'rod':
            width_ok = True
            circular_noise_reject = False
            edge_sliver_reject = False
            if rod_diameter_mm is not None and np.isfinite(rod_diameter_mm) and (rod_diameter_mm > 0.0):
                width_ok = width_mm_raw <= float(rod_diameter_mm) * float(stage_params.get('rod_width_max_factor', 1.85))
                circular_noise_reject = entry.get('mask_source', '') == 'binary_component' and m['circularity'] >= float(stage_params.get('circular_noise_min_circularity', 0.65)) and (m['aspect_ratio'] <= float(stage_params.get('circular_noise_max_aspect_ratio', 2.4))) and (width_mm_raw >= float(rod_diameter_mm) * float(stage_params.get('circular_noise_max_diameter_factor', 1.35)))
                edge_sliver_reject = rod_edge_sliver_reject(bbox_local, gray_roi.shape, rectified_fill_ratio, stage_params)
            keep = stage_params['min_area_px'] <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params['min_solidity'] and (fill_ratio >= stage_params.get('min_fill_ratio', 0.0)) and (stage_params['min_aspect_ratio'] <= m['aspect_ratio'] <= stage_params['max_aspect_ratio']) and (stage_params['min_length_mm'] <= length_mm <= stage_params['max_length_mm']) and width_ok and (not circular_noise_reject) and (not edge_sliver_reject)
        elif object_mode == 'disc':
            d_true = float(disc_diameter_mm)
            keep = stage_params['min_area_px'] <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params['min_solidity'] and (m['circularity'] >= stage_params['min_circularity']) and (m['aspect_ratio'] <= stage_params['aspect_tol']) and (stage_params['min_diameter_factor'] * d_true <= circle_diameter_mm <= stage_params['max_diameter_factor'] * d_true)
            obj_diameter_mm = circle_diameter_mm
        elif object_mode == 'sphere':
            d_true = float(sphere_diameter_mm)
            keep = stage_params['min_area_px'] <= m['area'] <= stage_params['max_area_px'] and m['solidity'] >= stage_params['min_solidity'] and (m['circularity'] >= stage_params['min_circularity']) and (m['aspect_ratio'] <= stage_params['aspect_tol']) and (stage_params['min_diameter_factor'] * d_true <= circle_diameter_mm <= stage_params['max_diameter_factor'] * d_true)
            obj_diameter_mm = circle_diameter_mm
        if not keep:
            continue
        dets.append({'cx_raw': cx_track, 'cy_raw': cy_track, 'cx_box_raw': cx_global, 'cy_box_raw': cy_global, 'cx_rect_raw': float(m['cx_rect'] + x0), 'cy_rect_raw': float(m['cy_rect'] + y0), 'box_raw': box_global, 'area_px2': float(m['area']), 'solidity': float(m['solidity']), 'aspect_ratio': float(m['aspect_ratio']), 'circularity': float(m['circularity']), 'fill_ratio': float(fill_ratio), 'rectified_fill_ratio': float(rectified_fill_ratio), 'mask_source': entry.get('mask_source', ''), 'binary_threshold': float(rod_dbg['threshold_used']) if rod_dbg is not None else np.nan, 'width_raw_mm': float(width_mm_raw), 'box_regularized': int(box_regularized), 'fill_poly_local': pair_fill_local.copy() if pair_fill_local is not None else None, 'centre_rectified_px': centre_rectified, 'box_rectified_px': box_rectified, 'length_mm': float(length_mm), 'width_mm': float(width_mm), 'angle_deg': float(angle_deg), 'diameter_mm': obj_diameter_mm})
        dets.sort(key=lambda d: math.hypot(d.get('cx_box_raw', d['cx_raw']) - pred_x, d.get('cy_box_raw', d['cy_raw']) - pred_y))
    debug = {'roi_box': (x0, y0, x1, y1), 'bw': bw_filled if object_mode == 'rod' else bw, 'gray_roi': gray_roi}
    if object_mode == 'rod':
        debug['roi_binary_background'] = rod_dbg['background_roi']
        debug['roi_binary_deviation'] = rod_dbg['deviation_map']
        debug['roi_binary_blur'] = rod_dbg['blur']
        debug['roi_binary_bg'] = rod_dbg['bg']
        debug['roi_binary_map'] = rod_dbg['bright_map']
        debug['roi_binary_raw'] = rod_dbg['bw_raw']
        debug['roi_binary_clean'] = rod_dbg['bw']
        debug['roi_binary_filled'] = bw_filled
        debug['binary_threshold_used'] = float(rod_dbg['threshold_used'])
    elif object_mode == 'sphere' and sphere_color_mode == 'red':
        debug['roi_hsv'] = red_dbg['hsv']
        debug['roi_hsv_mask'] = red_dbg['hsv_mask']
        debug['roi_red_excess'] = red_dbg['red_excess']
        debug['roi_red_excess_blur'] = red_dbg['red_excess_blur']
        debug['roi_rex_mask'] = red_dbg['rex_mask']
    else:
        debug['roi_blur'] = aux_a
        debug['roi_edges'] = aux_b
        debug['roi_close'] = aux_c
    return (dets, debug)

def choose_initial_detection(dets, init_click, object_mode):
    """Handle the choose initial detection step used by this script."""
    if not dets:
        return None
    if object_mode != 'rod':
        return dets[0]
    click_x, click_y = (float(init_click[0]), float(init_click[1]))
    near_limit_px = 45.0
    scored = []
    for det in dets:
        dist = math.hypot(det.get('cx_box_raw', det['cx_raw']) - click_x, det.get('cy_box_raw', det['cy_raw']) - click_y)
        score = dist
        length_mm = float(det.get('length_mm', 0.0))
        rectified_fill = min(max(float(det.get('rectified_fill_ratio', 0.0)), 0.0), 1.0)
        score -= 2.8 * min(length_mm, DEFAULT_ROD_LENGTH_MM)
        score -= 9.0 * rectified_fill
        if det.get('mask_source', '') == 'paired_faces':
            score -= 7.0
        if length_mm < 0.45 * DEFAULT_ROD_LENGTH_MM:
            score += 8.0
        scored.append((dist, score, det))
    nearby = [item for item in scored if item[0] <= near_limit_px]
    pool = nearby if nearby else scored
    pool.sort(key=lambda item: (item[1], item[0]))
    return pool[0][2]

def relative_motion_from_first_mm(first_centre_rectified_px, centre_rectified_px, px_per_mm):
    """Handle the relative motion from first mm step used by this script."""
    x0_mm = px_to_mm_rectified(float(first_centre_rectified_px[0]), px_per_mm)
    y0_mm = px_to_mm_rectified(float(first_centre_rectified_px[1]), px_per_mm)
    x_mm = px_to_mm_rectified(float(centre_rectified_px[0]), px_per_mm)
    y_mm = px_to_mm_rectified(float(centre_rectified_px[1]), px_per_mm)
    return (x0_mm - x_mm, y_mm - y0_mm)

def rod_detection_backtracks(track, det, px_per_mm):
    """Handle the rod detection backtracks step used by this script."""
    if track.object_mode != 'rod':
        return False
    if track.max_disp_mm < DEFAULT_ROD_BACKTRACK_MIN_PROGRESS_MM:
        return False
    cand_x_rel_mm, cand_y_rel_mm = relative_motion_from_first_mm(track.first_centre_rectified_px, det['centre_rectified_px'], px_per_mm)
    back_x_mm = track.max_x_rel_mm - cand_x_rel_mm
    back_y_mm = track.max_y_rel_mm - cand_y_rel_mm
    if back_x_mm > DEFAULT_ROD_SEVERE_BACKTRACK_X_MM or back_y_mm > DEFAULT_ROD_SEVERE_BACKTRACK_Y_MM:
        return True
    if track.misses >= 1 and (back_x_mm > DEFAULT_ROD_BACKTRACK_TOL_X_MM or back_y_mm > DEFAULT_ROD_BACKTRACK_TOL_Y_MM):
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
        self.fill_ratio = float(det.get('fill_ratio', np.nan))
        self.mask_source = det.get('mask_source', '')
        self.binary_threshold = float(det.get('binary_threshold', np.nan))
        self.width_raw_mm = float(det.get('width_raw_mm', np.nan))
        self.box_regularized = int(det.get('box_regularized', 0))
        self.centre_rectified_px = np.array(det['centre_rectified_px'], dtype=float)
        self.box_rectified_px = np.array(det['box_rectified_px'], dtype=float)
        self.length_mm = float(det['length_mm'])
        self.width_mm = float(det['width_mm'])
        self.angle_deg = float(det['angle_deg'])
        self.diameter_mm = det['diameter_mm']
        self.first_centre_rectified_px = self.centre_rectified_px.copy()
        self.first_time_s = float(time_s)
        self.max_x_rel_mm = 0.0
        self.max_y_rel_mm = 0.0
        self.max_disp_mm = 0.0
        self.misses = 0
        self.kf = create_kalman(self.cx_raw, self.cy_raw)
        self.kalman_x_raw = float(det['cx_raw'])
        self.kalman_y_raw = float(det['cy_raw'])

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
        self.rod_anchor_uv = None
        self.cx_raw = float(self.cx_box_raw)
        self.cy_raw = float(self.cy_box_raw)
        self.centre_rectified_px = transform_points_homography(np.array([[self.cx_raw, self.cy_raw]]), H)[0]
        self.first_centre_rectified_px = self.centre_rectified_px.copy()
        self.max_x_rel_mm = 0.0
        self.max_y_rel_mm = 0.0
        self.max_disp_mm = 0.0
        self.kf = create_kalman(self.cx_raw, self.cy_raw)
        self.kalman_x_raw = float(self.cx_raw)
        self.kalman_y_raw = float(self.cy_raw)

    def refresh_progress_bounds(self, px_per_mm):
        """Refresh the progress bounds used by this script."""
        if self.object_mode != 'rod':
            return
        x_rel_mm, y_rel_mm = relative_motion_from_first_mm(self.first_centre_rectified_px, self.centre_rectified_px, px_per_mm)
        self.max_x_rel_mm = max(float(self.max_x_rel_mm), float(x_rel_mm))
        self.max_y_rel_mm = max(float(self.max_y_rel_mm), float(y_rel_mm))
        self.max_disp_mm = max(float(self.max_disp_mm), float(math.hypot(x_rel_mm, y_rel_mm)))

    def update(self, det, px_per_mm=None):
        """Handle the update step used by this script."""
        self.cx_box_raw = float(det.get('cx_box_raw', det['cx_raw']))
        self.cy_box_raw = float(det.get('cy_box_raw', det['cy_raw']))
        if self.object_mode == 'rod':
            self.cx_raw = float(self.cx_box_raw)
            self.cy_raw = float(self.cy_box_raw)
        else:
            self.cx_raw = 0.7 * self.cx_raw + 0.3 * float(det['cx_raw'])
            self.cy_raw = 0.7 * self.cy_raw + 0.3 * float(det['cy_raw'])
        self.box_raw = det['box_raw']
        self.area_px2 = float(det['area_px2'])
        self.solidity = float(det['solidity'])
        self.aspect_ratio = float(det['aspect_ratio'])
        self.circularity = float(det['circularity'])
        self.fill_ratio = float(det.get('fill_ratio', np.nan))
        self.mask_source = det.get('mask_source', '')
        self.binary_threshold = float(det.get('binary_threshold', np.nan))
        self.width_raw_mm = float(det.get('width_raw_mm', np.nan))
        self.box_regularized = int(det.get('box_regularized', 0))
        self.centre_rectified_px = np.array(det['centre_rectified_px'], dtype=float)
        self.box_rectified_px = np.array(det['box_rectified_px'], dtype=float)
        self.length_mm = float(det['length_mm'])
        self.width_mm = float(det['width_mm'])
        self.angle_deg = float(det['angle_deg'])
        self.diameter_mm = det['diameter_mm']
        self.correct(self.cx_raw, self.cy_raw)
        self.misses = 0
        if px_per_mm is not None:
            self.refresh_progress_bounds(px_per_mm)

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
    df['x_rel_mm'] = x0 - x_base
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
    save_plot(dfp['t_rel_s'], dfp['x_rel_mm'], 'time from first detection (s)', 'x displacement (mm, left positive)', f'{object_mode} x displacement vs time', reg('x_vs_time.png'))
    save_plot(dfp['t_rel_s'], dfp['y_rel_mm'], 'time from first detection (s)', 'y displacement (mm, down positive)', f'{object_mode} y displacement vs time', reg('y_vs_time.png'))
    save_plot(dfp['t_rel_s'], dfp['disp_mm'], 'time from first detection (s)', 'displacement (mm)', f'{object_mode} displacement vs time', reg('displacement_vs_time.png'))
    save_plot(dfp['t_rel_s'], dfp['speed_mm_s'], 'time from first detection (s)', 'speed (mm/s)', f'{object_mode} speed vs time', reg('speed_vs_time.png'))
    plt.figure()
    plt.plot(dfp['x_rel_mm'], dfp['y_rel_mm'])
    plt.xlabel('x displacement (mm, left positive)')
    plt.ylabel('y displacement (mm, down positive)')
    plt.title(f'{object_mode} trajectory')
    plt.axis('equal')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid(True)
    tp = reg('trajectory.png')
    plt.savefig(tp, dpi=200, bbox_inches='tight')
    plt.close()
    save_plot([dfp['t_rel_s'], dfp['t_rel_s']], [dfp['vx_mm_s'], dfp['vy_mm_s']], 'time from first detection (s)', 'velocity (mm/s)', f'{object_mode} velocity components vs time', reg('vx_vy_vs_time.png'), legend=['vx', 'vy'])
    save_plot([dfp['t_rel_s'], dfp['t_rel_s']], [dfp['ax_mm_s2'], dfp['ay_mm_s2']], 'time from first detection (s)', 'acceleration (mm/s^2)', f'{object_mode} acceleration components vs time', reg('ax_ay_vs_time.png'), legend=['ax', 'ay'])
    if object_mode == 'rod':
        save_plot(dfp['t_rel_s'], dfp['angle_smooth_deg'], 'time from first detection (s)', 'rod angle (deg)', 'rod angle vs time', reg('angle_vs_time.png'))
        save_plot(dfp['t_rel_s'], dfp['omega_deg_s'], 'time from first detection (s)', 'angular velocity (deg/s)', 'rod angular velocity vs time', reg('omega_vs_time.png'))
        plt.figure()
        plt.plot(dfp['y_rel_mm'], dfp['angle_smooth_deg'], '.-')
        plt.xlabel('y displacement (mm, down positive)')
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
        return {'video_path': DEFAULT_VIDEO_PATH, 'calibration_json': DEFAULT_CALIBRATION_JSON, 'out_dir': DEFAULT_OUT_DIR, 'real_fps': DEFAULT_REAL_FPS, 'frame_stride': DEFAULT_FRAME_STRIDE, 'start_frame': DEFAULT_START_FRAME, 'object_mode': DEFAULT_OBJECT_MODE, 'rod_diameter_mm': DEFAULT_ROD_DIAMETER_MM, 'sphere_color_mode': DEFAULT_SPHERE_COLOR_MODE, 'disc_diameter_mm': DEFAULT_DISC_DIAMETER_MM, 'sphere_diameter_mm': DEFAULT_SPHERE_DIAMETER_MM, 'use_preset_click': DEFAULT_USE_PRESET_CLICK, 'preset_click': DEFAULT_PRESET_CLICK, 'save_debug_images': DEFAULT_SAVE_DEBUG_IMAGES, 'num_debug_images': DEFAULT_NUM_DEBUG_IMAGES, 'enable_rod_background_subtraction': DEFAULT_ENABLE_ROD_BACKGROUND_SUBTRACTION, 'rod_background_seconds': DEFAULT_ROD_BACKGROUND_SECONDS, 'rod_background_samples': DEFAULT_ROD_BACKGROUND_SAMPLES, 'rod_background_margin_frames': DEFAULT_ROD_BACKGROUND_MARGIN_FRAMES, 'rod_background_keep_fraction': DEFAULT_ROD_BACKGROUND_KEEP_FRACTION, 'enable_imshow': ENABLE_IMSHOW}
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
    rod_diameter_mm = DEFAULT_ROD_DIAMETER_MM
    disc_diameter_mm = DEFAULT_DISC_DIAMETER_MM
    sphere_diameter_mm = DEFAULT_SPHERE_DIAMETER_MM
    if object_mode == 'rod':
        rod_diameter_mm = float(input(f'Rod diameter in mm [{DEFAULT_ROD_DIAMETER_MM}]: ').strip() or DEFAULT_ROD_DIAMETER_MM)
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
    return {'video_path': video_path, 'calibration_json': calibration_json, 'out_dir': out_dir, 'real_fps': real_fps, 'frame_stride': frame_stride, 'start_frame': start_frame, 'object_mode': object_mode, 'rod_diameter_mm': rod_diameter_mm, 'sphere_color_mode': sphere_color_mode, 'disc_diameter_mm': disc_diameter_mm, 'sphere_diameter_mm': sphere_diameter_mm, 'use_preset_click': use_preset_click, 'preset_click': preset_click, 'save_debug_images': save_debug_images, 'num_debug_images': num_debug_images, 'enable_rod_background_subtraction': DEFAULT_ENABLE_ROD_BACKGROUND_SUBTRACTION, 'rod_background_seconds': DEFAULT_ROD_BACKGROUND_SECONDS, 'rod_background_samples': DEFAULT_ROD_BACKGROUND_SAMPLES, 'rod_background_margin_frames': DEFAULT_ROD_BACKGROUND_MARGIN_FRAMES, 'rod_background_keep_fraction': DEFAULT_ROD_BACKGROUND_KEEP_FRACTION, 'enable_imshow': ENABLE_IMSHOW}

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
    rod_background_gray = None
    rod_background_info = None
    if cfg['object_mode'] == 'rod' and bool(cfg.get('enable_rod_background_subtraction', DEFAULT_ENABLE_ROD_BACKGROUND_SUBTRACTION)):
        background_seconds = float(cfg.get('rod_background_seconds', DEFAULT_ROD_BACKGROUND_SECONDS))
        if background_seconds > 0.0:
            background_window_frames = max(1, int(round(background_seconds * fps_used)))
        else:
            background_window_frames = 0
        if background_window_frames > 0:
            rod_background_gray, rod_background_info = build_rod_average_background_gray(cap, cfg['start_frame'], background_window_frames, int(cfg.get('rod_background_samples', DEFAULT_ROD_BACKGROUND_SAMPLES)), margin_frames=int(cfg.get('rod_background_margin_frames', DEFAULT_ROD_BACKGROUND_MARGIN_FRAMES)), keep_fraction=float(cfg.get('rod_background_keep_fraction', DEFAULT_ROD_BACKGROUND_KEEP_FRACTION)))
        else:
            rod_background_info = {'status': 'disabled', 'reason': 'rod_background_seconds<=0'}
        if rod_background_gray is not None:
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_background_average.png'), np.clip(rod_background_gray, 0, 255).astype(np.uint8))
        if rod_background_info is not None:
            with open(os.path.join(cfg['out_dir'], 'debug_background_info.json'), 'w', encoding='utf-8') as f:
                json.dump(rod_background_info, f, indent=2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg['start_frame'])
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Could not read start frame from video.')
    acquire_params = get_param_block(cfg['object_mode'], cfg['sphere_color_mode'], 'acquire')
    if cfg['use_preset_click']:
        init_click = cfg['preset_click']
    else:
        mode_label = cfg['object_mode'] if cfg['object_mode'] != 'sphere' else f"{cfg['sphere_color_mode']} sphere"
        if cfg['object_mode'] == 'rod':
            start_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            start_binary = preprocess_rod_binary_roi(start_gray, acquire_params, background_gray_roi=rod_background_gray)['bw']
            init_click = pick_initial_click_binary(start_binary, mode_label)
        else:
            init_click = pick_initial_click_raw(frame, mode_label)
    dets, debug = detect_in_raw_roi(frame, init_click[0], init_click[1], H, px_per_mm, cfg['object_mode'], cfg['sphere_color_mode'], cfg['disc_diameter_mm'], cfg['sphere_diameter_mm'], acquire_params, rod_anchor_uv=None, rod_diameter_mm=cfg['rod_diameter_mm'], rod_background_gray=rod_background_gray)
    if len(dets) == 0:
        cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_raw.png'), frame)
        if 'roi_binary_map' in debug:
            if debug.get('roi_binary_background') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_background.png'), debug['roi_binary_background'])
            if debug.get('roi_binary_deviation') is not None:
                cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_deviation.png'), debug['roi_binary_deviation'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_map.png'), debug['roi_binary_map'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_raw.png'), debug['roi_binary_raw'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_clean.png'), debug['roi_binary_clean'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_binary_filled.png'), debug['roi_binary_filled'])
        elif 'roi_blur' in debug:
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_blur.png'), debug['roi_blur'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_edges.png'), debug['roi_edges'])
            cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_roi_close.png'), debug['roi_close'])
        cv2.imwrite(os.path.join(cfg['out_dir'], 'debug_start_bw.png'), debug['bw'])
        raise RuntimeError('No valid contour found near the selected start point on the binarised frame. Adjust start frame or rod binary settings.')
    init_det = choose_initial_detection(dets, init_click, cfg['object_mode'])
    if cfg['object_mode'] == 'rod':
        debug['roi_binary_filled'] = debug['roi_binary_clean'].copy()
        fill_poly = init_det.get('fill_poly_local')
        if fill_poly is not None:
            cv2.drawContours(debug['roi_binary_filled'], [fill_poly.astype(np.int32).reshape(-1, 1, 2)], -1, 255, thickness=cv2.FILLED)
    track = ObjectTrack(init_det, time_s=0.0, object_mode=cfg['object_mode'])
    if cfg['object_mode'] == 'rod':
        track.set_rod_anchor_from_click(init_click, H)
    h_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_mp4 = os.path.join(cfg['out_dir'], 'annotated_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_mp4, fourcc, float(file_fps / cfg['frame_stride']), (w_raw, h_raw), True)
    out_binary_mp4 = os.path.join(cfg['out_dir'], 'annotated_binary.mp4')
    binary_vw = cv2.VideoWriter(out_binary_mp4, fourcc, float(file_fps / cfg['frame_stride']), (w_raw, h_raw), True) if cfg['object_mode'] == 'rod' else None
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
        if cfg['object_mode'] == 'rod' and track.misses > 0:
            stage_params['roi_half_size_px'] = min(int(stage_params['roi_half_size_px'] + track.misses * stage_params.get('miss_roi_growth_px', 0)), int(stage_params.get('miss_max_roi_half_size_px', stage_params['roi_half_size_px'])))
            stage_params['max_assoc_dist_px'] = float(stage_params['max_assoc_dist_px']) + float(track.misses) * float(stage_params.get('miss_assoc_growth_px', 0.0))
        dets, debug = detect_in_raw_roi(frame, pred_x, pred_y, H, px_per_mm, cfg['object_mode'], cfg['sphere_color_mode'], cfg['disc_diameter_mm'], cfg['sphere_diameter_mm'], stage_params, rod_anchor_uv=track.rod_anchor_uv if cfg['object_mode'] == 'rod' else None, rod_diameter_mm=cfg['rod_diameter_mm'], rod_background_gray=rod_background_gray)
        ann_binary = None
        if cfg['object_mode'] == 'rod':
            full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            full_binary = preprocess_rod_binary_roi(full_gray, stage_params, background_gray_roi=rod_background_gray)
            ann_binary = gray_to_bgr(full_binary['bw'])
        chosen_det = None
        best_score = np.inf
        for d in dets:
            det_x = float(d.get('cx_box_raw', d['cx_raw']))
            det_y = float(d.get('cy_box_raw', d['cy_raw']))
            dist = math.hypot(pred_x - det_x, pred_y - det_y)
            if cfg['object_mode'] == 'rod':
                dtheta = abs(d['angle_deg'] - track.angle_deg)
                dtheta = min(dtheta, 180.0 - dtheta)
                dlen = abs(d['length_mm'] - track.length_mm)
                score = dist + 0.25 * dtheta + 1.5 * dlen
                rectified_fill = min(max(float(d.get('rectified_fill_ratio', 0.0)), 0.0), 1.0)
                score -= 7.0 * rectified_fill
                nominal_length = float(stage_params.get('rod_length_target_mm', DEFAULT_ROD_LENGTH_MM))
                short_penalty = float(stage_params.get('rod_short_length_penalty_mm', 10.0))
                if d['length_mm'] < 0.45 * nominal_length:
                    score += short_penalty
                if d.get('mask_source', '') == 'paired_faces':
                    score -= float(stage_params.get('rod_pair_bonus', 6.0))
                elif d['length_mm'] < 0.6 * nominal_length:
                    score += float(stage_params.get('rod_binary_short_penalty', 7.0))
            else:
                score = dist
            if score < best_score:
                best_score = score
                chosen_det = d
        found_this_frame = False
        if chosen_det is not None and math.hypot(pred_x - chosen_det.get('cx_box_raw', chosen_det['cx_raw']), pred_y - chosen_det.get('cy_box_raw', chosen_det['cy_raw'])) <= stage_params['max_assoc_dist_px']:
            if cfg['object_mode'] == 'rod' and rod_detection_backtracks(track, chosen_det, px_per_mm):
                chosen_det = None
            else:
                track.update(chosen_det, px_per_mm=px_per_mm)
                found_this_frame = True
        if not found_this_frame:
            chosen_det = None
            track.misses += 1
            track.cx_raw = float(track.kalman_x_raw)
            track.cy_raw = float(track.kalman_y_raw)
            if cfg['object_mode'] == 'rod':
                track.cx_box_raw = float(track.kalman_x_raw)
                track.cy_box_raw = float(track.kalman_y_raw)
                if track.max_disp_mm >= DEFAULT_ROD_BACKTRACK_MIN_PROGRESS_MM and track.misses >= DEFAULT_ROD_TAIL_MISS_STOP:
                    print('Rod tracking ended after sustained loss near the trajectory frontier.')
                    break
        if cfg['object_mode'] == 'rod':
            debug['roi_binary_filled'] = debug['roi_binary_clean'].copy()
            fill_poly = chosen_det.get('fill_poly_local') if found_this_frame and chosen_det is not None else None
            if fill_poly is not None:
                cv2.drawContours(debug['roi_binary_filled'], [fill_poly.astype(np.int32).reshape(-1, 1, 2)], -1, 255, thickness=cv2.FILLED)
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
        x_rel_mm = x0_mm - x_kalman_mm
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
                if ann_binary is not None:
                    cv2.putText(ann_binary, f'theta={track.angle_deg:.1f} deg', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(ann_binary, f'L={track.length_mm:.1f} W={track.width_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(ann, f'D={track.diameter_mm:.1f} mm', (int(round(kf_cx)) + 8, int(round(kf_cy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
        row = {'frame_idx': frame_idx, 'out_idx': out_idx, 'time_s': t_s, 'found_this_frame': int(found_this_frame), 'cx_raw_px': meas_cx, 'cy_raw_px': meas_cy, 'cx_box_raw_px': track.cx_box_raw, 'cy_box_raw_px': track.cy_box_raw, 'cx_kalman_raw_px': kf_cx, 'cy_kalman_raw_px': kf_cy, 'x_rectified_px': x_rect_px, 'y_rectified_px': y_rect_px, 'x_mm': x_mm, 'y_mm': y_mm, 'x_kalman_mm': x_kalman_mm, 'y_kalman_mm': y_kalman_mm, 'x0_mm': x0_mm, 'y0_mm': y0_mm, 'x_rel_mm': x_rel_mm, 'y_rel_mm': y_rel_mm, 'disp_mm': disp_mm, 'area_px2': track.area_px2, 'solidity': track.solidity, 'aspect_ratio': track.aspect_ratio, 'circularity': track.circularity, 'fill_ratio': track.fill_ratio, 'mask_source': track.mask_source, 'binary_threshold': track.binary_threshold, 'width_raw_mm': track.width_raw_mm, 'box_regularized': track.box_regularized, 'misses': track.misses, 'p0x_raw': track.box_raw[0, 0], 'p0y_raw': track.box_raw[0, 1], 'p1x_raw': track.box_raw[1, 0], 'p1y_raw': track.box_raw[1, 1], 'p2x_raw': track.box_raw[2, 0], 'p2y_raw': track.box_raw[2, 1], 'p3x_raw': track.box_raw[3, 0], 'p3y_raw': track.box_raw[3, 1]}
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
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_map_{out_idx:04d}.png'), debug['roi_binary_map'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_raw_{out_idx:04d}.png'), debug['roi_binary_raw'])
                cv2.imwrite(os.path.join(cfg['out_dir'], f'roi_binary_clean_{out_idx:04d}.png'), debug['roi_binary_clean'])
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
            if ann_binary is not None:
                cv2.imshow('annotated binary', ann_binary)
            else:
                cv2.imshow('annotated raw', ann)
            if cv2.waitKey(1) & 255 == 27:
                break
        frame_idx += 1
        out_idx += 1
    cap.release()
    vw.release()
    if binary_vw is not None:
        binary_vw.release()
    cv2.destroyAllWindows()
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
