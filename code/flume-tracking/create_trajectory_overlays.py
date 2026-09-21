"""
Validation/overlay builder: create_trajectory_overlays.py

Purpose
-------
Creates still-image and video overlays from processed flume trajectories.

Inputs
------
Processed tracks and corresponding background/still images.

Outputs
-------
Static trajectory overlays and annotated videos.

Methodological notes
--------------------
Overlay plots are used for visual validation of the tracker against the original motion.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parent
VENDOR_DIR = WORKSPACE / '.vendor_video_libs'
if VENDOR_DIR.exists():
    sys.path.append(str(VENDOR_DIR))
import cv2
import numpy as np
OUT_ROOT = WORKSPACE / 'pxl_sphere_trim_tracking'

def load_case_dirs() -> list[Path]:
    """Load the case dirs used by this script."""
    case_dirs: list[Path] = []
    for case_summary in sorted(OUT_ROOT.glob('*/case_summary.json')):
        try:
            data = json.loads(case_summary.read_text(encoding='utf-8'))
        except Exception:
            continue
        if data.get('status') == 'ok':
            case_dirs.append(case_summary.parent)
    return case_dirs

def load_tracks(csv_path: Path) -> list[dict]:
    """Load the tracks used by this script."""
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    tracks: list[dict] = []
    for row in rows:
        try:
            tracks.append({'frame_idx': int(float(row['frame_idx'])), 'out_idx': int(float(row['out_idx'])), 'found': int(float(row['found_this_frame'])), 'cx_raw': float(row['cx_raw_px']), 'cy_raw': float(row['cy_raw_px']), 'cx_kalman': float(row['cx_kalman_raw_px']), 'cy_kalman': float(row['cy_kalman_raw_px'])})
        except Exception:
            continue
    return tracks

def build_found_track_segments(tracks: list[dict], gap_tolerance: int=1) -> list[list[dict]]:
    """Build the found track segments used by this script."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    prev_frame_idx: int | None = None
    for row in tracks:
        if row['found'] != 1:
            if current:
                segments.append(current)
                current = []
            prev_frame_idx = None
            continue
        frame_idx = int(row['frame_idx'])
        if current and prev_frame_idx is not None and (frame_idx - prev_frame_idx > gap_tolerance):
            segments.append(current)
            current = []
        current.append(row)
        prev_frame_idx = frame_idx
    if current:
        segments.append(current)
    return [seg for seg in segments if len(seg) >= 2]

def segment_points(segment: list[dict]) -> list[tuple[int, int]]:
    """Handle the segment points step used by this script."""
    return [(int(round(row['cx_raw'])), int(round(row['cy_raw']))) for row in segment]

def get_base_image(case_dir: Path) -> np.ndarray:
    """Handle the get base image step used by this script."""
    tracker_dir = case_dir / 'tracker_output'
    background_path = tracker_dir / 'debug_background_average.png'
    if background_path.exists():
        img = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    cap = cv2.VideoCapture(str(case_dir / 'trimmed_oriented.mp4'))
    try:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    finally:
        cap.release()
    raise RuntimeError(f'Could not load background or first frame for {case_dir.name}')

def draw_static_overlay(base_img: np.ndarray, segments: list[list[tuple[int, int]]], label: str) -> np.ndarray:
    """Handle the draw static overlay step used by this script."""
    overlay = base_img.copy()
    shade = np.zeros_like(overlay)
    cv2.rectangle(shade, (0, 0), (overlay.shape[1], 56), (0, 0, 0), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(overlay, 1.0, shade, 0.35, 0.0)
    for seg in segments:
        pts = np.asarray(seg, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (40, 220, 255), 3, lineType=cv2.LINE_AA)
    if segments:
        start_pt = segments[0][0]
        end_pt = segments[-1][-1]
        cv2.circle(overlay, start_pt, 8, (0, 200, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, start_pt, 12, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(overlay, end_pt, 8, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, end_pt, 12, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(overlay, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return overlay

def draw_video_overlay(case_dir: Path, segment_rows: list[dict], output_path: Path, label: str) -> None:
    """Handle the draw video overlay step used by this script."""
    video_path = case_dir / 'trimmed_oriented.mp4'
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not open writer: {output_path}')
    rows_by_frame = {int(row['frame_idx']): row for row in segment_rows}
    found_points: list[tuple[int, int]] = []
    prev_found_frame: int | None = None
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            row = rows_by_frame.get(frame_idx)
            if row is not None:
                if prev_found_frame is not None and frame_idx - prev_found_frame > 1:
                    found_points.append(None)
                found_points.append((int(round(row['cx_raw'])), int(round(row['cy_raw']))))
                prev_found_frame = frame_idx
            overlay = frame.copy()
            previous_point: tuple[int, int] | None = None
            for point in found_points:
                if point is None:
                    previous_point = None
                    continue
                if previous_point is not None:
                    cv2.line(overlay, previous_point, point, (40, 220, 255), 3, lineType=cv2.LINE_AA)
                previous_point = point
            if previous_point is not None:
                cv2.circle(overlay, previous_point, 8, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            shade = np.zeros_like(overlay)
            cv2.rectangle(shade, (0, 0), (overlay.shape[1], 44), (0, 0, 0), thickness=cv2.FILLED)
            overlay = cv2.addWeighted(overlay, 1.0, shade, 0.3, 0.0)
            cv2.putText(overlay, label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

def process_case(case_dir: Path) -> dict:
    """Handle the process case step used by this script."""
    tracker_dir = case_dir / 'tracker_output'
    tracks = load_tracks(tracker_dir / 'tracks_raw.csv')
    if not tracks:
        raise RuntimeError(f'No tracks found in {tracker_dir}')
    case_data = json.loads((case_dir / 'case_summary.json').read_text(encoding='utf-8'))
    sphere_mode = case_data.get('tracker_result', {}).get('sphere_color_mode', case_data.get('kind', 'sphere'))
    label = f'{case_dir.name} trajectory overlay ({sphere_mode})'
    base_img = get_base_image(case_dir)
    track_segments = build_found_track_segments(tracks)
    if track_segments:
        primary_segment_rows = max(track_segments, key=len)
        segments = [segment_points(primary_segment_rows)]
    else:
        primary_segment_rows = [row for row in tracks if row['found'] == 1]
        segments = []
        if len(primary_segment_rows) >= 2:
            segments = [segment_points(primary_segment_rows)]
    static_overlay = draw_static_overlay(base_img, segments, label)
    static_path = tracker_dir / 'trajectory_overlay.png'
    cv2.imwrite(str(static_path), static_overlay)
    video_path = tracker_dir / 'trajectory_overlay.mp4'
    draw_video_overlay(case_dir, primary_segment_rows, video_path, label)
    return {'case': case_dir.name, 'static_overlay': str(static_path), 'video_overlay': str(video_path), 'segments': len(track_segments), 'primary_segment_length': len(primary_segment_rows)}

def main() -> None:
    """Run the main entry point for this script."""
    results = []
    for case_dir in load_case_dirs():
        results.append(process_case(case_dir))
        print(f'{case_dir.name}: overlays created')
    summary_path = OUT_ROOT / 'trajectory_overlay_summary.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {summary_path}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
