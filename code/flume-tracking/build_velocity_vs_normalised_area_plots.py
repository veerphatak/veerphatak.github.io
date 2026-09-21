"""
Plot builder: build_velocity_vs_normalised_area_plots.py

Purpose
-------
Builds velocity versus normalised projected-area plots for rods and discs.

Inputs
------
Rod/disc tracker outputs containing velocity and projected-area data.

Outputs
-------
Scatter plots and CSV tables for velocity-area analysis.

Methodological notes
--------------------
Projected area is treated as a proxy for orientation-dependent hydrodynamic exposure to the flow.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
SOURCE_ROOT = Path('C:\\IP Work\\Water Tunnel Flow Overview')
TRACKER_ROOT = SOURCE_ROOT / 'tracker_outputs'
OUTPUT_ROOT = Path(os.environ.get('WATER_TUNNEL_VELOCITY_AREA_OUTPUT_ROOT', 'C:\\IP Work\\Water Tunnel Velocity vs Normalised Area Corrected'))
RUN_OUTPUT_ROOT = OUTPUT_ROOT / 'run_outputs'
SUMMARY_DIR = OUTPUT_ROOT / 'summary'
PLOT_NAME = 'velocity_vs_normalised_area.png'
CSV_NAME = 'velocity_vs_normalised_area.csv'
SUMMARY_NAME = 'velocity_vs_normalised_area_summary.csv'
MAX_FRAME_GAP_FOR_VELOCITY = float(os.environ.get('WATER_TUNNEL_MAX_FRAME_GAP_FOR_VELOCITY', '5.5'))

def parse_float(value: str | None) -> float | None:
    """Handle the parse float step used by this script."""
    if value is None or value == '':
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number

def parse_int(value: str | None) -> int | None:
    """Handle the parse int step used by this script."""
    if value is None or value == '':
        return None
    try:
        return int(float(value))
    except ValueError:
        return None

def iter_target_run_dirs() -> list[Path]:
    """Handle the iter target run dirs step used by this script."""
    run_dirs: list[Path] = []
    for group_dir in sorted(TRACKER_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        lower = group_dir.name.lower()
        if 'sphere' in lower:
            continue
        if 'rod' not in lower and 'disc' not in lower:
            continue
        for run_dir in sorted(group_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / 'tracks_processed.csv').exists():
                run_dirs.append(run_dir)
    return run_dirs

def relative_run_parts(run_dir: Path) -> tuple[str, str]:
    """Handle the relative run parts step used by this script."""
    group_name = run_dir.parent.name
    run_name = run_dir.name
    return (group_name, run_name)

def detect_run_fps(run_dir: Path) -> tuple[float | None, str | None]:
    """Detect the run fps used by this script."""
    for name in ('annotated_raw.mp4', 'annotated_binary.mp4'):
        video_path = run_dir / name
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if math.isfinite(fps) and fps > 1.0:
                return (fps, name)
        finally:
            cap.release()
    return (None, None)

def gradient_from_arrays(t: np.ndarray, x: np.ndarray, max_gap_s: float | None) -> np.ndarray:
    """Handle the gradient from arrays step used by this script."""
    valid = np.isfinite(t) & np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if int(valid.sum()) < 2:
        return out
    valid_idx = np.flatnonzero(valid)
    t_valid = t[valid]
    x_valid = x[valid]
    if max_gap_s is None:
        dt_valid = np.diff(t_valid)
        dt_valid = dt_valid[np.isfinite(dt_valid) & (dt_valid > 0)]
        max_gap_s = float(np.median(dt_valid) * 1.5) if len(dt_valid) else None
    seg_start = 0
    for i in range(1, len(valid_idx) + 1):
        at_end = i == len(valid_idx)
        gap_break = False
        if not at_end and max_gap_s is not None:
            gap_break = t_valid[i] - t_valid[i - 1] > max_gap_s
        if at_end or gap_break:
            seg_slice = slice(seg_start, i)
            seg_idx = valid_idx[seg_slice]
            seg_t = t_valid[seg_slice]
            seg_x = x_valid[seg_slice]
            if len(seg_idx) >= 2:
                out[seg_idx] = np.gradient(seg_x, seg_t)
            seg_start = i
    return out

def collect_points(processed_csv: Path) -> tuple[list[dict[str, float]], float]:
    """Handle the collect points step used by this script."""
    run_dir = processed_csv.parent
    actual_fps, fps_source = detect_run_fps(run_dir)
    if actual_fps is None:
        raise RuntimeError(f'Could not detect usable FPS from annotated video in {run_dir}')
    points: list[dict[str, float]] = []
    rows: list[dict[str, float]] = []
    max_area = 0.0
    with processed_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            found = parse_int(row.get('found_this_frame'))
            if found != 1:
                continue
            area_px2 = parse_float(row.get('area_px2'))
            if area_px2 is None or area_px2 <= 0.0:
                continue
            out_idx = parse_int(row.get('out_idx'))
            if out_idx is None:
                out_idx = parse_int(row.get('frame_idx'))
            if out_idx is None:
                continue
            x_mm = parse_float(row.get('x_rel_smooth_mm'))
            y_mm = parse_float(row.get('y_rel_smooth_mm'))
            if x_mm is None or y_mm is None:
                x_mm = parse_float(row.get('x_rel_mm'))
                y_mm = parse_float(row.get('y_rel_mm'))
            if x_mm is None or y_mm is None:
                x_mm = parse_float(row.get('x_kalman_mm'))
                y_mm = parse_float(row.get('y_kalman_mm'))
            if x_mm is None or y_mm is None:
                x_mm = parse_float(row.get('x_mm'))
                y_mm = parse_float(row.get('y_mm'))
            if x_mm is None or y_mm is None:
                continue
            rows.append({'frame_idx': float(parse_int(row.get('frame_idx')) or out_idx), 'out_idx': float(out_idx), 'area_px2': area_px2, 'x_mm': x_mm, 'y_mm': y_mm, 'angle_deg': parse_float(row.get('angle_deg')) or float('nan')})
            max_area = max(max_area, area_px2)
    if max_area <= 0.0:
        raise RuntimeError(f'No valid positive quadrilateral area found in {processed_csv}')
    rows.sort(key=lambda p: p['out_idx'])
    t = np.asarray([(row['out_idx'] - rows[0]['out_idx']) / actual_fps for row in rows], dtype=float)
    x = np.asarray([row['x_mm'] for row in rows], dtype=float)
    y = np.asarray([row['y_mm'] for row in rows], dtype=float)
    max_gap_s = MAX_FRAME_GAP_FOR_VELOCITY / actual_fps
    vx = gradient_from_arrays(t, x, max_gap_s=max_gap_s)
    vy = gradient_from_arrays(t, y, max_gap_s=max_gap_s)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    for row, time_s, speed_mm_s in zip(rows, t, speed):
        if not math.isfinite(speed_mm_s):
            continue
        points.append({'frame_idx': row['frame_idx'], 'time_s': float(time_s), 'area_px2': row['area_px2'], 'normalised_area': row['area_px2'] / max_area, 'speed_mm_s': float(speed_mm_s), 'angle_deg': row['angle_deg'], 'actual_fps_used': float(actual_fps), 'actual_fps_source': fps_source or ''})
    if not points:
        raise RuntimeError(f'No usable corrected velocity points found in {processed_csv}')
    return (points, max_area)

def write_point_csv(csv_path: Path, points: list[dict[str, float]]) -> None:
    """Write the point csv used by this script."""
    fieldnames = ['frame_idx', 'time_s', 'area_px2', 'normalised_area', 'speed_mm_s', 'angle_deg', 'actual_fps_used', 'actual_fps_source']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow({'frame_idx': int(point['frame_idx']), 'time_s': point['time_s'], 'area_px2': point['area_px2'], 'normalised_area': point['normalised_area'], 'speed_mm_s': point['speed_mm_s'], 'angle_deg': point['angle_deg'], 'actual_fps_used': point['actual_fps_used'], 'actual_fps_source': point['actual_fps_source']})

def save_scatter_plot(plot_path: Path, run_name: str, points: list[dict[str, float]], max_area: float) -> None:
    """Save the scatter plot used by this script."""
    xs = [point['normalised_area'] for point in points]
    ys = [point['speed_mm_s'] for point in points]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    ax.scatter(xs, ys, s=16, alpha=0.72, color='#1f77b4', edgecolors='none')
    ax.set_xlim(0.0, 1.0)
    ymin = min(ys)
    ymax = max(ys)
    if math.isclose(ymin, ymax):
        pad = max(1.0, abs(ymax) * 0.05)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel('Normalised quadrilateral area, A / Amax')
    ax.set_ylabel('Instantaneous velocity (mm/s)')
    ax.set_title(run_name.replace('_', ' '))
    ax.grid(True, alpha=0.28)
    ax.text(0.98, 0.02, f'Amax = {max_area:.2f} px^2\nn = {len(points)}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.8'})
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

def main() -> int:
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for run_dir in iter_target_run_dirs():
        processed_csv = run_dir / 'tracks_processed.csv'
        points, max_area = collect_points(processed_csv)
        group_name, run_name = relative_run_parts(run_dir)
        out_dir = RUN_OUTPUT_ROOT / group_name / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        point_csv = out_dir / CSV_NAME
        plot_path = out_dir / PLOT_NAME
        write_point_csv(point_csv, points)
        save_scatter_plot(plot_path, run_name, points, max_area)
        group_summary_dir = SUMMARY_DIR / group_name
        group_summary_dir.mkdir(parents=True, exist_ok=True)
        (group_summary_dir / f'{run_name}_{PLOT_NAME}').write_bytes(plot_path.read_bytes())
        (group_summary_dir / f'{run_name}_{CSV_NAME}').write_bytes(point_csv.read_bytes())
        summary_rows.append({'group': group_name, 'run_name': run_name, 'source_run_dir': str(run_dir), 'processed_csv': str(processed_csv), 'plot_path': str(plot_path), 'point_csv': str(point_csv), 'point_count': len(points), 'max_area_px2': max_area})
    summary_csv = SUMMARY_DIR / SUMMARY_NAME
    with summary_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'run_name', 'source_run_dir', 'processed_csv', 'plot_path', 'point_csv', 'point_count', 'max_area_px2'])
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {'source_root': str(SOURCE_ROOT), 'tracker_root': str(TRACKER_ROOT), 'output_root': str(OUTPUT_ROOT), 'run_output_root': str(RUN_OUTPUT_ROOT), 'summary_dir': str(SUMMARY_DIR), 'plot_name': PLOT_NAME, 'csv_name': CSV_NAME, 'run_count': len(summary_rows)}
    (SUMMARY_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    raise SystemExit(main())
