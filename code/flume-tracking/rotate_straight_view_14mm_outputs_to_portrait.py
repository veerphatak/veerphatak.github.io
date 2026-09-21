"""
Post-processing utility: rotate_straight_view_14mm_outputs_to_portrait.py

Purpose
-------
Rotates completed 14 mm straight-view outputs into portrait orientation and updates coordinates accordingly.

Inputs
------
Existing 14 mm output folder.

Outputs
-------
Rotated images and rewritten CSV coordinate outputs.

Methodological notes
--------------------
Image rotation and coordinate rotation are applied together so figures and numeric data remain consistent.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import csv
import math
import os
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
ROOT = 'C:\\IP Work\\Straight View Sphere Slip Overlays Rebuilt\\14mm'

def rotate_xy_clockwise(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """Rotate the xy clockwise used by this script."""
    return (float(height - 1 - y), float(x))

def rotate_image_clockwise(path: str) -> None:
    """Rotate the image clockwise used by this script."""
    if not os.path.exists(path):
        return
    image = Image.open(path)
    rotated = image.transpose(Image.Transpose.ROTATE_270)
    rotated.save(path)

def process_run_csv(run_dir: str) -> Dict[str, object]:
    """Handle the process run csv step used by this script."""
    bg_path = os.path.join(run_dir, 'average_background.png')
    width, height = Image.open(bg_path).size
    for name in ['average_background.png', 'journey_overlay.png', 'isolated_particle_positions_overlay.png']:
        rotate_image_clockwise(os.path.join(run_dir, name))
    csv_path = os.path.join(run_dir, 'trajectory_points.csv')
    with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    rotated_rows: List[Dict[str, str]] = []
    for row in rows:
        x_px, y_px = rotate_xy_clockwise(float(row['x_px']), float(row['y_px']), width, height)
        xs_px, ys_px = rotate_xy_clockwise(float(row['x_smooth_px']), float(row['y_smooth_px']), width, height)
        row['x_px'] = f'{x_px:.4f}'
        row['y_px'] = f'{y_px:.4f}'
        row['x_smooth_px'] = f'{xs_px:.4f}'
        row['y_smooth_px'] = f'{ys_px:.4f}'
        rotated_rows.append(row)
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rotated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rotated_rows)
    xs = np.array([float(r['x_smooth_px']) for r in rotated_rows], dtype=np.float32)
    ys = np.array([float(r['y_smooth_px']) for r in rotated_rows], dtype=np.float32)
    frames = np.array([int(float(r['frame_idx'])) for r in rotated_rows], dtype=np.int32)
    path_length = float(np.sum(np.sqrt(np.sum(np.diff(np.column_stack([xs, ys]), axis=0) ** 2, axis=1)))) if len(xs) > 1 else 0.0
    return {'run_name': os.path.basename(run_dir), 'csv_path': csv_path, 'first_frame': int(frames[0]), 'last_frame': int(frames[-1]), 'n_points': int(len(frames)), 'path_length_px': path_length, 'start_x_px': float(xs[0]), 'start_y_px': float(ys[0]), 'end_x_px': float(xs[-1]), 'end_y_px': float(ys[-1]), 'xs': xs, 'ys': ys}

def rewrite_type_csvs(results: List[Dict[str, object]]) -> None:
    """Handle the rewrite type csvs step used by this script."""
    summary_path = os.path.join(ROOT, '14mm_journey_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['run_name', 'video_path', 'hint_folder', 'seed_frame', 'first_frame', 'last_frame', 'n_points', 'path_length_px', 'start_x_px', 'start_y_px', 'end_x_px', 'end_y_px'])
        for result in results:
            writer.writerow([result['run_name'], '', '', '', result['first_frame'], result['last_frame'], result['n_points'], f"{result['path_length_px']:.4f}", f"{result['start_x_px']:.4f}", f"{result['start_y_px']:.4f}", f"{result['end_x_px']:.4f}", f"{result['end_y_px']:.4f}"])
    lateral_summary = os.path.join(ROOT, '14mm_lateral_slip_summary.csv')
    with open(lateral_summary, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['run_name', 'start_x_px', 'end_x_px', 'start_y_px', 'end_y_px', 'net_lateral_drift_px', 'max_lateral_span_px'])
        for result in results:
            xs = result['xs']
            ys = result['ys']
            writer.writerow([result['run_name'], f'{xs[0]:.4f}', f'{xs[-1]:.4f}', f'{ys[0]:.4f}', f'{ys[-1]:.4f}', f'{xs[-1] - xs[0]:.4f}', f'{float(xs.max()) - float(xs.min()):.4f}'])
    grid = np.linspace(0.0, 1.0, 101)
    interpolated_x = []
    interpolated_y = []
    run_names = []
    for result in results:
        xs = result['xs']
        ys = result['ys']
        coords = np.column_stack([xs, ys])
        if len(coords) < 2:
            continue
        step = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(step)])
        if float(s[-1]) <= 0.0:
            continue
        s_norm = s / s[-1]
        interpolated_x.append(np.interp(grid, s_norm, xs))
        interpolated_y.append(np.interp(grid, s_norm, ys))
        run_names.append(result['run_name'])
    if not interpolated_x:
        return
    xs_stack = np.stack(interpolated_x, axis=0)
    ys_stack = np.stack(interpolated_y, axis=0)
    mean_x = xs_stack.mean(axis=0)
    mean_y = ys_stack.mean(axis=0)
    spread = np.sqrt(((xs_stack - mean_x) ** 2 + (ys_stack - mean_y) ** 2).mean(axis=0))
    progress_csv = os.path.join(ROOT, '14mm_progress_aligned_spread.csv')
    with open(progress_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['progress', 'mean_x_px', 'mean_y_px', 'rms_spread_px'])
        for idx, g in enumerate(grid):
            writer.writerow([f'{g:.4f}', f'{mean_x[idx]:.4f}', f'{mean_y[idx]:.4f}', f'{spread[idx]:.4f}'])
    lateral_progress_csv = os.path.join(ROOT, '14mm_lateral_slip_by_progress.csv')
    with open(lateral_progress_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['progress', 'mean_lateral_px'] + [f'{name}_lateral_px' for name in run_names])
        for idx, g in enumerate(grid):
            row = [f'{g:.4f}', '0.0000']
            row.extend((f'{xs_stack[ridx, idx] - mean_x[idx]:.4f}' for ridx in range(xs_stack.shape[0])))
            writer.writerow(row)

def main() -> None:
    """Run the main entry point for this script."""
    for name in ['14mm_average_background.png', '14mm_all_6_journeys_overlay.png', '14mm_all_6_particle_positions_overlay.png']:
        rotate_image_clockwise(os.path.join(ROOT, name))
    run_dirs = [os.path.join(ROOT, name) for name in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, name)) and name.endswith('_raw_retracked')]
    run_dirs.sort()
    results = [process_run_csv(run_dir) for run_dir in run_dirs]
    rewrite_type_csvs(results)

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
