"""
Plot builder: build_instantaneous_velocity_vs_time_plots.py

Purpose
-------
Generates per-run instantaneous speed versus time plots and matching CSV exports.

Inputs
------
Processed trajectory folders from tracker outputs.

Outputs
-------
Publication-ready velocity-time figures and tabulated plotting data.

Methodological notes
--------------------
Velocity is reconstructed from calibrated displacement and acquisition frame rate; smoothing is used only to clarify trends, not to replace raw values.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUTPUT_ROOT = Path(os.environ.get('VELOCITY_TIME_OUTPUT_ROOT', 'C:\\IP Work\\Instantaneous Velocity vs Time Plots'))
SMOOTH_WINDOW_POINTS = int(os.environ.get('VELOCITY_TIME_SMOOTH_WINDOW_POINTS', '9'))

@dataclass(frozen=True)
class PackageDef:
    """Store the data needed by the PackageDef structure."""
    category: str
    batch_label: str
    tracker_root: Path
PACKAGES = [PackageDef(category='water_tunnel', batch_label='Water Tunnel Flow Overview', tracker_root=Path('C:\\IP Work\\Water Tunnel Flow Overview\\tracker_outputs')), PackageDef(category='flume_bottom_view', batch_label='10mm sphere bottom view runs', tracker_root=Path('C:\\IP Work\\10mm sphere bottom view runs\\tracker_outputs')), PackageDef(category='flume_bottom_view', batch_label='14mm sphere bottom view runs', tracker_root=Path('C:\\IP Work\\14mm sphere bottom view runs\\tracker_outputs')), PackageDef(category='flume_side_view', batch_label='24 April Flume Side Views Processed', tracker_root=Path('C:\\IP Work\\24 April Flume Side Views Processed\\tracker_outputs')), PackageDef(category='flume_side_view', batch_label='Straight View Sphere Drops Processed', tracker_root=Path('C:\\IP Work\\Straight View Sphere Drops Processed\\tracker_outputs'))]

def safe_slug(text: str) -> str:
    """Handle the safe slug step used by this script."""
    return ''.join((ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in text)).strip().replace(' ', '_')

def find_run_dirs(tracker_root: Path) -> list[Path]:
    """Handle the find run dirs step used by this script."""
    if not tracker_root.exists():
        return []
    run_dirs: list[Path] = []
    seen: set[str] = set()
    for run_dir in tracker_root.rglob('*'):
        if not run_dir.is_dir():
            continue
        parts_lower = {part.lower() for part in run_dir.parts}
        name_lower = run_dir.name.lower()
        if 'summary_files' in parts_lower:
            continue
        if '__attempt_' in str(run_dir).lower():
            continue
        if '_tracker_output' not in name_lower:
            continue
        key = str(run_dir).lower()
        if key not in seen:
            seen.add(key)
            run_dirs.append(run_dir)
    return sorted(run_dirs)

def pick_time_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Handle the pick time column step used by this script."""
    for col in ('t_rel_s', 'time_s', 't_s'):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().any():
                return (col, series)
    for frame_col in ('out_idx', 'frame_idx', 'frame', 'frame_index'):
        if frame_col in df.columns:
            series = pd.to_numeric(df[frame_col], errors='coerce') / 30.0
            if series.notna().any():
                return (f'{frame_col}/30', series)
    return (None, None)

def pick_speed_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Handle the pick speed column step used by this script."""
    if 'speed_mm_s' in df.columns:
        series = pd.to_numeric(df['speed_mm_s'], errors='coerce')
        if series.notna().any():
            return ('speed_mm_s', series)
    if 'speed_m_s' in df.columns:
        series = pd.to_numeric(df['speed_m_s'], errors='coerce') * 1000.0
        if series.notna().any():
            return ('speed_m_s*1000', series)
    if 'vx_mm_s' in df.columns and 'vy_mm_s' in df.columns:
        vx = pd.to_numeric(df['vx_mm_s'], errors='coerce')
        vy = pd.to_numeric(df['vy_mm_s'], errors='coerce')
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if pd.Series(speed).notna().any():
            return ('sqrt(vx_mm_s^2+vy_mm_s^2)', pd.Series(speed, index=df.index))
    if 'vx_m_s' in df.columns and 'vy_m_s' in df.columns:
        vx = pd.to_numeric(df['vx_m_s'], errors='coerce')
        vy = pd.to_numeric(df['vy_m_s'], errors='coerce')
        speed = np.sqrt(vx ** 2 + vy ** 2) * 1000.0
        if pd.Series(speed).notna().any():
            return ('sqrt(vx_m_s^2+vy_m_s^2)*1000', pd.Series(speed, index=df.index))
    return (None, None)

def choose_smoothing_window(n_points: int) -> int:
    """Handle the choose smoothing window step used by this script."""
    if n_points <= 2:
        return 1
    window = min(SMOOTH_WINDOW_POINTS, n_points)
    if window % 2 == 0:
        window -= 1
    return max(window, 3)

def smooth_velocity_series(speed_series: pd.Series) -> tuple[pd.Series, int]:
    """Handle the smooth velocity series step used by this script."""
    n_points = int(len(speed_series))
    window = choose_smoothing_window(n_points)
    if window <= 1:
        return (speed_series.astype(float).copy(), 1)
    smoothed = pd.to_numeric(speed_series, errors='coerce').rolling(window=window, center=True, min_periods=1).mean()
    return (smoothed, window)

def prepare_velocity_time_data(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """Handle the prepare velocity time data step used by this script."""
    df = pd.read_csv(csv_path)
    time_name, time_series = pick_time_column(df)
    speed_name, speed_series = pick_speed_column(df)
    if time_series is None or speed_series is None:
        raise ValueError('Could not find usable time or speed columns.')
    plot_df = pd.DataFrame({'time_s': pd.to_numeric(time_series, errors='coerce'), 'instantaneous_velocity_mm_s': pd.to_numeric(speed_series, errors='coerce')})
    if 'found_this_frame' in df.columns:
        plot_df['found_this_frame'] = pd.to_numeric(df['found_this_frame'], errors='coerce')
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['time_s', 'instantaneous_velocity_mm_s']).copy()
    plot_df = plot_df.sort_values('time_s').drop_duplicates('time_s')
    if plot_df.empty:
        raise ValueError('No finite time/velocity rows remained after cleaning.')
    smoothed, smooth_window = smooth_velocity_series(plot_df['instantaneous_velocity_mm_s'])
    plot_df['smoothed_velocity_mm_s'] = smoothed
    meta = {'time_column_used': time_name, 'speed_column_used': speed_name, 'smoothing_window_points': int(smooth_window), 'n_points': int(len(plot_df)), 'time_min_s': float(plot_df['time_s'].min()), 'time_max_s': float(plot_df['time_s'].max()), 'speed_min_mm_s': float(plot_df['instantaneous_velocity_mm_s'].min()), 'speed_max_mm_s': float(plot_df['instantaneous_velocity_mm_s'].max()), 'speed_mean_mm_s': float(plot_df['instantaneous_velocity_mm_s'].mean()), 'smoothed_speed_min_mm_s': float(plot_df['smoothed_velocity_mm_s'].min()), 'smoothed_speed_max_mm_s': float(plot_df['smoothed_velocity_mm_s'].max()), 'smoothed_speed_mean_mm_s': float(plot_df['smoothed_velocity_mm_s'].mean())}
    return (plot_df, meta)

def plot_velocity_time(plot_df: pd.DataFrame, title: str, out_path: Path, smooth_window: int) -> None:
    """Handle the plot velocity time step used by this script."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(plot_df['time_s'].to_numpy(dtype=float), plot_df['instantaneous_velocity_mm_s'].to_numpy(dtype=float), color='#7aa6ff', linewidth=1.4, alpha=0.75, label='raw instantaneous velocity')
    ax.plot(plot_df['time_s'].to_numpy(dtype=float), plot_df['smoothed_velocity_mm_s'].to_numpy(dtype=float), color='#0b2c84', linewidth=2.4, label=f'smoothed velocity ({smooth_window}-point moving average)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('instantaneous velocity (mm/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main() -> int:
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    package_summaries: list[dict] = []
    for package in PACKAGES:
        run_dirs = find_run_dirs(package.tracker_root)
        package_summaries.append({'category': package.category, 'batch_label': package.batch_label, 'tracker_root': str(package.tracker_root), 'run_count_detected': int(len(run_dirs))})
        package_out_root = OUTPUT_ROOT / package.category / safe_slug(package.batch_label)
        package_out_root.mkdir(parents=True, exist_ok=True)
        for run_dir in run_dirs:
            csv_path = run_dir / 'tracks_processed.csv'
            run_out_dir = package_out_root / run_dir.name
            run_out_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_df, meta = prepare_velocity_time_data(csv_path)
                csv_out = run_out_dir / 'instantaneous_velocity_vs_time_with_smooth.csv'
                png_out = run_out_dir / 'instantaneous_velocity_vs_time_smoothed_overlay.png'
                plot_df.to_csv(csv_out, index=False)
                plot_velocity_time(plot_df, f'{run_dir.name} velocity vs time', png_out, smooth_window=int(meta['smoothing_window_points']))
                summary_rows.append({'status': 'ok', 'category': package.category, 'batch_label': package.batch_label, 'run_name': run_dir.name, 'source_run_dir': str(run_dir), 'source_tracks_processed_csv': str(csv_path), 'output_dir': str(run_out_dir), 'output_csv': str(csv_out), 'output_png': str(png_out), **meta})
            except Exception as exc:
                (run_out_dir / 'error.txt').write_text(str(exc), encoding='utf-8')
                summary_rows.append({'status': 'error', 'category': package.category, 'batch_label': package.batch_label, 'run_name': run_dir.name, 'source_run_dir': str(run_dir), 'source_tracks_processed_csv': str(csv_path), 'output_dir': str(run_out_dir), 'error': str(exc)})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_ROOT / 'velocity_vs_time_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    manifest = {'output_root': str(OUTPUT_ROOT), 'packages': package_summaries, 'total_runs_detected': int(sum((item['run_count_detected'] for item in package_summaries))), 'successful_runs': int((summary_df['status'] == 'ok').sum()) if not summary_df.empty else 0, 'failed_runs': int((summary_df['status'] == 'error').sum()) if not summary_df.empty else 0, 'summary_csv': str(summary_path)}
    (OUTPUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    raise SystemExit(main())
