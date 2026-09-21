"""
Plot builder: build_streamwise_velocity_vs_time_plots.py

Purpose
-------
Generates streamwise velocity versus time plots using corrected FPS handling and optional smoothing.

Inputs
------
Processed trajectory folders and inferred/defined flow-speed metadata.

Outputs
-------
Streamwise velocity figures and CSV exports.

Methodological notes
--------------------
The downstream direction is inferred from trajectory progression and compared against the bulk-flow reference.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUTPUT_ROOT = Path(os.environ.get('STREAMWISE_VELOCITY_TIME_OUTPUT_ROOT', 'C:\\IP Work\\Streamwise Velocity vs Time Plots'))
SMOOTH_WINDOW_POINTS = int(os.environ.get('STREAMWISE_VELOCITY_TIME_SMOOTH_WINDOW_POINTS', '9'))
MAX_FRAME_GAP_FOR_VELOCITY = float(os.environ.get('STREAMWISE_MAX_FRAME_GAP_FOR_VELOCITY', '5.5'))
FLUME_FLOW_MM_S = float(os.environ.get('FLUME_BULK_FLOW_MM_S', '250.0'))
WATER_TUNNEL_3IN_MM_S = float(os.environ.get('WATER_TUNNEL_3IN_MM_S', str(3.0 * 25.4)))
WATER_TUNNEL_45IN_MM_S = float(os.environ.get('WATER_TUNNEL_45IN_MM_S', str(4.5 * 25.4)))

@dataclass(frozen=True)
class PackageDef:
    """Store the data needed by the PackageDef structure."""
    category: str
    batch_label: str
    tracker_root: Path
PACKAGES = [PackageDef(category='water_tunnel', batch_label='Water Tunnel Flow Overview', tracker_root=Path('C:\\IP Work\\Water Tunnel Flow Overview\\tracker_outputs')), PackageDef(category='flume_bottom_view', batch_label='10mm sphere bottom view runs', tracker_root=Path('C:\\IP Work\\10mm sphere bottom view runs\\tracker_outputs')), PackageDef(category='flume_bottom_view', batch_label='14mm sphere bottom view runs', tracker_root=Path('C:\\IP Work\\14mm sphere bottom view runs\\tracker_outputs')), PackageDef(category='flume_side_view', batch_label='24 April Flume Side Views Processed', tracker_root=Path('C:\\IP Work\\24 April Flume Side Views Processed\\tracker_outputs')), PackageDef(category='flume_side_view', batch_label='Straight View Sphere Drops Processed', tracker_root=Path('C:\\IP Work\\Straight View Sphere Drops Processed\\tracker_outputs'))]
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'grid.linewidth': 0.6})

def safe_slug(text: str) -> str:
    """Handle the safe slug step used by this script."""
    return ''.join((ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in text)).strip().replace(' ', '_')

def humanize_run_name(name: str) -> str:
    """Handle the humanize run name step used by this script."""
    cleaned = name
    for token in ('_tracker_output', '_binarized', '_widthlocked_5mm', '_output_refined_5mm'):
        cleaned = cleaned.replace(token, '')
    return cleaned.replace('_', ' ')

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
        if 'summary_files' in parts_lower:
            continue
        if '__attempt_' in str(run_dir).lower():
            continue
        if '_tracker_output' not in run_dir.name.lower():
            continue
        key = str(run_dir).lower()
        if key not in seen:
            seen.add(key)
            run_dirs.append(run_dir)
    return sorted(run_dirs)

def infer_bulk_flow_mm_s(package: PackageDef, run_dir: Path) -> float:
    """Handle the infer bulk flow mm s step used by this script."""
    if package.category == 'water_tunnel':
        parent = run_dir.parent.name
        if '3_Inchps' in parent:
            return WATER_TUNNEL_3IN_MM_S
        return WATER_TUNNEL_45IN_MM_S
    return FLUME_FLOW_MM_S

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
            if np.isfinite(fps) and fps > 1.0:
                return (fps, name)
        finally:
            cap.release()
    return (None, None)

def pick_time_series(df: pd.DataFrame, actual_fps: float | None) -> tuple[str | None, pd.Series | None]:
    """Handle the pick time series step used by this script."""
    if actual_fps is not None:
        for frame_col in ('out_idx', 'frame_idx', 'frame', 'frame_index'):
            if frame_col in df.columns:
                series = pd.to_numeric(df[frame_col], errors='coerce')
                if series.notna().any():
                    return (f'{frame_col}/{actual_fps:.6f}', (series - series.dropna().iloc[0]) / actual_fps)
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

def pick_xy_position_series(df: pd.DataFrame) -> tuple[tuple[str, str] | None, tuple[pd.Series, pd.Series] | None]:
    """Handle the pick xy position series step used by this script."""
    for x_col, y_col in (('x_rel_smooth_mm', 'y_rel_smooth_mm'), ('x_rel_mm', 'y_rel_mm'), ('x_kalman_mm', 'y_kalman_mm'), ('x_mm', 'y_mm')):
        if x_col in df.columns and y_col in df.columns:
            x_series = pd.to_numeric(df[x_col], errors='coerce')
            y_series = pd.to_numeric(df[y_col], errors='coerce')
            if x_series.notna().any() and y_series.notna().any():
                return ((x_col, y_col), (x_series, y_series))
    return (None, None)

def pick_x_position_series(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Handle the pick x position series step used by this script."""
    names, series_pair = pick_xy_position_series(df)
    if names is None or series_pair is None:
        return (None, None)
    return (names[0], series_pair[0])

def gradient_from_position(series: pd.Series, time_series: pd.Series, max_gap_s: float | None=None) -> pd.Series:
    """Handle the gradient from position step used by this script."""
    t = pd.to_numeric(time_series, errors='coerce').to_numpy(dtype=float)
    x = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(x)
    if valid.sum() < 2:
        raise ValueError('Not enough valid position/time points to derive velocity.')
    vx = np.full_like(x, np.nan, dtype=float)
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
                vx[seg_idx] = np.gradient(seg_x, seg_t)
            seg_start = i
    return pd.Series(vx, index=series.index)

def derive_streamwise_velocity(df: pd.DataFrame, time_series: pd.Series, max_gap_s: float | None=None) -> tuple[str, pd.Series]:
    """Handle the derive streamwise velocity step used by this script."""
    x_name, x_series = pick_x_position_series(df)
    if x_series is None:
        raise ValueError('Could not find usable streamwise position or velocity columns.')
    return (f'gradient({x_name})', gradient_from_position(x_series, time_series, max_gap_s=max_gap_s))

def choose_smoothing_window(n_points: int) -> int:
    """Handle the choose smoothing window step used by this script."""
    if n_points <= 2:
        return 1
    window = min(SMOOTH_WINDOW_POINTS, n_points)
    if window % 2 == 0:
        window -= 1
    return max(window, 3)

def smooth_series(series: pd.Series) -> tuple[pd.Series, int]:
    """Handle the smooth series step used by this script."""
    n_points = int(len(series))
    window = choose_smoothing_window(n_points)
    if window <= 1:
        return (series.astype(float).copy(), 1)
    smoothed = pd.to_numeric(series, errors='coerce').rolling(window=window, center=True, min_periods=1).mean()
    return (smoothed, window)

def style_axes(ax: plt.Axes) -> None:
    """Handle the style axes step used by this script."""
    ax.grid(True, axis='both', alpha=0.22, color='#7f8c8d')
    ax.set_axisbelow(True)

def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save the figure used by this script."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def determine_downstream_sign(df: pd.DataFrame) -> tuple[float, str]:
    """Handle the determine downstream sign step used by this script."""
    x_name, x_series = pick_x_position_series(df)
    if x_series is not None:
        xs = pd.to_numeric(x_series, errors='coerce').dropna()
        if len(xs) >= 2:
            delta = float(xs.iloc[-1] - xs.iloc[0])
            if abs(delta) > 1e-09:
                return (1.0 if delta > 0 else -1.0, f'sign(net {x_name} displacement)')
    if 'vx_mm_s' in df.columns:
        vx = pd.to_numeric(df['vx_mm_s'], errors='coerce').dropna()
        if not vx.empty:
            med = float(vx.median())
            if abs(med) > 1e-09:
                return (1.0 if med > 0 else -1.0, 'sign(median vx_mm_s)')
    return (1.0, 'default_positive')

def prepare_streamwise_plot_data(csv_path: Path, bulk_flow_mm_s: float) -> tuple[pd.DataFrame, dict]:
    """Handle the prepare streamwise plot data step used by this script."""
    df = pd.read_csv(csv_path)
    run_dir = csv_path.parent
    found_series = pd.to_numeric(df['found_this_frame'], errors='coerce') if 'found_this_frame' in df.columns else pd.Series(1, index=df.index, dtype=float)
    work_df = df[found_series.fillna(0) == 1].copy()
    if work_df.empty:
        raise ValueError('No found-frame rows available for streamwise plot.')
    actual_fps, fps_source = detect_run_fps(run_dir)
    time_name, time_series = pick_time_series(work_df, actual_fps)
    if time_series is None:
        raise ValueError('Could not find usable time column.')
    max_gap_s = MAX_FRAME_GAP_FOR_VELOCITY / actual_fps if actual_fps is not None and actual_fps > 1.0 else None
    velocity_name, velocity_series = derive_streamwise_velocity(work_df, time_series, max_gap_s=max_gap_s)
    sign_factor, sign_reason = determine_downstream_sign(work_df)
    raw_streamwise = pd.to_numeric(velocity_series, errors='coerce') * float(sign_factor)
    constrained_streamwise = raw_streamwise.clip(lower=0.0, upper=bulk_flow_mm_s)
    plot_df = pd.DataFrame({'time_s': pd.to_numeric(time_series, errors='coerce'), 'streamwise_velocity_raw_mm_s': raw_streamwise, 'streamwise_velocity_mm_s': constrained_streamwise, 'streamwise_sign_factor': float(sign_factor)})
    plot_df['found_this_frame'] = 1
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['time_s', 'streamwise_velocity_mm_s']).copy()
    plot_df = plot_df.sort_values('time_s').drop_duplicates('time_s')
    if plot_df.empty:
        raise ValueError('No finite time/streamwise-velocity rows remained after cleaning.')
    smoothed, smooth_window = smooth_series(plot_df['streamwise_velocity_mm_s'])
    plot_df['smoothed_streamwise_velocity_mm_s'] = smoothed.clip(lower=0.0, upper=bulk_flow_mm_s)
    plot_df['bulk_flow_reference_mm_s'] = bulk_flow_mm_s
    meta = {'time_column_used': time_name, 'streamwise_velocity_column_used': velocity_name, 'actual_fps_used': float(actual_fps) if actual_fps is not None else np.nan, 'actual_fps_source': fps_source or '', 'max_gap_s_used': float(max_gap_s) if max_gap_s is not None else np.nan, 'streamwise_sign_factor': float(sign_factor), 'streamwise_sign_reason': sign_reason, 'streamwise_constrained_to_bulk': True, 'smoothing_window_points': int(smooth_window), 'n_points': int(len(plot_df)), 'time_min_s': float(plot_df['time_s'].min()), 'time_max_s': float(plot_df['time_s'].max()), 'streamwise_velocity_min_mm_s': float(plot_df['streamwise_velocity_mm_s'].min()), 'streamwise_velocity_max_mm_s': float(plot_df['streamwise_velocity_mm_s'].max()), 'streamwise_velocity_mean_mm_s': float(plot_df['streamwise_velocity_mm_s'].mean()), 'smoothed_streamwise_velocity_min_mm_s': float(plot_df['smoothed_streamwise_velocity_mm_s'].min()), 'smoothed_streamwise_velocity_max_mm_s': float(plot_df['smoothed_streamwise_velocity_mm_s'].max()), 'smoothed_streamwise_velocity_mean_mm_s': float(plot_df['smoothed_streamwise_velocity_mm_s'].mean())}
    return (plot_df, meta)

def plot_streamwise_velocity_time(plot_df: pd.DataFrame, bulk_flow_mm_s: float, title: str, out_path: Path, smooth_window: int, actual_fps_used: float | None) -> None:
    """Handle the plot streamwise velocity time step used by this script."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(plot_df['time_s'].to_numpy(dtype=float), plot_df['streamwise_velocity_mm_s'].to_numpy(dtype=float), s=16, alpha=0.35, color='#9bbcf7', linewidths=0, rasterized=True, label='tracked frames')
    ax.plot(plot_df['time_s'].to_numpy(dtype=float), plot_df['streamwise_velocity_mm_s'].to_numpy(dtype=float), color='#9bbcf7', linewidth=1.1, alpha=0.75, label='bounded streamwise trace')
    ax.plot(plot_df['time_s'].to_numpy(dtype=float), plot_df['smoothed_streamwise_velocity_mm_s'].to_numpy(dtype=float), color='#0b2c84', linewidth=2.7, label=f'smoothed trend ({smooth_window}-point moving average)')
    ax.axhline(0.0, color='#666666', linewidth=1.0, linestyle='--', label='zero')
    ax.axhline(float(bulk_flow_mm_s), color='#d47a00', linewidth=1.6, linestyle=':', label=f'bulk flow ({bulk_flow_mm_s:.1f} mm/s)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('streamwise velocity (mm/s)')
    ax.set_title(title)
    style_axes(ax)
    ymax = max(float(bulk_flow_mm_s), float(plot_df['smoothed_streamwise_velocity_mm_s'].max()), float(plot_df['streamwise_velocity_mm_s'].max()))
    ax.set_ylim(0.0, ymax * 1.05 if ymax > 0 else 1.0)
    note_bits = [f'{len(plot_df)} tracked frames', 'range constrained to 0..bulk flow']
    if actual_fps_used is not None and np.isfinite(actual_fps_used):
        note_bits.insert(0, f'{actual_fps_used:.2f} fps')
    ax.text(0.99, 0.03, ' | '.join(note_bits), ha='right', va='bottom', transform=ax.transAxes, fontsize=8.5, color='#555555')
    ax.legend(frameon=False, ncol=2, loc='upper left')
    save_figure(fig, out_path)

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
            bulk_flow_mm_s = infer_bulk_flow_mm_s(package, run_dir)
            try:
                plot_df, meta = prepare_streamwise_plot_data(csv_path, float(bulk_flow_mm_s))
                csv_out = run_out_dir / 'streamwise_velocity_vs_time_with_smooth.csv'
                png_out = run_out_dir / 'streamwise_velocity_vs_time_smoothed_overlay.png'
                plot_df['bulk_flow_reference_mm_s'] = float(bulk_flow_mm_s)
                plot_df.to_csv(csv_out, index=False)
                plot_streamwise_velocity_time(plot_df, bulk_flow_mm_s=float(bulk_flow_mm_s), title=f'{humanize_run_name(run_dir.name)} streamwise velocity vs time', out_path=png_out, smooth_window=int(meta['smoothing_window_points']), actual_fps_used=float(meta['actual_fps_used']) if pd.notna(meta['actual_fps_used']) else None)
                summary_rows.append({'status': 'ok', 'category': package.category, 'batch_label': package.batch_label, 'run_name': run_dir.name, 'source_run_dir': str(run_dir), 'source_tracks_processed_csv': str(csv_path), 'output_dir': str(run_out_dir), 'output_csv': str(csv_out), 'output_png': str(png_out), 'bulk_flow_mm_s': float(bulk_flow_mm_s), **meta})
            except Exception as exc:
                (run_out_dir / 'error.txt').write_text(str(exc), encoding='utf-8')
                summary_rows.append({'status': 'error', 'category': package.category, 'batch_label': package.batch_label, 'run_name': run_dir.name, 'source_run_dir': str(run_dir), 'source_tracks_processed_csv': str(csv_path), 'output_dir': str(run_out_dir), 'bulk_flow_mm_s': float(bulk_flow_mm_s), 'error': str(exc)})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_ROOT / 'streamwise_velocity_vs_time_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    manifest = {'output_root': str(OUTPUT_ROOT), 'packages': package_summaries, 'total_runs_detected': int(sum((item['run_count_detected'] for item in package_summaries))), 'successful_runs': int((summary_df['status'] == 'ok').sum()) if not summary_df.empty else 0, 'failed_runs': int((summary_df['status'] == 'error').sum()) if not summary_df.empty else 0, 'summary_csv': str(summary_path), 'flume_bulk_flow_mm_s': float(FLUME_FLOW_MM_S), 'water_tunnel_3inch_bulk_flow_mm_s': float(WATER_TUNNEL_3IN_MM_S), 'water_tunnel_45inch_bulk_flow_mm_s': float(WATER_TUNNEL_45IN_MM_S)}
    (OUTPUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    raise SystemExit(main())
