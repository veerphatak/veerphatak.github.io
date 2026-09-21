"""
Analysis and plot builder: build_particle_type_overall_velocity_analysis.py

Purpose
-------
Pools final runs by particle type and compares overall velocity behaviour across spheres, rods and discs.

Inputs
------
Processed water-tunnel and flume trajectory outputs.

Outputs
-------
Combined velocity summaries, particle-type comparison figures and CSV exports.

Methodological notes
--------------------
Runs are grouped by particle type and flow condition to support cross-particle comparisons without mixing incompatible reference speeds.

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
OUTPUT_ROOT = Path(os.environ.get('PARTICLE_TYPE_VELOCITY_ANALYSIS_ROOT', 'C:\\IP Work\\Particle Type Overall Velocity Analysis'))
AREA_BINS = int(os.environ.get('PARTICLE_TYPE_AREA_BINS', '20'))
TIME_BINS = int(os.environ.get('PARTICLE_TYPE_TIME_BINS', '24'))
FLUME_FLOW_MM_S = float(os.environ.get('FLUME_BULK_FLOW_MM_S', '250.0'))
WATER_TUNNEL_3IN_MM_S = float(os.environ.get('WATER_TUNNEL_3IN_MM_S', str(3.0 * 25.4)))
WATER_TUNNEL_45IN_MM_S = float(os.environ.get('WATER_TUNNEL_45IN_MM_S', str(4.5 * 25.4)))
MAX_FRAME_GAP_FOR_VELOCITY = float(os.environ.get('PARTICLE_TYPE_MAX_FRAME_GAP_FOR_VELOCITY', '5.5'))
ERROR_BAR_MIN_RUNS = int(os.environ.get('PARTICLE_TYPE_ERROR_BAR_MIN_RUNS', '2'))
ERROR_BAR_CAPSIZE = float(os.environ.get('PARTICLE_TYPE_ERROR_BAR_CAPSIZE', '2.6'))
ERROR_BAR_MARKER_SIZE = float(os.environ.get('PARTICLE_TYPE_ERROR_BAR_MARKER_SIZE', '3.6'))

@dataclass(frozen=True)
class SourceDef:
    """Store the data needed by the SourceDef structure."""
    environment_group: str
    batch_label: str
    tracker_root: Path
SOURCES = [SourceDef(environment_group='water_tunnel', batch_label='Water Tunnel Flow Overview', tracker_root=Path('C:\\IP Work\\Water Tunnel Flow Overview\\tracker_outputs')), SourceDef(environment_group='flume_spheres', batch_label='10mm sphere bottom view runs', tracker_root=Path('C:\\IP Work\\10mm sphere bottom view runs\\tracker_outputs')), SourceDef(environment_group='flume_spheres', batch_label='14mm sphere bottom view runs', tracker_root=Path('C:\\IP Work\\14mm sphere bottom view runs\\tracker_outputs')), SourceDef(environment_group='flume_spheres', batch_label='24 April Flume Side Views Processed', tracker_root=Path('C:\\IP Work\\24 April Flume Side Views Processed\\tracker_outputs')), SourceDef(environment_group='flume_spheres', batch_label='Straight View Sphere Drops Processed', tracker_root=Path('C:\\IP Work\\Straight View Sphere Drops Processed\\tracker_outputs'))]
COLOUR_CYCLE = ['#0b5fff', '#e07a00', '#1b9e77', '#c0392b', '#7a4dd8', '#8c564b', '#17a2b8']
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'grid.linewidth': 0.6})

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

def infer_particle_type(run_name: str) -> str | None:
    """Handle the infer particle type step used by this script."""
    name = run_name.lower()
    if 'rod' in name:
        return 'rod'
    if 'disc' in name:
        return 'disc'
    if '10mm' in name and 'sphere' in name:
        return '10 mm sphere'
    if '14mm' in name and 'sphere' in name:
        return '14 mm sphere'
    return None

def infer_batch_subgroup(source: SourceDef, run_dir: Path) -> str:
    """Handle the infer batch subgroup step used by this script."""
    if source.environment_group == 'water_tunnel':
        parent_name = run_dir.parent.name
        if '3_Inchps' in parent_name:
            return '3 inch/s'
        if '4.5_Inchps' in parent_name:
            return '4.5 inch/s'
        return source.batch_label
    lower = source.batch_label.lower()
    if 'bottom view' in lower:
        return 'bottom view'
    if '24 april' in lower:
        return 'side view (24 April)'
    if 'straight view' in lower:
        return 'straight view'
    return source.batch_label

def infer_bulk_flow_mm_s(source: SourceDef, run_dir: Path) -> float:
    """Handle the infer bulk flow mm s step used by this script."""
    if source.environment_group == 'water_tunnel':
        parent_name = run_dir.parent.name
        if '3_Inchps' in parent_name:
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

def choose_time_series(df: pd.DataFrame, actual_fps: float | None) -> tuple[str | None, pd.Series | None]:
    """Handle the choose time series step used by this script."""
    if actual_fps is not None:
        for frame_col in ('out_idx', 'frame_idx', 'frame', 'frame_index'):
            if frame_col in df.columns:
                s = pd.to_numeric(df[frame_col], errors='coerce')
                if s.notna().any():
                    return (f'{frame_col}/{actual_fps:.6f}', (s - s.dropna().iloc[0]) / actual_fps)
    for col in ('t_rel_s', 'time_s', 't_s'):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            if s.notna().any():
                return (col, s)
    for frame_col in ('out_idx', 'frame_idx', 'frame', 'frame_index'):
        if frame_col in df.columns:
            s = pd.to_numeric(df[frame_col], errors='coerce') / 30.0
            if s.notna().any():
                return (f'{frame_col}/30', s)
    return (None, None)

def choose_xy_series(df: pd.DataFrame) -> tuple[tuple[str, str] | None, tuple[pd.Series, pd.Series] | None]:
    """Handle the choose xy series step used by this script."""
    for x_col, y_col in (('x_rel_smooth_mm', 'y_rel_smooth_mm'), ('x_rel_mm', 'y_rel_mm'), ('x_kalman_mm', 'y_kalman_mm'), ('x_mm', 'y_mm')):
        if x_col in df.columns and y_col in df.columns:
            xs = pd.to_numeric(df[x_col], errors='coerce')
            ys = pd.to_numeric(df[y_col], errors='coerce')
            if xs.notna().any() and ys.notna().any():
                return ((x_col, y_col), (xs, ys))
    return (None, None)

def choose_x_series(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Handle the choose x series step used by this script."""
    xy_names, xy_series = choose_xy_series(df)
    if xy_names is not None and xy_series is not None:
        return (xy_names[0], xy_series[0])
    return (None, None)

def gradient_from_position(pos: pd.Series, time_series: pd.Series, max_gap_s: float | None=None) -> pd.Series:
    """Handle the gradient from position step used by this script."""
    t = pd.to_numeric(time_series, errors='coerce').to_numpy(dtype=float)
    x = pd.to_numeric(pos, errors='coerce').to_numpy(dtype=float)
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
    return pd.Series(vx, index=pos.index)

def derive_streamwise_velocity(df: pd.DataFrame, time_series: pd.Series, max_gap_s: float | None=None) -> tuple[str, pd.Series]:
    """Handle the derive streamwise velocity step used by this script."""
    x_name, x_series = choose_x_series(df)
    if x_series is None:
        raise ValueError('Could not find usable x position or streamwise velocity column.')
    return (f'gradient({x_name})', gradient_from_position(x_series, time_series, max_gap_s=max_gap_s))

def determine_downstream_sign(df: pd.DataFrame) -> tuple[float, str]:
    """Handle the determine downstream sign step used by this script."""
    x_name, x_series = choose_x_series(df)
    if x_series is not None:
        xs = pd.to_numeric(x_series, errors='coerce').dropna()
        if len(xs) >= 2:
            delta = float(xs.iloc[-1] - xs.iloc[0])
            if abs(delta) > 1e-09:
                return (1.0 if delta > 0 else -1.0, f'sign(net {x_name} displacement)')
    if 'vx_mm_s' in df.columns:
        vx = pd.to_numeric(df['vx_mm_s'], errors='coerce').dropna()
        if not vx.empty:
            median_vx = float(vx.median())
            if abs(median_vx) > 1e-09:
                return (1.0 if median_vx > 0 else -1.0, 'sign(median vx_mm_s)')
    return (1.0, 'default_positive')

def derive_speed_series(df: pd.DataFrame, time_series: pd.Series, max_gap_s: float | None=None) -> tuple[str, pd.Series]:
    """Handle the derive speed series step used by this script."""
    xy_names, xy_series = choose_xy_series(df)
    if xy_names is None or xy_series is None:
        raise ValueError('Could not find usable x/y position columns to derive speed.')
    x_name, y_name = xy_names
    x_series, y_series = xy_series
    vx = gradient_from_position(x_series, time_series, max_gap_s=max_gap_s)
    vy = gradient_from_position(y_series, time_series, max_gap_s=max_gap_s)
    speed = np.sqrt(pd.to_numeric(vx, errors='coerce') ** 2 + pd.to_numeric(vy, errors='coerce') ** 2)
    return (f'sqrt(gradient({x_name})^2+gradient({y_name})^2)', pd.Series(speed, index=df.index))

def prepare_run_points(run_dir: Path, source: SourceDef) -> tuple[pd.DataFrame, dict]:
    """Handle the prepare run points step used by this script."""
    csv_path = run_dir / 'tracks_processed.csv'
    df = pd.read_csv(csv_path)
    particle_type = infer_particle_type(run_dir.name)
    if particle_type is None:
        raise ValueError(f'Could not infer particle type from run name: {run_dir.name}')
    found_series_full = pd.to_numeric(df['found_this_frame'], errors='coerce') if 'found_this_frame' in df.columns else pd.Series(1, index=df.index, dtype=float)
    work_df = df[found_series_full.fillna(0) == 1].copy()
    if work_df.empty:
        raise ValueError('No found-frame rows available for velocity calculation.')
    if 'area_px2' not in work_df.columns:
        raise ValueError('Missing area_px2 column for normalised area analysis.')
    actual_fps, fps_source = detect_run_fps(run_dir)
    time_name, time_series = choose_time_series(work_df, actual_fps)
    if time_series is None:
        raise ValueError('Could not find usable time column.')
    max_gap_s = MAX_FRAME_GAP_FOR_VELOCITY / actual_fps if actual_fps is not None and actual_fps > 1.0 else None
    speed_name, speed_series = derive_speed_series(work_df, time_series, max_gap_s=max_gap_s)
    streamwise_name, streamwise_series = derive_streamwise_velocity(work_df, time_series, max_gap_s=max_gap_s)
    sign_factor, sign_reason = determine_downstream_sign(work_df)
    bulk_flow = float(infer_bulk_flow_mm_s(source, run_dir))
    streamwise_series_raw = pd.to_numeric(streamwise_series, errors='coerce') * float(sign_factor)
    streamwise_series = streamwise_series_raw.clip(lower=0.0, upper=bulk_flow)
    area_series = pd.to_numeric(work_df['area_px2'], errors='coerce')
    found_series = pd.Series(1, index=work_df.index, dtype=float)
    points = pd.DataFrame({'time_s': pd.to_numeric(time_series, errors='coerce'), 'instantaneous_velocity_mm_s': pd.to_numeric(speed_series, errors='coerce'), 'streamwise_velocity_raw_mm_s': streamwise_series_raw, 'streamwise_velocity_mm_s': streamwise_series, 'area_px2': area_series, 'found_this_frame': found_series})
    points = points.replace([np.inf, -np.inf], np.nan)
    points = points[points['found_this_frame'].fillna(0) == 1].copy()
    points = points.dropna(subset=['time_s', 'instantaneous_velocity_mm_s', 'streamwise_velocity_mm_s', 'area_px2'])
    if points.empty:
        raise ValueError('No usable found-frame points after filtering.')
    max_area = float(points['area_px2'].max())
    if not np.isfinite(max_area) or max_area <= 0.0:
        raise ValueError('Non-positive max area; cannot compute normalised area.')
    points['normalised_area'] = points['area_px2'] / max_area
    points['environment_group'] = source.environment_group
    points['source_batch'] = source.batch_label
    points['subgroup'] = infer_batch_subgroup(source, run_dir)
    points['particle_type'] = particle_type
    points['run_name'] = run_dir.name
    points['bulk_flow_mm_s'] = infer_bulk_flow_mm_s(source, run_dir)
    points['streamwise_sign_factor'] = float(sign_factor)
    points['normalised_velocity'] = points['instantaneous_velocity_mm_s'] / bulk_flow
    points['normalised_streamwise_velocity'] = points['streamwise_velocity_mm_s'] / bulk_flow
    meta = {'run_name': run_dir.name, 'source_batch': source.batch_label, 'subgroup': infer_batch_subgroup(source, run_dir), 'particle_type': particle_type, 'time_column_used': time_name, 'speed_column_used': speed_name, 'streamwise_velocity_column_used': streamwise_name, 'actual_fps_used': float(actual_fps) if actual_fps is not None else np.nan, 'actual_fps_source': fps_source or '', 'max_gap_s_used': float(max_gap_s) if max_gap_s is not None else np.nan, 'streamwise_sign_factor': float(sign_factor), 'streamwise_sign_reason': sign_reason, 'streamwise_constrained_to_bulk': True, 'bulk_flow_mm_s': bulk_flow, 'max_area_px2': max_area, 'n_points': int(len(points))}
    return (points, meta)

def binned_summary_curve(points: pd.DataFrame, x_col: str, y_col: str, edges: np.ndarray) -> pd.DataFrame:
    """Handle the binned summary curve step used by this script."""
    x_series = pd.to_numeric(points[x_col], errors='coerce')
    y_series = pd.to_numeric(points[y_col], errors='coerce')
    valid = x_series.notna() & y_series.notna()
    rows: list[dict] = []
    for idx in range(len(edges) - 1):
        left, right = (edges[idx], edges[idx + 1])
        if idx == len(edges) - 2:
            mask = valid & (x_series >= left) & (x_series <= right)
        else:
            mask = valid & (x_series >= left) & (x_series < right)
        count = int(mask.sum())
        if count == 0:
            continue
        y_sel = y_series.loc[mask].to_numpy(dtype=float)
        run_medians = pd.DataFrame({'run_name': points.loc[mask, 'run_name'].astype(str), 'metric': y_series.loc[mask].to_numpy(dtype=float)}).groupby('run_name', sort=False)['metric'].median().dropna()
        run_count = int(run_medians.shape[0])
        run_median_mean = float(run_medians.mean()) if run_count else float(np.nan)
        run_median_median = float(run_medians.median()) if run_count else float(np.nan)
        run_median_q25 = float(run_medians.quantile(0.25)) if run_count else float(np.nan)
        run_median_q75 = float(run_medians.quantile(0.75)) if run_count else float(np.nan)
        run_median_std = float(run_medians.std(ddof=1)) if run_count >= 2 else float(np.nan)
        run_median_sem = float(run_median_std / np.sqrt(run_count)) if run_count >= 2 else float(np.nan)
        rows.append({'bin_left': float(left), 'bin_right': float(right), 'bin_center': float((left + right) * 0.5), 'count': count, 'q25': float(np.nanpercentile(y_sel, 25)), 'median': float(np.nanmedian(y_sel)), 'q75': float(np.nanpercentile(y_sel, 75)), 'run_count': run_count, 'run_median_mean': run_median_mean, 'run_median_median': run_median_median, 'run_median_q25': run_median_q25, 'run_median_q75': run_median_q75, 'run_median_std': run_median_std, 'run_median_sem': run_median_sem, 'error_bar_supported': bool(run_count >= ERROR_BAR_MIN_RUNS and np.isfinite(run_median_sem))})
    return pd.DataFrame(rows)

def add_summary_line_and_error_bars(ax: plt.Axes, summary: pd.DataFrame, color: str, label: str, fill_alpha: float, line_width: float) -> None:
    """Handle the add summary line and error bars step used by this script."""
    x = summary['bin_center'].to_numpy(dtype=float)
    pooled_q25 = summary['q25'].to_numpy(dtype=float)
    pooled_q75 = summary['q75'].to_numpy(dtype=float)
    plotted_y = summary['run_median_mean'].to_numpy(dtype=float)
    ax.fill_between(x, pooled_q25, pooled_q75, color=color, alpha=fill_alpha, linewidth=0)
    ax.plot(x, plotted_y, color=color, linewidth=line_width, label=label)
    run_counts = summary['run_count'].to_numpy(dtype=float)
    sem = summary['run_median_sem'].to_numpy(dtype=float)
    err_mask = (run_counts >= ERROR_BAR_MIN_RUNS) & np.isfinite(plotted_y) & np.isfinite(sem)
    if np.any(err_mask):
        ax.errorbar(x[err_mask], plotted_y[err_mask], yerr=sem[err_mask], fmt='o', markersize=ERROR_BAR_MARKER_SIZE, capsize=ERROR_BAR_CAPSIZE, elinewidth=0.95, capthick=0.95, markerfacecolor='white', markeredgecolor=color, markeredgewidth=0.95, color=color, alpha=0.95, linestyle='none')

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

def plot_area_metric(points: pd.DataFrame, y_col: str, ylabel: str, title: str, out_path: Path, plot_type: str, y_ref: float | None=None) -> pd.DataFrame:
    """Handle the plot area metric step used by this script."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    subgroups = list(dict.fromkeys(points['subgroup'].tolist()))
    edges = np.linspace(0.0, 1.0, AREA_BINS + 1)
    summary_rows: list[pd.DataFrame] = []
    ax.scatter(points['normalised_area'].to_numpy(dtype=float), points[y_col].to_numpy(dtype=float), s=6, alpha=0.045, color='#7f8c8d', linewidths=0, rasterized=True, label='all tracked frames')
    for idx, subgroup in enumerate(subgroups):
        grp = points[points['subgroup'] == subgroup]
        color = COLOUR_CYCLE[idx % len(COLOUR_CYCLE)]
        summary = binned_summary_curve(grp, 'normalised_area', y_col, edges)
        if not summary.empty:
            summary['series'] = subgroup
            summary['plot_type'] = plot_type
            summary_rows.append(summary)
            add_summary_line_and_error_bars(ax=ax, summary=summary, color=color, label=subgroup, fill_alpha=0.12, line_width=2.0)
    overall_summary = binned_summary_curve(points, 'normalised_area', y_col, edges)
    if not overall_summary.empty:
        overall_summary['series'] = 'overall'
        overall_summary['plot_type'] = plot_type
        summary_rows.append(overall_summary)
        add_summary_line_and_error_bars(ax=ax, summary=overall_summary, color='black', label='overall', fill_alpha=0.08, line_width=2.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('normalised area')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)
    if y_ref is not None:
        ax.axhline(float(y_ref), color='#d47a00', linestyle=':', linewidth=1.4, alpha=0.9)
    ax.text(0.99, 0.03, f"{points['run_name'].nunique()} runs | {len(points)} frames | bars: between-run SEM", ha='right', va='bottom', transform=ax.transAxes, fontsize=9, color='#444444', bbox={'facecolor': 'white', 'alpha': 0.82, 'edgecolor': 'none', 'pad': 2.5})
    ax.legend(frameon=False, ncol=2, loc='upper left')
    save_figure(fig, out_path)
    return pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

def plot_velocity_area(points: pd.DataFrame, title: str, out_path: Path) -> pd.DataFrame:
    """Handle the plot velocity area step used by this script."""
    return plot_area_metric(points=points, y_col='instantaneous_velocity_mm_s', ylabel='instantaneous velocity (mm/s)', title=title, out_path=out_path, plot_type='instantaneous_velocity_vs_normalised_area')

def plot_normalised_velocity_area(points: pd.DataFrame, title: str, out_path: Path) -> pd.DataFrame:
    """Handle the plot normalised velocity area step used by this script."""
    return plot_area_metric(points=points, y_col='normalised_velocity', ylabel='normalised velocity (-)', title=title, out_path=out_path, plot_type='normalised_velocity_vs_normalised_area', y_ref=1.0)

def plot_time_metric(points: pd.DataFrame, y_col: str, ylabel: str, title: str, out_path: Path, plot_type: str, y_ref: float | None=None) -> pd.DataFrame:
    """Handle the plot time metric step used by this script."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    subgroups = list(dict.fromkeys(points['subgroup'].tolist()))
    max_time = float(points['time_s'].max())
    edges = np.linspace(0.0, max(max_time, 1e-09), TIME_BINS + 1)
    summary_rows: list[pd.DataFrame] = []
    ax.scatter(points['time_s'].to_numpy(dtype=float), points[y_col].to_numpy(dtype=float), s=6, alpha=0.04, color='#7f8c8d', linewidths=0, rasterized=True, label='all tracked frames')
    for idx, subgroup in enumerate(subgroups):
        grp = points[points['subgroup'] == subgroup]
        color = COLOUR_CYCLE[idx % len(COLOUR_CYCLE)]
        summary = binned_summary_curve(grp, 'time_s', y_col, edges)
        if not summary.empty:
            summary['series'] = subgroup
            summary['plot_type'] = plot_type
            summary_rows.append(summary)
            add_summary_line_and_error_bars(ax=ax, summary=summary, color=color, label=subgroup, fill_alpha=0.12, line_width=2.1)
    overall_summary = binned_summary_curve(points, 'time_s', y_col, edges)
    if not overall_summary.empty:
        overall_summary['series'] = 'overall'
        overall_summary['plot_type'] = plot_type
        summary_rows.append(overall_summary)
        add_summary_line_and_error_bars(ax=ax, summary=overall_summary, color='black', label='overall', fill_alpha=0.08, line_width=2.5)
    ax.axhline(0.0, color='#666666', linestyle='--', linewidth=1.0)
    if y_ref is not None:
        ax.axhline(float(y_ref), color='#d47a00', linestyle=':', linewidth=1.4, alpha=0.9)
    ax.set_xlabel('time from first detection (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)
    ax.text(0.01, 0.97, f"{points['run_name'].nunique()} runs | {len(points)} frames | bars: between-run SEM", ha='left', va='top', transform=ax.transAxes, fontsize=9, color='#444444', bbox={'facecolor': 'white', 'alpha': 0.82, 'edgecolor': 'none', 'pad': 2.5})
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save_figure(fig, out_path)
    return pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

def plot_streamwise_time(points: pd.DataFrame, title: str, out_path: Path) -> pd.DataFrame:
    """Handle the plot streamwise time step used by this script."""
    return plot_time_metric(points=points, y_col='streamwise_velocity_mm_s', ylabel='streamwise velocity (mm/s)', title=title, out_path=out_path, plot_type='streamwise_velocity_vs_time')

def plot_normalised_streamwise_time(points: pd.DataFrame, title: str, out_path: Path) -> pd.DataFrame:
    """Handle the plot normalised streamwise time step used by this script."""
    return plot_time_metric(points=points, y_col='normalised_streamwise_velocity', ylabel='normalised streamwise velocity (-)', title=title, out_path=out_path, plot_type='normalised_streamwise_velocity_vs_time', y_ref=1.0)

def main() -> int:
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_points: list[pd.DataFrame] = []
    run_summaries: list[dict] = []
    for source in SOURCES:
        for run_dir in find_run_dirs(source.tracker_root):
            try:
                points, meta = prepare_run_points(run_dir, source)
                all_points.append(points)
                run_summaries.append({'status': 'ok', 'environment_group': source.environment_group, 'source_batch': source.batch_label, 'source_run_dir': str(run_dir), **meta})
            except Exception as exc:
                run_summaries.append({'status': 'error', 'environment_group': source.environment_group, 'source_batch': source.batch_label, 'source_run_dir': str(run_dir), 'run_name': run_dir.name, 'error': str(exc)})
    if not all_points:
        raise RuntimeError('No usable run points were collected.')
    pooled = pd.concat(all_points, ignore_index=True)
    pooled_csv = OUTPUT_ROOT / 'all_points_pooled.csv'
    pooled.to_csv(pooled_csv, index=False)
    summary_df = pd.DataFrame(run_summaries)
    summary_csv = OUTPUT_ROOT / 'run_collection_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    plot_summaries: list[dict] = []
    for (environment_group, particle_type), grp in pooled.groupby(['environment_group', 'particle_type'], dropna=False):
        out_dir = OUTPUT_ROOT / environment_group / safe_slug(str(particle_type))
        out_dir.mkdir(parents=True, exist_ok=True)
        pooled_points_csv = out_dir / 'pooled_points.csv'
        grp.to_csv(pooled_points_csv, index=False)
        vel_area_png = out_dir / 'instantaneous_velocity_vs_normalised_area_by_particle_type.png'
        streamwise_png = out_dir / 'streamwise_velocity_vs_time_overall.png'
        norm_vel_area_png = out_dir / 'normalised_velocity_vs_normalised_area_by_particle_type.png'
        norm_streamwise_png = out_dir / 'normalised_streamwise_velocity_vs_time_overall.png'
        vel_area_summary_csv = out_dir / 'instantaneous_velocity_vs_normalised_area_binned_summary.csv'
        streamwise_summary_csv = out_dir / 'streamwise_velocity_vs_time_binned_summary.csv'
        norm_vel_area_summary_csv = out_dir / 'normalised_velocity_vs_normalised_area_binned_summary.csv'
        norm_streamwise_summary_csv = out_dir / 'normalised_streamwise_velocity_vs_time_binned_summary.csv'
        title_prefix = f"{environment_group.replace('_', ' ')} | {particle_type}"
        vel_area_summary = plot_velocity_area(grp, f'{title_prefix}: instantaneous velocity vs normalised area', vel_area_png)
        streamwise_summary = plot_streamwise_time(grp, f'{title_prefix}: streamwise velocity vs time', streamwise_png)
        norm_vel_area_summary = plot_normalised_velocity_area(grp, f'{title_prefix}: normalised velocity vs normalised area', norm_vel_area_png)
        norm_streamwise_summary = plot_normalised_streamwise_time(grp, f'{title_prefix}: normalised streamwise velocity vs time', norm_streamwise_png)
        vel_area_summary.to_csv(vel_area_summary_csv, index=False)
        streamwise_summary.to_csv(streamwise_summary_csv, index=False)
        norm_vel_area_summary.to_csv(norm_vel_area_summary_csv, index=False)
        norm_streamwise_summary.to_csv(norm_streamwise_summary_csv, index=False)
        plot_summaries.append({'environment_group': environment_group, 'particle_type': particle_type, 'n_runs': int(grp['run_name'].nunique()), 'n_points': int(len(grp)), 'subgroups': sorted(set(grp['subgroup'].tolist())), 'output_dir': str(out_dir), 'pooled_points_csv': str(pooled_points_csv), 'velocity_area_plot': str(vel_area_png), 'streamwise_time_plot': str(streamwise_png), 'normalised_velocity_area_plot': str(norm_vel_area_png), 'normalised_streamwise_time_plot': str(norm_streamwise_png), 'velocity_area_binned_summary_csv': str(vel_area_summary_csv), 'streamwise_time_binned_summary_csv': str(streamwise_summary_csv), 'normalised_velocity_area_binned_summary_csv': str(norm_vel_area_summary_csv), 'normalised_streamwise_time_binned_summary_csv': str(norm_streamwise_summary_csv)})
    plot_summary_df = pd.DataFrame(plot_summaries)
    plot_summary_csv = OUTPUT_ROOT / 'plot_summary.csv'
    plot_summary_df.to_csv(plot_summary_csv, index=False)
    manifest = {'output_root': str(OUTPUT_ROOT), 'pooled_points_csv': str(pooled_csv), 'run_collection_summary_csv': str(summary_csv), 'plot_summary_csv': str(plot_summary_csv), 'successful_runs': int((summary_df['status'] == 'ok').sum()) if not summary_df.empty else 0, 'failed_runs': int((summary_df['status'] == 'error').sum()) if not summary_df.empty else 0, 'error_bar_basis': 'between-run SEM of run-level bin medians', 'error_bar_min_runs': int(ERROR_BAR_MIN_RUNS), 'plot_groups': plot_summaries}
    (OUTPUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    raise SystemExit(main())
