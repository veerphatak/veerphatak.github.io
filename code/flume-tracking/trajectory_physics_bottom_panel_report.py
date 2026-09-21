"""
Physics analysis script: trajectory_physics_bottom_panel_report.py

Purpose
-------
Computes the bottom-view flume sphere physics summary.

Inputs
------
Processed bottom-view tracks and flow/particle-property settings.

Outputs
-------
Bottom-view summary tables and figures.

Methodological notes
--------------------
The analysis emphasises plan-view streamwise/lateral motion and aggregates repeated trials using mean and standard deviation.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUTPUT_ROOT = Path('C:\\IP Work\\Bottom_Panel_Physics_Reports')
BATCHES = []
FPS = 240.0
SMOOTHING_WINDOW = 9
MIN_POINTS = 6
SAVE_PER_TRIAL_KINEMATICS = True
BULK_FLOW_SPEED_M_S = 0.25
BULK_FLOW_SPEED_MM_S = BULK_FLOW_SPEED_M_S * 1000.0
RHO_F = 1000.0
RHO_PP = 905.0

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Handle the find col step used by this script."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def safe_scalar(value) -> float:
    """Handle the safe scalar step used by this script."""
    if value is None:
        return np.nan
    if isinstance(value, (np.floating, float, np.integer, int)):
        return float(value) if np.isfinite(value) else np.nan
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan

def infer_particle_info(folder_name: str) -> dict:
    """Handle the infer particle info step used by this script."""
    name = folder_name.lower()
    if '10mm' in name and 'sphere' in name:
        diameter_m = 0.01
        return {'particle_type': '10 mm sphere', 'shape': 'sphere', 'material': 'polypropylene', 'density_kg_m3': RHO_PP, 'diameter_m': diameter_m}
    if '14mm' in name and 'sphere' in name:
        diameter_m = 0.014
        return {'particle_type': '14 mm sphere', 'shape': 'sphere', 'material': 'polypropylene', 'density_kg_m3': RHO_PP, 'diameter_m': diameter_m}
    return {'particle_type': 'sphere', 'shape': 'sphere', 'material': 'polypropylene', 'density_kg_m3': RHO_PP, 'diameter_m': 0.01}

def resolve_trial_name(trial_dir: Path) -> str:
    """Handle the resolve trial name step used by this script."""
    if trial_dir.name != 'tracker_output':
        return trial_dir.name
    case_dir = trial_dir.parent
    case_summary = case_dir / 'case_summary.json'
    if case_summary.exists():
        try:
            data = pd.read_json(case_summary, typ='series')
            video_name = data.get('video_name')
            if isinstance(video_name, str) and video_name.strip():
                return Path(video_name).stem
        except Exception:
            pass
    return case_dir.name

def smooth_series(arr: np.ndarray, window: int=SMOOTHING_WINDOW) -> np.ndarray:
    """Handle the smooth series step used by this script."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 5:
        return arr.copy()
    window = min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if window < 3:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode='edge')
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode='valid')

def split_found_segments(df: pd.DataFrame, found_col: str) -> list[pd.DataFrame]:
    """Handle the split found segments step used by this script."""
    segments: list[pd.DataFrame] = []
    start_idx = None
    last_frame = None
    frames = pd.to_numeric(df['frame'], errors='coerce').to_numpy(dtype=float)
    found = pd.to_numeric(df[found_col], errors='coerce').fillna(0).astype(int).to_numpy()
    for i, is_found in enumerate(found):
        frame = int(frames[i])
        if is_found != 1:
            if start_idx is not None:
                segments.append(df.iloc[start_idx:i].copy())
                start_idx = None
            last_frame = None
            continue
        if start_idx is None:
            start_idx = i
        elif last_frame is not None and frame - last_frame > 1:
            segments.append(df.iloc[start_idx:i].copy())
            start_idx = i
        last_frame = frame
    if start_idx is not None:
        segments.append(df.iloc[start_idx:].copy())
    return [seg.reset_index(drop=True) for seg in segments if len(seg) > 0]

def load_tracks(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Load the tracks used by this script."""
    df = pd.read_csv(csv_path)
    frame_col = find_col(df, ['frame_idx', 'out_idx', 'frame', 'frame_index'])
    time_col = find_col(df, ['t_rel_s', 'time_s', 't_s'])
    x_mm_col = find_col(df, ['x_kalman_mm', 'x_mm'])
    y_mm_col = find_col(df, ['y_kalman_mm', 'y_mm'])
    found_col = find_col(df, ['found_this_frame', 'found'])
    if x_mm_col is None or y_mm_col is None:
        raise ValueError(f'No usable mm-position columns found in {csv_path.name}')
    out = pd.DataFrame({'frame': pd.to_numeric(df[frame_col], errors='coerce') if frame_col else np.arange(len(df)), 't_s': pd.to_numeric(df[time_col], errors='coerce') if time_col else np.nan, 'x_mm': pd.to_numeric(df[x_mm_col], errors='coerce'), 'y_mm': pd.to_numeric(df[y_mm_col], errors='coerce')})
    if found_col:
        out['found_this_frame'] = pd.to_numeric(df[found_col], errors='coerce').fillna(0).astype(int)
    out = out.dropna(subset=['frame', 'x_mm', 'y_mm']).copy()
    out['frame'] = out['frame'].astype(int)
    out = out.sort_values('frame').drop_duplicates('frame').reset_index(drop=True)
    if out['t_s'].isna().all():
        out['t_s'] = out['frame'] / FPS
    else:
        missing = out['t_s'].isna()
        out.loc[missing, 't_s'] = out.loc[missing, 'frame'] / FPS
    analysis_source = 'all_rows'
    if 'found_this_frame' in out.columns:
        segments = split_found_segments(out, 'found_this_frame')
        if segments:
            longest = max(segments, key=len)
            if len(longest) >= MIN_POINTS:
                out = longest.copy()
                analysis_source = 'longest_found_segment'
            else:
                found_only = out[out['found_this_frame'] == 1].copy().reset_index(drop=True)
                if len(found_only) >= MIN_POINTS:
                    out = found_only
                    analysis_source = 'all_found_rows'
    if len(out) < MIN_POINTS:
        raise ValueError(f'Too few valid points in {csv_path.name}: {len(out)}')
    return (out.reset_index(drop=True), analysis_source)

def pca_axes(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Handle the pca axes step used by this script."""
    coords = np.column_stack([x_mm, y_mm])
    coords = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    primary = vh[0] / np.linalg.norm(vh[0])
    secondary = vh[1] / np.linalg.norm(vh[1])
    net = np.array([x_mm[-1] - x_mm[0], y_mm[-1] - y_mm[0]], dtype=float)
    if np.dot(primary, net) < 0:
        primary = -primary
        secondary = -secondary
    return (primary, secondary)

def compute_kinematics(tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute the kinematics used by this script."""
    t = tracks['t_s'].to_numpy(dtype=float)
    x_mm = tracks['x_mm'].to_numpy(dtype=float)
    y_mm = tracks['y_mm'].to_numpy(dtype=float)
    x0 = float(x_mm[0])
    y0 = float(y_mm[0])
    x_rel = x_mm - x0
    y_rel = y_mm - y0
    r_rel = np.sqrt(x_rel ** 2 + y_rel ** 2)
    x_s = smooth_series(x_rel)
    y_s = smooth_series(y_rel)
    r_s = np.sqrt(x_s ** 2 + y_s ** 2)
    vx = np.gradient(x_s, t)
    vy = np.gradient(y_s, t)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    accel = np.sqrt(ax ** 2 + ay ** 2)
    primary_axis, secondary_axis = pca_axes(x_s, y_s)
    along_mm = x_s * primary_axis[0] + y_s * primary_axis[1]
    cross_mm = x_s * secondary_axis[0] + y_s * secondary_axis[1]
    v_along = np.gradient(along_mm, t)
    v_cross = np.gradient(cross_mm, t)
    dx = np.diff(x_s)
    dy = np.diff(y_s)
    path_length_mm = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    net_dx_mm = float(x_s[-1] - x_s[0])
    net_dy_mm = float(y_s[-1] - y_s[0])
    net_disp_mm = float(np.sqrt(net_dx_mm ** 2 + net_dy_mm ** 2))
    tortuosity = path_length_mm / net_disp_mm if net_disp_mm > 1e-09 else np.nan
    duration_s = float(t[-1] - t[0])
    mean_speed_mm_s = safe_scalar(np.nanmean(speed))
    max_speed_mm_s = safe_scalar(np.nanmax(speed))
    net_speed_mm_s = float(net_disp_mm / duration_s) if duration_s > 1e-09 else np.nan
    mean_abs_along_axis_speed_mm_s = safe_scalar(np.nanmean(np.abs(v_along)))
    mean_abs_cross_axis_speed_mm_s = safe_scalar(np.nanmean(np.abs(v_cross)))
    summary = {'duration_s': duration_s, 'bulk_flow_speed_m_s': float(BULK_FLOW_SPEED_M_S), 'bulk_flow_speed_mm_s': float(BULK_FLOW_SPEED_MM_S), 'trajectory_length_mm': path_length_mm, 'net_displacement_mm': net_disp_mm, 'net_x_displacement_mm': net_dx_mm, 'net_y_displacement_mm': net_dy_mm, 'x_span_mm': float(np.max(x_s) - np.min(x_s)), 'y_span_mm': float(np.max(y_s) - np.min(y_s)), 'mean_speed_mm_s': mean_speed_mm_s, 'max_speed_mm_s': max_speed_mm_s, 'net_displacement_speed_mm_s': net_speed_mm_s, 'mean_speed_over_bulk_flow': safe_scalar(mean_speed_mm_s / BULK_FLOW_SPEED_MM_S), 'max_speed_over_bulk_flow': safe_scalar(max_speed_mm_s / BULK_FLOW_SPEED_MM_S), 'net_displacement_speed_over_bulk_flow': safe_scalar(net_speed_mm_s / BULK_FLOW_SPEED_MM_S), 'mean_vx_mm_s': safe_scalar(np.nanmean(vx)), 'mean_vy_mm_s': safe_scalar(np.nanmean(vy)), 'mean_abs_vx_mm_s': safe_scalar(np.nanmean(np.abs(vx))), 'mean_abs_vy_mm_s': safe_scalar(np.nanmean(np.abs(vy))), 'mean_abs_along_axis_speed_mm_s': mean_abs_along_axis_speed_mm_s, 'mean_abs_cross_axis_speed_mm_s': mean_abs_cross_axis_speed_mm_s, 'mean_accel_mm_s2': safe_scalar(np.nanmean(accel)), 'max_accel_mm_s2': safe_scalar(np.nanmax(accel)), 'max_radial_distance_mm': float(np.nanmax(r_s)), 'mean_radial_distance_mm': safe_scalar(np.nanmean(r_s)), 'radial_rms_mm': float(np.sqrt(np.nanmean(r_s ** 2))), 'principal_axis_span_mm': float(np.max(along_mm) - np.min(along_mm)), 'cross_axis_span_mm': float(np.max(cross_mm) - np.min(cross_mm)), 'cross_axis_std_mm': safe_scalar(np.nanstd(cross_mm, ddof=1) if len(cross_mm) > 1 else np.nan), 'cross_axis_rms_mm': float(np.sqrt(np.nanmean(cross_mm ** 2))), 'anisotropy_ratio': float((np.max(along_mm) - np.min(along_mm)) / max(np.max(cross_mm) - np.min(cross_mm), 1e-09)), 'tortuosity': tortuosity, 'primary_axis_x': float(primary_axis[0]), 'primary_axis_y': float(primary_axis[1]), 'secondary_axis_x': float(secondary_axis[0]), 'secondary_axis_y': float(secondary_axis[1])}
    kin = pd.DataFrame({'frame': tracks['frame'].to_numpy(dtype=int), 't_s': t, 'x_rel_mm': x_s, 'y_rel_mm': y_s, 'radial_distance_mm': r_s, 'along_axis_mm': along_mm, 'cross_axis_mm': cross_mm, 'vx_mm_s': vx, 'vy_mm_s': vy, 'v_along_mm_s': v_along, 'v_cross_mm_s': v_cross, 'speed_mm_s': speed, 'ax_mm_s2': ax, 'ay_mm_s2': ay, 'accel_mm_s2': accel})
    return (kin, summary)

def analyse_trial(trial_dir: Path, batch_label: str) -> tuple[dict, pd.DataFrame]:
    """Handle the analyse trial step used by this script."""
    trial_name = resolve_trial_name(trial_dir)
    info = infer_particle_info(trial_name)
    tracks, analysis_source = load_tracks(trial_dir / 'tracks_processed.csv')
    kin, summary = compute_kinematics(tracks)
    result = {'trial': trial_name, 'batch_label': batch_label, 'analysis_source': analysis_source, 'particle_type': info['particle_type'], 'shape': info['shape'], 'material': info['material'], 'density_kg_m3': info['density_kg_m3'], 'diameter_m': info['diameter_m'], 'n_points': int(len(kin)), **summary}
    kin['trial'] = trial_name
    kin['batch_label'] = batch_label
    return (result, kin)

def aggregate_with_std(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    """Handle the aggregate with std step used by this script."""
    grouped = df.groupby(group_col, dropna=False)
    out = grouped.agg(n_trials=('trial', 'count'), **{f'{metric}_mean': (metric, 'mean') for metric in metrics}, **{f'{metric}_std': (metric, lambda s: np.nanstd(pd.to_numeric(s, errors='coerce'), ddof=1) if len(pd.to_numeric(s, errors='coerce').dropna()) > 1 else np.nan) for metric in metrics}).reset_index()
    return out

def plot_by_trial(df: pd.DataFrame, col: str, ylabel: str, path: Path, title_suffix: str | None=None) -> None:
    """Handle the plot by trial step used by this script."""
    sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col]).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(sub))
    ax.bar(x, sub[col].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(sub['trial'], rotation=70, ha='right')
    ax.set_ylabel(ylabel)
    title = f'{ylabel} by trial'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_centered_trajectories(kinematics: list[pd.DataFrame], path: Path, title_suffix: str | None=None) -> None:
    """Handle the plot centered trajectories step used by this script."""
    if not kinematics:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for kin in kinematics:
        ax.plot(kin['x_rel_mm'], kin['y_rel_mm'], linewidth=1.8, label=str(kin['trial'].iloc[0]))
        ax.scatter([kin['x_rel_mm'].iloc[0]], [kin['y_rel_mm'].iloc[0]], s=24, color='green')
        ax.scatter([kin['x_rel_mm'].iloc[-1]], [kin['y_rel_mm'].iloc[-1]], s=24, color='red')
    ax.set_xlabel('x displacement (mm)')
    ax.set_ylabel('y displacement (mm)')
    title = 'Centered bottom-view trajectories'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def write_interpretation_notes(out_dir: Path) -> None:
    """Write the interpretation notes used by this script."""
    text = '# Bottom-View Interpretation Notes\n\nThis report is for underside-view videos, not side-view videos.\n\n- `x_rel_mm` and `y_rel_mm` are in-plane displacements in the viewing panel image coordinates.\n- `principal_axis_span_mm` is the span of the trajectory along its dominant PCA direction.\n- `cross_axis_span_mm` and `cross_axis_std_mm` quantify lateral dispersion perpendicular to that dominant path.\n- `radial_rms_mm` describes dispersion away from the release point in the bottom-view plane.\n- The reference bulk flow speed used here is `0.25 m/s` (`250 mm/s`).\n- `*_over_bulk_flow` columns compare in-plane particle speed metrics with that bulk-flow reference; they are not full 3D slip-velocity estimates.\n- No side-view-only drag, settling, or vertical-transport labels are used here.\n'
    (out_dir / 'interpretation_notes.md').write_text(text, encoding='utf-8')

def collect_trial_dirs(source_dirs: list[Path]) -> list[Path]:
    """Handle the collect trial dirs step used by this script."""
    trial_dirs: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        if source_dir.is_dir() and (source_dir / 'tracks_processed.csv').exists():
            trial_dirs.append(source_dir)
            continue
        for child in sorted(source_dir.iterdir()):
            if child.is_dir() and (child / 'tracks_processed.csv').exists():
                trial_dirs.append(child)
    return trial_dirs

def process_batch(batch_def: dict, out_dir: Path) -> None:
    """Handle the process batch step used by this script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir / 'processed_kinematics'
    if SAVE_PER_TRIAL_KINEMATICS:
        processed_dir.mkdir(exist_ok=True)
    metrics = ['trajectory_length_mm', 'net_displacement_mm', 'x_span_mm', 'y_span_mm', 'principal_axis_span_mm', 'cross_axis_span_mm', 'cross_axis_std_mm', 'cross_axis_rms_mm', 'radial_rms_mm', 'mean_speed_mm_s', 'mean_speed_over_bulk_flow', 'net_displacement_speed_mm_s', 'net_displacement_speed_over_bulk_flow', 'mean_abs_cross_axis_speed_mm_s', 'max_speed_mm_s', 'tortuosity']
    rows = []
    logs = []
    all_kinematics: list[pd.DataFrame] = []
    for trial_dir in collect_trial_dirs(batch_def['source_dirs']):
        trial_name = resolve_trial_name(trial_dir)
        try:
            summary, kin = analyse_trial(trial_dir, batch_def['label'])
            rows.append(summary)
            all_kinematics.append(kin)
            logs.append({'trial': trial_name, 'status': 'ok', 'message': ''})
            if SAVE_PER_TRIAL_KINEMATICS:
                kin.to_csv(processed_dir / f'{trial_name}_kinematics.csv', index=False)
        except Exception as exc:
            logs.append({'trial': trial_name, 'status': 'error', 'message': str(exc)})
    trials = pd.DataFrame(rows)
    logs_df = pd.DataFrame(logs)
    trials.to_csv(out_dir / 'full_bottomview_results_by_trial.csv', index=False)
    logs_df.to_csv(out_dir / 'processing_log.csv', index=False)
    if not trials.empty:
        by_particle = aggregate_with_std(trials, 'particle_type', metrics)
        by_particle.to_csv(out_dir / 'full_bottomview_results_by_particle.csv', index=False)
        plot_by_trial(trials, 'cross_axis_span_mm', 'Cross-axis span (mm)', out_dir / 'cross_axis_span_by_trial.png', batch_def['label'])
        plot_by_trial(trials, 'cross_axis_rms_mm', 'Cross-axis RMS (mm)', out_dir / 'cross_axis_rms_by_trial.png', batch_def['label'])
        plot_by_trial(trials, 'radial_rms_mm', 'Radial RMS (mm)', out_dir / 'radial_rms_by_trial.png', batch_def['label'])
        plot_by_trial(trials, 'trajectory_length_mm', 'Trajectory length (mm)', out_dir / 'trajectory_length_by_trial.png', batch_def['label'])
        plot_by_trial(trials, 'mean_speed_mm_s', 'Mean speed (mm/s)', out_dir / 'mean_speed_by_trial.png', batch_def['label'])
        plot_by_trial(trials, 'mean_speed_over_bulk_flow', 'Mean speed / bulk flow', out_dir / 'mean_speed_over_bulk_flow_by_trial.png', batch_def['label'])
        plot_centered_trajectories(all_kinematics, out_dir / 'centered_trajectories.png', batch_def['label'])
    else:
        pd.DataFrame().to_csv(out_dir / 'full_bottomview_results_by_particle.csv', index=False)
    write_interpretation_notes(out_dir)

def main() -> None:
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for batch_def in BATCHES:
        out_dir = OUTPUT_ROOT / batch_def['label']
        process_batch(batch_def, out_dir)
    script_path = Path(__file__)
    (OUTPUT_ROOT / script_path.name).write_text(script_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Saved bottom-view physics reports to: {OUTPUT_ROOT}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
