"""
Plot builder: build_rod_velocity_orientation_analysis.py

Purpose
-------
Builds pooled rod-only plots linking velocity with orientation angle and projected span normal to flow.

Inputs
------
Rod tracker output folders with orientation and trajectory data.

Outputs
-------
Rod orientation/velocity figures and summary CSV files.

Methodological notes
--------------------
Rod angle is folded relative to the flow direction to compare geometrically equivalent orientations.

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
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE_ROOT = Path('C:\\IP Work\\Water Tunnel Flow Overview')
TRACKER_ROOT = SOURCE_ROOT / 'tracker_outputs'
OUTPUT_ROOT = Path('C:\\IP Work\\Water Tunnel Rod Orientation Analysis')
RUN_OUTPUT_ROOT = OUTPUT_ROOT / 'run_outputs'
SUMMARY_DIR = OUTPUT_ROOT / 'summary'
METRICS_CSV = 'rod_velocity_orientation_metrics.csv'
PLOT_ANGLE = 'velocity_vs_angle_to_flow.png'
PLOT_SPAN = 'velocity_vs_projected_span_normal_to_flow.png'
SUMMARY_CSV = 'rod_velocity_orientation_summary.csv'

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

def rod_run_dirs() -> list[Path]:
    """Handle the rod run dirs step used by this script."""
    runs: list[Path] = []
    for group_dir in sorted(TRACKER_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        if 'rod' not in group_dir.name.lower():
            continue
        for run_dir in sorted(group_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / 'tracks_processed.csv').exists():
                runs.append(run_dir)
    return runs

def fold_angle_to_flow_deg(angle_deg: float) -> float:
    """Handle the fold angle to flow deg step used by this script."""
    theta = angle_deg % 180.0
    if theta > 90.0:
        theta = 180.0 - theta
    return theta

def projected_span_normal_to_flow_mm(length_mm: float, width_mm: float, angle_to_flow_deg: float) -> float:
    """Handle the projected span normal to flow mm step used by this script."""
    theta = math.radians(angle_to_flow_deg)
    return abs(length_mm * math.sin(theta)) + abs(width_mm * math.cos(theta))

def collect_metrics(processed_csv: Path) -> list[dict[str, float]]:
    """Handle the collect metrics step used by this script."""
    metrics: list[dict[str, float]] = []
    with processed_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            found = parse_int(row.get('found_this_frame'))
            if found != 1:
                continue
            speed_mm_s = parse_float(row.get('speed_mm_s'))
            length_mm = parse_float(row.get('length_mm'))
            width_mm = parse_float(row.get('width_mm'))
            angle_smooth_deg = parse_float(row.get('angle_smooth_deg'))
            angle_raw_deg = parse_float(row.get('angle_deg'))
            if speed_mm_s is None or length_mm is None or width_mm is None:
                continue
            angle_source_deg = angle_smooth_deg if angle_smooth_deg is not None else angle_raw_deg
            if angle_source_deg is None:
                continue
            angle_to_flow_deg = fold_angle_to_flow_deg(angle_source_deg)
            span_normal_mm = projected_span_normal_to_flow_mm(length_mm, width_mm, angle_to_flow_deg)
            metrics.append({'frame_idx': float(parse_int(row.get('frame_idx')) or 0), 'time_s': parse_float(row.get('time_s')) or 0.0, 'speed_mm_s': speed_mm_s, 'length_mm': length_mm, 'width_mm': width_mm, 'angle_deg_raw': angle_raw_deg if angle_raw_deg is not None else float('nan'), 'angle_deg_smooth': angle_smooth_deg if angle_smooth_deg is not None else float('nan'), 'angle_to_flow_deg': angle_to_flow_deg, 'projected_span_normal_to_flow_mm': span_normal_mm})
    return metrics

def pearson(xs: list[float], ys: list[float]) -> float:
    """Handle the pearson step used by this script."""
    if len(xs) < 3:
        return float('nan')
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum(((x - mx) * (y - my) for x, y in zip(xs, ys)))
    denx = sum(((x - mx) ** 2 for x in xs))
    deny = sum(((y - my) ** 2 for y in ys))
    if denx <= 0.0 or deny <= 0.0:
        return float('nan')
    return num / math.sqrt(denx * deny)

def write_metrics_csv(csv_path: Path, metrics: list[dict[str, float]]) -> None:
    """Write the metrics csv used by this script."""
    fieldnames = ['frame_idx', 'time_s', 'speed_mm_s', 'length_mm', 'width_mm', 'angle_deg_raw', 'angle_deg_smooth', 'angle_to_flow_deg', 'projected_span_normal_to_flow_mm']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({'frame_idx': int(metric['frame_idx']), 'time_s': metric['time_s'], 'speed_mm_s': metric['speed_mm_s'], 'length_mm': metric['length_mm'], 'width_mm': metric['width_mm'], 'angle_deg_raw': metric['angle_deg_raw'], 'angle_deg_smooth': metric['angle_deg_smooth'], 'angle_to_flow_deg': metric['angle_to_flow_deg'], 'projected_span_normal_to_flow_mm': metric['projected_span_normal_to_flow_mm']})

def save_scatter(plot_path: Path, xs: list[float], ys: list[float], xlabel: str, title: str, xlim: tuple[float, float] | None=None) -> None:
    """Save the scatter used by this script."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    ax.scatter(xs, ys, s=16, alpha=0.72, color='#1f77b4', edgecolors='none')
    if xlim is not None:
        ax.set_xlim(*xlim)
    ymin = min(ys)
    ymax = max(ys)
    if math.isclose(ymin, ymax):
        pad = max(1.0, abs(ymax) * 0.05)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Instantaneous velocity (mm/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.text(0.98, 0.02, f'n = {len(xs)}\nr = {pearson(xs, ys):.3f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.8'})
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

def save_group_scatter(summary_subdir: Path, name: str, xs: list[float], ys: list[float], xlabel: str, xlim=None) -> None:
    """Save the group scatter used by this script."""
    if not xs or not ys:
        return
    save_scatter(summary_subdir / name, xs, ys, xlabel=xlabel, title=name.replace('_', ' ').replace('.png', ''), xlim=xlim)

def main() -> int:
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    grouped_metrics: dict[str, list[dict[str, float]]] = {}
    all_metrics: list[dict[str, float]] = []
    for run_dir in rod_run_dirs():
        group_name = run_dir.parent.name
        run_name = run_dir.name
        processed_csv = run_dir / 'tracks_processed.csv'
        metrics = collect_metrics(processed_csv)
        if not metrics:
            continue
        out_dir = RUN_OUTPUT_ROOT / group_name / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = out_dir / METRICS_CSV
        write_metrics_csv(metrics_csv, metrics)
        xs_angle = [m['angle_to_flow_deg'] for m in metrics]
        xs_span = [m['projected_span_normal_to_flow_mm'] for m in metrics]
        ys = [m['speed_mm_s'] for m in metrics]
        save_scatter(out_dir / PLOT_ANGLE, xs_angle, ys, xlabel='Angle to flow (deg, folded to 0-90)', title=f"{run_name.replace('_', ' ')}: velocity vs angle to flow", xlim=(0.0, 90.0))
        save_scatter(out_dir / PLOT_SPAN, xs_span, ys, xlabel='Projected span normal to flow (mm)', title=f"{run_name.replace('_', ' ')}: velocity vs projected span")
        grouped_metrics.setdefault(group_name, []).extend(metrics)
        all_metrics.extend(metrics)
        summary_rows.append({'group': group_name, 'run_name': run_name, 'source_run_dir': str(run_dir), 'processed_csv': str(processed_csv), 'metrics_csv': str(metrics_csv), 'plot_angle': str(out_dir / PLOT_ANGLE), 'plot_projected_span': str(out_dir / PLOT_SPAN), 'point_count': len(metrics), 'angle_speed_pearson_r': pearson(xs_angle, ys), 'projected_span_speed_pearson_r': pearson(xs_span, ys), 'angle_to_flow_min_deg': min(xs_angle), 'angle_to_flow_max_deg': max(xs_angle), 'projected_span_min_mm': min(xs_span), 'projected_span_max_mm': max(xs_span)})
    for group_name, metrics in grouped_metrics.items():
        subdir = SUMMARY_DIR / group_name
        subdir.mkdir(parents=True, exist_ok=True)
        xs_angle = [m['angle_to_flow_deg'] for m in metrics]
        xs_span = [m['projected_span_normal_to_flow_mm'] for m in metrics]
        ys = [m['speed_mm_s'] for m in metrics]
        save_group_scatter(subdir, 'pooled_velocity_vs_angle_to_flow.png', xs_angle, ys, 'Angle to flow (deg, folded to 0-90)', xlim=(0.0, 90.0))
        save_group_scatter(subdir, 'pooled_velocity_vs_projected_span_normal_to_flow.png', xs_span, ys, 'Projected span normal to flow (mm)')
    if all_metrics:
        xs_angle = [m['angle_to_flow_deg'] for m in all_metrics]
        xs_span = [m['projected_span_normal_to_flow_mm'] for m in all_metrics]
        ys = [m['speed_mm_s'] for m in all_metrics]
        save_group_scatter(SUMMARY_DIR, 'all_rods_velocity_vs_angle_to_flow.png', xs_angle, ys, 'Angle to flow (deg, folded to 0-90)', xlim=(0.0, 90.0))
        save_group_scatter(SUMMARY_DIR, 'all_rods_velocity_vs_projected_span_normal_to_flow.png', xs_span, ys, 'Projected span normal to flow (mm)')
    with (SUMMARY_DIR / SUMMARY_CSV).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'run_name', 'source_run_dir', 'processed_csv', 'metrics_csv', 'plot_angle', 'plot_projected_span', 'point_count', 'angle_speed_pearson_r', 'projected_span_speed_pearson_r', 'angle_to_flow_min_deg', 'angle_to_flow_max_deg', 'projected_span_min_mm', 'projected_span_max_mm'])
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {'source_root': str(SOURCE_ROOT), 'tracker_root': str(TRACKER_ROOT), 'output_root': str(OUTPUT_ROOT), 'run_output_root': str(RUN_OUTPUT_ROOT), 'summary_dir': str(SUMMARY_DIR), 'run_count': len(summary_rows), 'plots': [PLOT_ANGLE, PLOT_SPAN], 'metrics_csv': METRICS_CSV, 'angle_definition': 'Folded rod angle relative to horizontal flow axis, mapped to 0-90 degrees.', 'projected_span_definition': 'Projected rod span normal to flow = |L*sin(theta)| + |W*cos(theta)| using length_mm, width_mm, and folded angle to flow.'}
    (SUMMARY_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    raise SystemExit(main())
