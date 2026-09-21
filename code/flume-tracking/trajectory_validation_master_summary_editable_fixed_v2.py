"""
Validation summary script: trajectory_validation_master_summary_editable_fixed_v2.py

Purpose
-------
Aggregates validation outputs into a master summary table and plots.

Inputs
------
Validation-output folder tree.

Outputs
-------
Master validation CSVs and summary figures.

Methodological notes
--------------------
Summary statistics are used to document tracking quality and identify runs requiring manual review.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
BATCH_OUTPUT_DIR = 'C:\\Users\\User\\Downloads\\trajectory_validation_manual_outputs'
MASTER_OUT_DIR = None

def infer_particle_type(name):
    """Handle the infer particle type step used by this script."""
    n = name.lower()
    if '10mm' in n and 'sphere' in n:
        return '10 mm sphere'
    if '14mm' in n and 'sphere' in n:
        return '14 mm sphere'
    if 'disc' in n:
        return 'disc'
    if 'rod' in n:
        return 'rod'
    return 'other'

def infer_run_number(name):
    """Handle the infer run number step used by this script."""
    m = re.search('(?:release|drop)_(\\d+)', name.lower())
    return int(m.group(1)) if m else np.nan

def main():
    """Run the main entry point for this script."""
    batch_dir = BATCH_OUTPUT_DIR
    out_dir = MASTER_OUT_DIR or os.path.join(batch_dir, 'master_summary')
    os.makedirs(out_dir, exist_ok=True)
    summary_files = glob.glob(os.path.join(batch_dir, '*', 'trajectory_validation_summary_full_video.csv'))
    rows = []
    for sf in summary_files:
        folder_name = os.path.basename(os.path.dirname(sf))
        try:
            row = pd.read_csv(sf).iloc[0].to_dict()
            row['folder_name'] = folder_name
            row['particle_type'] = infer_particle_type(folder_name)
            row['run_number'] = infer_run_number(folder_name)
            rows.append(row)
        except Exception:
            continue
    if not rows:
        raise RuntimeError('No summary CSV files found in the batch output directory')
    df = pd.DataFrame(rows).sort_values(['particle_type', 'run_number', 'folder_name']).reset_index(drop=True)
    master_csv = os.path.join(out_dir, 'trajectory_validation_master_summary.csv')
    concise_csv = os.path.join(out_dir, 'trajectory_validation_master_summary_concise.csv')
    df.to_csv(master_csv, index=False)
    concise_cols = [c for c in ['folder_name', 'particle_type', 'run_number', 'video_frames', 'visible_frames_used', 'detected_frames', 'detection_coverage_pct', 'mean_pixel_error', 'median_pixel_error', 'max_pixel_error', 'frame_step', 'foreground_mode', 'source_video', 'sample_strategy', 'sample_positions_target', 'sampled_positions_used', 'trajectory_points_available', 'trajectory_points_sampled'] if c in df.columns]
    df[concise_cols].to_csv(concise_csv, index=False)
    plt.figure(figsize=(12, 5))
    plt.bar(df['folder_name'], df['mean_pixel_error'])
    plt.xticks(rotation=75, ha='right')
    plt.ylabel('Mean pixel error')
    plt.title('Mean pixel error by trial')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mean_pixel_error_by_trial.png'), dpi=200)
    plt.close()
    plt.figure(figsize=(12, 5))
    plt.bar(df['folder_name'], df['detection_coverage_pct'])
    plt.xticks(rotation=75, ha='right')
    plt.ylabel('Detection coverage (%)')
    plt.title('Detection coverage by trial')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'detection_coverage_by_trial.png'), dpi=200)
    plt.close()
    agg = df.groupby('particle_type', dropna=False).agg(mean_pixel_error=('mean_pixel_error', 'mean'), median_pixel_error=('median_pixel_error', 'mean'), mean_detection_coverage=('detection_coverage_pct', 'mean'), n_trials=('folder_name', 'count')).reset_index()
    agg.to_csv(os.path.join(out_dir, 'particle_type_aggregates.csv'), index=False)
    plt.figure(figsize=(8, 5))
    plt.bar(agg['particle_type'], agg['mean_pixel_error'])
    plt.ylabel('Mean pixel error')
    plt.title('Mean pixel error by particle type')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mean_pixel_error_by_particle_type.png'), dpi=200)
    plt.close()
    print('Saved master summary outputs to:', out_dir)

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
