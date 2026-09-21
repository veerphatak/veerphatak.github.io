"""
Batch runner: run_tuned_sphere_batch.py

Purpose
-------
Runs the tuned water-tunnel sphere tracker across the final sphere cases.

Inputs
------
Case definitions containing video path, colour mode, seed frame and initial click.

Outputs
-------
Standardised tracker output folders for each sphere run.

Methodological notes
--------------------
Batch settings preserve the manually tuned parameters used for the reported final analysis.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
WORKSPACE = Path('C:\\IP Work\\New project')
SCRIPT_PATH = WORKSPACE / 'Kalman_Tracker_Spheres_binarized.py'
DOCS_BATCH_DIR = Path('C:\\IP Work\\4.5_Inchps_Spheres_Binarised')
CALIBRATION_JSON = Path('C:\\Users\\User\\Downloads\\rod_calibration_snap_fit\\calibration.json')
CASES = [{'label': '10mm_2', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\10mm sphere release 2.mp4'), 'out_dir': DOCS_BATCH_DIR / '10mm_sphere_release_2_binarized_tracker_output', 'sphere_color_mode': 'red', 'sphere_diameter_mm': 10.0, 'start_frame': 0, 'preset_click': (742, 754)}, {'label': '10mm_3', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\10mm sphere release 3.mp4'), 'out_dir': DOCS_BATCH_DIR / '10mm_sphere_release_3_binarized_tracker_output', 'sphere_color_mode': 'red', 'sphere_diameter_mm': 10.0, 'start_frame': 135, 'preset_click': (1055, 281)}, {'label': '10mm_4', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\10mm sphere release 4.mp4'), 'out_dir': DOCS_BATCH_DIR / '10mm_sphere_release_4_binarized_tracker_output', 'sphere_color_mode': 'red', 'sphere_diameter_mm': 10.0, 'start_frame': 315, 'preset_click': (1074, 39)}, {'label': '10mm_5', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\10mm sphere release 5.mp4'), 'out_dir': DOCS_BATCH_DIR / '10mm_sphere_release_5_binarized_tracker_output', 'sphere_color_mode': 'red', 'sphere_diameter_mm': 10.0, 'start_frame': 101, 'preset_click': (666, 971)}, {'label': '14mm_1', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\14mm sphere release 1.mp4'), 'out_dir': DOCS_BATCH_DIR / '14mm_sphere_release_1_binarized_tracker_output', 'sphere_color_mode': 'white', 'sphere_diameter_mm': 14.0, 'start_frame': 120, 'preset_click': (739, 725)}, {'label': '14mm_2', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\14mm sphere release 2.mp4'), 'out_dir': DOCS_BATCH_DIR / '14mm_sphere_release_2_binarized_tracker_output', 'sphere_color_mode': 'white', 'sphere_diameter_mm': 14.0, 'start_frame': 75, 'preset_click': (822, 837)}, {'label': '14mm_3', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\14mm sphere release 3.mp4'), 'out_dir': DOCS_BATCH_DIR / '14mm_sphere_release_3_binarized_tracker_output', 'sphere_color_mode': 'white', 'sphere_diameter_mm': 14.0, 'start_frame': 230, 'preset_click': (994, 988)}, {'label': '14mm_4', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\14mm sphere release 4.mp4'), 'out_dir': DOCS_BATCH_DIR / '14mm_sphere_release_4_binarized_tracker_output', 'sphere_color_mode': 'white', 'sphere_diameter_mm': 14.0, 'start_frame': 340, 'preset_click': (1033, 815)}, {'label': '14mm_5', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\14mm sphere release 5.mp4'), 'out_dir': DOCS_BATCH_DIR / '14mm_sphere_release_5_binarized_tracker_output', 'sphere_color_mode': 'white', 'sphere_diameter_mm': 14.0, 'start_frame': 85, 'preset_click': (1022, 777)}]

def load_tracker_module():
    """Load the tracker module used by this script."""
    spec = importlib.util.spec_from_file_location('sphere_tracker_batch_mod', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def summarize_case(case):
    """Handle the summarize case step used by this script."""
    raw_csv = case['out_dir'] / 'tracks_raw.csv'
    processed_csv = case['out_dir'] / 'tracks_processed.csv'
    df = pd.read_csv(raw_csv)
    found_col = None
    if 'found_this_frame' in df.columns:
        found_col = 'found_this_frame'
    elif 'found' in df.columns:
        found_col = 'found'
    found = int(df[found_col].fillna(0).astype(int).sum()) if found_col else None
    summary = {'label': case['label'], 'video_path': str(case['video_path']), 'out_dir': str(case['out_dir']), 'sphere_color_mode': case['sphere_color_mode'], 'sphere_diameter_mm': float(case['sphere_diameter_mm']), 'start_frame': int(case['start_frame']), 'preset_click': list(case['preset_click']), 'rows': int(len(df)), 'found': found, 'missed': int(len(df) - found) if found is not None else None, 'found_column': found_col, 'raw_csv': str(raw_csv), 'processed_csv': str(processed_csv)}
    return summary

def run_case(module, case):
    """Run the case used by this script."""
    module.USE_RUNTIME_PROMPTS = False
    module.ENABLE_IMSHOW = False
    module.DEFAULT_VIDEO_PATH = str(case['video_path'])
    module.DEFAULT_CALIBRATION_JSON = str(CALIBRATION_JSON)
    module.DEFAULT_OUT_DIR = str(case['out_dir'])
    module.DEFAULT_REAL_FPS = 240.0
    module.DEFAULT_FRAME_STRIDE = 1
    module.DEFAULT_START_FRAME = int(case['start_frame'])
    module.DEFAULT_OBJECT_MODE = 'sphere'
    module.DEFAULT_SPHERE_COLOR_MODE = case['sphere_color_mode']
    module.DEFAULT_SPHERE_DIAMETER_MM = float(case['sphere_diameter_mm'])
    module.DEFAULT_USE_PRESET_CLICK = True
    module.DEFAULT_PRESET_CLICK = tuple(case['preset_click'])
    module.DEFAULT_SAVE_DEBUG_IMAGES = True
    module.DEFAULT_NUM_DEBUG_IMAGES = 12
    module.DEFAULT_START_SEARCH_FRAMES = 25
    case['out_dir'].mkdir(parents=True, exist_ok=True)
    module.main()
    return summarize_case(case)

def main():
    """Run the main entry point for this script."""
    summary_only = '--summary-only' in sys.argv
    DOCS_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, DOCS_BATCH_DIR / SCRIPT_PATH.name)
    module = None if summary_only else load_tracker_module()
    results = {'ignored_cases': [{'label': '10mm_1', 'reason': 'User marked this case as an anomaly and asked to ignore it.', 'video_path': 'C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\10mm sphere release 1.mp4'}], 'cases': []}
    for case in CASES:
        if summary_only:
            print(f"Summarizing {case['label']}...")
            summary = summarize_case(case)
        else:
            print(f"Running {case['label']}...")
            summary = run_case(module, case)
        print(f"Completed {case['label']}: rows={summary['rows']} found={summary['found']} missed={summary['missed']}")
        results['cases'].append(summary)
    summary_path = DOCS_BATCH_DIR / 'batch_summary.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote summary to {summary_path}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
