"""
Batch runner: run_45_rod_batch.py

Purpose
-------
Runs the 4.5 inch/s rod tracking batch.

Inputs
------
Run table containing 4.5 inch/s rod videos, start frames and seed coordinates.

Outputs
-------
Processed rod output folders for the higher-speed cases.

Methodological notes
--------------------
The same rod tracker is reused so comparisons between flow speeds are not biased by different processing methods.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import importlib.util
import json
import shutil
from pathlib import Path
import pandas as pd
WORKSPACE = Path('C:\\IP Work\\New project')
ROOT_OUT_DIR = Path('C:\\IP Work\\4.5_Inchps_rod_binarised_widthlocked_5mm_fullbatch_v2')
CALIBRATION_JSON = Path('C:\\Users\\User\\Downloads\\rod_calibration_snap_fit\\calibration.json')
ROD_SCRIPT_PATH = WORKSPACE / 'Kalman_Tracker_Rod_binarized.py'
ROD_CASES = [{'label': 'rod_drop_1', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\Calibration + rod drop 1.mp4'), 'out_dir': ROOT_OUT_DIR / 'rod_drop_1_binarized_tracker_output_widthlocked_5mm'}, {'label': 'rod_drop_2', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\rod drop 2.mp4'), 'out_dir': ROOT_OUT_DIR / 'rod_drop_2_binarized_tracker_output_widthlocked_5mm'}, {'label': 'rod_drop_3', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\rod drop 3.mp4'), 'out_dir': ROOT_OUT_DIR / 'rod_drop_3_binarized_tracker_output_widthlocked_5mm'}, {'label': 'rod_drop_4', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\rod drop 4.mp4'), 'out_dir': ROOT_OUT_DIR / 'rod_drop_4_binarized_tracker_output_widthlocked_5mm'}, {'label': 'rod_drop_5', 'video_path': Path('C:\\Users\\User\\Downloads\\4.5_inch_per_second_flow\\rod drop 5.mp4'), 'out_dir': ROOT_OUT_DIR / 'rod_drop_5_binarized_tracker_output_widthlocked_5mm'}]

def load_module(script_path, module_name):
    """Load the module used by this script."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def load_seed_from_output(seed_output_dir):
    """Load the seed from output used by this script."""
    raw_csv = Path(seed_output_dir) / 'tracks_raw.csv'
    df = pd.read_csv(raw_csv)
    row = df.iloc[0]
    return {'start_frame': int(row['frame_idx']), 'preset_click': (int(round(float(row['cx_raw_px']))), int(round(float(row['cy_raw_px'])))), 'seed_source': str(seed_output_dir)}

def resolve_case(case):
    """Handle the resolve case step used by this script."""
    seed = load_seed_from_output(case['out_dir'])
    resolved = case.copy()
    resolved['start_frame'] = int(seed['start_frame'])
    resolved['preset_click'] = tuple(seed['preset_click'])
    resolved['seed_source'] = seed['seed_source']
    return resolved

def configure_rod_module(module, case):
    """Handle the configure rod module step used by this script."""
    module.USE_RUNTIME_PROMPTS = False
    module.ENABLE_IMSHOW = False
    module.DEFAULT_VIDEO_PATH = str(case['video_path'])
    module.DEFAULT_CALIBRATION_JSON = str(CALIBRATION_JSON)
    module.DEFAULT_OUT_DIR = str(case['out_dir'])
    module.DEFAULT_REAL_FPS = 240.0
    module.DEFAULT_FRAME_STRIDE = 1
    module.DEFAULT_START_FRAME = int(case['start_frame'])
    module.DEFAULT_OBJECT_MODE = 'rod'
    module.DEFAULT_ROD_DIAMETER_MM = 5.0
    module.DEFAULT_USE_PRESET_CLICK = True
    module.DEFAULT_PRESET_CLICK = tuple(case['preset_click'])
    module.DEFAULT_SAVE_DEBUG_IMAGES = True
    module.DEFAULT_NUM_DEBUG_IMAGES = 12

def summarize_run(case):
    """Handle the summarize run step used by this script."""
    raw_csv = case['out_dir'] / 'tracks_raw.csv'
    processed_csv = case['out_dir'] / 'tracks_processed.csv'
    df = pd.read_csv(raw_csv)
    found = int(df['found_this_frame'].fillna(0).astype(int).sum())
    out = {'name': case['out_dir'].name, 'video_path': str(case['video_path']), 'start_frame': int(case['start_frame']), 'preset_click': list(case['preset_click']), 'out_dir': str(case['out_dir']), 'status': 'ok', 'rows': int(len(df)), 'found_frames': found, 'missed_frames': int(len(df) - found), 'seed_source': case['seed_source'], 'raw_csv': str(raw_csv), 'processed_csv': str(processed_csv)}
    if 'length_mm' in df.columns:
        out['length_median_mm'] = float(pd.to_numeric(df['length_mm'], errors='coerce').median())
    if 'width_mm' in df.columns:
        out['width_median_mm'] = float(pd.to_numeric(df['width_mm'], errors='coerce').median())
    if 'box_regularized' in df.columns:
        out['regularized_frames'] = int(pd.to_numeric(df['box_regularized'], errors='coerce').fillna(0).astype(int).sum())
    if 'mask_source' in df.columns:
        out['mask_sources'] = {str(k): int(v) for k, v in df['mask_source'].fillna('nan').value_counts().items()}
    return out

def main():
    """Run the main entry point for this script."""
    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROD_SCRIPT_PATH, ROOT_OUT_DIR / ROD_SCRIPT_PATH.name)
    shutil.copy2(WORKSPACE / 'run_45_rod_batch.py', ROOT_OUT_DIR / 'run_45_rod_batch.py')
    rod_module = load_module(ROD_SCRIPT_PATH, 'rod_tracker_45inch_batch')
    results = []
    for raw_case in ROD_CASES:
        case = resolve_case(raw_case)
        case['out_dir'].mkdir(parents=True, exist_ok=True)
        print(f"Running {case['label']} from frame {case['start_frame']} click={case['preset_click']}...")
        configure_rod_module(rod_module, case)
        rod_module.main()
        summary = summarize_run(case)
        print(f"Completed {case['label']}: rows={summary['rows']} found={summary['found_frames']} missed={summary['missed_frames']}")
        results.append(summary)
    summary_path = ROOT_OUT_DIR / 'batch_summary.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote summary to {summary_path}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
