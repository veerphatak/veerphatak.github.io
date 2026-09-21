"""
Batch runner: run_3inch_rod_disc_batch.py

Purpose
-------
Runs the final 3 inch/s rod and disc water-tunnel tracking batch.

Inputs
------
Run table containing videos, calibration files, start frames and seed points.

Outputs
-------
Processed rod/disc output folders containing trajectories, videos and diagnostics.

Methodological notes
--------------------
All cases are processed through the same tracker configuration structure to maintain repeatability across particle types.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import pandas as pd
WORKSPACE = Path('C:\\IP Work\\New project')
ROOT_OUT_DIR = Path('C:\\IP Work\\3_Inchps_Rod_Disc_Binarised')
ROD_OUT_DIR = ROOT_OUT_DIR / 'rods'
DISC_OUT_DIR = ROOT_OUT_DIR / 'discs'
CALIBRATION_JSON = Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\rod_calibration\\calibration.json')
OLD_OUTPUTS_ROOT = Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow')
ROD_SCRIPT_PATH = WORKSPACE / 'Kalman_Tracker_Rod_binarized.py'
DISC_SCRIPT_PATH = WORKSPACE / 'Kalman_Tracker_Disc_binarized.py'
IGNORED_CASES = [{'label': 'disc_drop_2', 'reason': 'User marked this case as an anomaly because there is no disc in the video.', 'video_path': 'C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 2.mp4'}]
ROD_CASES = [{'label': 'rod_drop_1', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Callibration + Rod drop 1.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_1_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_1_tracker_output'}, {'label': 'rod_drop_2', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 2.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_2_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_2_updated_tracker_output'}, {'label': 'rod_drop_3', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 3.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_3_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_3_updated_tracker_output'}, {'label': 'rod_drop_4', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 4.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_4_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_4_updated_tracker_output'}, {'label': 'rod_drop_5', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 5.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_5_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_5_updated_tracker_output'}, {'label': 'rod_drop_6', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 6.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_6_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_6_updated_tracker_output'}, {'label': 'rod_drop_7', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 7.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_7_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_7_updated_tracker_output'}, {'label': 'rod_drop_8', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 8.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_8_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_8_updated_tracker_output'}, {'label': 'rod_drop_9', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 9.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_9_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_9_updated_tracker_output'}, {'label': 'rod_drop_10', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Rod drop 10.mp4'), 'out_dir': ROD_OUT_DIR / 'rod_drop_10_binarized_tracker_output_widthlocked_5mm', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'rod_drop_10_updated_tracker_output'}]
DISC_CASES = [{'label': 'disc_drop_1', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Callibration + Disc drop 1.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_1_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_1_tracker_output'}, {'label': 'disc_drop_3', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 3.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_3_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_3_tracker_output'}, {'label': 'disc_drop_4', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 4.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_4_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_4_tracker_output'}, {'label': 'disc_drop_5', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 5.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_5_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_5_tracker_output'}, {'label': 'disc_drop_6', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 6.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_6_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_6_part_2_tracker_output'}, {'label': 'disc_drop_7', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 7.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_7_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_7_part_2_tracker_output'}, {'label': 'disc_drop_8', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 8.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_8_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_8_tracker_output'}, {'label': 'disc_drop_9', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 9.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_9_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_9_tracker_output'}, {'label': 'disc_drop_10', 'video_path': Path('C:\\Users\\User\\Downloads\\3_inch_per_second_flow\\Disc drop 10.mp4'), 'out_dir': DISC_OUT_DIR / 'disc_drop_10_binarized_tracker_output', 'seed_output_dir': OLD_OUTPUTS_ROOT / 'disc_drop_10_tracker_output'}]

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

def resolve_seed(case):
    """Handle the resolve seed step used by this script."""
    if 'seed_output_dir' in case:
        seed = load_seed_from_output(case['seed_output_dir'])
    else:
        seed = {'start_frame': int(case['start_frame']), 'preset_click': tuple(case['preset_click']), 'seed_source': case.get('seed_source', 'manual')}
    resolved = case.copy()
    resolved['start_frame'] = int(seed['start_frame'])
    resolved['preset_click'] = tuple(seed['preset_click'])
    resolved['seed_source'] = seed['seed_source']
    return resolved

def summarize_run(case):
    """Handle the summarize run step used by this script."""
    raw_csv = case['out_dir'] / 'tracks_raw.csv'
    processed_csv = case['out_dir'] / 'tracks_processed.csv'
    df = pd.read_csv(raw_csv)
    found_col = 'found_this_frame' if 'found_this_frame' in df.columns else 'found' if 'found' in df.columns else None
    found = int(df[found_col].fillna(0).astype(int).sum()) if found_col else None
    return {'label': case['label'], 'video_path': str(case['video_path']), 'out_dir': str(case['out_dir']), 'start_frame': int(case['start_frame']), 'preset_click': list(case['preset_click']), 'seed_source': case['seed_source'], 'rows': int(len(df)), 'found': found, 'missed': int(len(df) - found) if found is not None else None, 'raw_csv': str(raw_csv), 'processed_csv': str(processed_csv)}

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
    if hasattr(module, 'DEFAULT_START_SEARCH_FRAMES'):
        module.DEFAULT_START_SEARCH_FRAMES = 25

def configure_disc_module(module, case):
    """Handle the configure disc module step used by this script."""
    module.USE_RUNTIME_PROMPTS = False
    module.ENABLE_IMSHOW = False
    module.DEFAULT_VIDEO_PATH = str(case['video_path'])
    module.DEFAULT_CALIBRATION_JSON = str(CALIBRATION_JSON)
    module.DEFAULT_OUT_DIR = str(case['out_dir'])
    module.DEFAULT_REAL_FPS = 240.0
    module.DEFAULT_FRAME_STRIDE = 1
    module.DEFAULT_START_FRAME = int(case['start_frame'])
    module.DEFAULT_OBJECT_MODE = 'disc'
    module.DEFAULT_DISC_DIAMETER_MM = 20.0
    module.DEFAULT_USE_PRESET_CLICK = True
    module.DEFAULT_PRESET_CLICK = tuple(case['preset_click'])
    module.DEFAULT_SAVE_DEBUG_IMAGES = True
    module.DEFAULT_NUM_DEBUG_IMAGES = 12
    if hasattr(module, 'DEFAULT_START_SEARCH_FRAMES'):
        module.DEFAULT_START_SEARCH_FRAMES = 25

def run_cases(module, cases, configure_fn):
    """Run the cases used by this script."""
    results = []
    for raw_case in cases:
        case = resolve_seed(raw_case)
        case['out_dir'].mkdir(parents=True, exist_ok=True)
        print(f"Running {case['label']} from frame {case['start_frame']} click={case['preset_click']}...")
        configure_fn(module, case)
        module.main()
        summary = summarize_run(case)
        print(f"Completed {case['label']}: rows={summary['rows']} found={summary['found']} missed={summary['missed']}")
        results.append(summary)
    return results

def filter_cases(cases, include_labels):
    """Handle the filter cases step used by this script."""
    if not include_labels:
        return list(cases)
    wanted = {label.strip() for label in include_labels if label.strip()}
    return [case for case in cases if case['label'] in wanted]

def summarize_existing_cases(cases):
    """Handle the summarize existing cases step used by this script."""
    results = []
    for raw_case in cases:
        case = resolve_seed(raw_case)
        raw_csv = case['out_dir'] / 'tracks_raw.csv'
        if raw_csv.exists():
            results.append(summarize_run(case))
    return results

def main():
    """Run the main entry point for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--rods-only', action='store_true')
    parser.add_argument('--discs-only', action='store_true')
    parser.add_argument('--include', type=str, default='')
    args = parser.parse_args()
    include_labels = [part for part in args.include.split(',') if part.strip()]
    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DISC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROD_SCRIPT_PATH, ROD_OUT_DIR / ROD_SCRIPT_PATH.name)
    shutil.copy2(DISC_SCRIPT_PATH, DISC_OUT_DIR / DISC_SCRIPT_PATH.name)
    shutil.copy2(WORKSPACE / 'run_3inch_rod_disc_batch.py', ROOT_OUT_DIR / 'run_3inch_rod_disc_batch.py')
    rod_module = load_module(ROD_SCRIPT_PATH, 'rod_tracker_3inch_batch')
    disc_module = load_module(DISC_SCRIPT_PATH, 'disc_tracker_3inch_batch')
    selected_rod_cases = filter_cases(ROD_CASES, include_labels)
    selected_disc_cases = filter_cases(DISC_CASES, include_labels)
    if args.discs_only:
        selected_rod_cases = []
    if args.rods_only:
        selected_disc_cases = []
    if selected_rod_cases:
        run_cases(rod_module, selected_rod_cases, configure_rod_module)
    if selected_disc_cases:
        run_cases(disc_module, selected_disc_cases, configure_disc_module)
    rod_results = summarize_existing_cases(ROD_CASES)
    disc_results = summarize_existing_cases(DISC_CASES)
    summary = {'calibration_json': str(CALIBRATION_JSON), 'output_root': str(ROOT_OUT_DIR), 'rod_output_root': str(ROD_OUT_DIR), 'disc_output_root': str(DISC_OUT_DIR), 'ignored_cases': IGNORED_CASES, 'rod_cases': rod_results, 'disc_cases': disc_results}
    summary_path = ROOT_OUT_DIR / 'batch_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote summary to {summary_path}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
