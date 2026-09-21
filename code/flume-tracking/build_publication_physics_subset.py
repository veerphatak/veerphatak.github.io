"""
Export utility: build_publication_physics_subset.py

Purpose
-------
Extracts the final publication-focused subset from the larger physics output folder.

Inputs
------
Full physics report outputs.

Outputs
-------
Reduced tables, selected figures and metric notes for direct report inclusion.

Methodological notes
--------------------
The subset keeps only the metrics used in the final report to avoid overwhelming the appendix.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import json
import shutil
from pathlib import Path
import pandas as pd
INPUT_ROOT = Path('C:\\IP Work\\Open_Channel_Physics_Reports_3_and_4.5_Inchps')
OUTPUT_ROOT = Path('C:\\IP Work\\Open_Channel_Physics_Publication_Subset')
WORKSPACE = Path('C:\\IP Work\\New project')
TRIAL_COLUMNS = ['trial', 'batch_label', 'flow_speed_in_s', 'particle_type', 'shape', 'duration_s', 'trajectory_length_mm', 'net_displacement_mm', 'net_streamwise_displacement_mm', 'net_vertical_displacement_mm', 'tortuosity', 'mean_speed_mm_s', 'terminal_velocity_mm_s', 'mean_streamwise_velocity_mm_s', 'mean_vertical_velocity_mm_s', 'slip_velocity_mm_s', 'Re', 'Cd_eff', 'Cl_eff', 'Archimedes_Ar', 'Galileo_Ga', 'Stokes_number_flow', 'Froude_terminal', 'terminal_to_flow_ratio', 'streamwise_advective_ratio', 'vertical_transport_ratio']
PARTICLE_COLUMNS = ['particle_type', 'n_trials', 'terminal_velocity_mm_s_mean', 'mean_speed_mm_s_mean', 'mean_streamwise_velocity_mm_s_mean', 'slip_velocity_mm_s_mean', 'tortuosity_mean', 'Re_mean', 'Cd_eff_mean', 'Cl_eff_mean', 'Archimedes_Ar_mean', 'Galileo_Ga_mean', 'Stokes_number_flow_mean', 'Froude_terminal_mean', 'terminal_to_flow_ratio_mean', 'streamwise_advective_ratio_mean', 'terminal_velocity_mm_s_std', 'mean_speed_mm_s_std', 'mean_streamwise_velocity_mm_s_std', 'slip_velocity_mm_s_std', 'tortuosity_std', 'Re_std', 'Cd_eff_std', 'Cl_eff_std', 'Archimedes_Ar_std', 'Galileo_Ga_std', 'Stokes_number_flow_std', 'Froude_terminal_std', 'terminal_to_flow_ratio_std', 'streamwise_advective_ratio_std']
SHAPE_COLUMNS = ['shape', 'n_trials', 'terminal_velocity_mm_s_mean', 'mean_speed_mm_s_mean', 'slip_velocity_mm_s_mean', 'Re_mean', 'Cd_eff_mean', 'Archimedes_Ar_mean', 'Stokes_number_flow_mean', 'terminal_to_flow_ratio_mean', 'terminal_velocity_mm_s_std', 'mean_speed_mm_s_std', 'slip_velocity_mm_s_std', 'Re_std', 'Cd_eff_std', 'Archimedes_Ar_std', 'Stokes_number_flow_std', 'terminal_to_flow_ratio_std']
PARTICLE_FLOW_COLUMNS = ['particle_flow_group', 'n_trials', 'terminal_velocity_mm_s_mean', 'mean_speed_mm_s_mean', 'mean_streamwise_velocity_mm_s_mean', 'slip_velocity_mm_s_mean', 'tortuosity_mean', 'Re_mean', 'Cd_eff_mean', 'Cl_eff_mean', 'Archimedes_Ar_mean', 'Galileo_Ga_mean', 'Stokes_number_flow_mean', 'Froude_terminal_mean', 'terminal_to_flow_ratio_mean', 'streamwise_advective_ratio_mean', 'terminal_velocity_mm_s_std', 'mean_speed_mm_s_std', 'mean_streamwise_velocity_mm_s_std', 'slip_velocity_mm_s_std', 'tortuosity_std', 'Re_std', 'Cd_eff_std', 'Cl_eff_std', 'Archimedes_Ar_std', 'Galileo_Ga_std', 'Stokes_number_flow_std', 'Froude_terminal_std', 'terminal_to_flow_ratio_std', 'streamwise_advective_ratio_std']
BATCH_PLOTS = ['terminal_velocity_by_particle.png', 'mean_speed_by_particle.png', 'slip_velocity_by_particle.png', 'stokes_number_by_particle.png', 'terminal_to_flow_ratio_by_particle.png', 'drag_coefficient_by_particle.png', 'terminal_velocity_by_trial.png', 'streamwise_velocity_by_trial.png', 'slip_velocity_by_trial.png', 'drag_curve_comparison.png']
COMBINED_PLOTS = ['terminal_velocity_by_particle_flow.png', 'mean_speed_by_particle_flow.png', 'slip_velocity_by_particle_flow.png', 'stokes_number_by_particle_flow.png', 'terminal_to_flow_ratio_by_particle_flow.png', 'drag_curve_comparison.png']

def copy_file(src: Path, dst: Path):
    """Handle the copy file step used by this script."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def subset_columns(df: pd.DataFrame, columns):
    """Handle the subset columns step used by this script."""
    keep = [col for col in columns if col in df.columns]
    return df.loc[:, keep].copy()

def write_metric_notes(path: Path):
    """Write the metric notes used by this script."""
    text = '# Publication Subset Metric Notes\n\nThis folder keeps a reduced set of the most publication-useful transport metrics from the\nfull physics report.\n\n## Trial-level metrics retained\n\n- `trajectory_length_mm`: total travelled path length\n- `net_displacement_mm`: straight-line displacement from start to finish\n- `net_streamwise_displacement_mm`: total left-right transport over the run\n- `net_vertical_displacement_mm`: total vertical transport over the run\n- `tortuosity`: path length divided by net displacement\n- `mean_speed_mm_s`: mean trajectory speed\n- `terminal_velocity_mm_s`: late-stage PCA-fit terminal/settling speed\n- `mean_streamwise_velocity_mm_s`: average streamwise transport velocity\n- `mean_vertical_velocity_mm_s`: average vertical transport velocity\n- `slip_velocity_mm_s`: mismatch between flow speed and net streamwise transport speed\n- `Re`: terminal Reynolds number\n- `Cd_eff`: effective drag coefficient\n- `Cl_eff`: effective lift coefficient inferred from curvature\n- `Archimedes_Ar`: buoyancy-viscosity-gravity scaling\n- `Galileo_Ga`: square-root form of Archimedes number\n- `Stokes_number_flow`: particle response time scaled by flow speed and size\n- `Froude_terminal`: terminal-speed Froude number\n- `terminal_to_flow_ratio`: terminal speed divided by imposed flow speed\n- `streamwise_advective_ratio`: net streamwise transport speed divided by imposed flow speed\n- `vertical_transport_ratio`: net vertical transport speed divided by imposed flow speed\n\n## Why these were kept\n\nThese metrics give a compact summary of:\n- transport magnitude\n- path efficiency / wandering\n- flow-particle coupling\n- inertial / viscous scaling\n- drag / lift behavior\n\nThe full report folder remains the source for all secondary metrics and the complete\nper-trial kinematics.\n'
    path.write_text(text, encoding='utf-8')

def build_batch_subset(batch_label: str):
    """Build the batch subset used by this script."""
    src_dir = INPUT_ROOT / batch_label
    dst_dir = OUTPUT_ROOT / batch_label
    dst_dir.mkdir(parents=True, exist_ok=True)
    trial_df = pd.read_csv(src_dir / 'full_physics_results_by_trial.csv')
    particle_df = pd.read_csv(src_dir / 'full_physics_results_by_particle.csv')
    shape_df = pd.read_csv(src_dir / 'full_physics_results_by_shape.csv')
    log_df = pd.read_csv(src_dir / 'processing_log.csv')
    subset_columns(trial_df, TRIAL_COLUMNS).to_csv(dst_dir / 'publication_metrics_by_trial.csv', index=False)
    subset_columns(particle_df, PARTICLE_COLUMNS).to_csv(dst_dir / 'publication_metrics_by_particle.csv', index=False)
    subset_columns(shape_df, SHAPE_COLUMNS).to_csv(dst_dir / 'publication_metrics_by_shape.csv', index=False)
    log_df.to_csv(dst_dir / 'processing_log.csv', index=False)
    figs_dir = dst_dir / 'figures'
    figs_dir.mkdir(exist_ok=True)
    copied_plots = []
    for plot_name in BATCH_PLOTS:
        src_plot = src_dir / plot_name
        if src_plot.exists():
            copy_file(src_plot, figs_dir / plot_name)
            copied_plots.append(plot_name)
    return {'label': batch_label, 'source_dir': str(src_dir), 'output_dir': str(dst_dir), 'n_trials': int(len(trial_df)), 'n_processed': int((log_df['status'] == 'processed').sum()), 'n_skipped': int((log_df['status'] == 'skipped').sum()), 'plots_copied': copied_plots}

def build_combined_subset():
    """Build the combined subset used by this script."""
    src_dir = INPUT_ROOT / 'combined'
    dst_dir = OUTPUT_ROOT / 'combined'
    dst_dir.mkdir(parents=True, exist_ok=True)
    trial_df = pd.read_csv(src_dir / 'full_physics_results_by_trial.csv')
    particle_flow_df = pd.read_csv(src_dir / 'full_physics_results_by_particle_flow.csv')
    flow_df = pd.read_csv(src_dir / 'full_physics_results_by_flow.csv')
    log_df = pd.read_csv(src_dir / 'processing_log.csv')
    subset_columns(trial_df, TRIAL_COLUMNS).to_csv(dst_dir / 'publication_metrics_by_trial.csv', index=False)
    subset_columns(particle_flow_df, PARTICLE_FLOW_COLUMNS).to_csv(dst_dir / 'publication_metrics_by_particle_flow.csv', index=False)
    flow_df.to_csv(dst_dir / 'publication_metrics_by_flow.csv', index=False)
    log_df.to_csv(dst_dir / 'processing_log.csv', index=False)
    figs_dir = dst_dir / 'figures'
    figs_dir.mkdir(exist_ok=True)
    copied_plots = []
    for plot_name in COMBINED_PLOTS:
        src_plot = src_dir / plot_name
        if src_plot.exists():
            copy_file(src_plot, figs_dir / plot_name)
            copied_plots.append(plot_name)
    return {'label': 'combined', 'source_dir': str(src_dir), 'output_dir': str(dst_dir), 'n_trials': int(len(trial_df)), 'n_processed': int((log_df['status'] == 'processed').sum()), 'n_skipped': int((log_df['status'] == 'skipped').sum()), 'plots_copied': copied_plots}

def main():
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    batch_summaries = [build_batch_subset('3_Inchps'), build_batch_subset('4.5_Inchps'), build_combined_subset()]
    write_metric_notes(OUTPUT_ROOT / 'publication_subset_metric_notes.md')
    copy_file(INPUT_ROOT / 'equations_reference.md', OUTPUT_ROOT / 'equations_reference.md')
    script_path = WORKSPACE / 'build_publication_physics_subset.py'
    copy_file(script_path, OUTPUT_ROOT / script_path.name)
    manifest = {'input_root': str(INPUT_ROOT), 'output_root': str(OUTPUT_ROOT), 'batch_summaries': batch_summaries}
    (OUTPUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote publication subset to: {OUTPUT_ROOT}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
