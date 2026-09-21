Project Appendix Code
=====================

Purpose
-------
This folder contains the Python scripts used to process the mesoplastic particle
trajectory videos, generate validation overlays, calculate derived physics
quantities, and produce the final report figures. The appendix is organised to
allow the processing route to be followed from raw videos to final plots.

Recommended workflow
--------------------
1. Run the relevant tracker script or batch runner for the particle type.
2. Check the annotated videos and validation overlays to confirm that the detected
   trajectory follows the visible particle rather than reflections, bubbles or hands.
3. Run the physics-analysis scripts to calculate velocities and derived metrics.
4. Run the plot-building scripts to regenerate the figures used in the report.
5. Use the exported CSV files to audit any plotted values.

Core tracking scripts
---------------------
- Kalman_Tracker_Spheres_binarized.py: sphere tracking using colour/grayscale
  segmentation and Kalman smoothing.
- Kalman_Tracker_Rod_binarized.py: rod tracking using binarised contours,
  fitted quadrilaterals and Kalman smoothing.
- Kalman_Tracker_Disc_binarized.py: disc tracking using binarised contours,
  projected-area/orientation extraction and Kalman smoothing.

Batch and pipeline runners
--------------------------
- run_tuned_sphere_batch.py: final tuned water-tunnel sphere batch.
- run_3inch_rod_disc_batch.py: final 3 inch/s rod and disc batch.
- run_45_rod_batch.py: final 4.5 inch/s rod batch.
- run_24april_sideview_sphere_pipeline.py: complete side-view flume sphere
  pipeline.
- run_25april_bottom_panel_pipeline.py: complete bottom-view flume sphere
  pipeline.

Analysis and plotting scripts
-----------------------------
- trajectory_physics_open_channel_full_report.py: full physics summary for the
  open-channel/flume trajectory data.
- trajectory_physics_bottom_panel_report.py: bottom-view flume physics summary.
- build_instantaneous_velocity_vs_time_plots.py: instantaneous speed-time plots.
- build_streamwise_velocity_vs_time_plots.py: streamwise velocity-time plots.
- build_velocity_vs_normalised_area_plots.py: velocity versus projected-area
  plots for rods and discs.
- build_rod_velocity_orientation_analysis.py: rod orientation versus velocity
  analysis.
- build_particle_type_overall_velocity_analysis.py: pooled comparison by
  particle type.
- build_publication_physics_subset.py: reduced output package for direct report
  inclusion.

Validation and utility scripts
------------------------------
- create_trajectory_overlays.py: static/video trajectory overlays.
- build_straight_view_slip_overlays_from_raw.py: straight-view slip overlay
  generation from raw videos.
- trajectory_validation_overlay_full_video_editable_fixed_v3.py: full-video
  validation overlays.
- trajectory_validation_master_summary_editable_fixed_v2.py: master validation
  summary.
- rotate_straight_view_14mm_outputs_to_portrait.py: orientation correction for
  the 14 mm straight-view outputs.

Installation
------------
Python 3.10 or later is recommended. From this folder, install the required
packages with:

    pip install -r requirements.txt

The main third-party packages are NumPy, pandas, OpenCV, matplotlib, Pillow and
openpyxl. Some scripts also use standard-library modules such as pathlib, json,
math, statistics and shutil.

Reproducibility notes
---------------------
Several scripts contain absolute file paths because the processing was carried
out on the project workstation. To rerun the code on another machine, update the
input video paths, calibration paths and output folders near the top of the
script before execution. Where batch files are used, the run tables document the
videos, start frames and initial particle locations used for the final analysis.

Commenting approach
-------------------
The scripts have been annotated for appendix use. Comments focus on processing
logic, calibration assumptions, physical interpretation and quality-control
steps. Obvious line-by-line comments have been avoided so that the code remains
readable and not excessive.

Quality-control checks
----------------------
For each processed run, the recommended checks are:
- confirm that the particle is detected continuously through the reported section;
- inspect annotated videos and still overlays against the original video;
- check that the calibration grid or homography is appropriate for the camera view;
- verify that velocity plots do not contain artefacts caused by missed detections;
- compare repeated trials before using pooled particle-type trends.
