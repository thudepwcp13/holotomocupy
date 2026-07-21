ROOT=/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/z1_scan_step0

python compare_z1_roi_metrics.py \
    "$ROOT/merged_z1_results.h5" \
    --output-dir "$ROOT/z1_roi_metrics" \
    --keys proj_beta_roi proj_delta_roi \
    --pixel-size 45 \
    --pixel-unit nm \
    --center auto \
    --ring-radii auto \
    --ring-count 6 \
    --spoke-harmonic auto \
    --overwrite