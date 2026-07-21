ROOT=/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/z1_scan_step0
z1_start=0.12620
z1_stop=0.12790
z1_step=0.00010
# Geometry
ENERGY=19.55
DISTANCE=0.116379
PIXEL_SIZE=44.761e-9

#ENERGY=29.63
#DISTANCE=0.05455659
#PIXEL_SIZE=55e-9

python merge_z1_nfp_results.py \
    "$ROOT" \
    --z1-range "${z1_start}:${z1_stop}:${z1_step}" \
    --output merged_z1_results.h5 \
    --flat-pred-mode propagate \
    --energy-kev ${ENERGY} \
    --distance-m ${DISTANCE} \
    --pixel-size-m ${PIXEL_SIZE} \
    --overwrite