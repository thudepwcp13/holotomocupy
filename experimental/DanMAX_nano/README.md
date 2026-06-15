# DanMAX nano experiment adapter

This directory contains the DanMAX-specific adapter for the `holotomocupy` experimental pipeline.

The reference pipeline in `experimental/Y350a_dist1234/` is ESRF-ID16A oriented.  Its `step0.py` performs near-field ptychography (NFP) probe calibration and writes a probe HDF5 file for the later iterative reconstruction.  The DanMAX detector and motor metadata layout is different, so this directory starts with a dedicated step-0 implementation.

## Step 0: DanMAX near-field ptychography sanity check and optional NFP

Expected input files:

- `dark_file`: HDF5 file with detector frames at `/entry/measurement/orca`
- `flat_file`: HDF5 file with detector frames at `/entry/measurement/orca`
- `sample_file`: HDF5 file with detector frames at `/entry/measurement/orca`
- `sample_file`: object motion coordinates at `/entry/measurement/tom_sam_x` and `/entry/measurement/tom_y`

Typical shapes are `100 × 2592 × 3712` for dark, flat, and sample stacks.  Dark and flat are averaged over the repeated frames.  Sample frames are interpreted as the NFP scan positions, with the corresponding motor coordinates converted to object-plane pixels.

Start with the safe sanity-check mode:

```bash
cd experimental/DanMAX_nano
python step0.py config_step0.conf
```

This validates the HDF5 layout, checks dark/flat/sample shapes, computes the centered crop, converts `tom_sam_x` and `tom_y` into pixel shifts, writes correction statistics, and stores a small `corrected_preview` stack in `h5_out`.

After the sanity check looks correct, edit `config_step0.conf` and set:

```ini
run_reconstruction=true
```

Then launch the actual NFP reconstruction, typically one MPI rank per GPU:

```bash
mpirun -n <ngpus> python step0.py config_step0.conf
```

## Important configuration fields

- `energy`, `z1`, `focustodetectordistance`, and `detector_pixelsize` are kept explicit because DanMAX HDF5 metadata may not yet be standardized for this reconstruction pipeline.
- `position_unit` controls how `tom_sam_x` and `tom_y` are interpreted.  Supported values are `m`, `mm`, `um`, `nm`, and `px`.
- `pos_row_sign` and `pos_col_sign` allow flipping the motor-to-detector sign convention without editing the script.
- `n=0` uses the largest centered square crop inside the detector frame; otherwise the code uses an `n × n` centered crop.

## Output HDF5 contents

In sanity-check mode, `h5_out` contains:

- `dark_mean`
- `flat_mean`
- `flat_minus_dark`
- `pos`
- `tom_sam_x`
- `tom_y`
- `corrected_preview` if enabled
- geometry and crop metadata as HDF5 attributes

In reconstruction mode, the same output is extended with:

- `prb_amp`
- `prb_phase`
- `proj_delta`
- `proj_beta`
- `pos_err`

## Steps 1–6 status

Only step 0 is implemented here for the DanMAX HDF5 layout.  The next adapter work should map DanMAX tomography acquisitions into the common `/exchange/*` HDF5 layout used by `steps15.py`, then reuse the existing step-6 BH reconstruction as much as possible.
