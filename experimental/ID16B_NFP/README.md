# ID16B_NFP

Near-field ptychography step-0 reader/reconstruction for the ID16B multi-key HDF5 layout.

## Expected layout

- Dark: one fixed 3-D dataset, averaged over the first `dark_nframes` frames.
- Flat: one image per scan-dependent key, such as `/${n}.1/measurement/pco1`.
- Sample: one image and two scalar motor positions per scan-dependent key.
- Images may be `H x W` or `1 x H x W`. This workflow requires `2048 x 2048` and performs no crop or padding.

The `${n}` placeholder is replaced by each integer from the inclusive scan-ID specifications. For example, `flat_scan_ids=1:10` expands to IDs 1 through 10, including 10.

## Run

Sanity check:

```bash
DARK_FILE=/data/dark.h5 \
FLAT_FILE=/data/flat.h5 \
SAMPLE_FILE=/data/sample.h5 \
bash run_step0.sh sanity
```

NFP reconstruction:

```bash
DARK_FILE=/data/dark.h5 \
FLAT_FILE=/data/flat.h5 \
SAMPLE_FILE=/data/sample.h5 \
bash run_step0.sh nfp
```

Select a subset using the actual sample scan IDs:

```bash
FRAME_IDS=1:256:4 bash run_step0.sh nfp
```

The output HDF5 stores the expanded flat/sample scan IDs, selected scan IDs, motor values, converted positions, probe, projection, and position corrections.

`prb` and `proj` are replicated MPI variables in `RecNFP`; this workflow writes the rank-0 copy only and does not concatenate duplicate copies from multiple ranks.
