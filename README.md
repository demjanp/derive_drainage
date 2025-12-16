# Derive drainage network

Command-line tool to stage CopDEM GLO-30 data for an AOI and derive a potential drainage network using WhiteboxTools. The pipeline performs global breach/fill conditioning and global flow routing/accumulation on a single conditioned DEM, with boundary outlets enforced for clipped AOIs.

## Quick start

```bash
python -m derive_drainage.cli \
  -aoi path/to/aoi.gpkg \
  -output path/to/output \
  --crs <projected EPSG> \
  --stream-thresholds-m2 100000,400000
```

Key options:
- `--buffer-km`: AOI buffer distance (km) for staging (default 1.0).
- `--stream-thresholds-m2`: Comma-separated contributing-area thresholds in square meters (default `100000,400000`; two outputs generated). 
- `--sea-ring-cells`: Boundary ring width (cells) forced to a low sea level to create outlets (default 0; set >0 to enable).
- `--sea-collar-cells`: Pad the DEM with a low-elevation collar (cells) to guarantee outlets outside the clipped extent (default 2; set 0 to disable).
- `--sea-level-offset`: Meters below DEM minimum to set the boundary ring/sea mask (default 50 m). Optionally provide `--sea-mask` polygons to lower instead of NoData.
- `--breach-*` / `--fill-*`: Whitebox BreachDepressionsLeastCost and FillDepressions parameters (defaults tuned to be conservative: search 10 m, max cost 200, flat increment 0.05; fill max depth 1 m).
- `--min-stream-length-m`: Minimum stream segment length to keep (meters, default 100).
- `--snap-tolerance-m`: Endpoint snapping tolerance for streams (default 60 m).
- `--threads`: Max WhiteboxTools worker threads (default: tool-defined/all cores).

WhiteboxTools binary: install once per environment if it is not already present:
```bash
python - <<'PY'
from whitebox import whitebox_tools as wt
wt.download_wbt()
PY
```

## Pipeline (required structure)

1) Stage and reproject CopDEM tiles to the target CRS and mosaic into a single DEM.
2) Enforce outlets for clipped AOIs by padding with a low-elevation collar and optionally lowering a boundary ring/sea mask to a low sea level.
3) Global hybrid breach-then-fill conditioning (Whitebox `BreachDepressionsLeastCost` then `FillDepressions`), writing signed/absolute modification depths.
4) Global flow routing and accumulation on the conditioned DEM only (`D8Pointer`, `D8FlowAccumulation` to m² contributing area).
5) Stream extraction per m² threshold (`ExtractStreams`, `StreamLinkIdentifier`, `StrahlerStreamOrder`, `RasterStreamsToVector`), dissolve/endpoint-snap, remove short segments, and merge into one GPKG.
6) QA: global modification rasters plus global summary CSV/JSON (max breach depth, max fill depth, percent modified area).

## Outputs (under `-output`)

- `dem_conditioned.tif`: Conditioned DEM used for routing and stream extraction.
- `streams_<threshold>.gpkg`: Stream vectors for each requested threshold (two by default), with `thr_m2`, `order`, `length_m`, `acc_m2`.
- `run_metadata.json`: Parameters, inputs, and key outputs (plus basic conditioning stats).
- `processing.log`: Processing log.
- `cache/`: Intermediate rasters/vectors needed while running the workflow (mosaic, flow pointer, flow accumulation, stream rasters) kept separate from final outputs.
