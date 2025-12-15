# Derive drainage network

Command-line tool to stage CopDEM GLO-30 data for an AOI and derive a drainage (potential stream) network.

## Usage

```bash
python -m derive_drainage.cli -aoi path/to/aoi.gpkg -output path/to/output --crs <projected EPSG> [options]
```

Key options:
 - `--buffer-km`: AOI buffer distance (km) used for staging (default 1.0).
 - `--stream-accum-threshold`: Flow accumulation threshold (cells) to start a stream (default 400).
 - `--min-stream-length-m`: Minimum stream segment length to keep (meters, default 100).
 - `--keep-temp`: Keep the temporary directory instead of removing it.

## Outputs (in `-output`)

- `hydro/dem_mosaic.tif`: Mosaic of the reprojected DEM.
- `hydro/dem_filled.tif`: Depression-filled DEM mosaic.
- `hydro/flow_direction.tif`: D8 flow direction raster.
- `hydro/flow_accumulation.tif`: Flow accumulation raster (cell counts).
- `streams.gpkg` (layer `streams`): Vectorized potential stream network in the requested CRS.
- `run_metadata.json`: Metadata describing parameters, inputs, and outputs.
- `processing.log`: Processing log.
