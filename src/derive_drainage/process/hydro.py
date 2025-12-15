"""
Hydrology helpers to derive flow direction, accumulation, and drainage network rasters.
"""

from __future__ import annotations

import heapq
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.merge import merge
from rasterio.transform import Affine
from tqdm import tqdm

LOG = logging.getLogger(__name__)

# Clockwise D8 neighbor offsets (row, col): N, NE, E, SE, S, SW, W, NW
NEIGHBORS: list[tuple[int, int]] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def mosaic_dem_tiles(tile_paths: List[Path]) -> tuple[np.ndarray, Affine, Dict]:
    """
    Mosaic DEM tiles into a single array (first band) and return data, transform, and profile.
    """
    if not tile_paths:
        raise ValueError("No DEM tiles provided for mosaicking.")

    datasets = [rasterio.open(p) for p in tile_paths]
    base_meta = datasets[0].meta.copy()
    mosaic, out_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    arr = np.asarray(mosaic[0], dtype="float32")
    profile = base_meta
    profile.update(
        {
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": out_transform,
            "dtype": "float32",
        }
    )
    return arr, out_transform, profile


def _priority_flood_fill(dem: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """
    Fill depressions using a priority-flood algorithm.
    """
    rows, cols = dem.shape
    filled = dem.copy()
    visited = np.zeros_like(dem, dtype=bool)
    pq: list[tuple[float, int, int]] = []

    # Seed with border cells
    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                continue
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                heapq.heappush(pq, (filled[r, c], r, c))
                visited[r, c] = True

    while pq:
        elev, r, c = heapq.heappop(pq)
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            if nodata_mask[nr, nc] or visited[nr, nc]:
                continue
            visited[nr, nc] = True
            n_elev = filled[nr, nc]
            new_elev = max(n_elev, elev)
            filled[nr, nc] = new_elev
            heapq.heappush(pq, (new_elev, nr, nc))

    filled[nodata_mask] = np.nan
    return filled


def fill_depressions(dem: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """
    Fill DEM depressions to enforce drainage.
    """
    if dem.ndim != 2:
        raise ValueError("DEM array must be 2D.")
    if nodata_value is None or np.isnan(nodata_value):
        nodata_mask = np.isnan(dem)
    else:
        nodata_mask = (dem == nodata_value) | np.isnan(dem)

    if np.all(nodata_mask):
        raise ValueError("DEM contains only nodata values.")

    return _priority_flood_fill(dem, nodata_mask)


def compute_flow_direction_d8(filled_dem: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """
    Compute D8 flow directions. Returns int8 array with values 0..7 for neighbor index, -1 for sinks/nodata.
    """
    rows, cols = filled_dem.shape
    flow_dir = np.full_like(filled_dem, fill_value=-1, dtype="int8")

    if nodata_value is None or np.isnan(nodata_value):
        nodata_mask = np.isnan(filled_dem)
    else:
        nodata_mask = (filled_dem == nodata_value) | np.isnan(filled_dem)

    for r in tqdm(range(rows), desc="Flow direction rows", unit="row"):
        for c in range(cols):
            if nodata_mask[r, c]:
                continue
            z = filled_dem[r, c]
            best_drop = -np.inf
            best_dir = -1
            for idx, (dr, dc) in enumerate(NEIGHBORS):
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    continue
                if nodata_mask[nr, nc]:
                    continue
                drop = z - filled_dem[nr, nc]
                if drop > best_drop:
                    best_drop = drop
                    best_dir = idx
            if best_dir >= 0 and best_drop >= 0:
                flow_dir[r, c] = best_dir

    flow_dir[nodata_mask] = -1
    return flow_dir


def compute_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """
    Compute flow accumulation (cell counts) from flow direction grid.
    """
    rows, cols = flow_dir.shape
    acc = np.ones_like(flow_dir, dtype="float32")
    inflow_count = np.zeros_like(flow_dir, dtype="int32")

    for r in tqdm(range(rows), desc="Accum inflow count", unit="row"):
        for c in range(cols):
            direction = flow_dir[r, c]
            if direction < 0:
                continue
            dr, dc = NEIGHBORS[direction]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and flow_dir[nr, nc] >= 0:
                inflow_count[nr, nc] += 1

    stack: list[tuple[int, int]] = []
    for r in tqdm(range(rows), desc="Seed sources", unit="row"):
        for c in range(cols):
            if flow_dir[r, c] >= 0 and inflow_count[r, c] == 0:
                stack.append((r, c))

    total_cells = int(np.count_nonzero(flow_dir >= 0))
    pbar = tqdm(total=total_cells, desc="Flow accumulation", unit="cell")
    while stack:
        r, c = stack.pop()
        pbar.update(1)
        direction = flow_dir[r, c]
        if direction >= 0:
            dr, dc = NEIGHBORS[direction]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and flow_dir[nr, nc] >= 0:
                acc[nr, nc] += acc[r, c]
                inflow_count[nr, nc] -= 1
                if inflow_count[nr, nc] == 0:
                    stack.append((nr, nc))
    pbar.close()

    acc[flow_dir < 0] = 0.0
    return acc


def _write_raster(path: Path, array: np.ndarray, profile: Dict) -> Path:
    """
    Write a single-band raster with the provided profile.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = profile.copy()
    meta.update(
        {
            "count": 1,
            "height": array.shape[0],
            "width": array.shape[1],
            "dtype": str(array.dtype),
        }
    )
    nodata_val = meta.get("nodata")
    if nodata_val is None:
        nodata_val = np.nan
    meta["nodata"] = nodata_val
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array, 1)
    return path


def derive_drainage_from_tiles(
    tile_paths: List[Path],
    crs_obj: CRS | int | str,
    output_dir: Path,
    stream_threshold_cells: int = 400,
    min_stream_length_m: float = 100.0,
) -> Dict:
    """
    Build drainage products from DEM tiles and write outputs to the output directory.
    """
    if not tile_paths:
        raise ValueError("No DEM tiles found for drainage derivation.")

    LOG.info("Mosaicking %d DEM tiles for hydrology", len(tile_paths))
    dem, transform, profile = mosaic_dem_tiles(tile_paths)
    cell_area = abs(transform.a * transform.e - transform.b * transform.d) or abs(transform.a * transform.e)

    hydro_dir = output_dir / "hydro"
    mosaic_path = hydro_dir / "dem_mosaic.tif"

    LOG.info("Writing mosaicked DEM to %s", mosaic_path)
    _write_raster(mosaic_path, dem, profile)

    LOG.info("Filling depressions")
    filled_dem = fill_depressions(dem, profile.get("nodata"))

    LOG.info("Computing flow direction (D8)")
    flow_dir = compute_flow_direction_d8(filled_dem, profile.get("nodata"))

    LOG.info("Computing flow accumulation")
    flow_accum = compute_flow_accumulation(flow_dir)

    LOG.info("Extracting stream mask with threshold %d cells", stream_threshold_cells)
    stream_mask = flow_accum >= float(stream_threshold_cells)

    filled_path = hydro_dir / "dem_filled.tif"
    flow_dir_path = hydro_dir / "flow_direction.tif"
    flow_accum_path = hydro_dir / "flow_accumulation.tif"

    _write_raster(filled_path, filled_dem, profile)
    _write_raster(flow_dir_path, flow_dir.astype("int16"), profile | {"nodata": -1})
    _write_raster(flow_accum_path, flow_accum.astype("float32"), profile | {"nodata": 0.0})

    return {
        "filled_dem": filled_path,
        "dem_mosaic": mosaic_path,
        "flow_dir": flow_dir_path,
        "flow_accum": flow_accum_path,
        "stream_mask": stream_mask,
        "transform": transform,
        "crs": CRS.from_user_input(crs_obj),
        "cell_area": cell_area,
        "flow_accum_array": flow_accum,
        "min_stream_length_m": min_stream_length_m,
        "stream_threshold_cells": stream_threshold_cells,
    }
