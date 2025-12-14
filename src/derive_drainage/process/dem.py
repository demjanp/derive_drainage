"""
DEM utilities for reprojection and tiling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry.base import BaseGeometry


def reproject_dem(src_path: Path, dst_path: Path, dst_crs: str | int) -> Path:
    """
    Reproject a DEM to the target CRS.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return dst_path


def _grid_range(start: float, stop: float, step: float) -> Iterable[float]:
    """
    Yield positions spaced by step that fully cover [start, stop].
    """
    start_idx = math.floor(start / step)
    stop_idx = math.ceil(stop / step)
    for idx in range(start_idx, stop_idx):
        yield idx * step


def tile_dem(
    dem_path: Path,
    aoi_geom_proj: BaseGeometry,
    out_dir: Path,
    tile_size_m: float = 10_000.0,
    overlap_m: float = 2_500.0,
) -> list[Path]:
    """
    Tile the DEM into overlapping chunks covering the AOI.

    Each tile covers a 10x10 km core with a 2.5 km buffer on all sides
    (resulting in 15x15 km tile extents).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = aoi_geom_proj.bounds  # minx, miny, maxx, maxy in projected CRS
    tile_paths: list[Path] = []

    with rasterio.open(dem_path) as src:
        dataset_window = Window(col_off=0, row_off=0, width=src.width, height=src.height)
        x_positions = list(_grid_range(bounds[0], bounds[2], tile_size_m))
        y_positions = list(_grid_range(bounds[1], bounds[3], tile_size_m))

        for ix, x0 in enumerate(x_positions):
            for iy, y0 in enumerate(y_positions):
                x1 = x0 + tile_size_m
                y1 = y0 + tile_size_m
                buffered_window = from_bounds(
                    left=x0 - overlap_m,
                    bottom=y0 - overlap_m,
                    right=x1 + overlap_m,
                    top=y1 + overlap_m,
                    transform=src.transform,
                )
                window = buffered_window.intersection(dataset_window)
                if window.width <= 0 or window.height <= 0:
                    continue
                window = window.round_offsets().round_lengths()
                transform = window_transform(window, src.transform)
                tile_meta = src.meta.copy()
                tile_meta.update(
                    {
                        "transform": transform,
                        "width": int(window.width),
                        "height": int(window.height),
                    }
                )
                tile_path = out_dir / f"tile_x{ix}_y{iy}.tif"
                with rasterio.open(tile_path, "w", **tile_meta) as dst:
                    dst.write(src.read(window=window))
                tile_paths.append(tile_path)

    return tile_paths
