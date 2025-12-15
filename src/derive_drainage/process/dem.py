"""
DEM utilities for reprojection and tiling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import matplotlib.tri as mtri
from tqdm import tqdm


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


def erase_features_from_dem_tiles(
    tile_paths: List[Path],
    crs_obj: CRS | str | int,
    osm_tiles: List[Path],
    gdw_gdf_proj: gpd.GeoDataFrame,
    reservoir_gdf_proj: gpd.GeoDataFrame,
) -> None:
    """
    Rasterize OSM/GDW/reservoir features and erase them (set to NaN) from DEM tiles in-place.
    """
    crs = CRS.from_user_input(crs_obj)
    osm_by_stem: Dict[str, Path] = {p.stem: p for p in osm_tiles}

    for tile_path in tile_paths:
        tile_stem = tile_path.stem
        osm_tile_path = osm_by_stem.get(tile_stem)
        if osm_tile_path is None:
            osm_tile_gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
        else:
            osm_tile_gdf = gpd.read_file(osm_tile_path)
            if osm_tile_gdf.crs is None:
                osm_tile_gdf = osm_tile_gdf.set_crs(crs)

        with rasterio.open(tile_path) as src:
            tile_bounds = src.bounds
            tile_geom = box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)
            relevant_gdw = gdw_gdf_proj[gdw_gdf_proj.intersects(tile_geom)] if not gdw_gdf_proj.empty else gdw_gdf_proj
            relevant_res = (
                reservoir_gdf_proj[reservoir_gdf_proj.intersects(tile_geom)]
                if not reservoir_gdf_proj.empty
                else reservoir_gdf_proj
            )
            if (
                osm_tile_gdf.empty
                and (relevant_gdw is None or getattr(relevant_gdw, "empty", True))
                and (relevant_res is None or getattr(relevant_res, "empty", True))
            ):
                continue
            geoms = []
            if not osm_tile_gdf.empty:
                geoms.extend([g for g in osm_tile_gdf.geometry if g is not None and not g.is_empty])
            if relevant_gdw is not None and not getattr(relevant_gdw, "empty", True):
                geoms.extend([g for g in relevant_gdw.geometry if g is not None and not g.is_empty])
            if relevant_res is not None and not getattr(relevant_res, "empty", True):
                geoms.extend([g for g in relevant_res.geometry if g is not None and not g.is_empty])
            if not geoms:
                continue
            mask = rasterize(
                [(geom, 1) for geom in geoms],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype="uint8",
            )
            data = src.read(1).astype("float32")
            data[mask == 1] = np.nan
            meta = src.meta.copy()
            meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(tile_path, "w", **meta) as dst:
            dst.write(data, 1)


def fill_tile_nodata_natural_neighbor(tile_paths: List[Path]) -> None:
    """
    Fill NaN holes in DEM tiles using natural neighbor interpolation in-place.
    """
    for tile_path in tqdm(tile_paths, desc="Fill nodata (tiles)", unit="tile"):
        with rasterio.open(tile_path) as src:
            band = src.read(1).astype("float32")
            nodata_val = src.nodata
            mask = np.isnan(band) if nodata_val is None else (band == nodata_val) | np.isnan(band)
            if not mask.any():
                continue
            valid_mask = ~mask
            if not valid_mask.any():
                raise ValueError(f"Tile {tile_path} contains only NaN after erasure; cannot interpolate.")

            rows_valid, cols_valid = np.nonzero(valid_mask)
            xs_valid, ys_valid = rasterio.transform.xy(src.transform, rows_valid, cols_valid)
            values = band[rows_valid, cols_valid]

            rows_target, cols_target = np.nonzero(mask)
            xs_target, ys_target = rasterio.transform.xy(src.transform, rows_target, cols_target)

            points = np.column_stack([np.asarray(xs_valid), np.asarray(ys_valid)])
            targets = np.column_stack([np.asarray(xs_target), np.asarray(ys_target)])

            triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
            interpolator = mtri.LinearTriInterpolator(triangulation, values)
            filled_values = interpolator(targets[:, 0], targets[:, 1])
            if np.isnan(filled_values).any():
                raise RuntimeError(f"Natural neighbor interpolation left NaNs in tile {tile_path}")

            filled_band = band.copy()
            filled_band[rows_target, cols_target] = filled_values
            meta = src.meta.copy()
            meta.update({"dtype": "float32", "nodata": None})

        tmp_path = tile_path.with_suffix(".tmp_fill.tif")
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(filled_band, 1)
        tmp_path.replace(tile_path)


def clip_tiles_to_core(tile_paths: List[Path], overlap_m: float) -> None:
    """
    Clip overlapped tiles back to their core (remove overlap on all sides) in-place.
    """
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            bounds = src.bounds
            core_left = bounds.left + overlap_m
            core_right = bounds.right - overlap_m
            core_bottom = bounds.bottom + overlap_m
            core_top = bounds.top - overlap_m
            if core_left >= core_right or core_bottom >= core_top:
                raise ValueError(f"Overlap {overlap_m} exceeds tile extent for {tile_path}")
            core_window = from_bounds(core_left, core_bottom, core_right, core_top, src.transform)
            dataset_window = Window(col_off=0, row_off=0, width=src.width, height=src.height)
            window = core_window.intersection(dataset_window).round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                raise ValueError(f"Computed core window is empty for {tile_path}")
            transform = window_transform(window, src.transform)
            meta = src.meta.copy()
            meta.update(
                {
                    "transform": transform,
                    "width": int(window.width),
                    "height": int(window.height),
                }
            )
            data = src.read(window=window)

        tmp_path = tile_path.with_suffix(".tmp_core.tif")
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(data)
        tmp_path.replace(tile_path)
