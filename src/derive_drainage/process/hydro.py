"""
Hydrology pipeline built around WhiteboxTools for global conditioning and flow routing.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio import features
from rasterio.merge import merge
from rasterio.transform import Affine
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from whitebox.whitebox_tools import WhiteboxTools

LOG = logging.getLogger(__name__)


def _write_raster(path: Path, array: np.ndarray, profile: Dict, transform: Affine | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = profile.copy()
    meta.update(
        {
            "height": array.shape[0],
            "width": array.shape[1],
            "count": 1,
            "dtype": str(array.dtype),
        }
    )
    if transform is not None:
        meta["transform"] = transform
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array, 1)
    return path


def _require_projected_crs(crs_obj) -> CRS:
    crs_checked = CRS.from_user_input(crs_obj)
    if not crs_checked.is_projected:
        raise ValueError(f"Projected CRS in linear units is required; got {crs_checked.to_string()}")
    return crs_checked


def _compute_cell_area_m2(transform: Affine, crs) -> float:
    """
    Compute pixel area in square meters, requiring a projected CRS.
    """
    crs_checked = _require_projected_crs(crs)
    base_area = abs(transform.a * transform.e - transform.b * transform.d) or abs(transform.a * transform.e)
    factor = 1.0
    if crs_checked.axis_info:
        info = crs_checked.axis_info[0]
        if info.unit_conversion_factor:
            factor = info.unit_conversion_factor
    return base_area * (factor ** 2)


def _build_wbt(work_dir: Path, max_procs: int | None) -> WhiteboxTools:
    def _new() -> WhiteboxTools:
        w = WhiteboxTools()
        w.set_verbose_mode(True)
        w.workdir = str(work_dir)
        if max_procs is not None and max_procs > 0:
            w.set_max_procs(max_procs)
        return w

    wbt = _new()
    exe_path = Path(wbt.exe_path)
    if not exe_path.exists():
        from whitebox import whitebox_tools as wt

        wt.download_wbt()
        wbt = _new()
        exe_path = Path(wbt.exe_path)
        if not exe_path.exists():
            raise RuntimeError(
                f"WhiteboxTools executable not found at {exe_path}. Download it with `python - <<'PY'\\nfrom whitebox import whitebox_tools as wt\\nwt.download_wbt()\\nPY` or set WBT_PATH to the executable."
            )
    LOG.info("Using WhiteboxTools at %s", exe_path)
    return wbt


def _require_created(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} was not created at {path}")


def _prepare_mosaic(dem_paths: Sequence[Path], mosaic_path: Path) -> tuple[Path, Dict]:
    """
    Ensure inputs share a single grid; write a mosaic for downstream steps.
    """
    if not dem_paths:
        raise ValueError("No DEM inputs provided for mosaicking.")
    if len(dem_paths) == 1:
        source = Path(dem_paths[0])
        mosaic_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, mosaic_path)
        with rasterio.open(source) as src:
            profile = src.profile.copy()
        return mosaic_path, profile

    datasets = [rasterio.open(Path(p)) for p in dem_paths]
    base_meta = datasets[0].meta.copy()
    mosaic, out_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    arr = np.asarray(mosaic[0], dtype="float32")
    profile = base_meta
    nodata_val = base_meta.get("nodata", -32768.0)
    profile.update(
        {
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": out_transform,
            "dtype": "float32",
            "nodata": nodata_val,
        }
    )
    _write_raster(mosaic_path, arr, profile, transform=out_transform)
    return mosaic_path, profile


def _pad_dem_with_sea_collar(
    dem_path: Path, output_path: Path, collar_cells: int, sea_level_offset: float
) -> tuple[Path, float, int]:
    """
    Expand the DEM with a low-elevation collar to provide outlets beyond the clipped extent.
    """
    if collar_cells <= 0:
        raise ValueError("collar_cells must be positive to pad a DEM.")
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    valid_mask = np.isfinite(arr)
    if nodata is not None:
        valid_mask &= ~np.isclose(arr, nodata)
    if not np.any(valid_mask):
        raise ValueError("DEM has no valid data to derive sea level for collar padding.")
    min_val = float(np.nanmin(arr[valid_mask]))
    sea_level = min_val - abs(sea_level_offset)

    pad = collar_cells
    padded = np.pad(arr, pad_width=pad, mode="constant", constant_values=sea_level)
    new_transform = transform * Affine.translation(-pad, -pad)
    new_profile = profile.copy()
    new_profile.update(
        {
            "height": padded.shape[0],
            "width": padded.shape[1],
            "transform": new_transform,
        }
    )
    _write_raster(output_path, padded, new_profile, transform=new_transform)
    return output_path, sea_level, pad


def _impose_outlets(
    dem_path: Path, output_path: Path, ring_cells: int, sea_level_offset: float, sea_mask: Path | None
) -> tuple[Path, float]:
    """
    Create spill outlets by lowering a boundary ring and optional sea mask to a low sea level.
    """
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    valid_mask = np.isfinite(arr)
    if nodata is not None:
        valid_mask &= ~np.isclose(arr, nodata)
    if not np.any(valid_mask):
        raise ValueError("DEM has no valid data to derive sea level for outlet carving.")
    min_val = float(np.nanmin(arr[valid_mask]))
    sea_level = min_val - abs(sea_level_offset)

    arr_out = arr.copy()
    if ring_cells > 0:
        h, w = arr.shape
        ring = max(1, min(ring_cells, h // 2, w // 2))
        ring_mask = np.zeros_like(arr_out, dtype=bool)
        ring_mask[:ring, :] = True
        ring_mask[-ring:, :] = True
        ring_mask[:, :ring] = True
        ring_mask[:, -ring:] = True
        arr_out[ring_mask] = sea_level

    if sea_mask is not None and sea_mask.exists():
        gdf = gpd.read_file(sea_mask)
        if not gdf.empty:
            if gdf.crs and profile.get("crs") and gdf.crs != profile["crs"]:
                gdf = gdf.to_crs(profile["crs"])
            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
            if shapes:
                mask = features.rasterize(shapes=shapes, out_shape=arr_out.shape, transform=transform, fill=0, all_touched=True)
                arr_out[mask == 1] = sea_level

    _write_raster(output_path, arr_out.astype("float32"), profile, transform=transform)
    return output_path, sea_level


def _run_breach(wbt: WhiteboxTools, src: Path, dst: Path, dist: float, max_cost: float | None, flat_increment: float | None) -> None:
    dist_arg = int(dist)
    max_cost_arg = int(max_cost) if max_cost is not None else None
    wbt.breach_depressions_least_cost(
        dem=str(src),
        output=str(dst),
        dist=dist_arg,
        max_cost=max_cost_arg,
        min_dist=True,
        flat_increment=flat_increment,
        fill=False,
    )
    _require_created(dst, "BreachDepressionsLeastCost output")


def _run_fill_depressions(
    wbt: WhiteboxTools, src: Path, dst: Path, flat_increment: float | None, max_depth: float | None
) -> None:
    wbt.fill_depressions(
        dem=str(src),
        output=str(dst),
        fix_flats=True,
        flat_increment=flat_increment,
        max_depth=max_depth,
    )
    _require_created(dst, "FillDepressions output")


def _threshold_tag(threshold_m2: float) -> str:
    safe = f"{threshold_m2:.2f}".rstrip("0").rstrip(".")
    return safe.replace(".", "p")


def _snap_coord(value: float, tolerance: float) -> float:
    return round(value / tolerance) * tolerance


def _snap_line_endpoints(geom: LineString, tolerance: float) -> LineString:
    coords = list(geom.coords)
    snapped: list[tuple[float, float]] = []
    for idx, (x, y) in enumerate(coords):
        if idx == 0 or idx == len(coords) - 1:
            snapped.append((_snap_coord(x, tolerance), _snap_coord(y, tolerance)))
        else:
            snapped.append((x, y))
    return LineString(snapped)


def _dissolve_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    dissolved_rows: list[dict] = []
    for (thr, order), group in gdf.groupby(["thr_m2", "order"]):
        if group.empty:
            continue
        geom_union = unary_union(group.geometry)
        try:
            merged = linemerge(geom_union)
        except ValueError:
            merged = geom_union
        geoms: Iterable[LineString]
        if merged.is_empty:
            continue
        if isinstance(merged, LineString):
            geoms = [merged]
        else:
            geoms = [geom for geom in merged.geoms if isinstance(geom, LineString)]
        for geom in geoms:
            dissolved_rows.append({"thr_m2": thr, "order": order, "geometry": geom})
    if not dissolved_rows:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(dissolved_rows, geometry="geometry", crs=gdf.crs)


def _sample_raster(path: Path, coords: list[tuple[float, float]]) -> list[float]:
    with rasterio.open(path) as src:
        values = list(src.sample(coords))
    return [float(v[0]) if len(v) > 0 else float("nan") for v in values]


def _write_empty_streams_gpkg(gpkg_path: Path, crs, threshold: float, reason: str) -> None:
    """
    Persist an empty streams layer so the expected output exists even when no cells meet a threshold.
    """
    LOG.warning(
        "No stream vectors for threshold %.2f m2 (%s); writing empty layer to %s", threshold, reason, gpkg_path
    )
    empty = gpd.GeoDataFrame(
        {"thr_m2": [], "order": [], "length_m": [], "acc_m2": []},
        geometry=[],
        crs=crs,
    )
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    empty.to_file(gpkg_path, layer="streams", driver="GPKG")


def _cleanup_stream_scratch(tag: str, streams_dir: Path, scratch_paths: Sequence[Path]) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".cpg", ".prj"]:
        sidecar = streams_dir / f"streams_{tag}{suffix}"
        if sidecar.exists():
            sidecar.unlink()
    for scratch in scratch_paths:
        if scratch.exists():
            scratch.unlink()


def _extract_streams(
    flow_accum_cells: Path,
    flow_accum_m2: Path,
    flow_pointer: Path,
    thresholds_m2: Sequence[float],
    min_length_m: float,
    snap_tolerance_m: float,
    streams_dir: Path,
    output_streams_dir: Path,
    threads: int | None,
) -> list[Path]:
    streams_dir.mkdir(parents=True, exist_ok=True)
    output_streams_dir.mkdir(parents=True, exist_ok=True)

    wbt = _build_wbt(streams_dir, threads)

    per_threshold_vectors: list[Path] = []

    with rasterio.open(flow_pointer) as ptr_src:
        ptr_crs = ptr_src.crs

    for threshold in thresholds_m2:
        if threshold <= 0:
            raise ValueError("Stream threshold must be positive.")
        tag = _threshold_tag(threshold)
        stream_raster = streams_dir / f"streams_{tag}.tif"
        link_raster = streams_dir / f"stream_links_{tag}.tif"
        order_raster = streams_dir / f"strahler_order_{tag}.tif"
        shp_path = streams_dir / f"streams_{tag}.shp"
        gpkg_path = output_streams_dir / f"streams_{tag}.gpkg"

        scratch_paths = [stream_raster, link_raster, order_raster]
        try:
            # Threshold is provided directly in m^2 since the accumulation raster is already in catchment-area units.
            wbt.extract_streams(flow_accum=str(flow_accum_m2), output=str(stream_raster), threshold=float(threshold))
            _require_created(stream_raster, "ExtractStreams output")

            with rasterio.open(stream_raster) as ssrc:
                stream_data = ssrc.read(1)
                if np.count_nonzero(stream_data) == 0:
                    _write_empty_streams_gpkg(gpkg_path, ptr_crs, threshold, "no stream cells found")
                    per_threshold_vectors.append(gpkg_path)
                    continue

            wbt.stream_link_identifier(d8_pntr=str(flow_pointer), streams=str(stream_raster), output=str(link_raster))
            _require_created(link_raster, "StreamLinkIdentifier output")
            wbt.strahler_stream_order(d8_pntr=str(flow_pointer), streams=str(stream_raster), output=str(order_raster))
            _require_created(order_raster, "StrahlerStreamOrder output")
            wbt.raster_streams_to_vector(streams=str(stream_raster), d8_pntr=str(flow_pointer), output=str(shp_path))
            if not shp_path.exists():
                _write_empty_streams_gpkg(gpkg_path, ptr_crs, threshold, "RasterStreamsToVector produced no output")
                per_threshold_vectors.append(gpkg_path)
                continue

            gdf = gpd.read_file(shp_path)
            if gdf.crs is None and ptr_crs is not None:
                gdf = gdf.set_crs(ptr_crs)
            if gdf.empty:
                _write_empty_streams_gpkg(gpkg_path, ptr_crs, threshold, "no vectors after read")
                per_threshold_vectors.append(gpkg_path)
                continue
            gdf = gdf.explode(ignore_index=True)
            gdf["thr_m2"] = threshold

            midpoint_coords = [geom.interpolate(0.5, normalized=True).coords[0] for geom in gdf.geometry]
            end_coords = [geom.coords[-1] for geom in gdf.geometry]
            order_vals = _sample_raster(order_raster, midpoint_coords)
            acc_vals = _sample_raster(flow_accum_m2, end_coords)

            gdf["order"] = [int(v) if not math.isnan(v) else -1 for v in order_vals]
            gdf["acc_m2"] = acc_vals
            gdf["geometry"] = [LineString(geom.coords) for geom in gdf.geometry]
            gdf["geometry"] = [_snap_line_endpoints(geom, snap_tolerance_m) for geom in gdf.geometry]
            gdf["length_m"] = [geom.length for geom in gdf.geometry]
            gdf = gdf[gdf["length_m"] >= min_length_m]

            dissolved = _dissolve_lines(gdf)
            if dissolved.crs is None and ptr_crs is not None:
                dissolved = dissolved.set_crs(ptr_crs)
            if dissolved.empty:
                _write_empty_streams_gpkg(gpkg_path, ptr_crs, threshold, "no dissolved features")
                per_threshold_vectors.append(gpkg_path)
                continue
            dissolved["thr_m2"] = threshold
            dissolved["order"] = [int(val) for val in dissolved["order"]]
            dissolved["geometry"] = [_snap_line_endpoints(geom, snap_tolerance_m) for geom in dissolved.geometry]
            dissolved["length_m"] = [geom.length for geom in dissolved.geometry]
            downstream_coords = [geom.coords[-1] for geom in dissolved.geometry]
            dissolved["acc_m2"] = _sample_raster(flow_accum_m2, downstream_coords)
            dissolved = dissolved[dissolved["length_m"] >= min_length_m]

            dissolved = dissolved.reset_index(drop=True)
            dissolved.to_file(gpkg_path, layer="streams", driver="GPKG")

            per_threshold_vectors.append(gpkg_path)
        finally:
            _cleanup_stream_scratch(tag, streams_dir, scratch_paths)

    return per_threshold_vectors


def derive_drainage(
    dem_paths: Sequence[Path],
    crs_obj: CRS | int | str,
    output_dir: Path,
    cache_dir: Path,
    stream_thresholds_m2: Sequence[float],
    min_stream_length_m: float,
    breach_search_dist: float,
    breach_max_cost: float | None,
    breach_flat_increment: float | None,
    fill_flat_increment: float | None,
    snap_tolerance_m: float,
    fill_max_depth: float | None,
    sea_ring_cells: int,
    sea_level_offset: float,
    sea_mask: Path | None = None,
    sea_collar_cells: int = 0,
    threads: int | None = None,
) -> Dict:
    """
    Hydrologic conditioning followed by global flow routing and stream extraction on a single merged DEM.
    """
    crs_checked = _require_projected_crs(crs_obj)
    hydro_dir = cache_dir / "hydro"
    hydro_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dem_mosaic_path, mosaic_profile = _prepare_mosaic(dem_paths, hydro_dir / "dem_mosaic.tif")
    mosaic_crs = mosaic_profile.get("crs")
    if mosaic_crs is None:
        raise ValueError("DEM mosaic lacks a CRS; reproject inputs to a projected CRS before deriving drainage.")
    mosaic_crs_checked = _require_projected_crs(mosaic_crs)
    if mosaic_crs_checked != crs_checked:
        LOG.warning("DEM mosaic CRS %s differs from requested CRS %s; proceeding with mosaic CRS for processing", mosaic_crs_checked, crs_checked)
    base_height = mosaic_profile["height"]
    base_width = mosaic_profile["width"]
    base_transform = mosaic_profile["transform"]

    dem_for_conditioning = dem_mosaic_path
    dem_with_outlets: Path | None = None
    sea_level: float | None = None
    collar_pad: int = 0
    collared_dem_path: Path | None = None

    if sea_collar_cells > 0:
        dem_for_conditioning, sea_level, collar_pad = _pad_dem_with_sea_collar(
            dem_path=dem_mosaic_path,
            output_path=hydro_dir / "dem_with_collar.tif",
            collar_cells=sea_collar_cells,
            sea_level_offset=sea_level_offset,
        )
        collared_dem_path = dem_for_conditioning

    if sea_ring_cells > 0 or (sea_mask is not None and sea_mask.exists()):
        dem_with_outlets, sea_level = _impose_outlets(
            dem_path=dem_for_conditioning,
            output_path=hydro_dir / "dem_with_outlets.tif",
            ring_cells=sea_ring_cells,
            sea_level_offset=sea_level_offset,
            sea_mask=sea_mask,
        )
        dem_for_conditioning = dem_with_outlets
    fill_depth_limit = None if fill_max_depth is None or fill_max_depth <= 0 else fill_max_depth

    wbt = _build_wbt(hydro_dir, threads)

    breached_path = hydro_dir / "dem_breached.tif"
    conditioned_tmp_path = hydro_dir / "dem_conditioned_full.tif"

    wbt.breach_depressions_least_cost(
        dem=str(dem_for_conditioning),
        output=str(breached_path),
        dist=int(breach_search_dist),
        max_cost=int(breach_max_cost) if breach_max_cost is not None else None,
        min_dist=True,
        flat_increment=breach_flat_increment,
        fill=False,
    )
    _require_created(breached_path, "Global BreachDepressionsLeastCost output")

    wbt.fill_depressions(
        dem=str(breached_path),
        output=str(conditioned_tmp_path),
        fix_flats=True,
        flat_increment=fill_flat_increment,
        max_depth=fill_depth_limit,
    )
    _require_created(conditioned_tmp_path, "Global FillDepressions output")

    # Final conditioned DEM written to output_dir with NaN nodata.
    with rasterio.open(conditioned_tmp_path) as src_cond, rasterio.open(dem_for_conditioning) as src_raw:
        cond_arr = src_cond.read(1).astype("float32")
        raw_arr = src_raw.read(1).astype("float32")
        cond_profile = src_cond.profile.copy()
        nodata_vals: list[float] = []
        if src_cond.nodata is not None:
            nodata_vals.append(float(src_cond.nodata))
        if src_raw.nodata is not None:
            nodata_vals.append(float(src_raw.nodata))
        nodata_vals.extend([-32768.0, -3.4028235e38])
        for nd in nodata_vals:
            cond_arr = np.where(np.isclose(cond_arr, nd), np.nan, cond_arr)
            raw_arr = np.where(np.isclose(raw_arr, nd), np.nan, raw_arr)
        cond_arr = np.where(~np.isfinite(cond_arr), np.nan, cond_arr)
        raw_arr = np.where(~np.isfinite(raw_arr), np.nan, raw_arr)
        cond_arr = np.where(np.isfinite(raw_arr), cond_arr, np.nan)

    if collar_pad > 0:
        cond_arr = cond_arr[collar_pad:-collar_pad, collar_pad:-collar_pad]
        raw_arr = raw_arr[collar_pad:-collar_pad, collar_pad:-collar_pad]
        cond_profile.update(
            {
                "height": base_height,
                "width": base_width,
                "transform": base_transform,
            }
        )

    conditioned_dem_path = output_dir / "dem_conditioned.tif"
    cond_profile.update({"nodata": np.nan, "dtype": "float32"})
    diff_arr = cond_arr - raw_arr
    diff_arr = np.where(~np.isfinite(raw_arr), np.nan, diff_arr)
    if fill_depth_limit and fill_depth_limit > 0:
        diff_arr = np.where(diff_arr > fill_depth_limit, fill_depth_limit, diff_arr)
        cond_arr = raw_arr + diff_arr

    _write_raster(conditioned_dem_path, cond_arr, cond_profile, transform=cond_profile["transform"])

    valid_mask = np.isfinite(raw_arr)
    if np.count_nonzero(valid_mask) == 0:
        max_fill_depth = 0.0
        max_breach_depth = 0.0
        percent_modified = 0.0
    else:
        diff_valid = diff_arr[valid_mask]
        max_fill_depth = float(np.nanmax(diff_valid)) if diff_valid.size else 0.0
        min_diff = np.nanmin(diff_valid) if diff_valid.size else 0.0
        max_breach_depth = float(-min_diff) if min_diff < 0 else 0.0
        percent_modified = float(np.count_nonzero(diff_valid != 0) * 100.0 / diff_valid.size)
    global_stats = {
        "max_fill_depth": max_fill_depth,
        "max_breach_depth": max_breach_depth,
        "percent_modified": percent_modified,
    }

    wbt = _build_wbt(hydro_dir, threads)

    flow_pointer_path = hydro_dir / "flow_pointer.tif"
    flow_accum_m2_path = hydro_dir / "flow_accum_m2.tif"
    flow_accum_cells_path = hydro_dir / "flow_accum_cells.tif"

    wbt.d8_pointer(dem=str(conditioned_dem_path), output=str(flow_pointer_path), esri_pntr=False)
    _require_created(flow_pointer_path, "D8Pointer output")
    # Use D-Infinity accumulation directly in catchment-area (m^2) units to avoid cell-area conversion errors.
    wbt.d_inf_flow_accumulation(
        i=str(conditioned_dem_path),
        output=str(flow_accum_m2_path),
        out_type="ca",
        log=False,
        clip=False,
        pntr=False,
    )
    _require_created(flow_accum_m2_path, "DInfFlowAccumulation output")

    with rasterio.open(flow_accum_m2_path) as src:
        acc_m2 = src.read(1).astype("float64")
        cell_area_m2 = _compute_cell_area_m2(src.transform, src.crs)
        acc_cells = np.where(np.isfinite(acc_m2), acc_m2 / cell_area_m2, np.nan)
        acc_profile = src.profile.copy()
        acc_profile.update({"dtype": "float32", "nodata": 0.0})
    _write_raster(flow_accum_cells_path, acc_cells.astype("float32"), acc_profile, transform=acc_profile["transform"])

    streams_cache_dir = hydro_dir / "streams"
    per_threshold_vectors = _extract_streams(
        flow_accum_cells=flow_accum_cells_path,
        flow_accum_m2=flow_accum_m2_path,
        flow_pointer=flow_pointer_path,
        thresholds_m2=stream_thresholds_m2,
        min_length_m=min_stream_length_m,
        snap_tolerance_m=snap_tolerance_m,
        streams_dir=streams_cache_dir,
        output_streams_dir=output_dir,
        threads=threads,
    )

    return {
        "dem_mosaic": dem_mosaic_path,
        "dem_with_outlets": dem_with_outlets if dem_with_outlets else None,
        "conditioned_dem": conditioned_dem_path,
        "flow_pointer": flow_pointer_path,
        "flow_accum_cells": flow_accum_cells_path,
        "flow_accum_m2": flow_accum_m2_path,
        "per_threshold_vectors": per_threshold_vectors,
        "global_stats": global_stats,
        "cell_area_m2": cell_area_m2,
        "transform": cond_profile["transform"],
        "crs": mosaic_crs_checked,
        "collared_dem": collared_dem_path,
        "sea_level": sea_level,
        "sea_ring_cells": sea_ring_cells if sea_ring_cells > 0 else None,
        "sea_collar_cells": sea_collar_cells if sea_collar_cells > 0 else None,
    }

# Backward compatibility for older imports.
derive_drainage_from_tiles = derive_drainage
