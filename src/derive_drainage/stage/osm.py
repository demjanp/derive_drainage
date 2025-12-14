"""
Stage OSM features for a bounding box using osmnx.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd

LOG = logging.getLogger(__name__)


def _configure_osmnx(cache_dir: Path) -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.log_console = False
    ox.settings.timeout = 120


def stage_osm(bbox_4326: Tuple[float, float, float, float], cache_dir: Path, tags: Dict[str, object]) -> gpd.GeoDataFrame:
    """
    Fetch OSM features intersecting bbox (minx, miny, maxx, maxy) using provided tags.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    overpass_cache = cache_dir / "osm_cache"
    overpass_cache.mkdir(parents=True, exist_ok=True)
    _configure_osmnx(overpass_cache)

    minx, miny, maxx, maxy = bbox_4326  # west, south, east, north
    west, south, east, north = minx, miny, maxx, maxy
    fetch = getattr(ox.features, "features_from_bbox", None)
    if not callable(fetch):
        fetch = getattr(ox, "features_from_bbox", None)
    if not callable(fetch):
        raise RuntimeError("osmnx.features_from_bbox is unavailable; please upgrade osmnx.")

    def bbox_area_km2(north_val: float, south_val: float, east_val: float, west_val: float) -> float:
        lat_km = abs(north_val - south_val) * 111.32
        mean_lat = 0.5 * (north_val + south_val)
        lon_km = abs(east_val - west_val) * 111.32 * max(math.cos(math.radians(mean_lat)), 0.01)
        return lat_km * lon_km

    area_km2 = bbox_area_km2(north, south, east, west)
    target_km2 = 5.0
    subdivisions = max(1, math.ceil(math.sqrt(area_km2 / target_km2))) if area_km2 > target_km2 else 1

    lat_step = (north - south) / subdivisions
    lon_step = (east - west) / subdivisions
    LOG.info(
        "OSM bbox area %.2f km^2 split into %d sub-queries (target %.1f km^2)",
        area_km2,
        subdivisions,
        target_km2,
    )

    fragments: list[gpd.GeoDataFrame] = []
    for i in range(subdivisions):
        sub_s = south + i * lat_step
        sub_n = north if i == subdivisions - 1 else sub_s + lat_step
        for j in range(subdivisions):
            sub_w = west + j * lon_step
            sub_e = east if j == subdivisions - 1 else sub_w + lon_step
            gdf_sub = fetch((sub_w, sub_s, sub_e, sub_n), tags)
            if gdf_sub is None or getattr(gdf_sub, "empty", True):
                continue
            fragments.append(gdf_sub)

    if not fragments:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(pd.concat(fragments, ignore_index=True), crs=fragments[0].crs)

    if gdf is None or getattr(gdf, "empty", True):
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf)

    if not isinstance(getattr(gdf, "geometry", None), gpd.GeoSeries):
        geom_col = None
        for candidate in ("geometry", "geom", "the_geom", "wkb_geometry"):
            if candidate in gdf.columns:
                geom_col = candidate
                break
        if geom_col:
            gdf = gdf.set_geometry(geom_col)

    gdf = gdf[gdf.geometry.notna()].copy()

    def _fix_geom(geom):
        if geom is None or geom.is_empty:
            return None
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            return None
        return geom

    gdf["geometry"] = gdf.geometry.map(_fix_geom)
    gdf = gdf[gdf.geometry.notna()].copy()

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf.reset_index(drop=True)
