"""
AOI helpers for buffering and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.ops import unary_union


def read_aoi(aoi_path: Path, layer: str | None) -> tuple[gpd.GeoDataFrame, object]:
    """
    Read the AOI layer and return dissolved GeoDataFrame and geometry.
    """
    gdf = gpd.read_file(aoi_path, layer=layer)
    if gdf.empty:
        raise ValueError("AOI layer is empty")
    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("AOI layer must contain polygonal geometries")
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    geom = unary_union(gdf.geometry)
    dissolved = gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs)
    dissolved = dissolved.to_crs(epsg=4326)
    return dissolved, dissolved.geometry.iloc[0]


def buffer_aoi(geom, buffer_km: float, crs_epsg: int) -> Tuple[object, object]:
    """
    Buffer the AOI geometry in the processing CRS and return projected and WGS84 copies.
    """
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    gdf_proj = gdf.to_crs(epsg=crs_epsg)
    buffered = gdf_proj.buffer(buffer_km * 1000.0).iloc[0]
    buffered_proj = gdf_proj.copy()
    buffered_proj.geometry = [buffered]
    buffered_wgs84 = buffered_proj.to_crs(epsg=4326).geometry.iloc[0]
    return buffered_proj.geometry.iloc[0], buffered_wgs84
