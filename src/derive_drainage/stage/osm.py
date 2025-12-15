"""
Stage OSM features for a bounding box using osmnx.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Tuple, List, Iterable

import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterio
from tqdm import tqdm
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, box
from shapely.ops import transform
from osmnx._errors import InsufficientResponseError

LOG = logging.getLogger(__name__)


def _configure_osmnx(cache_dir: Path) -> None:
	ox.settings.use_cache = True
	ox.settings.cache_folder = str(cache_dir)
	ox.settings.log_console = False
	ox.settings.timeout = 120


def stage_osm(bbox_4326: Tuple[float, float, float, float], cache_dir: Path, tags: Dict[str, object], tile_idx: int, n_tiles: int) -> gpd.GeoDataFrame:
	"""
	Fetch OSM features intersecting bbox (minx, miny, maxx, maxy) using provided tags.
	Keep roads with railway present or highway in {motorway, primary}, and keep buildings.
	"""
	cache_dir.mkdir(parents=True, exist_ok=True)
	overpass_cache = cache_dir / "osm_cache"
	overpass_cache.mkdir(parents=True, exist_ok=True)
	_configure_osmnx(overpass_cache)

	minx, miny, maxx, maxy = bbox_4326  # west, south, east, north
	west, south, east, north = minx, miny, maxx, maxy
	cache_key = f"osm_filtered_{west:.6f}_{south:.6f}_{east:.6f}_{north:.6f}.gpkg"
	cache_path = cache_dir / cache_key
	if cache_path.exists():
		return gpd.read_file(cache_path)

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
	target_km2 = 150.0
	subdivisions = max(1, math.ceil(math.sqrt(area_km2 / target_km2))) if area_km2 > target_km2 else 1

	lat_step = (north - south) / subdivisions
	lon_step = (east - west) / subdivisions
	fragments: list[gpd.GeoDataFrame] = []
	subtile_bboxes: list[Tuple[float, float, float, float]] = []
	for i in range(subdivisions):
		sub_s = south + i * lat_step
		sub_n = north if i == subdivisions - 1 else sub_s + lat_step
		for j in range(subdivisions):
			sub_w = west + j * lon_step
			sub_e = east if j == subdivisions - 1 else sub_w + lon_step
			subtile_bboxes.append((sub_w, sub_s, sub_e, sub_n))

	for sub_w, sub_s, sub_e, sub_n in tqdm(subtile_bboxes, desc="OSM tile %d/%d subqueries" % (tile_idx+1, n_tiles), unit="tile"):
		try:
			gdf_sub = fetch((sub_w, sub_s, sub_e, sub_n), tags)
		except InsufficientResponseError:
			LOG.info("No OSM features returned for sub-bbox (%.6f, %.6f, %.6f, %.6f); skipping", sub_w, sub_s, sub_e, sub_n)
			continue
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

	railway_col = gdf["railway"] if "railway" in gdf.columns else pd.Series(False, index=gdf.index)
	highway_col = gdf["highway"] if "highway" in gdf.columns else pd.Series("", index=gdf.index)
	building_col = gdf["building"] if "building" in gdf.columns else pd.Series("", index=gdf.index)
	mask_keep = railway_col.notna() | highway_col.isin(["motorway", "primary"]) | building_col.notna()
	gdf = gdf[mask_keep].copy()

	if gdf.empty:
		gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

	if gdf.crs is None:
		gdf = gdf.set_crs("EPSG:4326", allow_override=True)
	else:
		gdf = gdf.to_crs("EPSG:4326")

	gdf = gdf.reset_index(drop=True)
	# Only persist geometry to avoid schema issues from arbitrary OSM fields
	gdf = gdf[["geometry"]]
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	gdf.to_file(cache_path, driver="GPKG")
	return gdf


def _normalize_geometries(geoms: Iterable) -> list:
	"""
	Normalize mixed geometries into polygons/multipolygons and linestrings.
	"""
	normalized = []
	for geom in geoms:
		if geom is None or geom.is_empty:
			continue
		if not geom.is_valid:
			geom = geom.buffer(0)
		if geom.is_empty:
			continue
		gtype = geom.geom_type
		if gtype == "Polygon":
			normalized.append(MultiPolygon([geom]))
		elif gtype == "MultiPolygon":
			normalized.append(geom)
		elif gtype == "LineString":
			normalized.append(geom)
		elif gtype == "MultiLineString":
			normalized.extend([ls for ls in geom.geoms if ls is not None and not ls.is_empty])
	return [g for g in normalized if g.geom_type in ("MultiPolygon", "LineString")]


def stage_osm_tiles_for_dem(
	tile_paths: List[Path],
	dem_crs: CRS | int | str,
	cache_dir: Path,
	tags: Dict[str, object],
	out_dir: Path | None = None,
) -> List[Path]:
	"""
	Stage OSM data per DEM tile:
	- For each tile, convert bounds to WGS84, fetch OSM using provided tags.
	- Clip to tile extent, normalize geometries, and write per-tile GeoPackage.

	Returns list of written tile paths.
	"""
	out_dir = out_dir or cache_dir / "osm_tiles"
	out_dir.mkdir(parents=True, exist_ok=True)
	to_wgs84 = Transformer.from_crs(CRS.from_user_input(dem_crs), CRS.from_epsg(4326), always_xy=True).transform
	written: List[Path] = []

	for tile_idx, tile_path in enumerate(tile_paths):
		with rasterio.open(tile_path) as src:
			bounds = src.bounds
		tile_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
		tile_geom_wgs = transform(to_wgs84, tile_geom)
		minx, miny, maxx, maxy = tile_geom_wgs.bounds
		osm_tile_gdf = stage_osm((minx, miny, maxx, maxy), cache_dir, tags, tile_idx, len(tile_paths))
		if osm_tile_gdf.empty:
			continue
		osm_tile_proj = osm_tile_gdf.to_crs(CRS.from_user_input(dem_crs))
		clipped = osm_tile_proj.clip(tile_geom)
		normalized = _normalize_geometries(clipped.geometry)
		if not normalized:
			continue
		out_gdf = gpd.GeoDataFrame(geometry=normalized, crs=osm_tile_proj.crs)
		out_path = out_dir / f"{tile_path.stem}.gpkg"
		out_gdf.to_file(out_path, layer="osm", driver="GPKG")
		written.append(out_path)

	return written
