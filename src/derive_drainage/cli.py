from __future__ import annotations

import argparse
import logging
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, box
from shapely.ops import transform

from derive_drainage.common.aoi import buffer_aoi, read_aoi
from derive_drainage.common.logging import configure_logging
from derive_drainage.common.metadata import write_metadata
from derive_drainage.process.dem import reproject_dem, tile_dem
from derive_drainage.stage.copdem import stage_copdem_glo30
from derive_drainage.stage.gdw import stage_gdw
from derive_drainage.stage.osm import stage_osm

LOG = logging.getLogger(__name__)

DEFAULT_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
DEFAULT_COLLECTION = "cop-dem-glo-30-dged-cog"
DEFAULT_GDW_URL = "https://figshare.com/ndownloader/articles/25988293/versions/1"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage CopDEM GLO-30 and GDW data for an AOI.")
    parser.add_argument("-aoi", required=True, type=Path, help="AOI geopackage path")
    parser.add_argument("-output", required=True, type=Path, help="Output directory")

    parser.add_argument("--aoi-layer", dest="aoi_layer", default=None)
    parser.add_argument("--buffer-km", dest="buffer_km", type=float, default=1.0)
    parser.add_argument("--crs", dest="crs", type=int, default=3034)
    parser.add_argument("--gdw-url", dest="gdw_url", default=DEFAULT_GDW_URL)
    parser.add_argument("--stac-url", dest="stac_url", default=DEFAULT_STAC_URL)
    parser.add_argument("--collection", dest="collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--keep-temp", dest="keep_temp", action="store_true")

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)

    LOG.info("Reading AOI")
    aoi_gdf, aoi_geom = read_aoi(args.aoi, args.aoi_layer)
    crs_obj = CRS.from_user_input(args.crs)
    if not crs_obj.is_projected:
        raise ValueError(f"--crs must be a projected CRS in meters; got {args.crs}")
    aoi_proj = aoi_gdf.to_crs(crs_obj)
    aoi_proj_geom = aoi_proj.geometry.iloc[0]
    _buffered_proj, buffered_wgs84 = buffer_aoi(aoi_geom, args.buffer_km, args.crs)

    temp_cm = nullcontext(output_dir / "temp") if args.keep_temp else tempfile.TemporaryDirectory()
    with temp_cm as tmp:
        tmpdir = Path(tmp)
        tmpdir.mkdir(parents=True, exist_ok=True)

        dem_reproj = cache_dir / "copdem_reprojected.tif"
        tiles_dir = cache_dir / "dem_tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        existing_tiles = sorted(tiles_dir.glob("*.tif"))
        if existing_tiles:
            LOG.info("Found %d DEM tiles in cache; skipping CopDEM staging and tiling.", len(existing_tiles))
            stac_started = None
            copdem_info = {"item_ids": None, "asset_names": None, "clipped": None, "skipped_cache": True}
            tiles = existing_tiles
        else:
            LOG.info("Staging CopDEM via STAC")
            stac_started = datetime.utcnow().isoformat() + "Z"
            copdem_info = stage_copdem_glo30(buffered_wgs84, args.stac_url, args.collection, tmpdir, cache_dir)

            LOG.info("Reprojecting CopDEM to CRS %s", args.crs)
            reproject_dem(copdem_info["clipped"], dem_reproj, args.crs)

            LOG.info("Generating DEM tiles (10x10 km with 2.5 km overlap) into %s", tiles_dir)
            tiles = tile_dem(
                dem_path=dem_reproj,
                aoi_geom_proj=aoi_proj_geom,
                out_dir=tiles_dir,
                tile_size_m=10_000.0,
                overlap_m=2_500.0,
            )

        LOG.info("Staging GDW dams")
        gdw_started = datetime.utcnow().isoformat() + "Z"
        gdw_info = stage_gdw(buffered_wgs84, args.gdw_url, tmpdir, cache_dir)

        LOG.info("Staging OSM features per DEM tile (highway, railway, building)")
        osm_started = datetime.utcnow().isoformat() + "Z"
        osm_tags = {"highway": True, "railway": True, "building": True}

        osm_tiles_dir = cache_dir / "osm_tiles"
        osm_tiles_dir.mkdir(parents=True, exist_ok=True)
        osm_tile_paths: list[Path] = []
        to_wgs84 = Transformer.from_crs(crs_obj, CRS.from_epsg(4326), always_xy=True).transform

        def _normalize_geom(geom):
            if geom is None or geom.is_empty:
                return []
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                return []
            gtype = geom.geom_type
            if gtype == "Polygon":
                return [MultiPolygon([geom])]
            if gtype == "MultiPolygon":
                return [geom]
            if gtype == "LineString":
                return [geom]
            if gtype == "MultiLineString":
                return [ls for ls in geom.geoms if ls is not None and not ls.is_empty]
            return []

        for tile_path in tiles:
            with rasterio.open(tile_path) as src:
                tb = src.bounds
            tile_geom = box(tb.left, tb.bottom, tb.right, tb.top)
            tile_geom_wgs = transform(to_wgs84, tile_geom)
            minx, miny, maxx, maxy = tile_geom_wgs.bounds
            osm_tile_gdf = stage_osm((minx, miny, maxx, maxy), cache_dir, osm_tags)
            if osm_tile_gdf.empty:
                continue
            osm_tile_proj = osm_tile_gdf.to_crs(args.crs)
            clipped = osm_tile_proj.clip(tile_geom)
            norm_geoms = []
            for geom in clipped.geometry:
                norm_geoms.extend(_normalize_geom(geom))
            norm_geoms = [
                g for g in norm_geoms if g.geom_type in ("MultiPolygon", "LineString")
            ]
            if not norm_geoms:
                continue
            out_gdf = gpd.GeoDataFrame(geometry=norm_geoms, crs=osm_tile_proj.crs)
            out_path = osm_tiles_dir / f"{tile_path.stem}.gpkg"
            out_gdf.to_file(out_path, layer="osm", driver="GPKG")
            osm_tile_paths.append(out_path)

        param_dict = {}
        for key, value in vars(args).items():
            param_dict[key] = str(value) if isinstance(value, Path) else value

        metadata = {
            "stac": {
                "endpoint": args.stac_url,
                "collection": args.collection,
                "item_ids": copdem_info.get("item_ids"),
                "asset_names": copdem_info.get("asset_names"),
                "timestamp": stac_started,
                "skipped_cache": copdem_info.get("skipped_cache", False),
            },
            "gdw": {
                "url": args.gdw_url,
                "feature_count": gdw_info.get("feature_count"),
                "timestamp": gdw_started,
            },
            "osm": {
                "feature_count": len(osm_tile_paths),
                "timestamp": osm_started,
            },
            "parameters": param_dict,
            "outputs": {
                "log": str(output_dir / "processing.log"),
                "reprojected_dem": str(dem_reproj) if dem_reproj.exists() else None,
                "dem_tiles": [str(p) for p in tiles],
                "osm_tiles": [str(p) for p in osm_tile_paths],
            },
        }
        write_metadata(output_dir, metadata)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
