from __future__ import annotations

import argparse
import logging
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
from pyproj import CRS

from derive_drainage.common.aoi import buffer_aoi, read_aoi
from derive_drainage.common.logging import configure_logging
from derive_drainage.common.metadata import write_metadata
from derive_drainage.process.dem import clip_tiles_to_core, erase_features_from_dem_tiles, fill_tile_nodata_natural_neighbor, reproject_dem, tile_dem
from derive_drainage.stage.copdem import stage_copdem_glo30
from derive_drainage.stage.gdw import stage_gdw
from derive_drainage.stage.osm import stage_osm_tiles_for_dem

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
        tile_size_m = 10_000.0
        overlap_m = 2_500.0

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
                tile_size_m=tile_size_m,
                overlap_m=overlap_m,
            )

        LOG.info("Staging GDW dams")
        gdw_started = datetime.utcnow().isoformat() + "Z"
        gdw_info = stage_gdw(buffered_wgs84, args.gdw_url, tmpdir, cache_dir)
        gdw_gdf_proj = gdw_info["gdf"].to_crs(crs_obj) if not gdw_info["gdf"].empty else gdw_info["gdf"]
        reservoir_gdf = gdw_info.get("reservoir_gdf")
        reservoir_gdf_proj = (
            reservoir_gdf.to_crs(crs_obj) if reservoir_gdf is not None and not reservoir_gdf.empty else gpd.GeoDataFrame(geometry=[], crs=crs_obj)
        )

        LOG.info("Staging OSM features per DEM tile (highway, railway, building)")
        osm_started = datetime.utcnow().isoformat() + "Z"
        osm_tags = {"highway": True, "railway": True, "building": True}

        osm_tiles_dir = cache_dir / "osm_tiles"
        osm_tile_paths = stage_osm_tiles_for_dem(
            tile_paths=tiles,
            dem_crs=crs_obj,
            cache_dir=cache_dir,
            tags=osm_tags,
            out_dir=osm_tiles_dir,
        )

        param_dict = {}
        for key, value in vars(args).items():
            param_dict[key] = str(value) if isinstance(value, Path) else value

        LOG.info("Erasing OSM/GDW features from DEM tiles")
        erase_features_from_dem_tiles(
            tile_paths=tiles,
            crs_obj=crs_obj,
            osm_tiles=osm_tile_paths,
            gdw_gdf_proj=gdw_gdf_proj,
            reservoir_gdf_proj=reservoir_gdf_proj,
        )

        LOG.info("Filling erased voids via natural neighbor interpolation")
        fill_tile_nodata_natural_neighbor(tile_paths=tiles)

        LOG.info("Clipping DEM tiles to core (10x10 km) by removing overlap")
        clip_tiles_to_core(tile_paths=tiles, overlap_m=overlap_m)

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
