from __future__ import annotations

import argparse
import logging
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyproj import CRS

from derive_drainage.common.aoi import buffer_aoi, read_aoi
from derive_drainage.common.logging import configure_logging
from derive_drainage.common.metadata import write_metadata
from derive_drainage.process.dem import reproject_dem
from derive_drainage.process.hydro import derive_drainage_from_tiles
from derive_drainage.process.vectorize import stream_mask_to_lines
from derive_drainage.stage.copdem import stage_copdem_glo30

LOG = logging.getLogger(__name__)

DEFAULT_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
DEFAULT_COLLECTION = "cop-dem-glo-30-dged-cog"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage CopDEM GLO-30 data and derive drainage network for an AOI.")
    parser.add_argument("-aoi", required=True, type=Path, help="AOI geopackage path")
    parser.add_argument("-output", required=True, type=Path, help="Output directory")

    parser.add_argument("--aoi-layer", dest="aoi_layer", default=None)
    parser.add_argument("--buffer-km", dest="buffer_km", type=float, default=1.0)
    parser.add_argument("--crs", dest="crs", type=int, default=3034)
    parser.add_argument("--stac-url", dest="stac_url", default=DEFAULT_STAC_URL)
    parser.add_argument("--collection", dest="collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--keep-temp", dest="keep_temp", action="store_true")
    parser.add_argument("--stream-accum-threshold", dest="stream_accum_threshold", type=int, default=400, help="Flow accumulation threshold (cells) to start streams")
    parser.add_argument("--min-stream-length-m", dest="min_stream_length_m", type=float, default=100.0, help="Minimum stream length to retain (meters)")

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
        if dem_reproj.exists():
            LOG.info("Found reprojected DEM in cache; skipping CopDEM staging.")
            stac_started = None
            copdem_info = {"item_ids": None, "asset_names": None, "clipped": None, "skipped_cache": True}
        else:
            LOG.info("Staging CopDEM via STAC")
            stac_started = datetime.utcnow().isoformat() + "Z"
            copdem_info = stage_copdem_glo30(buffered_wgs84, args.stac_url, args.collection, tmpdir, cache_dir)

            LOG.info("Reprojecting CopDEM to CRS %s", args.crs)
            reproject_dem(copdem_info["clipped"], dem_reproj, args.crs)

        tiles = [dem_reproj]

        param_dict = {}
        for key, value in vars(args).items():
            param_dict[key] = str(value) if isinstance(value, Path) else value

        LOG.info("Generating drainage network from DEM tiles")
        drainage = derive_drainage_from_tiles(
            tile_paths=tiles,
            crs_obj=crs_obj,
            output_dir=output_dir,
            stream_threshold_cells=args.stream_accum_threshold,
            min_stream_length_m=args.min_stream_length_m,
        )

        streams_path = output_dir / "streams.gpkg"
        streams_gdf = stream_mask_to_lines(
            stream_mask=drainage["stream_mask"],
            transform=drainage["transform"],
            crs_obj=drainage["crs"],
            flow_accum=drainage["flow_accum_array"],
            cell_area=drainage["cell_area"],
            min_length_m=args.min_stream_length_m,
        )
        streams_written = None
        if streams_gdf.empty:
            LOG.warning("No stream segments found at threshold %d cells", args.stream_accum_threshold)
        else:
            streams_path.parent.mkdir(parents=True, exist_ok=True)
            streams_gdf.to_file(streams_path, layer="streams", driver="GPKG")
            streams_written = streams_path
            LOG.info("Wrote %d stream segments to %s", len(streams_gdf), streams_path)

        metadata = {
            "stac": {
                "endpoint": args.stac_url,
                "collection": args.collection,
                "item_ids": copdem_info.get("item_ids"),
                "asset_names": copdem_info.get("asset_names"),
                "timestamp": stac_started,
                "skipped_cache": copdem_info.get("skipped_cache", False),
            },
            "parameters": param_dict,
            "outputs": {
                "log": str(output_dir / "processing.log"),
                "reprojected_dem": str(dem_reproj) if dem_reproj.exists() else None,
                "dem_paths": [str(p) for p in tiles],
                "dem_mosaic": str(drainage.get("dem_mosaic")) if drainage.get("dem_mosaic") else None,
                "dem_filled": str(drainage.get("filled_dem")) if drainage.get("filled_dem") else None,
                "flow_direction": str(drainage.get("flow_dir")) if drainage.get("flow_dir") else None,
                "flow_accumulation": str(drainage.get("flow_accum")) if drainage.get("flow_accum") else None,
                "streams": str(streams_written) if streams_written else None,
            },
        }
        write_metadata(output_dir, metadata)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
