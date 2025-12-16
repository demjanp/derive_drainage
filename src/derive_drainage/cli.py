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
from derive_drainage.process.hydro import derive_drainage
from derive_drainage.stage.copdem import stage_copdem_glo30

LOG = logging.getLogger(__name__)


def _beep() -> None:
    """
    Emit an audible notification when processing finishes.
    Uses winsound on Windows; falls back to ASCII bell otherwise.
    """
    try:
        import winsound

        winsound.Beep(1200, 500)  # frequency Hz, duration ms
        return
    except Exception:
        pass
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass

DEFAULT_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
DEFAULT_COLLECTION = "cop-dem-glo-30-dged-cog"


def _parse_thresholds_m2(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one stream threshold (m2) is required.")
    return [float(p) for p in parts]


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
    # Defaults tuned for CopDEM to yield a fuller network on small/medium AOIs (units: square meters).
    # Lowered thresholds to better sustain continuous valleys when using D-Infinity accumulation (flow is split).
    parser.add_argument("--stream-thresholds-m2", dest="stream_thresholds_m2", default="5000,20000", help="Comma-separated contributing-area thresholds (m2) used to extract stream networks")
    parser.add_argument("--min-stream-length-m", dest="min_stream_length_m", type=float, default=100.0, help="Minimum stream length to retain (meters)")
    # More aggressive defaults to reduce flow cutoffs: longer search and higher cost ceiling.
    parser.add_argument("--breach-search-dist", dest="breach_search_dist", type=float, default=30.0, help="Search distance parameter for BreachDepressionsLeastCost (map units)")
    parser.add_argument("--breach-max-cost", dest="breach_max_cost", type=float, default=1000.0, help="Maximum breach cost for BreachDepressionsLeastCost")
    parser.add_argument("--breach-flat-increment", dest="breach_flat_increment", type=float, default=0.05, help="Flat increment used for breaching")
    parser.add_argument("--fill-flat-increment", dest="fill_flat_increment", type=float, default=0.001, help="Flat increment used for depression filling")
    # Limit fill depth by default to avoid excessive overfilling; set 0 to remove the cap.
    parser.add_argument("--fill-max-depth", dest="fill_max_depth", type=float, default=5.0, help="Optional maximum fill depth (map units). Use 0 to remove the cap.")
    parser.add_argument("--sea-ring-cells", dest="sea_ring_cells", type=int, default=0, help="Boundary ring width (cells) to force to a low sea level to provide outlets (0 disables).")
    parser.add_argument("--sea-level-offset", dest="sea_level_offset", type=float, default=50.0, help="Meters below the DEM minimum to set the boundary ring/sea mask (positive value).")
    parser.add_argument("--sea-mask", dest="sea_mask", type=Path, default=None, help="Optional polygon mask representing ocean/sea; cells under the mask are set to the low sea level.")
    parser.add_argument("--sea-collar-cells", dest="sea_collar_cells", type=int, default=2, help="Pad the DEM with a low-elevation collar of this width (cells) to guarantee outlets beyond the clipped extent (0 disables).")
    parser.add_argument("--snap-tolerance-m", dest="snap_tolerance_m", type=float, default=60.0, help="Endpoint snapping tolerance in meters for stream vectors")
    parser.add_argument("--threads", dest="threads", type=int, default=None, help="Maximum WhiteboxTools worker threads (default: tool-defined)")

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)

    stream_thresholds_m2 = _parse_thresholds_m2(args.stream_thresholds_m2)

    LOG.info("Reading AOI")
    aoi_gdf, aoi_geom = read_aoi(args.aoi, args.aoi_layer)
    crs_obj = CRS.from_user_input(args.crs)
    if not crs_obj.is_projected:
        raise ValueError(f"--crs must be a projected CRS in meters; got {args.crs}")
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

        dem_inputs = [dem_reproj]

        param_dict = {}
        for key, value in vars(args).items():
            param_dict[key] = str(value) if isinstance(value, Path) else value
        param_dict["stream_thresholds_m2"] = stream_thresholds_m2

        LOG.info("Generating drainage network with thresholds (m2): %s", stream_thresholds_m2)
        drainage = derive_drainage(
            dem_paths=dem_inputs,
            crs_obj=crs_obj,
            output_dir=output_dir,
            cache_dir=cache_dir,
            stream_thresholds_m2=stream_thresholds_m2,
            min_stream_length_m=args.min_stream_length_m,
            breach_search_dist=args.breach_search_dist,
            breach_max_cost=args.breach_max_cost,
            breach_flat_increment=args.breach_flat_increment,
            fill_flat_increment=args.fill_flat_increment,
            fill_max_depth=args.fill_max_depth,
            sea_ring_cells=args.sea_ring_cells,
            sea_level_offset=args.sea_level_offset,
            sea_mask=args.sea_mask,
            sea_collar_cells=args.sea_collar_cells,
            snap_tolerance_m=args.snap_tolerance_m,
            threads=args.threads,
        )

        per_threshold_streams = drainage.get("per_threshold_vectors", [])
        if per_threshold_streams:
            LOG.info("Wrote stream vectors: %s", ", ".join(str(p) for p in per_threshold_streams))
        else:
            LOG.warning("No stream vectors were generated.")

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
                "conditioned_dem": str(drainage.get("conditioned_dem")) if drainage.get("conditioned_dem") else None,
                "stream_vectors": [str(p) for p in per_threshold_streams],
                "sea_level": drainage.get("sea_level"),
                "sea_ring_cells": drainage.get("sea_ring_cells"),
                "sea_collar_cells": drainage.get("sea_collar_cells"),
            },
            "global_stats": drainage.get("global_stats"),
        }
        write_metadata(output_dir, metadata)
        # Audible cue when processing completes.
        _beep()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
