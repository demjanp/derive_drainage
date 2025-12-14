"""
Stage Copernicus DEM GLO-30 via Copernicus Data Space STAC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import rasterio
import time
import requests
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

LOG = logging.getLogger(__name__)


def _download_asset(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def _simplify_geom_for_stac(geom, tolerance: float = 0.001) -> BaseGeometry:
    """
    Simplify input geometry to reduce STAC payload size.
    """
    shp = geom if isinstance(geom, BaseGeometry) else shapely_shape(geom)
    unioned = unary_union(shp)
    simplified = unioned.simplify(tolerance, preserve_topology=True)
    return simplified


def _search_stac(stac_url: str, collection: str, geom, retries: int = 3, backoff: float = 5.0) -> List[dict]:
    search_url = stac_url.rstrip("/") + "/search"
    safe_geom = _simplify_geom_for_stac(geom)
    payload = {
        "collections": [collection],
        "intersects": mapping(safe_geom),
        "limit": 200,
    }
    features: List[dict] = []
    next_url = search_url
    body = payload
    headers = {"User-Agent": "derive-drainage/1.0", "Accept": "application/json"}
    while next_url:
        for attempt in range(retries + 1):
            if body is not None:
                resp = requests.post(next_url, json=body, timeout=90, headers=headers)
            else:
                resp = requests.get(next_url, timeout=90, headers=headers)
            if resp.status_code != 429:
                resp.raise_for_status()
                break
            if attempt == retries:
                resp.raise_for_status()
            sleep_for = backoff * (2 ** attempt)
            LOG.warning("STAC query rate limited (429). Retrying in %.1f seconds...", sleep_for)
            time.sleep(sleep_for)

        data = resp.json()
        features.extend(data.get("features", []))
        next_link = next((lnk for lnk in data.get("links", []) if lnk.get("rel") == "next"), None)
        if next_link:
            next_url = next_link.get("href")
            body = None
        else:
            next_url = None
    return features


def _resolve_href(href: str) -> str:
    """
    Normalize CopDEM s3 hrefs to publicly reachable HTTPS.
    """
    prefix = "s3://eodata/auxdata/CopDEM_COG/copernicus-dem-30m/"
    if href.startswith(prefix):
        tail = href[len(prefix) :]
        return f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{tail}"
    return href


def stage_copdem_glo30(
    aoi_geom_4326, stac_url: str, collection: str, tmpdir: Path, cache_dir: Path
) -> Dict:
    """
    Query STAC via HTTP, download intersecting COG assets (cached), mosaic, and clip to buffered AOI.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = _search_stac(stac_url, collection, aoi_geom_4326)

    asset_paths: List[Path] = []
    item_ids: List[str] = []
    asset_names: List[str] = []

    for idx, item in enumerate(items):
        assets = item.get("assets", {})
        asset_key = None
        for key, asset in assets.items():
            if not isinstance(asset, dict):
                continue
            media = str(asset.get("type", "")).lower()
            if "geotiff" in media:
                asset_key = key
                break
        if asset_key is None:
            for key, asset in assets.items():
                if isinstance(asset, dict):
                    asset_key = key
                    break
        if asset_key is None:
            continue
        asset = assets[asset_key]
        if not isinstance(asset, dict):
            continue
        href = asset.get("href")
        if href is None:
            continue
        href = _resolve_href(href)
        item_ids.append(item.get("id", f"item_{idx}"))
        asset_names.append(asset_key)
        cached_path = cache_dir / f"{item_ids[-1]}_{asset_key}.tif"
        if not cached_path.exists():
            _download_asset(href, cached_path)
        asset_paths.append(cached_path)

    if not asset_paths:
        raise RuntimeError("No CopDEM assets found for AOI")

    mosaic_path = Path(tmpdir) / "copdem_mosaic.tif"
    with rasterio.Env():
        datasets = [rasterio.open(p) for p in asset_paths]
        mosaic, out_trans = merge(datasets)
        out_meta = datasets[0].meta.copy()
        out_meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
            }
        )
        for ds in datasets:
            ds.close()
        with rasterio.open(mosaic_path, "w", **out_meta) as dst:
            dst.write(mosaic)

    clipped_path = Path(tmpdir) / "copdem_clipped.tif"
    with rasterio.open(mosaic_path) as src:
        geom = [mapping(aoi_geom_4326)]
        out_image, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        with rasterio.open(clipped_path, "w", **out_meta) as dst:
            dst.write(out_image)

    return {
        "downloaded": asset_paths,
        "mosaic": mosaic_path,
        "clipped": clipped_path,
        "item_ids": item_ids,
        "asset_names": asset_names,
    }
