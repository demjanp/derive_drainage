"""
Stage Global Dam Watch data.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import zipfile
from pathlib import Path
from typing import Dict

import geopandas as gpd
import requests
from bs4 import BeautifulSoup
from shapely.geometry import mapping

LOG = logging.getLogger(__name__)


def _download_zip(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=90, allow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def _resolve_sync_redirect(url: str) -> str:
    """
    Resolve Sync.com shared link to direct download if possible.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for meta in soup.find_all("meta"):
        if meta.get("http-equiv", "").lower() == "refresh":
            content = meta.get("content", "")
            if "url=" in content.lower():
                target = content.split("=", 1)[-1].strip()
                if target.startswith("https://"):
                    return target
    return url


def stage_gdw(aoi_geom_4326, gdw_url: str, tmpdir: Path, cache_dir: Path) -> Dict:
    """
    Download, extract, load dams, clip to buffered AOI, return info.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_name = gdw_url.split("/")[-1] or "gdw.zip"
    if not zip_name.lower().endswith(".zip"):
        zip_name += ".zip"
    zip_path = cache_dir / zip_name
    extract_dir = Path(tmpdir) / "gdw"

    if not zip_path.exists():
        dl_url = _resolve_sync_redirect(gdw_url)
        _download_zip(dl_url, zip_path)
    # Validate zip
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Downloaded GDW file is not a zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    shp_archives = list(extract_dir.rglob("*shp.zip"))
    if shp_archives:
        nested_dir = extract_dir / "gdw_shp"
        nested_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(shp_archives[0], "r") as inner:
            inner.extractall(nested_dir)
        shp_files = list(nested_dir.rglob("*.shp"))
    else:
        shp_files = list(extract_dir.rglob("*.shp"))
    shp_path = shp_files[0]
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)
    clipped = gdf.clip(gpd.GeoDataFrame(geometry=[aoi_geom_4326], crs="EPSG:4326"))

    out_path = Path(tmpdir) / "gdw_clipped.gpkg"
    clipped.to_file(out_path, layer="dams", driver="GPKG")

    return {
        "path": out_path,
        "gdf": clipped,
        "source_zip": zip_path,
        "feature_count": len(clipped),
    }
