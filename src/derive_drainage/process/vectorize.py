"""
Vectorization utilities to turn stream rasters into line features.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import Affine
from shapely.geometry import LineString
from tqdm import tqdm

LOG = logging.getLogger(__name__)

# 8-neighborhood offsets (row, col)
NEIGHBORS: list[tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


def _neighbor_indices(r: int, c: int, shape: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    rows, cols = shape
    for dr, dc in NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def skeletonize(stream_mask: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning to obtain a 1-pixel wide skeleton from a binary mask.
    """
    img = stream_mask.astype(np.uint8).copy()
    changed = True
    rows, cols = img.shape

    def transitions(r: int, c: int) -> int:
        p = [img[r + dr, c + dc] for dr, dc in NEIGHBORS]
        p.append(p[0])
        return sum((p[i] == 0 and p[i + 1] == 1) for i in range(8))

    def neighbor_sum(r: int, c: int) -> int:
        return sum(img[r + dr, c + dc] for dr, dc in NEIGHBORS)

    while changed:
        changed = False
        to_remove: list[tuple[int, int]] = []
        for r in tqdm(range(1, rows - 1), desc="Skeleton pass A", unit="row", leave=False):
            for c in range(1, cols - 1):
                if img[r, c] == 0:
                    continue
                n_sum = neighbor_sum(r, c)
                if n_sum < 2 or n_sum > 6:
                    continue
                if transitions(r, c) != 1:
                    continue
                if img[r - 1, c] * img[r, c + 1] * img[r + 1, c] == 0 and img[r, c + 1] * img[r + 1, c] * img[r, c - 1] == 0:
                    to_remove.append((r, c))
        if to_remove:
            changed = True
            for r, c in to_remove:
                img[r, c] = 0

        to_remove = []
        for r in tqdm(range(1, rows - 1), desc="Skeleton pass B", unit="row", leave=False):
            for c in range(1, cols - 1):
                if img[r, c] == 0:
                    continue
                n_sum = neighbor_sum(r, c)
                if n_sum < 2 or n_sum > 6:
                    continue
                if transitions(r, c) != 1:
                    continue
                if img[r - 1, c] * img[r, c + 1] * img[r, c - 1] == 0 and img[r - 1, c] * img[r + 1, c] * img[r, c - 1] == 0:
                    to_remove.append((r, c))
        if to_remove:
            changed = True
            for r, c in to_remove:
                img[r, c] = 0

    return img.astype(bool)


def _trace_segments(skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Trace skeleton pixels into ordered segments between junctions/endpoints.
    """
    rows, cols = skeleton.shape
    coords = {(r, c) for r, c in zip(*np.nonzero(skeleton))}
    if not coords:
        return []

    neighbors_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r, c in coords:
        neighbors = []
        for nr, nc in _neighbor_indices(r, c, (rows, cols)):
            if (nr, nc) in coords:
                neighbors.append((nr, nc))
        neighbors_map[(r, c)] = neighbors

    degree = {pt: len(neigh) for pt, neigh in neighbors_map.items()}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[list[tuple[int, int]]] = []

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        visited_edges.add(tuple(sorted([a, b])))

    def edge_seen(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return tuple(sorted([a, b])) in visited_edges

    def walk(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start]
        cur = start
        prev = None
        while True:
            path.append(nxt)
            add_edge(cur, nxt)
            prev = cur
            cur = nxt
            nbrs = [nb for nb in neighbors_map[cur] if nb != prev]
            if degree[cur] != 2 or not nbrs:
                break
            nxt = nbrs[0]
        return path

    # Trace from endpoints and junctions first
    start_nodes = [pt for pt, deg in degree.items() if deg != 2]
    for start in start_nodes:
        for nb in neighbors_map[start]:
            if edge_seen(start, nb):
                continue
            segments.append(walk(start, nb))

    # Trace remaining loops
    for pt in coords:
        for nb in neighbors_map[pt]:
            if edge_seen(pt, nb):
                continue
            segments.append(walk(pt, nb))

    return segments


def stream_mask_to_lines(
    stream_mask: np.ndarray,
    transform: Affine,
    crs_obj: CRS | int | str,
    flow_accum: np.ndarray | None,
    cell_area: float | None,
    min_length_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Convert a binary stream mask into a GeoDataFrame of LineStrings with attributes.
    """
    if flow_accum is not None and stream_mask.shape != flow_accum.shape:
        raise ValueError("stream_mask and flow_accum must have matching shapes.")

    skel = skeletonize(stream_mask)
    segments = _trace_segments(skel)
    if not segments:
        return gpd.GeoDataFrame(geometry=[], crs=CRS.from_user_input(crs_obj))

    geoms: list[LineString] = []
    lengths: list[float] = []
    upstream_cells: list[float] = []
    upstream_area: list[float] = []

    for coords in segments:
        rows_list, cols_list = zip(*coords)
        xs, ys = rasterio.transform.xy(transform, rows_list, cols_list, offset="center")  # type: ignore
        line = LineString(zip(xs, ys))
        length = float(line.length)
        if length < min_length_m:
            continue
        geoms.append(line)
        lengths.append(length)
        if flow_accum is not None:
            vals = flow_accum[rows_list, cols_list]
            upstream_val = float(np.nanmax(vals))
            upstream_cells.append(upstream_val)
            upstream_area.append(float(upstream_val * (cell_area or 0.0)))
        else:
            upstream_cells.append(float("nan"))
            upstream_area.append(float("nan"))

    gdf = gpd.GeoDataFrame(
        {
            "length_m": lengths,
            "upstream_cells": upstream_cells,
            "upstream_area_m2": upstream_area,
        },
        geometry=geoms,
        crs=CRS.from_user_input(crs_obj),
    )
    gdf = gdf.sort_values(by="upstream_cells", ascending=False).reset_index(drop=True)
    return gdf
