#!/usr/bin/env python
"""
Load lunar labels for segmentation training.

Key design decisions:
  - Craters are loaded as circular **Polygon** geometries using their DIAM_CIRC_IMG
    diameter from the Robbins database, so rasterisation produces filled circles.
  - Rille polygons from GeoPackage files are kept as-is (not reduced to centroids).
  - All geometries are stored in MOON_GEOG_CRS (lunar geographic degrees) so that
    rasterize_multilabel can reproject them to whatever CRS the raster uses.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

# Moon radius in metres – must match geo_utils.py
_MOON_RADIUS_M = 1_737_400.0


def _moon_geog_crs():
    """Return a pyproj CRS for the Moon in geographic (lon/lat) degrees."""
    from pyproj import CRS
    return CRS.from_proj4(
        f"+proj=longlat +a={_MOON_RADIUS_M} +b={_MOON_RADIUS_M} +no_defs"
    )


def _load_robbins_craters(csv_path: Path) -> gpd.GeoDataFrame:
    """
    Load the Robbins crater database and return circular Polygon geometries.

    The CSV uses:
      - LON_CIRC_IMG : longitude in 0–360 degrees (Moon-centric)
      - LAT_CIRC_IMG : latitude in -90–90 degrees
      - DIAM_CIRC_IMG: crater diameter in **kilometres**

    We convert longitude to -180–+180, compute a radius in degrees, and
    buffer each point into a circle.  The resulting GeoDataFrame carries
    MOON_GEOG_CRS so downstream code can reproject correctly.
    """
    logger.info(f"Loading Robbins crater CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"}
    if not required.issubset(df.columns):
        raise ValueError(f"Crater CSV missing columns: {required - set(df.columns)}")

    # Convert longitude from 0-360 to -180-+180
    df["lon"] = df["LON_CIRC_IMG"].copy()
    df.loc[df["lon"] > 180, "lon"] -= 360.0
    df["lat"] = df["LAT_CIRC_IMG"]

    # Diameter is in km → radius in km → radius in degrees (approximate)
    # arc = distance / R  →  degrees = (km * 1000) / R * (180/π)
    df["radius_deg"] = (
        (df["DIAM_CIRC_IMG"] / 2.0)  # radius in km
        * 1000.0                      # → metres
        / _MOON_RADIUS_M             # → radians
        * (180.0 / math.pi)          # → degrees
    )

    # Build circular polygons -------------------------------------------------
    # Use a coarse resolution (16 segments) for speed; at ~474 m/px a 1-km
    # crater is only ~2 px across so sub-pixel fidelity is unnecessary.
    geometries = [
        Point(lon, lat).buffer(r, resolution=16)
        for lon, lat, r in zip(df["lon"], df["lat"], df["radius_deg"])
    ]

    gdf = gpd.GeoDataFrame(
        df[["CRATER_ID", "DIAM_CIRC_IMG"]],
        geometry=geometries,
        crs=_moon_geog_crs(),
    )
    logger.info(f"  Created {len(gdf)} circular crater polygons")
    return gdf


def _load_generic_csv(csv_path: Path) -> gpd.GeoDataFrame | None:
    """Load a CSV with coordinate columns as Point geometry."""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    # Try standard column name pairs (case-insensitive)
    col_lower = {c.lower(): c for c in df.columns}
    for lon_key, lat_key in [
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("long", "lat"),
        ("lon_circ_img", "lat_circ_img"),
    ]:
        if lon_key in col_lower and lat_key in col_lower:
            lon_col = col_lower[lon_key]
            lat_col = col_lower[lat_key]

            lons = df[lon_col].copy()
            lats = df[lat_col].copy()

            # Robbins-style 0-360 → -180-+180
            if lons.max() > 180:
                lons.loc[lons > 180] -= 360.0

            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(lons, lats),
                crs=_moon_geog_crs(),
            )
            return gdf

    logger.warning(f"CSV {csv_path} has no recognised coordinate columns: {list(df.columns)}")
    return None


def load_all_labels(raw_dir: Path) -> dict:
    """
    Robustly loads downloaded lunar labels using recursive search for spatial files.

    Returns a dict of {class_name: GeoDataFrame} in MOON_GEOG_CRS.
    """
    labels = {}

    # Mapping from feature class to identifying keywords found in filenames
    feature_map = {
        "pit_skylight":         ["LUNAR_PIT_LOCATIONS", "ESSA_detections"],
        "wrinkle_ridge":        ["WRINKLE_RIDGES"],
        "lobate_scarp":         ["LOBATE_SCARPS"],
        "irregular_mare_patch": ["LUNAR_IMP_LOCATIONS"],
        "apollo_site":          ["ANTHROPOGENIC_OBJECTS"],
        "impact_crater":        ["lunar_crater_database_robbins", "robbins"],
        "candidate_rille":      ["marius_hills_rilles"],
    }

    # Collect all candidate files once
    all_files = list(raw_dir.rglob("*"))

    for class_name, patterns in feature_map.items():
        try:
            found_path = _find_best_file(all_files, patterns)

            if found_path is None:
                logger.warning(f"No spatial data found for '{class_name}'")
                continue

            logger.info(f"Loading {class_name} from {found_path}")
            gdf = _load_file(class_name, found_path)

            if gdf is None or len(gdf) == 0:
                logger.warning(f"No features loaded for '{class_name}' from {found_path}")
                continue

            # Ensure CRS is Moon geographic
            if gdf.crs is None:
                gdf.set_crs(_moon_geog_crs(), inplace=True)

            labels[class_name] = gdf
            logger.info(f"Loaded {len(gdf)} features for {class_name} "
                        f"(geom types: {gdf.geometry.geom_type.value_counts().to_dict()})")

        except Exception as e:
            logger.error(f"Error loading labels for {class_name}: {e}", exc_info=True)

    return labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_best_file(all_files: list, patterns: list[str]) -> Path | None:
    """Find the best matching file for a set of keyword patterns.

    Priority order: GeoPackage > CSV (with data) > XLSX > Shapefile.
    Skips empty files (≤50 bytes, which would be header-only CSVs).
    """
    for pattern in patterns:
        matches = [f for f in all_files if pattern.lower() in f.name.lower() and f.is_file()]

        # GeoPackage
        candidates = [m for m in matches if m.suffix.lower() == ".gpkg"]
        if candidates:
            return candidates[0]

        # CSV — skip empty stubs (header-only files are typically <50 bytes)
        csv_candidates = [m for m in matches if m.suffix.lower() == ".csv" and m.stat().st_size > 100]
        if csv_candidates:
            return csv_candidates[0]

        # XLSX
        xlsx_candidates = [m for m in matches if m.suffix.lower() == ".xlsx"]
        if xlsx_candidates:
            return xlsx_candidates[0]

        # Shapefile
        shp_candidates = [m for m in matches if m.suffix.lower() == ".shp"]
        if shp_candidates:
            return shp_candidates[0]

    return None


def _load_file(class_name: str, path: Path) -> gpd.GeoDataFrame | None:
    """Dispatch to the correct loader based on file type and class name."""
    moon_crs = _moon_geog_crs()
    suffix = path.suffix.lower()

    if class_name == "impact_crater" and suffix == ".csv":
        return _load_robbins_craters(path)

    if suffix == ".csv":
        return _load_generic_csv(path)

    if suffix == ".gpkg":
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(moon_crs, inplace=True)
        # Keep polygon/line geometries as-is — do NOT convert to centroids
        return gdf

    if suffix == ".shp":
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(moon_crs, inplace=True)
        return gdf

    if suffix == ".xlsx":
        df = pd.read_excel(path)
        col_lower = {c.lower(): c for c in df.columns}
        for lon_key, lat_key in [
            ("longitude", "latitude"), ("lon", "lat"),
            ("long", "lat"), ("lon_circ_img", "lat_circ_img"),
        ]:
            if lon_key in col_lower and lat_key in col_lower:
                lon_col, lat_col = col_lower[lon_key], col_lower[lat_key]
                lons = df[lon_col].copy()
                if lons.max() > 180:
                    lons.loc[lons > 180] -= 360.0
                return gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(lons, df[lat_col]),
                    crs=moon_crs,
                )
        logger.warning(f"XLSX {path} has no recognised coordinate columns")
        return None

    logger.warning(f"Unsupported file format: {path.suffix}")
    return None
