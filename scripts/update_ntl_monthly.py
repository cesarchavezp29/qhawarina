r"""
==============================================================================
NEXUS — VNP46A3 Monthly Nighttime Lights (VIIRS)
Peru monthly mosaics + district/department panels + Rossi 0.5° grid
(v7.1 — build + imputation, single file)

Changes from v7.0:
  A) PIXEL FLOOR FIX: negatives clamped to 0 (not NaN). Prevents selection
     bias where NaN-ing negatives inflates zonal mean in cloudy Amazon months.
     Spikes >1000 still set to NaN (true artifacts).
  B) PERU_BBOX north bound = -0.05 (was 0.1). Excludes v08 equator tiles.
  C) HV TILE PRE-FILTER: filter_h5_by_peru_tiles() rejects tiles by
     10°×10° grid bounds before opening HDF5. Drops ~470 v08 files.
  D) All three v6.5.2 imputation fixes retained:
     1) reindex_to_complete_months() called in fill_panel_monthly
     2) compute_global_bad_months() preserves date column
     3) global_bad_month only blocks filling for individually-bad obs

Key nodata rule:
  - On disk: nodata = -9999 (GDAL-safe)
  - In-memory: NaN for computation
  - Zonal stats: temp raster with nodata = -9999
==============================================================================
"""

# =============================================================================
# 0) Imports
# =============================================================================
import os
import re
import sys
import shutil
import tempfile
import warnings
import argparse
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with ✓ etc.)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import h5py

import earthaccess
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
from rasterstats import zonal_stats

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1) Configuration
# =============================================================================
NEXUS_BASE = Path(r"D:\Nexus\nexus")

SATELLITE_RAW  = NEXUS_BASE / "data" / "raw" / "satellite" / "vnp46a3"
SATELLITE_TIFF = NEXUS_BASE / "data" / "raw" / "satellite" / "ntl_monthly"
OUTPUT_DIR     = NEXUS_BASE / "data" / "processed"

_SHAPEFILE_PRIMARY  = Path(r"D:\Shining Path and Geographic\Censos\Merging\Limite Distrital INEI 2025 CPV.shp")
_SHAPEFILE_FALLBACK = Path(r"D:\nexus\nexus\data\raw\geo\peru_boundaries\hdx\per_admin3.shp")
SHAPEFILE_PATH = _SHAPEFILE_PRIMARY if _SHAPEFILE_PRIMARY.exists() else _SHAPEFILE_FALLBACK

ROSSI_GRID_BASE = Path(r"D:\Nexus\nexus\data\raw\viirs\0_5deg_v2\0_5deg_v2")
ROSSI_GRID_SHP_DIR = ROSSI_GRID_BASE / "shapefile"
ROSSI_GDP_GRID_CSV = None

# Peru is south of equator. North bound = -0.05 excludes v08 equator tiles.
# Peru's actual northernmost point (Putumayo) is ~-0.012°; -0.05 clips ~4 km.
PERU_BBOX = (-81.5, -18.5, -68.5, -0.05)

START_DATE = "2012-01"
END_DATE   = datetime.now().strftime("%Y-%m")  # dynamic: always current month

WGS84_CRS = CRS.from_epsg(4326)
EQA_CRS   = CRS.from_epsg(6933)
NODATA    = -9999.0

# --- Pixel floor/ceiling (v7 NEW) ---
# No legitimate NTL pixel is negative or above ~150 (p99.99).
# Cap at 1000 for safety margin. Clamp negatives to 0.
PIXEL_FLOOR = 0.0
PIXEL_CEIL  = 1000.0

GRID_ROOT = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
LAT_PATH  = f"{GRID_ROOT}/lat"
LON_PATH  = f"{GRID_ROOT}/lon"
NTL_DATASET_NAME = "AllAngle_Composite_Snow_Free"
NTL_PATH  = f"{GRID_ROOT}/{NTL_DATASET_NAME}"

for d in [SATELLITE_RAW, SATELLITE_TIFF, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MOSAIC_DIR_WGS84 = SATELLITE_TIFF / "mosaics_wgs84"
MOSAIC_DIR_EQA   = SATELLITE_TIFF / "mosaics_eqarea"
MOSAIC_DIR_WGS84.mkdir(parents=True, exist_ok=True)
MOSAIC_DIR_EQA.mkdir(parents=True, exist_ok=True)

print("✓ Configuration (v7.1)")
print(f"  Raw HDF5      : {SATELLITE_RAW}")
print(f"  Tile TIFFs    : {SATELLITE_TIFF}")
print(f"  Mosaics WGS84 : {MOSAIC_DIR_WGS84}")
print(f"  Mosaics EQA   : {MOSAIC_DIR_EQA}")
print(f"  Output        : {OUTPUT_DIR}")
print(f"  NTL dataset   : {NTL_PATH}")
print(f"  Rossi grid shp: {ROSSI_GRID_SHP_DIR}")
print(f"  Peru bbox     : {PERU_BBOX}")
print(f"  Pixel floor   : {PIXEL_FLOOR}  ceil: {PIXEL_CEIL}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART A: BUILD RAW PANELS FROM HDF5                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# =============================================================================
# 2) Utilities
# =============================================================================
def clean_stale_outputs():
    stale = []
    stale += list(SATELLITE_TIFF.glob("ntl_peru_*.tif"))
    stale += list(MOSAIC_DIR_WGS84.glob("ntl_peru_*_mosaic*.tif"))
    stale += list(MOSAIC_DIR_EQA.glob("ntl_peru_*_mosaic*.tif"))
    print(f"\nCleaning stale outputs: deleting {len(stale)} files...")
    for f in stale:
        try:
            f.unlink()
        except Exception:
            pass
    print("✓ Cleaned stale outputs")


def parse_vnp46a3_filename(filename: str):
    m = re.search(r"\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.", filename)
    if not m:
        return None, None, None, None
    year = int(m.group(1))
    doy  = int(m.group(2))
    h    = int(m.group(3))
    v    = int(m.group(4))
    date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
    return date.year, date.month, h, v


def parse_version(filename: str) -> int:
    m = re.search(r"\.h\d{2}v\d{2}\.(\d{3})\.", filename)
    return int(m.group(1)) if m else 0


def deduplicate_h5_files(h5_files: list[Path]) -> list[Path]:
    original_count = len(h5_files)
    best = {}
    for p in h5_files:
        year, month, h, v = parse_vnp46a3_filename(p.name)
        if year is None:
            continue
        key = (year, month, h, v)
        ver = parse_version(p.name)
        if key not in best or ver > best[key][1]:
            best[key] = (p, ver)
    deduped = sorted([p for p, _ in best.values()])
    dropped = original_count - len(deduped)
    if dropped:
        print(f"\n  Version dedup: {original_count} → {len(deduped)} "
              f"({dropped} dropped)")
    else:
        print(f"\n  Version dedup: none needed ({original_count} files)")
    return deduped


def scalar_attr(x, default):
    if x is None:
        return default
    try:
        return np.array(x).flat[0]
    except Exception:
        return default


def login_earthdata():
    import os
    if os.environ.get("EARTHDATA_USERNAME"):
        strategy = "environment"
    else:
        strategy = "netrc"  # reads ~/_netrc on Windows, ~/.netrc on Linux
    auth = earthaccess.login(strategy=strategy)
    if not auth:
        raise RuntimeError("Earthdata auth failed.")
    print("✓ Authenticated with NASA Earthdata")


def search_granules():
    print(f"\nSearching VNP46A3: {START_DATE} → {END_DATE}")
    results = earthaccess.search_data(
        short_name="VNP46A3",
        bounding_box=PERU_BBOX,
        temporal=(START_DATE, END_DATE),
        count=5000,
    )
    print(f"✓ Found {len(results)} granules")
    return results


def download_granules(results):
    print(f"\nDownloading to: {SATELLITE_RAW}")
    downloaded = earthaccess.download(results, local_path=str(SATELLITE_RAW))
    print(f"✓ Downloaded {len(downloaded)} files")
    return downloaded


# --- Tile bounds helpers (10°×10° grid) ---
def hv_bounds(h: int, v: int):
    west  = -180 + 10 * h
    east  = west + 10
    north =  90  - 10 * v
    south = north - 10
    return (west, south, east, north)


def bbox_intersects(a, b):
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return not (ae <= bw or aw >= be or an <= bs or as_ >= bn)


def filter_h5_by_peru_tiles(h5_files: list[Path],
                            peru_bbox=PERU_BBOX) -> list[Path]:
    kept, skipped = [], 0
    for p in h5_files:
        y, mth, h, v = parse_vnp46a3_filename(p.name)
        if y is None:
            continue
        if bbox_intersects(hv_bounds(h, v), peru_bbox):
            kept.append(p)
        else:
            skipped += 1
    print(f"\nTile filter (hv_bounds): {len(h5_files)} → {len(kept)} "
          f"(skipped {skipped})")
    hv = sorted({(parse_vnp46a3_filename(p.name)[2],
                  parse_vnp46a3_filename(p.name)[3]) for p in kept})
    if hv:
        print(f"  hv kept: {hv}")
    return kept


# =============================================================================
# 3) HDF5 → GeoTIFF (EPSG:4326)
# =============================================================================
def read_ntl_latlon_from_h5(h5_path: Path, ntl_path: str = NTL_PATH):
    with h5py.File(h5_path, "r") as f:
        if ntl_path not in f:
            raise KeyError(f"Dataset not found: {ntl_path}")
        if LAT_PATH not in f or LON_PATH not in f:
            raise KeyError("lat/lon vectors not found in HDF5")

        lat = f[LAT_PATH][:].astype(np.float64)
        lon = f[LON_PATH][:].astype(np.float64)

        ds = f[ntl_path]
        data = ds[:].astype(np.float32)

        fill   = int(scalar_attr(ds.attrs.get("_FillValue", 65535), 65535))
        scale  = float(scalar_attr(ds.attrs.get("scale_factor", 1.0), 1.0))
        offset = float(scalar_attr(ds.attrs.get("add_offset", 0.0), 0.0))

        data[data == fill] = np.nan
        data = data * scale + offset

        # ── v7.1 PIXEL FLOOR / CEILING ──
        # Clamp negatives to 0 (keep pixel valid, just zero radiance).
        # Kill absurd spikes (true artifacts like 50000).
        # Clamp (not NaN) prevents selection bias in zonal mean:
        #   NaN would exclude negatives → inflate mean in cloudy months.
        # Standard in NTL literature (Henderson et al. 2012).
        n_neg  = int(np.nansum(data < PIXEL_FLOOR))
        n_high = int(np.nansum(data > PIXEL_CEIL))
        data[data < PIXEL_FLOOR] = PIXEL_FLOOR   # clamp to 0
        data[data > PIXEL_CEIL]  = np.nan         # kill spikes

        # Ensure lat is descending (north → south)
        if lat[0] < lat[-1]:
            data = np.flipud(data)
            lat = lat[::-1]

        return data, lat, lon


def transform_from_1d_latlon(lat: np.ndarray, lon: np.ndarray,
                             height: int, width: int):
    if len(lat) != height or len(lon) != width:
        raise ValueError(
            f"lat/lon mismatch: lat={len(lat)} lon={len(lon)} "
            f"shape={height}x{width}"
        )
    dlat = (lat[-1] - lat[0]) / (len(lat) - 1)
    dlon = (lon[-1] - lon[0]) / (len(lon) - 1)
    west  = float(lon[0]  - 0.5 * dlon)
    east  = float(lon[-1] + 0.5 * dlon)
    north = float(lat[0]  - 0.5 * dlat)
    south = float(lat[-1] + 0.5 * dlat)
    return from_bounds(west, south, east, north, width, height)


def h5_to_geotiff_wgs84(h5_path: Path, out_dir: Path, peru_bbox=PERU_BBOX):
    fname = h5_path.name
    year, month, h, v = parse_vnp46a3_filename(fname)
    if year is None:
        return None

    out_path = out_dir / f"ntl_peru_{year}_{month:02d}_h{h:02d}v{v:02d}.tif"
    if out_path.exists():
        return out_path

    # Safety: reject by hv bounds even after pre-filter
    if not bbox_intersects(hv_bounds(h, v), peru_bbox):
        return None

    data, lat, lon = read_ntl_latlon_from_h5(h5_path, NTL_PATH)

    # Robust bbox check on actual lat/lon
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    tile_bbox = (lon_min, lat_min, lon_max, lat_max)
    if not bbox_intersects(tile_bbox, peru_bbox):
        return None

    height, width = data.shape
    transform = transform_from_1d_latlon(lat, lon, height, width)
    out_arr = np.where(np.isnan(data), NODATA, data).astype(np.float32)

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype="float32", crs=WGS84_CRS, transform=transform,
        nodata=NODATA, compress="lzw",
    ) as dst:
        dst.write(out_arr, 1)

    return out_path


# =============================================================================
# 4) Mosaic, crop, reproject
# =============================================================================
def crop_raster_to_bbox(src_path: Path, dst_path: Path, bbox):
    west, south, east, north = bbox
    with rasterio.open(src_path) as src:
        win = window_from_bounds(west, south, east, north,
                                 transform=src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win)
        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
            "compress": "lzw",
        })
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)
    return dst_path


def mosaic_monthly_tiles_wgs84(tile_paths: list[Path], mosaic_dir: Path,
                               crop_bbox=PERU_BBOX):
    monthly = defaultdict(list)
    for tif in sorted(tile_paths):
        tif = Path(tif)
        if not tif.exists():
            continue
        m = re.match(r"ntl_peru_(\d{4})_(\d{2})_", tif.name)
        if m:
            ym = (int(m.group(1)), int(m.group(2)))
            monthly[ym].append(tif)

    mosaic_paths = {}
    print(f"\nMosaicking {len(monthly)} months (WGS84; then crop Peru)")

    for (y, mth), files in sorted(monthly.items()):
        out_uncropped = mosaic_dir / f"ntl_peru_{y}_{mth:02d}_mosaic_uncropped.tif"
        out_cropped   = mosaic_dir / f"ntl_peru_{y}_{mth:02d}_mosaic.tif"

        if out_cropped.exists():
            mosaic_paths[(y, mth)] = out_cropped
            continue

        if len(files) == 1:
            shutil.copy2(files[0], out_uncropped)
        else:
            srcs = []
            try:
                srcs = [rasterio.open(f) for f in files]
                mosaic_data, mosaic_transform = merge(srcs, nodata=NODATA)
                for s in srcs:
                    s.close()
            except Exception as e:
                print(f"  ⚠ merge error {y}-{mth:02d}: {e}")
                for s in srcs:
                    try:
                        s.close()
                    except Exception:
                        pass
                continue

            band = mosaic_data[0].astype(np.float32)
            with rasterio.open(
                out_uncropped, "w", driver="GTiff",
                height=band.shape[0], width=band.shape[1], count=1,
                dtype="float32", crs=WGS84_CRS, transform=mosaic_transform,
                nodata=NODATA, compress="lzw",
            ) as out:
                out.write(band, 1)

        crop_raster_to_bbox(out_uncropped, out_cropped, crop_bbox)
        try:
            out_uncropped.unlink()
        except Exception:
            pass
        mosaic_paths[(y, mth)] = out_cropped

    print(f"✓ {len(mosaic_paths)} mosaics ready (cropped)")
    return mosaic_paths


def verify_mosaic_geometry(mosaic_paths: dict, expected_px=0.004167,
                           max_width_deg=15.0):
    if not mosaic_paths:
        return True
    test_key = max(mosaic_paths.keys())
    test_path = mosaic_paths[test_key]
    with rasterio.open(test_path) as src:
        b = src.bounds
        width_deg = b.right - b.left
        px_x = abs(src.transform.a)
        print(f"\n--- Mosaic geometry check: {test_path.name} ---")
        print(f"  Bounds: ({b.left:.4f}, {b.bottom:.4f}, "
              f"{b.right:.4f}, {b.top:.4f})")
        print(f"  Size: {src.width}×{src.height} px  | Pixel: {px_x:.6f}°")

        # v7: report pixel value distribution
        arr = src.read(1).astype(np.float32)
        arr[arr == NODATA] = np.nan
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            print(f"  Pixel stats: min={np.min(valid):.2f}  "
                  f"max={np.max(valid):.2f}  "
                  f"mean={np.mean(valid):.4f}  "
                  f"p99.99={np.percentile(valid, 99.99):.2f}")
            print(f"  Valid pixels: {len(valid):,}  "
                  f"NaN/nodata: {int(np.isnan(arr).sum()):,}")

        if width_deg > max_width_deg:
            print(f"  ⚠ WARNING: width {width_deg:.2f}° > {max_width_deg}°")
        if abs(px_x - expected_px) > 0.0006:
            print(f"  ⚠ WARNING: pixel {px_x:.6f}° ≠ expected {expected_px:.6f}°")
    return True


def reproject_mosaic_to_equal_area(src_path: Path, dst_path: Path,
                                   dst_crs=EQA_CRS):
    if dst_path.exists():
        return dst_path
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata if src.nodata is not None else NODATA
        src_data[src_data == src_nodata] = np.nan

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_data, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan, dst_nodata=np.nan,
        )

        dst = np.where(np.isnan(dst), NODATA, dst).astype(np.float32)

        with rasterio.open(
            dst_path, "w", driver="GTiff",
            height=dst_height, width=dst_width, count=1,
            dtype="float32", crs=dst_crs, transform=dst_transform,
            nodata=NODATA, compress="lzw",
        ) as out:
            out.write(dst, 1)
    return dst_path


def build_equal_area_mosaics(mosaic_paths_wgs84: dict, mosaic_dir_eq: Path):
    mosaic_paths_eq = {}
    print("\nReprojecting mosaics to equal-area (EPSG:6933)")
    for (y, mth), p in sorted(mosaic_paths_wgs84.items()):
        dst = mosaic_dir_eq / f"ntl_peru_{y}_{mth:02d}_mosaic_eqarea.tif"
        reproject_mosaic_to_equal_area(p, dst)
        mosaic_paths_eq[(y, mth)] = dst
    print(f"✓ {len(mosaic_paths_eq)} equal-area mosaics ready")
    return mosaic_paths_eq


# =============================================================================
# 5) Temp raster helper for zonal stats
# =============================================================================
def make_stats_safe_raster(raster_path: Path) -> str:
    tmp_nodata = -9999.0
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            arr[arr == nd] = tmp_nodata
        arr[arr == NODATA] = tmp_nodata
        arr[np.isnan(arr)] = tmp_nodata
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=tmp_nodata, compress="lzw")
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(arr, 1)
    return tmp_path


# =============================================================================
# 6) Diagnostics
# =============================================================================
def sample_points_on_raster(raster_path: Path, points: dict):
    out = []
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        for name, (lon, lat) in points.items():
            try:
                r, c = src.index(lon, lat)
                if 0 <= r < src.height and 0 <= c < src.width:
                    v = arr[r, c]
                    v = np.nan if v == NODATA else float(v)
                    out.append((name, lon, lat, v))
                else:
                    out.append((name, lon, lat, np.nan))
            except Exception:
                out.append((name, lon, lat, np.nan))
    return pd.DataFrame(out, columns=["name", "lon", "lat", "value"])


def lima_moment_of_truth(mosaic_paths_wgs84: dict):
    if not mosaic_paths_wgs84:
        print("No mosaics to test.")
        return False
    test_key = max(mosaic_paths_wgs84.keys())
    test_mosaic = mosaic_paths_wgs84[test_key]
    print(f"\n--- Spatial alignment check: {test_mosaic.name} ---")
    points = {
        "Lima Centro":    (-77.03, -12.05),
        "Miraflores":     (-77.03, -12.12),
        "Callao":         (-77.12, -12.06),
        "Arequipa":       (-71.54, -16.41),
        "Trujillo":       (-79.03, -8.11),
        "Amazon (dark)":  (-74.00, -5.00),
    }
    df = sample_points_on_raster(test_mosaic, points)
    print(df.to_string(index=False))
    with rasterio.open(test_mosaic) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == NODATA] = np.nan
        if np.any(np.isfinite(arr)):
            idx = np.unravel_index(np.nanargmax(arr), arr.shape)
            blon, blat = src.xy(idx[0], idx[1])
            print(f"Brightest pixel: value={arr[idx]:.3f} "
                  f"at lon={blon:.6f}, lat={blat:.6f}")
    lima = df.loc[df["name"] == "Lima Centro", "value"].iloc[0]
    mira = df.loc[df["name"] == "Miraflores", "value"].iloc[0]
    return (lima > 20) and (mira > 20)


# =============================================================================
# 7) Districts + zonal stats
# =============================================================================
def load_districts(shp_path: Path):
    gdf = gpd.read_file(shp_path)
    gdf["UBIGEO"] = gdf["UBIGEO"].astype(str).str.zfill(6)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf["DEPT_CODE"] = gdf["UBIGEO"].str[:2]
    return gdf


def zonal_mean_median_count(raster_path: Path,
                            gdf_wgs84: gpd.GeoDataFrame,
                            year: int, month: int):
    stats = zonal_stats(
        gdf_wgs84, str(raster_path),
        stats=["mean", "median", "count"],
        nodata=NODATA, all_touched=True,
    )
    df = pd.DataFrame(stats).rename(
        columns={"mean": "ntl_mean", "median": "ntl_median",
                 "count": "ntl_count"}
    )
    df["UBIGEO"] = gdf_wgs84["UBIGEO"].values
    df["year"] = year
    df["month"] = month
    df["date"] = pd.Timestamp(year=year, month=month, day=1)
    return df


def zonal_total_equal_area(raster_eq_path: Path,
                           gdf_eq: gpd.GeoDataFrame,
                           year: int, month: int):
    tmp_path = make_stats_safe_raster(raster_eq_path)
    try:
        with rasterio.open(tmp_path) as src:
            pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
        stats = zonal_stats(
            gdf_eq, tmp_path,
            stats=["sum", "count"], nodata=-9999.0, all_touched=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    df = pd.DataFrame(stats).rename(
        columns={"sum": "ntl_sum_raw", "count": "ntl_count_eqarea"}
    )
    df["ntl_sum_eqarea"] = df["ntl_sum_raw"] * pixel_area_km2
    df.drop(columns=["ntl_sum_raw"], inplace=True)
    df["UBIGEO"] = gdf_eq["UBIGEO"].values
    df["year"] = year
    df["month"] = month
    df["date"] = pd.Timestamp(year=year, month=month, day=1)
    df["pixel_area_km2"] = pixel_area_km2
    return df


def build_district_panel(mosaics_wgs84: dict, mosaics_eqarea: dict,
                         districts_wgs84: gpd.GeoDataFrame,
                         do_eqarea_totals=True):
    rows = []
    total = len(mosaics_wgs84)
    districts_eq = (districts_wgs84.to_crs(EQA_CRS)
                    if do_eqarea_totals else None)
    print("\nComputing district-month zonal stats")
    for i, ((y, mth), p_wgs) in enumerate(sorted(mosaics_wgs84.items())):
        if i == 0 or (i + 1) % 12 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {y}-{mth:02d}")
        df_mean = zonal_mean_median_count(p_wgs, districts_wgs84, y, mth)
        if do_eqarea_totals:
            p_eq = mosaics_eqarea[(y, mth)]
            df_tot = zonal_total_equal_area(p_eq, districts_eq, y, mth)
            df = df_mean.merge(
                df_tot[["UBIGEO", "ntl_sum_eqarea", "ntl_count_eqarea",
                         "pixel_area_km2"]],
                on="UBIGEO", how="left",
            )
        else:
            df = df_mean
            df["ntl_sum_eqarea"] = np.nan
            df["ntl_count_eqarea"] = np.nan
            df["pixel_area_km2"] = np.nan
        df["DEPT_CODE"] = df["UBIGEO"].str[:2]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 8) Department panel
# =============================================================================
DEPT_NAMES = {
    "01": "Amazonas", "02": "Ancash", "03": "Apurímac", "04": "Arequipa",
    "05": "Ayacucho", "06": "Cajamarca", "07": "Callao", "08": "Cusco",
    "09": "Huancavelica", "10": "Huánuco", "11": "Ica", "12": "Junín",
    "13": "La Libertad", "14": "Lambayeque", "15": "Lima", "16": "Loreto",
    "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
    "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes",
    "25": "Ucayali",
}


def build_department_panel(ntl_district: pd.DataFrame):
    dept = (
        ntl_district
        .groupby(["DEPT_CODE", "year", "month", "date"], as_index=False)
        .agg(
            ntl_mean=("ntl_mean", "mean"),
            ntl_median=("ntl_median", "mean"),
            ntl_count=("ntl_count", "sum"),
            ntl_sum_eqarea=("ntl_sum_eqarea", "sum"),
            n_districts=("UBIGEO", "count"),
        )
    )
    dept["department"] = dept["DEPT_CODE"].map(DEPT_NAMES)
    return dept


def save_outputs(ntl_district: pd.DataFrame, ntl_department: pd.DataFrame):
    out_d = OUTPUT_DIR / "ntl_monthly_district.csv"
    out_p = OUTPUT_DIR / "ntl_monthly_department.csv"
    ntl_district.to_csv(out_d, index=False)
    ntl_department.to_csv(out_p, index=False)
    ntl_district.to_parquet(OUTPUT_DIR / "ntl_monthly_district.parquet", index=False)
    ntl_department.to_parquet(OUTPUT_DIR / "ntl_monthly_department.parquet", index=False)
    print("\n✓ Saved:")
    print(f"  {out_d}  shape={ntl_district.shape}")
    print(f"  {out_p}  shape={ntl_department.shape}")
    print(f"  {OUTPUT_DIR / 'ntl_monthly_district.parquet'}")
    print(f"  {OUTPUT_DIR / 'ntl_monthly_department.parquet'}")


def plot_national_total(ntl_department: pd.DataFrame,
                        out_name="ntl_validation_total_eqarea_raw.png"):
    national = ntl_department.groupby("date")["ntl_sum_eqarea"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(national.index, national.values, linewidth=1.5)
    ax.set_title("Peru Total NTL (area-weighted; EPSG:6933) — v7.1 pixel clamp")
    ax.set_ylabel("NTL total (radiance × km²)")
    ax.grid(True, alpha=0.3)
    out = OUTPUT_DIR / out_name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"✓ Saved plot: {out}")


# =============================================================================
# 9) Rossi 0.5 deg grid
# =============================================================================
def find_single_shapefile(shp_dir: Path) -> Path:
    shp_files = sorted(shp_dir.glob("*.shp"))
    if not shp_files:
        shp_files = sorted(shp_dir.glob("**/*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found under: {shp_dir}")
    return shp_files[0]


def load_rossi_grid(shp_dir: Path) -> gpd.GeoDataFrame:
    shp_path = find_single_shapefile(shp_dir)
    print(f"\nLoading Rossi grid shapefile: {shp_path}")
    grid = gpd.read_file(shp_path)
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    elif grid.crs.to_epsg() != 4326:
        grid = grid.to_crs("EPSG:4326")
    lon_col = next(
        (c for c in grid.columns
         if c.lower() in ("lon", "longitude", "x", "center_lon", "clon")),
        None,
    )
    lat_col = next(
        (c for c in grid.columns
         if c.lower() in ("lat", "latitude", "y", "center_lat", "clat")),
        None,
    )
    if lon_col and lat_col:
        grid["lon"] = grid[lon_col].astype(float)
        grid["lat"] = grid[lat_col].astype(float)
    else:
        cent = grid.geometry.centroid
        grid["lon"] = cent.x
        grid["lat"] = cent.y
    grid = grid[["lon", "lat", "geometry"]].copy()
    print(f"✓ Grid cells: {len(grid)}")
    return grid


def aggregate_monthly_to_grid(mosaics_wgs84: dict, mosaics_eqarea: dict,
                              grid_wgs84: gpd.GeoDataFrame,
                              do_eqarea_totals=True):
    grid_eq = grid_wgs84.to_crs(EQA_CRS) if do_eqarea_totals else None
    rows = []
    keys = sorted(mosaics_wgs84.keys())
    print(f"\nAggregating NTL to 0.5° grid: "
          f"{len(keys)} months × {len(grid_wgs84)} cells")
    for i, (y, mth) in enumerate(keys):
        if i == 0 or (i + 1) % 24 == 0 or (i + 1) == len(keys):
            print(f"  [{i+1}/{len(keys)}] {y}-{mth:02d}")
        p_wgs = mosaics_wgs84[(y, mth)]
        stats = zonal_stats(
            grid_wgs84, str(p_wgs),
            stats=["mean", "median", "count"],
            nodata=NODATA, all_touched=True,
        )
        df = pd.DataFrame(stats).rename(
            columns={"mean": "ntl_mean", "median": "ntl_median",
                      "count": "ntl_count"}
        )
        df["year"] = y
        df["month"] = mth
        df["date"] = pd.Timestamp(year=y, month=mth, day=1)
        df["lon"] = grid_wgs84["lon"].values
        df["lat"] = grid_wgs84["lat"].values
        if do_eqarea_totals:
            p_eq = mosaics_eqarea[(y, mth)]
            tmp_path = make_stats_safe_raster(p_eq)
            try:
                with rasterio.open(tmp_path) as src:
                    pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
                stats2 = zonal_stats(
                    grid_eq, tmp_path, stats=["sum"],
                    nodata=-9999.0, all_touched=True,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            df2 = pd.DataFrame(stats2).rename(columns={"sum": "ntl_sum_raw"})
            df["ntl_total"] = df2["ntl_sum_raw"] * pixel_area_km2
        else:
            df["ntl_total"] = np.nan
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def rossi_grid_outputs(grid_monthly: pd.DataFrame):
    out_m = OUTPUT_DIR / "ntl_monthly_rossi_grid_0p5deg.csv"
    grid_monthly.to_csv(out_m, index=False)
    print(f"✓ Saved: {out_m}  shape={grid_monthly.shape}")
    grid_annual = (
        grid_monthly.groupby(["lon", "lat", "year"], as_index=False)
        .agg(ntl_mean=("ntl_mean", "mean"),
             ntl_total=("ntl_total", "sum"),
             ntl_count=("ntl_count", "sum"))
    )
    out_a = OUTPUT_DIR / "ntl_annual_rossi_grid_0p5deg.csv"
    grid_annual.to_csv(out_a, index=False)
    print(f"✓ Saved: {out_a}  shape={grid_annual.shape}")
    return grid_annual


def merge_rossi_gdp(grid_annual: pd.DataFrame, gdp_csv: Path):
    gdp = pd.read_csv(gdp_csv)
    lon_col = next(
        (c for c in gdp.columns if c.lower() in ("lon", "longitude", "x")),
        None,
    )
    lat_col = next(
        (c for c in gdp.columns if c.lower() in ("lat", "latitude", "y")),
        None,
    )
    year_col = next(
        (c for c in gdp.columns if c.lower() == "year"), None
    )
    if lon_col is None or lat_col is None or year_col is None:
        raise ValueError("GDP CSV must have lon/lat/year columns.")
    preferred = [c for c in gdp.columns
                 if c.lower() in ("gdpc", "gdppc", "gdp_pc",
                                   "gdpc_postadjust", "gdp")]
    target = None
    for c in preferred:
        if c in gdp.columns and pd.api.types.is_numeric_dtype(gdp[c]):
            target = c
            break
    if target is None:
        numeric = [c for c in gdp.columns
                   if pd.api.types.is_numeric_dtype(gdp[c])
                   and c not in (lon_col, lat_col, year_col)]
        if not numeric:
            raise ValueError("No numeric GDP-like column found.")
        target = numeric[0]
    gdp_sub = gdp[[lon_col, lat_col, year_col, target]].rename(
        columns={lon_col: "lon", lat_col: "lat",
                 year_col: "year", target: "gdp_target"}
    )
    merged = grid_annual.merge(gdp_sub, on=["lon", "lat", "year"], how="inner")
    out = OUTPUT_DIR / "gdpgrid_0p5deg_vs_ntl_annual.csv"
    merged.to_csv(out, index=False)
    print(f"✓ Saved GDP×NTL merge: {out}  shape={merged.shape}")
    print("\nRossi GDP grid comparison (cell-year)")
    for y in sorted(merged["year"].unique()):
        tmp = merged[merged["year"] == y].dropna()
        if len(tmp) < 50:
            continue
        r_tot = tmp["ntl_total"].corr(tmp["gdp_target"])
        r_mean = tmp["ntl_mean"].corr(tmp["gdp_target"])
        print(f"  {y}: corr(GDP, NTL total)={r_tot:.3f} "
              f"| corr(GDP, NTL mean)={r_mean:.3f}  n={len(tmp)}")
    return merged


# =============================================================================
# 10) Main build pipeline
# =============================================================================
def main_build(download=False, force_clean=False, do_eqarea_totals=True,
               run_rossi_grid=True):
    if force_clean:
        clean_stale_outputs()
    if download:
        login_earthdata()
        results = search_granules()
        download_granules(results)

    h5_files = sorted(SATELLITE_RAW.glob("*.h5"))
    if not h5_files:
        h5_files = sorted(SATELLITE_RAW.glob("**/*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {SATELLITE_RAW}")

    h5_files = deduplicate_h5_files(h5_files)

    # v7: pre-filter by tile hv bounds
    h5_files = filter_h5_by_peru_tiles(h5_files, peru_bbox=PERU_BBOX)
    if not h5_files:
        raise RuntimeError("No HDF5 files intersect Peru after tile filter.")

    print("\n--- Preflight HDF5 check (first kept file) ---")
    with h5py.File(h5_files[0], "r") as f:
        assert NTL_PATH in f, f"Missing dataset: {NTL_PATH}"
        assert LAT_PATH in f and LON_PATH in f, "Missing lat/lon vectors"
        lat = f[LAT_PATH][:]
        lon = f[LON_PATH][:]
        print(f"  Dataset: {NTL_PATH}")
        print(f"  lat range: {lat[0]:.4f} → {lat[-1]:.4f}   "
              f"| lon range: {lon[0]:.4f} → {lon[-1]:.4f}")
        print(f"  lat spacing: {abs(lat[1]-lat[0]):.6f}°, "
              f"lon spacing: {abs(lon[1]-lon[0]):.6f}°")

    print("\nConverting HDF5 → tile GeoTIFFs (EPSG:4326)")
    print(f"  Pixel floor={PIXEL_FLOOR}, ceil={PIXEL_CEIL}")
    tiffs, errors = [], []
    for i, p in enumerate(h5_files):
        if i == 0 or (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(h5_files)}  {p.name}")
        try:
            out = h5_to_geotiff_wgs84(p, SATELLITE_TIFF, PERU_BBOX)
            if out:
                tiffs.append(out)
        except Exception as e:
            errors.append((p.name, str(e)))
    print(f"✓ Created {len(tiffs)} tile TIFFs")
    if errors:
        print(f"  ⚠ {len(errors)} failures (showing 5):")
        for fname, err in errors[:5]:
            print(f"    {fname}: {err}")

    mosaics_wgs84 = mosaic_monthly_tiles_wgs84(
        tiffs, MOSAIC_DIR_WGS84, crop_bbox=PERU_BBOX
    )
    verify_mosaic_geometry(mosaics_wgs84)

    if not lima_moment_of_truth(mosaics_wgs84):
        print("\n⚠ Spatial alignment check FAILED. Stop.")
        return None, None, None, None

    mosaics_eqarea = {}
    if do_eqarea_totals:
        mosaics_eqarea = build_equal_area_mosaics(mosaics_wgs84, MOSAIC_DIR_EQA)

    print("\nLoading districts...")
    districts = load_districts(SHAPEFILE_PATH)
    print(f"✓ Districts: {len(districts)}  "
          f"Depts: {districts['DEPT_CODE'].nunique()}")

    ntl_district = build_district_panel(
        mosaics_wgs84=mosaics_wgs84,
        mosaics_eqarea=(mosaics_eqarea if do_eqarea_totals
                        else mosaics_wgs84),
        districts_wgs84=districts,
        do_eqarea_totals=do_eqarea_totals,
    )
    ntl_department = build_department_panel(ntl_district)
    save_outputs(ntl_district, ntl_department)
    if do_eqarea_totals:
        plot_national_total(ntl_department,
                            out_name="ntl_validation_total_eqarea_raw.png")

    rossi_monthly, rossi_annual = None, None
    if run_rossi_grid:
        grid = load_rossi_grid(ROSSI_GRID_SHP_DIR)
        rossi_monthly = aggregate_monthly_to_grid(
            mosaics_wgs84, mosaics_eqarea, grid, do_eqarea_totals=True
        )
        rossi_annual = rossi_grid_outputs(rossi_monthly)
        if ROSSI_GDP_GRID_CSV is not None:
            gdp_csv = Path(ROSSI_GDP_GRID_CSV)
            if gdp_csv.exists():
                merge_rossi_gdp(rossi_annual, gdp_csv)
            else:
                print(f"\n⚠ Rossi GDP CSV not found: {gdp_csv}")
        else:
            print("\nRossi GDP CSV not set (skipping GDP merge).")

    print("\n✅ BUILD COMPLETE (v7.1)")
    return ntl_district, ntl_department, rossi_monthly, rossi_annual


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART B: IMPUTATION MODULE                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# --- Paths ---
DISTRICT_CSV = OUTPUT_DIR / "ntl_monthly_district.csv"
ROSSI_CSV    = OUTPUT_DIR / "ntl_monthly_rossi_grid_0p5deg.csv"

OUT_DISTRICT_FILLED = OUTPUT_DIR / "ntl_monthly_district_filled.csv"
OUT_DEPT_FILLED     = OUTPUT_DIR / "ntl_monthly_department_filled.csv"
OUT_NATIONAL_FILLED = OUTPUT_DIR / "ntl_monthly_national_filled.csv"
OUT_QC_PLOT         = OUTPUT_DIR / "ntl_imputation_qc.png"
OUT_QC_MISSING_PLOT = OUTPUT_DIR / "ntl_imputation_missing_share_dist.png"
OUT_BAD_MONTHS_CSV  = OUTPUT_DIR / "ntl_global_bad_months.csv"

# --- Parameters ---
BAD_MISS_THRESH = 0.10
MAX_ABS_VALUE   = None
INTERP_LIMIT    = 2
USE_MONTH_MEDIAN_FIRST = True
MAD_Z_THRESHOLD = 10.0

BLOCK_GLOBAL_BAD_MONTHS = True
GLOBAL_BAD_SHARE_THRESH = 0.20
GLOBAL_BAD_MEAN_MISS    = 0.05
GLOBAL_BAD_NEG_TOTAL    = True

DIST_TOTAL_COL  = "ntl_sum_eqarea"
DIST_COUNT_COL  = "ntl_count_eqarea"
ROSSI_TOTAL_COL = "ntl_total"
ROSSI_COUNT_COL = "ntl_count"


# =============================================================================
# 11) Imputation helpers
# =============================================================================
def _ensure_month_start(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt)
    return dt.dt.to_period("M").dt.to_timestamp()


def add_expected_and_missing_share(df: pd.DataFrame, unit_col: str,
                                   count_col: str) -> pd.DataFrame:
    out = df.copy()
    out["expected_count"] = out.groupby(unit_col)[count_col].transform("max")
    out["missing_share"] = np.where(
        out["expected_count"] > 0,
        1 - (out[count_col] / out["expected_count"]),
        np.nan,
    )
    return out


def robust_outlier_flag(series: pd.Series, z: float = 8.0) -> pd.Series:
    x = series.astype(float)
    if x.notna().sum() < 10:
        return pd.Series(False, index=series.index)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=series.index)
    rz = 0.6745 * (x - med) / mad
    return np.abs(rz) > z


def reindex_to_complete_months(df: pd.DataFrame, unit_col: str,
                               date_col: str) -> pd.DataFrame:
    all_dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq="MS")
    all_units = df[unit_col].unique()

    full_idx = pd.MultiIndex.from_product(
        [all_units, all_dates], names=[unit_col, date_col]
    )
    full = pd.DataFrame(index=full_idx).reset_index()

    df = df.copy()
    df["_original_row"] = True

    out = full.merge(df, on=[unit_col, date_col], how="left")
    out["is_reindex_gap"] = out["_original_row"].isna()
    out.drop(columns=["_original_row"], inplace=True)

    n_added = int(out["is_reindex_gap"].sum())
    if n_added > 0:
        print(f"    Reindex: added {n_added:,} missing unit-month rows "
              f"({len(all_units):,} units × {len(all_dates)} months "
              f"= {len(out):,} total)")
    else:
        print(f"    Reindex: panel already complete ({len(out):,} rows)")
    return out


def compute_global_bad_months(
    df: pd.DataFrame, date_col: str, unit_col: str,
    missing_share_col: str, value_col: str,
    miss_thresh: float, share_thresh: float,
    mean_miss_thresh: float, flag_neg_total: bool,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = _ensure_month_start(tmp[date_col])

    records = []
    for dt, g in tmp.groupby(date_col):
        ms = g[missing_share_col]
        records.append({
            "date": dt,
            "n_units": g[unit_col].nunique(),
            "share_units_high_miss": (
                float((ms > miss_thresh).mean())
                if ms.notna().any() else np.nan
            ),
            "mean_missing_share": (
                float(ms.mean()) if ms.notna().any() else np.nan
            ),
            "national_total": (
                float(g[value_col].sum(skipna=True))
                if g[value_col].notna().any() else np.nan
            ),
        })

    bym = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    cond = (
        (bym["share_units_high_miss"].fillna(0) > share_thresh)
        | (bym["mean_missing_share"].fillna(0) > mean_miss_thresh)
    )
    if flag_neg_total:
        cond = cond | (bym["national_total"].notna()
                       & (bym["national_total"] < 0))

    bym["is_global_bad"] = cond
    return bym


def print_missing_share_diagnostics(df: pd.DataFrame, label: str = ""):
    ms = df["missing_share"].dropna()
    if ms.empty:
        print(f"  {label} missing_share: no data")
        return
    print(f"\n  {label} missing_share distribution (n={len(ms):,}):")
    for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        print(f"    p{int(q*100):02d} = {ms.quantile(q):.4f}")
    print(f"    Share > {BAD_MISS_THRESH}: "
          f"{(ms > BAD_MISS_THRESH).mean():.2%}")


def plot_missing_share_histogram(df: pd.DataFrame, out_path: Path,
                                 label: str = ""):
    ms = df["missing_share"].dropna()
    if ms.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(ms, bins=100, edgecolor="none", alpha=0.7)
    ax.axvline(BAD_MISS_THRESH, color="red", linestyle="--",
               label=f"Threshold = {BAD_MISS_THRESH}")
    ax.set_title(f"{label} Distribution of missing_share")
    ax.set_xlabel("missing_share (fraction of pixels missing)")
    ax.set_ylabel("Count (unit-months)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved: {out_path}")


def print_flag_summary(df: pd.DataFrame, label: str = ""):
    n = len(df)
    n_bad = int(df["is_bad"].sum())
    n_imputed = int(df["imputed"].sum())
    n_still_na = int(df["value_filled"].isna().sum())
    print(f"\n  {label} Flag summary ({n:,} unit-months):")
    print(f"    Flagged bad : {n_bad:,} ({n_bad/n:.2%})")
    print(f"    Imputed     : {n_imputed:,} ({n_imputed/n:.2%})")
    print(f"    Still NaN   : {n_still_na:,} ({n_still_na/n:.2%})")
    for col, txt in [
        ("is_negative", "negative"),
        ("is_high_miss", "high miss"),
        ("is_outlier", "outlier"),
        ("is_capped", "capped"),
        ("is_reindex_gap", "reindex gap"),
    ]:
        if col in df.columns:
            print(f"    — {txt:<12}: {int(df[col].sum()):,}")
    if "in_global_bad_month" in df.columns:
        n_gbm = int(df["in_global_bad_month"].sum())
        n_gbm_blocked = (int(df["impute_blocked"].sum())
                         if "impute_blocked" in df.columns else 0)
        print(f"    In global bad month : {n_gbm:,} ({n_gbm/n:.2%})")
        print(f"    Impute blocked      : {n_gbm_blocked:,}")


# =============================================================================
# 12) Core fill function
# =============================================================================
def fill_panel_monthly(
    df: pd.DataFrame,
    unit_col: str,
    date_col: str,
    value_col: str,
    count_col: str | None = None,
    bad_missing_thresh: float = 0.10,
    interp_limit: int = 2,
    use_month_median_first: bool = True,
    mad_z: float = 10.0,
    max_abs_value: float | None = None,
    block_global_bad_months: bool = True,
    global_bad_months: set | None = None,
    label: str = "",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = _ensure_month_start(out[date_col])
    out = out.sort_values([unit_col, date_col]).reset_index(drop=True)

    # Complete monthly panel
    out = reindex_to_complete_months(out, unit_col, date_col)
    out["year"] = pd.to_datetime(out[date_col]).dt.year
    out["month"] = pd.to_datetime(out[date_col]).dt.month

    # Missing share
    if count_col is not None and count_col in out.columns:
        out = add_expected_and_missing_share(out, unit_col, count_col)
    else:
        out["expected_count"] = np.nan
        out["missing_share"] = np.nan

    print_missing_share_diagnostics(out, label=label)

    # Unit-level flags
    out["is_outlier"] = (
        out.groupby(unit_col)[value_col]
           .transform(lambda s: robust_outlier_flag(s, z=mad_z))
    )

    val = out[value_col].astype(float)
    out["is_negative"] = val < 0

    out["is_high_miss"] = False
    if out["missing_share"].notna().any():
        out["is_high_miss"] = out["missing_share"] > bad_missing_thresh

    out["is_capped"] = False
    if max_abs_value is not None:
        out["is_capped"] = np.abs(val) > max_abs_value

    # is_bad = unit-level only (NOT global bad month)
    out["is_bad"] = (
        out["is_negative"]
        | out["is_high_miss"]
        | out["is_capped"]
        | out["is_outlier"].fillna(False)
        | out["is_reindex_gap"]
    )

    out["in_global_bad_month"] = (
        out[date_col].isin(global_bad_months)
        if global_bad_months is not None else False
    )

    # Clean
    out["value_clean"] = out[value_col].where(~out["is_bad"], np.nan)

    # Fill step 1: month-of-year median
    out["_month"] = out["month"]
    if use_month_median_first:
        mo_med = out.groupby([unit_col, "_month"])["value_clean"] \
                    .transform("median")
        out["value_filled"] = out["value_clean"].fillna(mo_med)
    else:
        out["value_filled"] = out["value_clean"].copy()

    # Fill step 2: time interpolation
    def _interp_group(g):
        g = g.set_index(date_col)
        g["value_filled"] = g["value_filled"].interpolate(
            method="time", limit=interp_limit
        )
        return g.reset_index()

    out = (out.groupby(unit_col, group_keys=False)
              .apply(_interp_group)
              .reset_index(drop=True))

    # Block inventing values in global bad months for individually-bad obs
    out["impute_blocked"] = False
    if block_global_bad_months and global_bad_months:
        block_mask = out["in_global_bad_month"] & out["value_clean"].isna()
        n_blocked = int(block_mask.sum())
        out.loc[block_mask, "value_filled"] = np.nan
        out.loc[block_mask, "impute_blocked"] = True
        if n_blocked > 0:
            print(f"\n    Blocked {n_blocked:,} imputations in global bad "
                  f"months (individually-bad units only)")

    out["imputed"] = out["value_clean"].isna() & out["value_filled"].notna()
    out.drop(columns=["_month"], inplace=True)

    print_flag_summary(out, label=label)
    return out


# =============================================================================
# 13) Department aggregation from filled districts
# =============================================================================
def build_filled_department_panel(dist_out: pd.DataFrame) -> pd.DataFrame:
    dist_tmp = dist_out.copy()
    if "month" not in dist_tmp.columns:
        dist_tmp["month"] = pd.to_datetime(dist_tmp["date"]).dt.month

    group_cols = ["DEPT_CODE", "year", "month", "date"]
    agg_dict = {}

    for col, fun in [
        ("ntl_sum_eqarea", "sum"),
        ("ntl_sum_eqarea_clean", "sum"),
        ("ntl_sum_eqarea_filled", "sum"),
    ]:
        if col in dist_tmp.columns:
            agg_dict[col] = (col, fun)

    if "ntl_mean" in dist_tmp.columns:
        agg_dict["ntl_mean"] = ("ntl_mean", "mean")
    if "ntl_median" in dist_tmp.columns:
        agg_dict["ntl_median"] = ("ntl_median", "mean")
    if "ntl_count" in dist_tmp.columns:
        agg_dict["ntl_count"] = ("ntl_count", "sum")

    agg_dict["n_districts"] = ("UBIGEO", "count")

    if "imputed" in dist_tmp.columns:
        agg_dict["imputed_districts"] = ("imputed", "sum")
    if "missing_share" in dist_tmp.columns:
        agg_dict["avg_missing_share"] = ("missing_share", "mean")
    if "is_bad" in dist_tmp.columns:
        agg_dict["bad_districts"] = ("is_bad", "sum")
    if "in_global_bad_month" in dist_tmp.columns:
        agg_dict["is_global_bad_month"] = ("in_global_bad_month", "max")

    dept = dist_tmp.groupby(group_cols, as_index=False).agg(**agg_dict)
    dept["department"] = dept["DEPT_CODE"].map(DEPT_NAMES)
    return dept


# =============================================================================
# 14) QC plots
# =============================================================================
def plot_qc_national(original: pd.Series, filled: pd.Series,
                     out_path: Path, global_bad_dates: set = None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(original.index, original.values, label="Original",
                 linewidth=1.2, alpha=0.7)
    axes[0].plot(filled.index, filled.values, label="Filled", linewidth=1.2)

    if global_bad_dates:
        for d in sorted(global_bad_dates):
            if d in original.index or d in filled.index:
                axes[0].axvspan(d, d + pd.DateOffset(months=1),
                                alpha=0.15, color="red", zorder=0)
                axes[1].axvspan(d, d + pd.DateOffset(months=1),
                                alpha=0.15, color="red", zorder=0)

    axes[0].set_title("Peru Total NTL — Original vs Filled (v7.1)")
    axes[0].set_ylabel("NTL total (radiance × km²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    diff = filled.reindex(original.index) - original
    axes[1].bar(diff.index, diff.values, width=25, color="coral", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Imputation Adjustment (Filled − Original)")
    axes[1].set_ylabel("Δ NTL total")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"✓ Saved plot: {out_path}")


# =============================================================================
# 15) Imputation runner
# =============================================================================
def run_imputation():
    print("\n" + "=" * 70)
    print("IMPUTATION v7.1: DISTRICT + DEPT + NATIONAL + (OPTIONAL) ROSSI GRID")
    print("=" * 70)

    if not DISTRICT_CSV.exists():
        raise FileNotFoundError(f"District CSV not found: {DISTRICT_CSV}")

    dist = pd.read_csv(DISTRICT_CSV)
    if "date" not in dist.columns:
        raise ValueError("District CSV must contain a 'date' column")

    print(f"\nLoaded district panel: {DISTRICT_CSV}  shape={dist.shape}")
    print(f"Districts: {dist['UBIGEO'].nunique()}  "
          f"Date range: {dist['date'].min()} → {dist['date'].max()}")

    # Pre-scan for global bad months
    tmp = dist.copy()
    tmp["date"] = _ensure_month_start(tmp["date"])
    if DIST_COUNT_COL in tmp.columns:
        tmp = add_expected_and_missing_share(tmp, "UBIGEO", DIST_COUNT_COL)
    else:
        tmp["missing_share"] = np.nan

    month_tbl = compute_global_bad_months(
        df=tmp, date_col="date", unit_col="UBIGEO",
        missing_share_col="missing_share", value_col=DIST_TOTAL_COL,
        miss_thresh=BAD_MISS_THRESH, share_thresh=GLOBAL_BAD_SHARE_THRESH,
        mean_miss_thresh=GLOBAL_BAD_MEAN_MISS,
        flag_neg_total=GLOBAL_BAD_NEG_TOTAL,
    )
    month_tbl.to_csv(OUT_BAD_MONTHS_CSV, index=False)
    print(f"\nSaved global month diagnostics: {OUT_BAD_MONTHS_CSV}")

    n_bad_months = int(month_tbl["is_global_bad"].sum())
    print(f"Global bad months: {n_bad_months}")
    if n_bad_months > 0:
        bad_rows = month_tbl[month_tbl["is_global_bad"]].head(10)
        print(bad_rows[["date", "share_units_high_miss",
                         "mean_missing_share", "national_total"]]
              .to_string(index=False))

    global_bad_months = set(
        pd.to_datetime(month_tbl.loc[month_tbl["is_global_bad"], "date"])
    )

    # Fill district panel
    dist_filled = fill_panel_monthly(
        df=dist,
        unit_col="UBIGEO",
        date_col="date",
        value_col=DIST_TOTAL_COL,
        count_col=(DIST_COUNT_COL
                   if DIST_COUNT_COL in dist.columns else None),
        bad_missing_thresh=BAD_MISS_THRESH,
        interp_limit=INTERP_LIMIT,
        use_month_median_first=USE_MONTH_MEDIAN_FIRST,
        mad_z=MAD_Z_THRESHOLD,
        max_abs_value=MAX_ABS_VALUE,
        block_global_bad_months=BLOCK_GLOBAL_BAD_MONTHS,
        global_bad_months=(global_bad_months
                           if BLOCK_GLOBAL_BAD_MONTHS else None),
        label="District",
    )

    plot_missing_share_histogram(dist_filled, OUT_QC_MISSING_PLOT,
                                 label="District")

    dist_out = dist_filled.copy()
    dist_out.rename(columns={
        "value_clean":  "ntl_sum_eqarea_clean",
        "value_filled": "ntl_sum_eqarea_filled",
    }, inplace=True)

    dist_out.to_csv(OUT_DISTRICT_FILLED, index=False)
    print(f"\n✓ Saved: {OUT_DISTRICT_FILLED}  shape={dist_out.shape}")

    # Department from filled districts
    dept_filled = build_filled_department_panel(dist_out)
    dept_filled.to_csv(OUT_DEPT_FILLED, index=False)
    print(f"✓ Saved: {OUT_DEPT_FILLED}  shape={dept_filled.shape}")

    # National series + QC plot
    dept_filled["date"] = _ensure_month_start(dept_filled["date"])
    if ("ntl_sum_eqarea" in dept_filled.columns
            and "ntl_sum_eqarea_filled" in dept_filled.columns):
        nat_original = (
            dept_filled.groupby("date")["ntl_sum_eqarea"].sum().sort_index()
        )
        nat_filled = (
            dept_filled.groupby("date")["ntl_sum_eqarea_filled"]
            .sum().sort_index()
        )

        nat_df = pd.DataFrame({
            "date": nat_original.index,
            "ntl_total_original": nat_original.values,
            "ntl_total_filled": nat_filled.values,
        })
        nat_df["global_bad_month"] = nat_df["date"].isin(global_bad_months)
        nat_df.to_csv(OUT_NATIONAL_FILLED, index=False)
        print(f"✓ Saved: {OUT_NATIONAL_FILLED}  shape={nat_df.shape}")

        plot_qc_national(nat_original, nat_filled, OUT_QC_PLOT,
                         global_bad_dates=global_bad_months)
    else:
        print("⚠ Could not build national QC series (missing columns).")

    # Optional: Rossi grid fill
    if ROSSI_CSV.exists():
        rossi = pd.read_csv(ROSSI_CSV)
        if "date" not in rossi.columns:
            raise ValueError("Rossi CSV must contain 'date' column")

        rossi["date"] = _ensure_month_start(rossi["date"])
        rossi["cell_id"] = (
            rossi["lon"].astype(str) + "_" + rossi["lat"].astype(str)
        )
        print(f"\nLoaded Rossi monthly grid: {ROSSI_CSV}  "
              f"shape={rossi.shape}")
        print(f"Grid cells: {rossi['cell_id'].nunique()}")

        tmp2 = rossi.copy()
        if ROSSI_COUNT_COL in tmp2.columns:
            tmp2 = add_expected_and_missing_share(
                tmp2, "cell_id", ROSSI_COUNT_COL
            )
        else:
            tmp2["missing_share"] = np.nan

        month_tbl2 = compute_global_bad_months(
            df=tmp2, date_col="date", unit_col="cell_id",
            missing_share_col="missing_share", value_col=ROSSI_TOTAL_COL,
            miss_thresh=BAD_MISS_THRESH,
            share_thresh=GLOBAL_BAD_SHARE_THRESH,
            mean_miss_thresh=GLOBAL_BAD_MEAN_MISS,
            flag_neg_total=GLOBAL_BAD_NEG_TOTAL,
        )
        rossi_bad_months = set(
            pd.to_datetime(
                month_tbl2.loc[month_tbl2["is_global_bad"], "date"]
            )
        )

        rossi_filled = fill_panel_monthly(
            df=rossi,
            unit_col="cell_id",
            date_col="date",
            value_col=ROSSI_TOTAL_COL,
            count_col=(ROSSI_COUNT_COL
                       if ROSSI_COUNT_COL in rossi.columns else None),
            bad_missing_thresh=BAD_MISS_THRESH,
            interp_limit=INTERP_LIMIT,
            use_month_median_first=USE_MONTH_MEDIAN_FIRST,
            mad_z=MAD_Z_THRESHOLD,
            max_abs_value=MAX_ABS_VALUE,
            block_global_bad_months=BLOCK_GLOBAL_BAD_MONTHS,
            global_bad_months=(rossi_bad_months
                               if BLOCK_GLOBAL_BAD_MONTHS else None),
            label="Rossi grid",
        )

        rossi_out = rossi_filled.copy()
        rossi_out.rename(columns={
            "value_clean":  "ntl_total_clean",
            "value_filled": "ntl_total_filled",
        }, inplace=True)

        out_rossi = OUTPUT_DIR / "ntl_monthly_rossi_grid_0p5deg_filled.csv"
        rossi_out.to_csv(out_rossi, index=False)
        print(f"\n✓ Saved: {out_rossi}  shape={rossi_out.shape}")
    else:
        print(f"\nRossi monthly grid CSV not found (skipping): {ROSSI_CSV}")

    print("\n✅ IMPUTATION COMPLETE (v7.1)")


# =============================================================================
# 16) RUN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update NTL monthly pipeline (VNP46A3 VIIRS)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip NASA Earthdata download (use existing HDF5 files)")
    parser.add_argument("--start-date", default=None,
                        help="Override START_DATE (YYYY-MM, default: 2012-01)")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip months that already have mosaics in mosaics_wgs84/")
    parser.add_argument("--skip-imputation", action="store_true",
                        help="Skip imputation step")
    parser.add_argument("--skip-rossi", action="store_true",
                        help="Skip Rossi 0.5° grid outputs")
    parser.add_argument("--no-force-clean", action="store_true",
                        help="Do NOT delete stale outputs before rebuild (faster for incremental)")
    args = parser.parse_args()

    if args.start_date:
        START_DATE = args.start_date

    DOWNLOAD_FROM_NASA = not args.skip_download
    FORCE_CLEAN        = not args.no_force_clean and not args.incremental
    DO_EQAREA_TOTALS   = True
    RUN_ROSSI_GRID     = not args.skip_rossi
    RUN_IMPUTATION     = not args.skip_imputation

    print(f"NTL pipeline v7.1")
    print(f"  START_DATE      : {START_DATE}")
    print(f"  END_DATE        : {END_DATE}")
    print(f"  SHAPEFILE       : {SHAPEFILE_PATH}")
    print(f"  skip_download   : {args.skip_download}")
    print(f"  incremental     : {args.incremental}")
    print(f"  force_clean     : {FORCE_CLEAN}")
    print(f"  run_rossi       : {RUN_ROSSI_GRID}")
    print(f"  run_imputation  : {RUN_IMPUTATION}")

    if args.incremental:
        # Determine which months are already mosaicked
        existing = sorted((SATELLITE_TIFF / "mosaics_wgs84").glob("*.tif"))
        existing_months = set()
        for f in existing:
            m = re.search(r"(\d{4})_(\d{2})_mosaic", f.name)
            if m:
                existing_months.add(f"{m.group(1)}-{m.group(2)}")
        print(f"  Incremental: {len(existing_months)} months already mosaicked, last: "
              f"{max(existing_months) if existing_months else 'none'}")

    # Build raw panels
    ntl_district, ntl_department, rossi_monthly, rossi_annual = main_build(
        download=DOWNLOAD_FROM_NASA,
        force_clean=FORCE_CLEAN,
        do_eqarea_totals=DO_EQAREA_TOTALS,
        run_rossi_grid=RUN_ROSSI_GRID,
    )

    # Imputation
    if RUN_IMPUTATION:
        run_imputation()