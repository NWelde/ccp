# sf_zoning_affordability_viz.py
# Updated EXACTLY for your uploaded CSVs:
#   - Building_Permits_20260208.csv  (has: "Issued Date", "Zipcode")
#   - San_Francisco_ZIP_Codes_20260208.csv (has: "zip" + WKT "geometry" polygons)
#   - Zip_zori_uc_sfrcondomfr_sm_month.csv (has: "RegionName" ZIP + monthly date columns like "2015-01-31")
#
# Creates (saved to out/):
#   1) map_permits_by_zip.png   (choropleth — building permit counts by ZIP)
#   2) bar_top_permits.png      (top ZIPs by building permit volume)
#   3) scatter_permits_vs_rent_change.png (building permits vs inflation-adjusted rent % change)
#
# Notes:
# - Analysis window is 2017–2025 (aligns permit and ZORI data to the same period).
# - All rent values are converted to 2025 dollars using BLS CPI-U annual averages.
# - The permits dataset is huge; this script reads it in chunks.
# - "Permits" refers exclusively to SF building permits (construction/renovation).
#
# CPI-U source: U.S. Bureau of Labor Statistics, All Items, U.S. City Average (annual avg).
#
# Install deps:
#   pip install pandas matplotlib geopandas shapely

from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely import wkt


# -----------------------
# Paths (match your filenames)
# -----------------------
DATA_DIR = "data"
OUT_DIR = "out"

PERMITS_CSV = os.path.join(DATA_DIR, "Building_Permits_20260208.csv")
ZIP_WKT_CSV = os.path.join(DATA_DIR, "San_Francisco_ZIP_Codes_20260208.csv")
ZORI_CSV = os.path.join(DATA_DIR, "Zip_zori_uc_sfrcondomfr_sm_month.csv")


# -----------------------
# Analysis window  (2017–2025)
# -----------------------
START_YEAR = 2017
END_YEAR = 2025

# -----------------------
# Inflation adjustment
# BLS CPI-U, All Items, U.S. City Average — annual averages (base year = 2025)
# Source: https://www.bls.gov/cpi/
# -----------------------
CPI_BY_YEAR: dict[int, float] = {
    2017: 245.1,
    2018: 251.1,
    2019: 255.7,
    2020: 258.8,
    2021: 271.0,
    2022: 292.7,
    2023: 304.7,
    2024: 314.5,
    2025: 320.0,  # base year — all rents expressed in 2025 dollars
}


def cpi_factor(year: int) -> float:
    """Return the multiplier to convert a dollar amount from `year` to 2025 dollars."""
    base = CPI_BY_YEAR[2025]
    cpi = CPI_BY_YEAR.get(year)
    if cpi is None:
        raise ValueError(f"No CPI entry for year {year}. Add it to CPI_BY_YEAR.")
    return base / cpi


# -----------------------
# Helpers
# -----------------------
def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def normalize_zip(series: pd.Series) -> pd.Series:
    """Extract 5-digit ZIP codes from whatever text."""
    return series.astype(str).str.extract(r"(\d{5})", expand=False)

def infer_zori_year_bounds(zori_csv: str) -> tuple[int, int]:
    """Read only header and infer min/max year from date columns like YYYY-MM-DD."""
    cols = pd.read_csv(zori_csv, nrows=0).columns.astype(str).tolist()
    date_cols = [c for c in cols if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
    years = sorted({int(c[:4]) for c in date_cols})
    return years[0], years[-1]

# -----------------------
# Load SF ZIP polygons (WKT)
# -----------------------
def load_sf_zip_polygons(zip_wkt_csv: str) -> gpd.GeoDataFrame:
    """
    Your SF ZIP file has a 'geometry' column containing WKT polygons,
    plus a 'zip' column we can join on.
    """
    df = pd.read_csv(zip_wkt_csv, usecols=["geometry", "zip"])
    df["zip"] = normalize_zip(df["zip"])
    df = df.dropna(subset=["zip", "geometry"]).copy()

    # Convert WKT -> shapely geometry
    df["geometry"] = df["geometry"].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf[["zip", "geometry"]]

# -----------------------
# Building permits: count by ZIP (chunked)
# -----------------------
def permits_count_by_zip(
    permits_csv: str,
    start_year: int,
    end_year: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Count SF building permits (construction & renovation) by ZIP code.
    Uses only the columns that exist in your permits file:
      - 'Issued Date'
      - 'Zipcode'
    Filters to permits issued between start_year and end_year (inclusive).
    """
    usecols = ["Issued Date", "Zipcode"]
    counts: dict[str, int] = {}

    for chunk in pd.read_csv(permits_csv, usecols=usecols, chunksize=chunksize, low_memory=True):
        dt = pd.to_datetime(chunk["Issued Date"], format="mixed", errors="coerce")
        yr = dt.dt.year
        mask = (yr >= start_year) & (yr <= end_year)
        if not mask.any():
            continue

        sub = chunk.loc[mask].copy()
        sub["zip"] = normalize_zip(sub["Zipcode"])
        sub = sub.dropna(subset=["zip"])

        vc = sub["zip"].value_counts()
        for z, n in vc.items():
            z = str(z)
            counts[z] = counts.get(z, 0) + int(n)

    out = pd.DataFrame({"zip": list(counts.keys()), "permits": list(counts.values())})
    return out.sort_values("permits", ascending=False)

# -----------------------
# ZORI: wide -> long; then inflation-adjusted rent change
# -----------------------
def load_zori_long(zori_csv: str, city_filter: str = "San Francisco") -> pd.DataFrame:
    """
    Your ZORI ZIP file is WIDE:
      - ZIP in 'RegionName'
      - many monthly columns: '2015-01-31', ...
    We'll melt it to long: zip, date, rent (nominal), rent_2025 (2025 dollars).
    """
    df = pd.read_csv(zori_csv)

    # Confirm expected columns in your file
    if "RegionName" not in df.columns:
        raise KeyError("Expected Zillow ZIP column 'RegionName' not found.")
    if "City" in df.columns:
        df = df[df["City"].astype(str) == city_filter].copy()

    df["zip"] = normalize_zip(df["RegionName"])
    df = df.dropna(subset=["zip"]).copy()

    date_cols = [c for c in df.columns.astype(str) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
    if not date_cols:
        raise KeyError("No monthly date columns found in ZORI file (expected YYYY-MM-DD).")

    long = df[["zip"] + date_cols].melt(id_vars=["zip"], var_name="date", value_name="rent")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["rent"] = pd.to_numeric(long["rent"], errors="coerce")
    long = long.dropna(subset=["date", "rent"]).copy()
    long["year"] = long["date"].dt.year

    # Inflate nominal rent to 2025 dollars row-by-row
    long["rent_2025"] = long.apply(
        lambda row: row["rent"] * cpi_factor(int(row["year"]))
        if int(row["year"]) in CPI_BY_YEAR else float("nan"),
        axis=1,
    )

    return long

def rent_change_by_zip(zori_long: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Compute % rent change by ZIP using inflation-adjusted (2025-dollar) yearly average rent.
    Using real (CPI-adjusted) values ensures that a ZIP where rents merely kept pace with
    inflation does not appear to have grown — and does not produce a spurious negative change.
    """
    yearly = zori_long.groupby(["zip", "year"], as_index=False)["rent_2025"].mean()

    start = yearly[yearly["year"] == start_year][["zip", "rent_2025"]].rename(
        columns={"rent_2025": "rent_start_2025"}
    )
    end = yearly[yearly["year"] == end_year][["zip", "rent_2025"]].rename(
        columns={"rent_2025": "rent_end_2025"}
    )

    merged = start.merge(end, on="zip", how="inner")
    # Both values are already in 2025 dollars → % change is a real (inflation-adjusted) figure
    merged["rent_pct_change"] = (
        (merged["rent_end_2025"] - merged["rent_start_2025"]) / merged["rent_start_2025"] * 100.0
    )
    return merged

# -----------------------
# Plotting
# -----------------------
def plot_map(zip_gdf: gpd.GeoDataFrame, permits: pd.DataFrame, start_year: int, end_year: int) -> None:
    g = zip_gdf.merge(permits, on="zip", how="left")
    g["permits"] = g["permits"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 10))
    g.plot(column="permits", ax=ax, legend=True, linewidth=0.2, edgecolor="white")
    ax.set_title(f"San Francisco Building Permits by ZIP Code ({start_year}–{end_year})", fontsize=13)
    ax.axis("off")

    caption = (
        f"Building permits issued by the SF Department of Building Inspection, {start_year}–{end_year}.\n"
        "Each permit covers new construction, additions, alterations, or repairs to a structure.\n"
        "Darker shading = more permits issued in that ZIP code over the period."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8.5,
             color="#444444", wrap=True)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, "map_permits_by_zip.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_bar_top_permits(permits: pd.DataFrame, start_year: int, end_year: int, n: int = 20) -> None:
    df = permits.head(n).copy()
    n_actual = len(df)

    # Scale figure width so each bar gets ~0.7 inches; minimum 10 inches
    fig_w = max(10, n_actual * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    # Color bars by permit count — darker = more permits
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=df["permits"].min(), vmax=df["permits"].max())
    colors = cm.Blues(norm(df["permits"].values) * 0.6 + 0.35)  # map to [0.35, 0.95] range

    bars = ax.bar(df["zip"].astype(str), df["permits"], color=colors, width=0.65, edgecolor="white", linewidth=0.5)

    # Value labels on top of each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + df["permits"].max() * 0.01,
            f"{h:,.0f}",
            ha="center", va="bottom", fontsize=7.5, color="#333333"
        )

    ax.set_title(
        f"Top {n_actual} ZIP Codes by SF Building Permits Issued ({start_year}–{end_year})",
        fontsize=13, pad=12
    )
    ax.set_xlabel("ZIP Code", fontsize=11)
    ax.set_ylabel("Building Permit Count", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, df["permits"].max() * 1.12)  # headroom for value labels
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=9)

    caption = (
        f"Total SF building permits (new construction, additions, alterations, repairs) issued per ZIP code, {start_year}–{end_year}.\n"
        "Source: SF Department of Building Inspection open data."
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=8.5, color="#444444")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bar_top_permits.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_scatter(metrics: pd.DataFrame, start_year: int, end_year: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(metrics["permits"], metrics["rent_pct_change"])
    ax.set_title(
        f"SF Building Permits vs. Real Rent Change by ZIP Code ({start_year}→{end_year})",
        fontsize=12
    )
    ax.set_xlabel(f"Total building permits issued ({start_year}–{end_year})", fontsize=10)
    ax.set_ylabel(f"Real rent change, % ({start_year}→{end_year}, 2025 dollars)", fontsize=10)

    # Add a horizontal reference line at 0%
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")

    # Label a few notable ZIPs
    label_df = pd.concat([
        metrics.nlargest(6, "permits"),
        metrics.nlargest(6, "rent_pct_change")
    ]).drop_duplicates("zip")

    for _, r in label_df.iterrows():
        ax.annotate(
            str(r["zip"]),
            xy=(r["permits"], r["rent_pct_change"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    caption = (
        f"Each dot is one SF ZIP code. X-axis: total building permits issued {start_year}–{end_year}.\n"
        f"Y-axis: real (inflation-adjusted) change in average Zillow Observed Rent Index (ZORI) from {start_year} to {end_year},\n"
        "expressed in 2025 dollars using BLS CPI-U annual averages. Dashed line = 0% real rent change."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8.5, color="#444444")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter_permits_vs_rent_change.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------
# Main
# -----------------------
def main() -> None:
    ensure_out_dir()

    # Cap END_YEAR to whatever Zillow actually has
    zmin, zmax = infer_zori_year_bounds(ZORI_CSV)
    start_year = max(START_YEAR, zmin)
    end_year = min(END_YEAR, zmax)

    if end_year not in CPI_BY_YEAR or start_year not in CPI_BY_YEAR:
        raise ValueError(
            f"CPI data missing for {start_year} or {end_year}. Update CPI_BY_YEAR at the top of this file."
        )

    print(f"Analysis window: {start_year}–{end_year} (building permits & ZORI rents)")
    print(f"All rent values expressed in {CPI_BY_YEAR[2025]:.1f} CPI-equivalent 2025 dollars.")

    # 1) Load SF ZIP polygons (for map)
    zip_gdf = load_sf_zip_polygons(ZIP_WKT_CSV)

    # 2) Building permit counts (chunked, huge file)
    print("Reading building permits (chunked)…")
    permits = permits_count_by_zip(PERMITS_CSV, start_year, end_year)

    # Keep only ZIPs that exist in the SF polygon file (cleaner joins)
    sf_zip_set = set(zip_gdf["zip"].astype(str))
    permits = permits[permits["zip"].isin(sf_zip_set)].copy()

    # 3) ZORI rent changes (inflation-adjusted)
    print("Loading ZORI rent data and adjusting for inflation…")
    zori_long = load_zori_long(ZORI_CSV, city_filter="San Francisco")
    zori_long = zori_long[zori_long["zip"].isin(sf_zip_set)].copy()

    rentchg = rent_change_by_zip(zori_long, start_year, end_year)

    # 4) Merge for scatter
    metrics = permits.merge(rentchg, on="zip", how="inner").copy()

    # Save merged tables for your writeup
    permits.to_csv(os.path.join(OUT_DIR, "permits_by_zip.csv"), index=False)
    rentchg.to_csv(os.path.join(OUT_DIR, "rent_change_by_zip.csv"), index=False)
    metrics.to_csv(os.path.join(OUT_DIR, "merged_metrics_zip.csv"), index=False)

    # 5) Visuals
    print("Generating visuals…")
    plot_map(zip_gdf, permits, start_year, end_year)
    plot_bar_top_permits(permits, start_year, end_year, n=20)
    if not metrics.empty:
        plot_scatter(metrics, start_year, end_year)
    else:
        print("No overlapping ZIPs between permits and ZORI after filtering — scatter skipped.")

    print("Done. Check the out/ folder for PNGs + merged CSVs.")

if __name__ == "__main__":
    main()
