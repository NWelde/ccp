# SF Building Permits & Rent Affordability Visualization (2017–2025)

Analyzes the relationship between San Francisco building permit activity and inflation-adjusted rent changes by ZIP code. All rent values are converted to 2025 dollars using BLS CPI-U annual averages.

## Output

The script generates three visualizations (saved to `out/`):

1. **map_permits_by_zip.png** — Choropleth map of building permit counts by ZIP code
2. **bar_top_permits.png** — Top 20 ZIP codes ranked by building permit volume
3. **scatter_permits_vs_rent_change.png** — Building permits vs. real (inflation-adjusted) rent change per ZIP

It also exports three CSVs to `out/`: `permits_by_zip.csv`, `rent_change_by_zip.csv`, and `merged_metrics_zip.csv`.

## Data

Place these three CSV files in a `data/` directory (not included in the repo due to size):

| File | Source | Description |
|------|--------|-------------|
| `Building_Permits_20260208.csv` | [SF Open Data](https://data.sfgov.org/) | Building permits with `Issued Date` and `Zipcode` columns |
| `San_Francisco_ZIP_Codes_20260208.csv` | [SF Open Data](https://data.sfgov.org/) | ZIP code boundaries with `zip` and WKT `geometry` columns |
| `Zip_zori_uc_sfrcondomfr_sm_month.csv` | [Zillow Research](https://www.zillow.com/research/data/) | ZORI (Zillow Observed Rent Index) by ZIP, wide format with monthly date columns |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib geopandas shapely
```

## Run

```bash
source .venv/bin/activate
python3 sf_affordability_viz.py
```

Output files will appear in `out/`.

## Notes

- "Permits" refers to SF Department of Building Inspection permits covering new construction, additions, alterations, and repairs.
- Inflation adjustment uses BLS CPI-U (All Items, U.S. City Average) annual averages. The conversion factors are defined in `CPI_BY_YEAR` at the top of the script.
- The analysis window is capped to years available in both the ZORI data and the CPI table (2017–2025).
