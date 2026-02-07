import pandas as pd

ASOF_DATE  = "2022-03-01"   # snapshot
START_DATE = "2021-09-01"   # growth start
END_DATE   = "2022-09-01"   # growth end

IN_PATH  = "data/raw/us_state_vaccinations.csv" 
OUT_PATH = "data/processed/vaccination_summary.csv"

valid_locations = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York State","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
    "District of Columbia","Puerto Rico"
]

# Load + clean
vac = pd.read_csv(IN_PATH)
vac["date"] = pd.to_datetime(vac["date"])

# Keep what we need
keep_cols = ["date", "location", "people_fully_vaccinated_per_hundred"]
missing = [c for c in keep_cols if c not in vac.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

vac = vac[keep_cols].copy()
vac = vac[vac["location"].isin(valid_locations)].copy()
vac = vac.rename(columns={"location": "state", "people_fully_vaccinated_per_hundred": "pct_full"})
vac["state"] = vac["state"].replace({"New York State": "New York"})

# Helper: grab last non null value on or before date
def grab_not_null(df: pd.DataFrame, target_date: str, out_col: str) -> pd.DataFrame:
    t = pd.to_datetime(target_date)
    tmp = df[(df["date"] <= t) & (df["pct_full"].notna())].sort_values(["state", "date"])
    out = tmp.groupby("state", as_index=False).tail(1)[["state", "pct_full"]]
    return out.rename(columns={"pct_full": out_col})

# 1) Max % fully vaccinated achieved
max_full = (
    vac[vac["pct_full"].notna()]
      .groupby("state", as_index=False)["pct_full"]
      .max()
      .rename(columns={"pct_full": "pct_full_max"})
)

# 2) % fully vaccinated as-of a date
asof_col = f"pct_full_asof_{ASOF_DATE}"
asof_full = grab_not_null(vac, ASOF_DATE, asof_col)

# 3) Growth over a date range (percentage points)
start_vals = grab_not_null(vac, START_DATE, f"pct_full_{START_DATE}")
end_vals   = grab_not_null(vac, END_DATE, f"pct_full_{END_DATE}")

growth = start_vals.merge(end_vals, on="state", how="outer")
growth_col = f"pct_full_growth_pp_{START_DATE}_to_{END_DATE}"
growth[growth_col] = growth[f"pct_full_{END_DATE}"] - growth[f"pct_full_{START_DATE}"]
growth = growth[["state", f"pct_full_{START_DATE}", f"pct_full_{END_DATE}", growth_col]]

# Combine
final = (
    max_full
    .merge(asof_full, on="state", how="left")
    .merge(growth, on="state", how="left")
)

final.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print(final.head())
