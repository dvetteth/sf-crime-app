# Streamlit heatmap for SF crime using 6‑month CSV chunks
# Filenames  sfcrime_YYYY_H1.csv  /  sfcrime_YYYY_H2.csv
# Run: streamlit run path/to/sf-crime-app.py

import os
import re
import pandas as pd
import streamlit as st
import pydeck as pdk

#map set up
DATA_DIR = "sf-crime/data/sixmonths"
DATE_COL = "Incident Date"  
LAT_COL  = "Latitude"
LON_COL  = "Longitude"
CAT_COL  = "Category"
DIST_COL = "PdDistrict"
NEIGH_COL = "Neighborhoods"  # optional; app hides if not present


#adjust map views
DEFAULT_VIEW = dict(lat=37.76, lon=-122.44, zoom=11, pitch=0, bearing=0)
MAX_POINTS_FOR_MAP = 250_000  #manage app load limits

st.set_page_config(page_title="SF Crime Heatmap", layout="wide")
st.title("SF Crime Heatmap")

#file set ups
def parse_semester_filename(filename: str):
    """
    Accept names like:
      sfcrime_2020_H1.csv
      sf_crime_2021_H2.csv
      anything_2019_H2.csv
    Returns (year:int, half:int) or (None, None) if no match.
    """

    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'(?P<year>\d{4})_H(?P<half>[12])$', base)
    if not m:
        return None, None
    return int(m.group("year")), int(m.group("half"))

def list_semester_files():
    """Return full paths of semester CSVs that match the pattern."""
    if not os.path.isdir(DATA_DIR):
        return []
    out = []
    for f in os.listdir(DATA_DIR):
        if not f.endswith(".csv"):
            continue
        y, h = parse_semester_filename(f)
        if y is None:
            continue
        out.append(os.path.join(DATA_DIR, f))
    return sorted(out)

def semesters_covering(start_ts: pd.Timestamp, end_ts: pd.Timestamp, files):
    """
    Given a date window [start, end], return the subset of semester files
    whose 6‑month span overlaps that window.
    """
    chosen = []
    for f in files:
        y, h = parse_semester_filename(f)
        if y is None:
            continue
        sem_start = pd.Timestamp(y, 1, 1) if h == 1 else pd.Timestamp(y, 7, 1)
        sem_end   = pd.Timestamp(y, 6, 30) if h == 1 else pd.Timestamp(y, 12, 31)
        if (sem_start <= end_ts) and (sem_end >= start_ts):
            chosen.append(f)
    return chosen

def discover_date_bounds(files):
    """
    First try filename‑based bounds. If that fails, scan DATE_COL only.
    Returns (min_ts, max_ts) or (None, None) if nothing can be inferred.
    """
    years = []
    for f in files:
        y, h = parse_semester_filename(f)
        if y is not None:
            years.append(y)
    if years:
        return pd.Timestamp(min(years), 1, 1), pd.Timestamp(max(years), 12, 31)

    # Fallback: read only the date column (fast & low‑memory)
    overall_min = None
    overall_max = None
    for f in files:
        try:
            s = pd.read_csv(f, usecols=[DATE_COL], parse_dates=[DATE_COL])[DATE_COL]
            fmin, fmax = s.min(), s.max()
            overall_min = fmin if overall_min is None else min(overall_min, fmin)
            overall_max = fmax if overall_max is None else max(overall_max, fmax)
        except Exception:
            continue
    return overall_min, overall_max

# cacheing

@st.cache_data(show_spinner=False)
def load_semester_csv(path: str) -> pd.DataFrame:
    """Load one CSV, parse dates, coerce lat/lon, drop invalids."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.dropna(subset=[LAT_COL, LON_COL])
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    return df.dropna(subset=[LAT_COL, LON_COL])

@st.cache_data(show_spinner=True)
def load_data_for_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Pick overlapping files, load & concat, then final date filter."""
    files = list_semester_files()
    if not files:
        return pd.DataFrame()
    selected = semesters_covering(start_ts, end_ts, files)
    if not selected:
        return pd.DataFrame()
    frames = [load_semester_csv(p) for p in sorted(selected)]
    df = pd.concat(frames, ignore_index=True)
    mask = (df[DATE_COL] >= start_ts) & (df[DATE_COL] <= end_ts)
    return df.loc[mask].reset_index(drop=True)

# filter helpers

def derive_filter_options(df: pd.DataFrame):
    cats = sorted(df[CAT_COL].dropna().unique()) if CAT_COL in df.columns else []
    dists = sorted(df[DIST_COL].dropna().unique()) if DIST_COL in df.columns else []
    neighs = sorted(df[NEIGH_COL].dropna().unique()) if NEIGH_COL in df.columns else []
    return cats, dists, neighs

def apply_filters(df: pd.DataFrame, cats=None, dists=None, neighs=None):
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if cats and CAT_COL in df.columns:
        mask &= df[CAT_COL].isin(cats)
    if dists and DIST_COL in df.columns:
        mask &= df[DIST_COL].isin(dists)
    if neighs and NEIGH_COL in df.columns:
        mask &= df[NEIGH_COL].isin(neighs)
    return df.loc[mask].copy()

# renderingmap

def shrink_for_map(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only lon/lat, dedupe, cast to float32, and cap row count."""
    if df.empty:
        return df
    small = df[[LON_COL, LAT_COL]].dropna().astype("float32")
    small = small.drop_duplicates()
    if len(small) > MAX_POINTS_FOR_MAP:
        small = small.sample(MAX_POINTS_FOR_MAP, random_state=42)
    return small

def render_map(coords_df: pd.DataFrame):
    view = pdk.ViewState(
    latitude=DEFAULT_VIEW["lat"],
    longitude=DEFAULT_VIEW["lon"],
    zoom=DEFAULT_VIEW["zoom"],
    pitch=DEFAULT_VIEW["pitch"],
    bearing=DEFAULT_VIEW.get("bearing", 0),
    )
    layer = pdk.Layer(
        "HeatmapLayer",
        data=coords_df,
        get_position=f"[{LON_COL}, {LAT_COL}]",
        aggregation="SUM",
        radius=200,
        opacity=0.8,
        threshold=0.01,
    )
    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer],
                             tooltip={"text": "Heat intensity by incident density"}))
    

# stream lit app dev

# 1) Verify data availability & get initial date bounds
all_files = list_semester_files()
if not all_files:
    st.error(f"No data files found in: {DATA_DIR}\n"
             f"Expected pattern: sfcrime_YYYY_H1.csv / sfcrime_YYYY_H2.csv")
    st.stop()

min_date, max_date = discover_date_bounds(all_files)
if min_date is None or max_date is None:
    st.error(f"Could not infer date bounds. Check filenames and that '{DATE_COL}' exists in CSVs.")
    st.stop()

# 2) Sidebar date filter
st.sidebar.header("Filters")
date_val = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if isinstance(date_val, (list, tuple)) and len(date_val) == 2:
    start_date = pd.to_datetime(date_val[0])
    end_date   = pd.to_datetime(date_val[1])
else:
    start_date, end_date = min_date, max_date

# 3) Load data for window
with st.spinner("Loading data…"):
    df = load_data_for_range(start_date, end_date)

if df.empty:
    st.warning("No records found for the selected date range.")
    st.stop()

# 4) Sidebar filters (built from loaded data)
cats_all, dists_all, neighs_all = derive_filter_options(df)
sel_cats  = st.sidebar.multiselect("Category", options=cats_all,  default=cats_all[:10] or cats_all)
sel_dists = st.sidebar.multiselect("Police District", options=dists_all, default=dists_all)
sel_neighs = st.sidebar.multiselect("Neighborhoods", options=neighs_all, default=neighs_all) if neighs_all else []

# 5) Apply non‑date filters
view = apply_filters(df, sel_cats, sel_dists, sel_neighs)

# 6) KPIs
c1, c2 = st.columns(2)
c1.metric("Incidents (filtered)", f"{len(view):,}")
c2.metric("Date window", f"{start_date.date()} → {end_date.date()}")

# 7) Map (send only coords; cap payload)
coords = shrink_for_map(view)
if len(view) > len(coords):
    st.info(f"Showing {len(coords):,} points on the map (downsampled from {len(view):,} to keep it fast).")
render_map(coords)

# 8) Summary + Download (use full filtered view)
with st.expander("Summary tables"):
    if CAT_COL in view.columns:
        st.write("Incidents by category")
        st.dataframe(view[CAT_COL].value_counts().rename("count").to_frame())
    if DIST_COL in view.columns:
        st.write("Incidents by district")
        st.dataframe(view[DIST_COL].value_counts().rename("count").to_frame())

st.download_button(
    "Download filtered CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name=f"sfcrime_filtered_{start_date.date()}_{end_date.date()}.csv",
    mime="text/csv",
)
