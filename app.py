# =====================================
# SPC / PC ANALYSIS DASHBOARD – FINAL
# Power BI style | Stable | Production
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PC / SPC Analysis Dashboard",
    layout="wide"
)

st.title("📊 PC / SPC Analysis Dashboard")
st.caption("Power BI style – Management View")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_main_data():
    url = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"
    return pd.read_csv(url)

@st.cache_data
def load_gauge_range():
    url = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
    return pd.read_csv(url)

df = load_main_data()
gauge_df = load_gauge_range()

# =========================
# NORMALIZE COLUMN NAMES
# =========================
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
gauge_df.columns = gauge_df.columns.str.strip().str.upper().str.replace(" ", "_")

# =========================
# QUALITY GROUP LOGIC
# =========================
df["QUALITY_GROUP"] = df["QUALITY_CODE"].astype(str)
df.loc[df["QUALITY_GROUP"].isin(["CQ00", "CQ06"]), "QUALITY_GROUP"] = "CQ00/CQ06"

# =========================
# PARSE GAUGE RANGE
# =========================
def parse_range(text):
    nums = re.findall(r"\d+\.\d+|\d+", str(text))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[-1])

gauge_ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r["RANGE_NAME"])
    if lo is not None:
        gauge_ranges.append((lo, hi, r["RANGE_NAME"]))

# =========================
# AUTO DETECT GAUGE COLUMN
# =========================
possible_gauge_cols = [
    "ORDER_GAUGE", "GAUGE", "THICKNESS", "ORDER_THICKNESS"
]

gauge_col = None
for c in possible_gauge_cols:
    if c in df.columns:
        gauge_col = c
        break

if gauge_col is None:
    st.error("❌ Cannot detect gauge / thickness column")
    st.stop()

df["GAUGE_VALUE"] = pd.to_numeric(df[gauge_col], errors="coerce")

def map_gauge_range(v):
    if pd.isna(v):
        return None
    for lo, hi, name in gauge_ranges:
        if lo <= v < hi:
            return name
    return None

df["GAUGE_RANGE"] = df["GAUGE_VALUE"].apply(map_gauge_range)

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔎 Filters")

material = st.sidebar.multiselect(
    "Classify Material",
    sorted(df["CLASSIFY_MATERIAL"].dropna().unique())
)

coating = st.sidebar.multiselect(
    "Metallic Coating Type",
    sorted(df["METALLIC_COATING_TYPE"].dropna().unique())
)

quality = st.sidebar.multiselect(
    "Quality Group",
    sorted(df["QUALITY_GROUP"].dropna().unique())
)

gauge_range = st.sidebar.multiselect(
    "Gauge Range",
    sorted(df["GAUGE_RANGE"].dropna().unique())
)

steel = st.sidebar.multiselect(
    "HR Steel Grade",
    sorted(df["HR_STEEL_GRADE"].dropna().unique())
)

# =========================
# APPLY FILTERS
# =========================
df_f = df.copy()

if material:
    df_f = df_f[df_f["CLASSIFY_MATERIAL"].isin(material)]

if coating:
    df_f = df_f[df_f["METALLIC_COATING_TYPE"].isin(coating)]

if quality:
    df_f = df_f[df_f["QUALITY_GROUP"].isin(quality)]

if gauge_range:
    df_f = df_f[df_f["GAUGE_RANGE"].isin(gauge_range)]

if steel:
    df_f = df_f[df_f["HR_STEEL_GRADE"].isin(steel)]

# =========================
# KPI SUMMARY
# =========================
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Coils", len(df_f))

with c2:
    st.metric("Valid SPC Groups (≥30)",
              df_f.groupby(
                  ["CLASSIFY_MATERIAL","METALLIC_COATING_TYPE","QUALITY_GROUP","GAUGE_RANGE","HR_STEEL_GRADE"]
              ).size().ge(30).sum())

with c3:
    st.metric("NG Rate (%)",
              round((df_f["QA_RESULT"] == "NG").mean() * 100, 2)
              if "QA_RESULT" in df_f.columns else 0)

# =========================
# MAIN TABLE – POWER BI STYLE
# =========================
st.subheader("📋 Data Table")

st.dataframe(
    df_f,
    use_container_width=True,
    height=520
)

# =========================
# MANAGEMENT NOTE
# =========================
with st.expander("📌 PC Analysis Logic – Management Note"):
    st.markdown("""
- Data grouped by **Classify Material, Metallic Coating Type, Quality Group, Gauge Range, HR Steel Grade**
- **CQ00 & CQ06 merged** due to equivalent quality behavior
- Thickness analyzed by **predefined gauge ranges**
- **SPC valid only when ≥ 30 coils**
- **QA logic is strict: 1 NG → FAIL**
""")
