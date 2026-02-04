import streamlit as st
import pandas as pd
import numpy as np
import re

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="PC Analysis Dashboard", layout="wide")

# ===============================
# LOAD DATA (FIXED)
# ===============================
@st.cache_data
def load_main_data():
    url = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
    return pd.read_csv(url)

@st.cache_data
def load_gauge_table():
    url = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"
    return pd.read_csv(url)

df = load_main_data()
gauge_df = load_gauge_table()

# ===============================
# QUALITY GROUP
# ===============================
df["QUALITY_GROUP"] = df["QUALITY_CODE"].apply(
    lambda x: "CQ00/06" if x in ["CQ00", "CQ06"] else x
)

# ===============================
# PARSE GAUGE RANGE
# ===============================
def parse_range(text):
    nums = re.findall(r"\d+\.\d+|\d+", str(text))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[-1])

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r["RANGE_NAME"])
    if lo is not None:
        ranges.append((lo, hi, r["RANGE_NAME"]))

def map_gauge(val):
    try:
        v = float(val)
    except:
        return "UNKNOWN"
    for lo, hi, name in ranges:
        if lo <= v < hi:
            return name
    return "OUT OF RANGE"

df["GAUGE_RANGE"] = df["ORDER_GAUGE"].apply(map_gauge)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Filters")

f = df.copy()

for col, label in [
    ("Classify material", "Classify Material"),
    ("METALLIC COATING TYPE", "Metallic Coating Type"),
    ("QUALITY_GROUP", "Quality Group"),
    ("GAUGE_RANGE", "Gauge Range"),
    ("HR STEEL GRADE", "HR Steel Grade"),
]:
    sel = st.sidebar.multiselect(label, sorted(f[col].dropna().unique()))
    if sel:
        f = f[f[col].isin(sel)]

# ===============================
# KPI
# ===============================
st.title("PC Analysis Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Total Coils", len(f))
c2.metric("Groups", f.groupby([
    "Classify material",
    "METALLIC COATING TYPE",
    "QUALITY_GROUP",
    "GAUGE_RANGE",
    "HR STEEL GRADE"
]).ngroups)
c3.metric("SPC Valid (≥30)", (f.groupby("GAUGE_RANGE").size() >= 30).sum())

# ===============================
# TABLE
# ===============================
st.dataframe(f, use_container_width=True, height=520)

# ===============================
# NOTE
# ===============================
st.markdown("""
### PC Analysis Logic – Management Note
- Group by **Classify material, Metallic Coating Type, Quality Group, Gauge Range, HR Steel Grade**
- **CQ00 + CQ06 merged**
- Thickness analyzed by **range**
- SPC valid only when **≥ 30 coils**
- QA rule: **1 NG → FAIL**
""")
