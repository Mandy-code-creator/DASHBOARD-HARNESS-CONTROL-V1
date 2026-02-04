# ================================
# FULL STREAMLIT APP – FINAL VERSION
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Material-level Hardness Detail", layout="wide")
st.title("📊 Material-level Hardness & Mechanical Detail")

# ================================
# GOOGLE SHEET LINKS
# ================================
MAIN_DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI"
    "/export?format=csv&gid=0"
)

GAUGE_MASTER_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM"
    "/export?format=csv&gid=0"
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_csv(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_csv(MAIN_DATA_URL)
gauge_master = load_csv(GAUGE_MASTER_URL)

# ================================
# FIND METALLIC COLUMN
# ================================
metal_col = next(
    (c for c in raw.columns if "METALLIC" in c.upper() and "COATING" in c.upper()),
    None
)
if metal_col is None:
    st.error("❌ METALLIC COATING TYPE column not found")
    st.stop()

raw["Metallic_Type"] = raw[metal_col]

# ================================
# RENAME COLUMNS
# ================================
df = raw.rename(columns={
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "Claasify material": "Rolling_Type",
    "TOP COATMASS": "Top_Coatmass",
    "ORDER GAUGE": "Order_Gauge",
    "COIL NO": "COIL_NO",
    "QUALITY_CODE": "Quality_Code",
    "Standard Hardness": "Std_Range_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE",
    "TENSILE_YIELD": "YS",
    "TENSILE_TENSILE": "TS",
    "TENSILE_ELONG": "EL",
})

# ================================
# QUALITY GROUP (CQ00 + CQ06)
# ================================
df["Quality_Group"] = np.where(
    df["Quality_Code"].isin(["CQ00", "CQ06"]),
    "CQ00/CQ06",
    df["Quality_Code"]
)

# ================================
# STANDARD RANGE
# ================================
def split_std(x):
    if isinstance(x, str) and "~" in x:
        try:
            lo, hi = x.split("~")
            return float(lo), float(hi)
        except:
            pass
    return np.nan, np.nan

df[["Std_Min", "Std_Max"]] = df["Std_Range_Text"].apply(
    lambda x: pd.Series(split_std(x))
)

# ================================
# NUMERIC
# ================================
for c in ["Order_Gauge", "Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# GAUGE RANGE MAPPING
# ================================
def parse_range(text):
    nums = [float(x) for x in text.replace("≦", "<=").replace("＜", "<").replace("≤", "<=").replace("≥", ">=").replace("＞", ">").replace("=", "").split() if x.replace('.', '', 1).isdigit()]
    return nums[0], nums[-1]

gauge_ranges = []
for _, r in gauge_master.iterrows():
    lo, hi = parse_range(r["Range_Name"])
    gauge_ranges.append((lo, hi, r["Range_Name"]))

def map_gauge_range(x):
    for lo, hi, name in gauge_ranges:
        if lo <= x < hi:
            return name
    return np.nan

df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge_range)

df = df.dropna(subset=["Gauge_Range"])

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🎛 FILTERS")

rolling = st.sidebar.radio("Classify Material", sorted(df["Rolling_Type"].unique()))
metal = st.sidebar.radio("Metallic Coating", sorted(df["Metallic_Type"].unique()))
qg = st.sidebar.radio("Quality Group", sorted(df["Quality_Group"].unique()))

df = df[
    (df["Rolling_Type"] == rolling) &
    (df["Metallic_Type"] == metal) &
    (df["Quality_Group"] == qg)
]

# ================================
# GROUP CONDITION ≥ 30
# ================================
GROUP_COLS = [
    "Rolling_Type",
    "Metallic_Type",
    "Quality_Group",
    "Gauge_Range",
    "Material"
]

count_df = (
    df.groupby(GROUP_COLS)
    .agg(N_Coils=("COIL_NO", "nunique"))
    .reset_index()
)

valid = count_df[count_df["N_Coils"] >= 30]

if valid.empty:
    st.warning("⚠️ No condition with ≥ 30 coils")
    st.stop()

# ================================
# MAIN LOOP (LOGIC GIỮ NGUYÊN)
# ================================
for _, cond in valid.iterrows():

    sub = df[
        (df["Rolling_Type"] == cond["Rolling_Type"]) &
        (df["Metallic_Type"] == cond["Metallic_Type"]) &
        (df["Quality_Group"] == cond["Quality_Group"]) &
        (df["Gauge_Range"] == cond["Gauge_Range"]) &
        (df["Material"] == cond["Material"])
    ].copy().sort_values("COIL_NO")

    lo, hi = sub[["Std_Min", "Std_Max"]].iloc[0]

    sub["NG_LAB"] = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["COIL_NG"] = sub["NG_LAB"] | sub["NG_LINE"]

    n_out = sub[sub["COIL_NG"]]["COIL_NO"].nunique()
    qa = "FAIL" if n_out > 0 else "PASS"

    st.markdown(
        f"""
        ## 🧱 {cond["Material"]} | {cond["Gauge_Range"]}
        - Rolling: **{cond["Rolling_Type"]}**
        - Metallic: **{cond["Metallic_Type"]}**
        - Quality: **{cond["Quality_Group"]}**
        - n = **{cond["N_Coils"]} coils**
        - ❌ Out = **{n_out}** → **{qa}**
        """
    )

    st.dataframe(sub, use_container_width=True)
