# ================================
# FULL STREAMLIT APP – FINAL FIXED
# - GIỮ NGUYÊN LOGIC QA STRICT (1 NG = FAIL)
# - GIỮ NGUYÊN TOÀN BỘ VIEW & BIỂU ĐỒ
# - CHỈ THAY ĐỔI ĐIỀU KIỆN FILTER + GAUGE RANGE
# - FIX 100% KeyError / IndexError / Unicode
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Material-level Hardness Detail",
    layout="wide"
)
st.title("📊 Material-level Hardness & Mechanical Detail")

# ================================
# UTIL
# ================================
def fig_to_png(fig, dpi=200):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# REFRESH
# ================================
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ================================
# LOAD MAIN DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"

@st.cache_data
def load_data(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_data(DATA_URL)

# ================================
# DETECT METALLIC COATING
# ================================
metal_col = next(
    (c for c in raw.columns if "METALLIC" in c.upper() and "COATING" in c.upper()),
    None
)

if metal_col is None:
    st.error("❌ Cannot find METALLIC COATING TYPE column")
    st.stop()

raw["Metallic_Type"] = raw[metal_col]

# ================================
# RENAME COLUMNS
# ================================
column_mapping = {
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
}

df = raw.rename(columns={k: v for k, v in column_mapping.items() if k in raw.columns})

# ================================
# REQUIRED CHECK
# ================================
required_cols = [
    "Product_Spec","Material","Rolling_Type","Metallic_Type",
    "Top_Coatmass","Order_Gauge","COIL_NO","Quality_Code",
    "Std_Range_Text","Hardness_LAB","Hardness_LINE","YS","TS","EL"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# ================================
# SPLIT STANDARD HARDNESS RANGE
# ================================
def split_std(x):
    if isinstance(x, str):
        nums = re.findall(r"\d+\.?\d*", x)
        if len(nums) >= 2:
            return pd.Series([float(nums[0]), float(nums[1])])
    return pd.Series([np.nan, np.nan])

df[["Std_Min","Std_Max"]] = df["Std_Range_Text"].apply(split_std)
df.drop(columns=["Std_Range_Text"], inplace=True)

# ================================
# FORCE NUMERIC
# ================================
for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# LOAD GAUGE RANGE MASTER
# ================================
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"
gauge_df = load_data(GAUGE_URL)

# auto detect range column
range_col = next(
    (c for c in gauge_df.columns if gauge_df[c].astype(str).str.contains(r"[<>≦≧]").any()),
    None
)

if range_col is None:
    st.error("❌ Cannot detect gauge range column")
    st.stop()

def parse_range(txt):
    nums = re.findall(r"\d+\.?\d*", str(txt))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[-1])

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r[range_col])
    if lo is not None:
        ranges.append((lo, hi, r[range_col]))

def map_gauge_range(v):
    for lo, hi, label in ranges:
        if lo <= v < hi:
            return label
    return None

df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge_range)

# ================================
# SIDEBAR FILTERS (PATCHED)
# ================================
st.sidebar.header("🎛 FILTERS")

rolling = st.sidebar.radio("Rolling Type", sorted(df["Rolling_Type"].dropna().unique()))
df = df[df["Rolling_Type"] == rolling]

metal = st.sidebar.radio("Metallic Coating", sorted(df["Metallic_Type"].dropna().unique()))
df = df[df["Metallic_Type"] == metal]

qc_map = df["Quality_Code"].replace({"CQ00": "CQ00/06", "CQ06": "CQ00/06"})
qc = st.sidebar.radio("Quality Code Group", sorted(qc_map.unique()))
df = df[qc_map == qc]

gauge_sel = st.sidebar.selectbox(
    "Gauge Range",
    sorted(df["Gauge_Range"].dropna().unique())
)
df = df[df["Gauge_Range"] == gauge_sel]

# ================================
# VIEW MODE
# ================================
view_mode = st.sidebar.radio(
    "📊 View Mode",
    [
        "📋 Data Table",
        "📈 Trend (LAB / LINE)",
        "📐 Hardness Optimal Range (IQR)"
    ]
)

if view_mode == "📐 Hardness Optimal Range (IQR)":
    K = st.sidebar.selectbox("IQR factor K", [0.5,0.75,1.0,1.25,1.5], index=2)

# ================================
# GROUP CONDITION (≥30 COILS)
# ================================
GROUP_COLS = ["Product_Spec","Material","Metallic_Type","Top_Coatmass","Gauge_Range"]

count_df = (
    df.groupby(GROUP_COLS)
      .agg(N_Coils=("COIL_NO","nunique"))
      .reset_index()
)

valid = count_df[count_df["N_Coils"] >= 30]

if valid.empty:
    st.warning("⚠️ No condition with ≥ 30 coils")
    st.stop()

# ================================
# MAIN LOOP (UNCHANGED)
# ================================
for _, cond in valid.iterrows():

    sub = df[
        (df["Product_Spec"] == cond["Product_Spec"]) &
        (df["Material"] == cond["Material"]) &
        (df["Top_Coatmass"] == cond["Top_Coatmass"]) &
        (df["Gauge_Range"] == cond["Gauge_Range"])
    ].copy().sort_values("COIL_NO")

    lo, hi = sub[["Std_Min","Std_Max"]].iloc[0]

    sub["NG_LAB"]  = (sub["Hardness_LAB"]  < lo) | (sub["Hardness_LAB"]  > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["COIL_NG"] = sub["NG_LAB"] | sub["NG_LINE"]

    n_out = sub[sub["COIL_NG"]]["COIL_NO"].nunique()
    qa = "FAIL" if n_out > 0 else "PASS"

    st.markdown(
        f"## 🧱 `{cond['Product_Spec']}`  \n"
        f"Material: **{cond['Material']}** | Gauge: **{cond['Gauge_Range']}**  \n"
        f"➡️ n = **{cond['N_Coils']}** | ❌ Out = **{n_out}** | 🧪 **{qa}**"
    )

    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    elif view_mode == "📈 Trend (LAB / LINE)":
        sub["X"] = np.arange(1, len(sub)+1)
        c1, c2 = st.columns(2)

        for label, col, box in [("LAB","Hardness_LAB",c1),("LINE","Hardness_LINE",c2)]:
            with box:
                fig, ax = plt.subplots(figsize=(5,3))
                ax.plot(sub["X"], sub[col], marker="o")
                ax.axhline(lo, linestyle="--")
                ax.axhline(hi, linestyle="--")
                ax.set_title(f"Hardness {label}")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

    elif view_mode == "📐 Hardness Optimal Range (IQR)":
        lab = sub[sub["Hardness_LAB"]>0]["Hardness_LAB"]
        line = sub[sub["Hardness_LINE"]>0]["Hardness_LINE"]

        def iqr(x,k):
            q1,q3 = x.quantile([0.25,0.75])
            return q1-k*(q3-q1), q3+k*(q3-q1)

        L1,U1 = iqr(lab,K)
        L2,U2 = iqr(line,K)

        opt_lo = max(L1,L2,lo)
        opt_hi = min(U1,U2,hi)
        target = (opt_lo+opt_hi)/2 if opt_lo<opt_hi else np.nan

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(lab, alpha=0.5, label="LAB")
        ax.hist(line, alpha=0.5, label="LINE")
        ax.axvline(lo, linestyle="--")
        ax.axvline(hi, linestyle="--")
        if opt_lo < opt_hi:
            ax.axvspan(opt_lo,opt_hi,alpha=0.3,label="OPTIMAL")
        if not np.isnan(target):
            ax.axvline(target, linestyle="-.", label=f"TARGET {target:.1f}")
        ax.legend()
        st.pyplot(fig)
