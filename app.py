# ================================
# FULL STREAMLIT APP – FINAL
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Material-level Hardness Detail", layout="wide")
st.title("📊 Material-level Hardness & Mechanical Detail")

# ================================
# UTILS
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
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# ================================
# METALLIC TYPE AUTO FIND
# ================================
metal_col = next(c for c in raw.columns if "METALLIC" in c.upper())
raw["Metallic_Type"] = raw[metal_col]

# ================================
# RENAME
# ================================
df = raw.rename(columns={
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "Claasify material": "Rolling_Type",
    "TOP COATMASS": "Top_Coatmass",
    "ORDER GAUGE": "Order_Gauge",
    "COIL NO": "COIL_NO",
    "QUALITY_CODE": "Quality_Code",
    "Standard Hardness": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE",
    "TENSILE_YIELD": "YS",
    "TENSILE_TENSILE": "TS",
    "TENSILE_ELONG": "EL",
})

# ================================
# STANDARD RANGE
# ================================
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))

# ================================
# NUMERIC
# ================================
for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# QUALITY GROUP (CQ00 + CQ06)
# ================================
df["Quality_Group"] = df["Quality_Code"].replace({
    "CQ00": "CQ00/CQ06",
    "CQ06": "CQ00/CQ06"
})

# ================================
# LOAD GAUGE RANGE TABLE
# ================================
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_gauge():
    return pd.read_csv(GAUGE_URL)

gauge_df = load_gauge()
gauge_df.columns = gauge_df.columns.str.strip()

gauge_col = next(c for c in gauge_df.columns if "RANGE" in c.upper())

def parse_range(txt):
    nums = re.findall(r"\d+\.\d+|\d+", str(txt))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[-1])

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r[gauge_col])
    if lo is not None:
        ranges.append((lo, hi, r[gauge_col]))

def map_gauge(val):
    for lo, hi, name in ranges:
        if lo <= val < hi:
            return name
    return None

df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge)
df = df.dropna(subset=["Gauge_Range"])

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🎛 FILTERS")

rolling = st.sidebar.radio("Rolling Type", sorted(df["Rolling_Type"].unique()))
metal   = st.sidebar.radio("Metallic Coating", sorted(df["Metallic_Type"].unique()))
qgroup  = st.sidebar.radio("Quality Group", sorted(df["Quality_Group"].unique()))

df = df[
    (df["Rolling_Type"] == rolling) &
    (df["Metallic_Type"] == metal) &
    (df["Quality_Group"] == qgroup)
]

view_mode = st.sidebar.radio(
    "📊 View Mode",
    ["📋 Data Table","📈 Trend (LAB / LINE)","📐 Hardness Optimal Range (IQR)"]
)

if view_mode == "📐 Hardness Optimal Range (IQR)":
    K = st.sidebar.selectbox("IQR K", [0.5,0.75,1,1.25,1.5], index=2)

# ================================
# GROUP CONDITION ≥30
# ================================
GROUP_COLS = [
    "Product_Spec","Material","Metallic_Type",
    "Quality_Group","Gauge_Range"
]

cnt = (
    df.groupby(GROUP_COLS)
      .agg(N=("COIL_NO","nunique"))
      .reset_index()
)

valid = cnt[cnt["N"] >= 30]
if valid.empty:
    st.warning("⚠️ No group with ≥ 30 coils")
    st.stop()

# ================================
# MAIN LOOP
# ================================
for _, g in valid.iterrows():

    sub = df[
        (df["Product_Spec"] == g["Product_Spec"]) &
        (df["Material"] == g["Material"]) &
        (df["Gauge_Range"] == g["Gauge_Range"])
    ].sort_values("COIL_NO")

    lo, hi = sub.iloc[0][["Std_Min","Std_Max"]]

    sub["NG_LAB"]  = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"]

    qa = "FAIL" if sub["NG"].any() else "PASS"

    st.markdown(
        f"""
### 🧱 {g['Product_Spec']}
**Material:** {g['Material']}  
**Gauge Range:** {g['Gauge_Range']}  
**Coils:** {g['N']} | **QA:** 🧪 **{qa}**
"""
    )

    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    elif view_mode == "📈 Trend (LAB / LINE)":
        x = np.arange(1,len(sub)+1)
        fig, ax = plt.subplots()
        ax.plot(x, sub["Hardness_LAB"], marker="o", label="LAB")
        ax.plot(x, sub["Hardness_LINE"], marker="s", label="LINE")
        ax.axhline(lo, ls="--")
        ax.axhline(hi, ls="--")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    else:
        lab = sub[sub["Hardness_LAB"]>0]["Hardness_LAB"]
        q1,q3 = lab.quantile([0.25,0.75])
        iqr = q3-q1
        st.write("IQR Range:", q1-K*iqr, "→", q3+K*iqr)
