# ============================================================
# SPC QUALITY DASHBOARD – POWER BI STYLE (MANAGEMENT VIEW)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SPC Quality Dashboard",
    layout="wide"
)

# ============================================================
# STYLE (POWER BI LOOK)
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    margin: 16px 0 6px 0;
}
.kpi-box {
    padding: 14px;
    border-radius: 10px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    text-align: center;
}
.kpi-title {
    font-size: 12px;
    color: #6B7280;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
}
.kpi-pass { color: #16A34A; }
.kpi-fail { color: #DC2626; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.title("📊 SPC Quality Dashboard – Management View")

st.info("""
**PC Analysis Logic – Management Note**

• Grouped by: Classify material, Metallic Coating Type, Quality Group, Gauge Range, HR Steel Grade  
• CQ00 and CQ06 are merged due to equivalent quality behavior  
• Thickness is analyzed by predefined gauge ranges (not single values)  
• SPC condition is valid only when ≥ 30 coils  
• QA logic is strict: **1 NG → FAIL**
""")

# ============================================================
# DATA SOURCES (GOOGLE SHEET)
# ============================================================
MAIN_DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
GAUGE_RANGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_csv(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

df = load_csv(MAIN_DATA_URL)
gauge_ref = load_csv(GAUGE_RANGE_URL)

# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
gauge_ref.columns = gauge_ref.columns.str.strip().str.upper().str.replace(" ", "_")

# ============================================================
# AUTO-DETECT IMPORTANT COLUMNS
# ============================================================
def find_col(keywords):
    for c in df.columns:
        for k in keywords:
            if k in c:
                return c
    return None

COL_MATERIAL = find_col(["HR_STEEL"])
COL_CLASSIFY = find_col(["CLAASIFY", "CLASSIFY"])
COL_METALLIC = find_col(["METALLIC"])
COL_QUALITY  = find_col(["QUALITY"])
COL_GAUGE    = find_col(["GAUGE", "THICKNESS"])
COL_COIL     = find_col(["COIL"])
COL_LAB      = find_col(["HARDNESS_冶金", "HARDNESS_LAB"])
COL_LINE     = find_col(["HARDNESS_鍍鋅", "HARDNESS_LINE"])
COL_STD      = find_col(["STANDARD"])

required = [COL_MATERIAL, COL_CLASSIFY, COL_METALLIC, COL_QUALITY, COL_GAUGE, COL_COIL, COL_LAB, COL_LINE, COL_STD]
if any(c is None for c in required):
    st.error("❌ Missing required columns in data file")
    st.stop()

# ============================================================
# QUALITY GROUP (CQ00 + CQ06)
# ============================================================
df["QUALITY_GROUP"] = df[COL_QUALITY].replace({"CQ00": "CQ00/CQ06", "CQ06": "CQ00/CQ06"})

# ============================================================
# PARSE STANDARD RANGE
# ============================================================
def parse_std(x):
    if isinstance(x, str) and "~" in x:
        a, b = x.split("~")
        return float(a), float(b)
    return np.nan, np.nan

df[["STD_MIN", "STD_MAX"]] = df[COL_STD].apply(lambda x: pd.Series(parse_std(x)))

# ============================================================
# GAUGE RANGE PARSER (SUPPORT: 0.28≦T＜0.35)
# ============================================================
def parse_range(text):
    nums = re.findall(r"\d+\.?\d*", str(text))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[-1])
    return None, None

gauge_ref[["G_MIN", "G_MAX"]] = gauge_ref.iloc[:,0].apply(
    lambda x: pd.Series(parse_range(x))
)

def map_gauge_range(val):
    for _, r in gauge_ref.iterrows():
        if r["G_MIN"] <= val < r["G_MAX"]:
            return r.iloc[0]
    return None

df["GAUGE_VALUE"] = pd.to_numeric(df[COL_GAUGE], errors="coerce")
df["GAUGE_RANGE"] = df["GAUGE_VALUE"].apply(map_gauge_range)

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("🎛 Filters")

f_class = st.sidebar.selectbox("Classify Material", sorted(df[COL_CLASSIFY].dropna().unique()))
f_metal = st.sidebar.selectbox("Metallic Coating", sorted(df[COL_METALLIC].dropna().unique()))
f_qual  = st.sidebar.selectbox("Quality Group", sorted(df["QUALITY_GROUP"].dropna().unique()))
f_gauge = st.sidebar.selectbox("Gauge Range", sorted(df["GAUGE_RANGE"].dropna().unique()))

df_f = df[
    (df[COL_CLASSIFY] == f_class) &
    (df[COL_METALLIC] == f_metal) &
    (df["QUALITY_GROUP"] == f_qual) &
    (df["GAUGE_RANGE"] == f_gauge)
].copy()

# ============================================================
# QA LOGIC
# ============================================================
df_f["NG_LAB"]  = (df_f[COL_LAB]  < df_f["STD_MIN"]) | (df_f[COL_LAB]  > df_f["STD_MAX"])
df_f["NG_LINE"] = (df_f[COL_LINE] < df_f["STD_MIN"]) | (df_f[COL_LINE] > df_f["STD_MAX"])
df_f["COIL_NG"] = df_f["NG_LAB"] | df_f["NG_LINE"]

total = df_f[COL_COIL].nunique()
ng_cnt = df_f[df_f["COIL_NG"]][COL_COIL].nunique()
qa = "FAIL" if ng_cnt > 0 else "PASS"

# ============================================================
# KPI SUMMARY
# ============================================================
st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Coils</div><div class='kpi-value'>{total}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>NG Coils</div><div class='kpi-value'>{ng_cnt}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(
        f"<div class='kpi-box'><div class='kpi-title'>QA Result</div>"
        f"<div class='kpi-value {'kpi-pass' if qa=='PASS' else 'kpi-fail'}'>{qa}</div></div>",
        unsafe_allow_html=True
    )

# ============================================================
# DATA TABLE (PRIORITY FOR MANAGEMENT)
# ============================================================
st.markdown('<div class="section-title">Detail Data</div>', unsafe_allow_html=True)

show_cols = [
    COL_COIL,
    COL_MATERIAL,
    "GAUGE_RANGE",
    "QUALITY_GROUP",
    COL_LAB,
    COL_LINE,
    "NG_LAB",
    "NG_LINE"
]

st.dataframe(
    df_f[show_cols].sort_values(COL_COIL),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# SPC CHART (ENGINEER VIEW)
# ============================================================
with st.expander("📈 SPC Trend (Engineering View)"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(df_f[COL_COIL], df_f[COL_LAB], marker="o", label="LAB")
    ax.plot(df_f[COL_COIL], df_f[COL_LINE], marker="s", label="LINE")
    ax.axhline(df_f["STD_MIN"].iloc[0], linestyle="--", color="red")
    ax.axhline(df_f["STD_MAX"].iloc[0], linestyle="--", color="red")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
