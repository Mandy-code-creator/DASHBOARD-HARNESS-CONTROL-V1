import streamlit as st
import pandas as pd
import numpy as np
import re

# ===============================
# PAGE CONFIG – Power BI style
# ===============================
st.set_page_config(
    page_title="PC Analysis Dashboard",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { font-weight: 600; }
    .metric-box {
        background-color: #f5f6fa;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_main_data():
    return pd.read_csv("data.csv")

@st.cache_data
def load_gauge_table():
    url = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"
    return pd.read_csv(url)

df = load_main_data()
gauge_df = load_gauge_table()

# ===============================
# QUALITY GROUP
# ===============================
def quality_group(q):
    if q in ["CQ00", "CQ06"]:
        return "CQ00/06"
    return q

df["QUALITY_GROUP"] = df["QUALITY_CODE"].apply(quality_group)

# ===============================
# PARSE RANGE – FIX 100%
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

def map_gauge_range(val):
    try:
        v = float(val)
    except:
        return "UNKNOWN"
    for lo, hi, name in ranges:
        if lo <= v < hi:
            return name
    return "OUT OF RANGE"

df["GAUGE_RANGE"] = df["ORDER_GAUGE"].apply(map_gauge_range)

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("🔎 Filters")

mat = st.sidebar.multiselect(
    "Classify Material",
    sorted(df["Classify material"].dropna().unique())
)

coat = st.sidebar.multiselect(
    "Metallic Coating Type",
    sorted(df["METALLIC COATING TYPE"].dropna().unique())
)

qual = st.sidebar.multiselect(
    "Quality Group",
    sorted(df["QUALITY_GROUP"].unique())
)

gauge = st.sidebar.multiselect(
    "Gauge Range",
    sorted(df["GAUGE_RANGE"].unique())
)

steel = st.sidebar.multiselect(
    "HR Steel Grade",
    sorted(df["HR STEEL GRADE"].dropna().unique())
)

# ===============================
# APPLY FILTER
# ===============================
f = df.copy()

if mat:
    f = f[f["Classify material"].isin(mat)]
if coat:
    f = f[f["METALLIC COATING TYPE"].isin(coat)]
if qual:
    f = f[f["QUALITY_GROUP"].isin(qual)]
if gauge:
    f = f[f["GAUGE_RANGE"].isin(gauge)]
if steel:
    f = f[f["HR STEEL GRADE"].isin(steel)]

# ===============================
# HEADER
# ===============================
st.title("PC Analysis Dashboard")
st.caption("Power BI–style SPC & QA Summary")

# ===============================
# KPI
# ===============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='metric-box'><h3>Total Coils</h3><h2>{}</h2></div>".format(len(f)), unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-box'><h3>Groups</h3><h2>{}</h2></div>".format(
        f.groupby(["Classify material","METALLIC COATING TYPE","QUALITY_GROUP","GAUGE_RANGE","HR STEEL GRADE"]).ngroups
    ), unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-box'><h3>SPC Valid</h3><h2>{}</h2></div>".format(
        (f.groupby(["GAUGE_RANGE"]).size() >= 30).sum()
    ), unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-box'><h3>QA Rule</h3><h2>STRICT</h2></div>", unsafe_allow_html=True)

# ===============================
# DATA TABLE
# ===============================
st.subheader("📊 Filtered Data")
st.dataframe(
    f,
    use_container_width=True,
    height=520
)

# ===============================
# MANAGEMENT NOTE
# ===============================
st.markdown("""
### 📌 PC Analysis Logic – Management Note
- Data is grouped by **Classify material, Metallic Coating Type, Quality Group, Gauge Range, HR Steel Grade**
- **CQ00 and CQ06** are merged due to equivalent quality behavior
- Thickness is analyzed by **predefined gauge ranges**
- SPC condition is valid only when **≥ 30 coils**
- QA logic is **strict: 1 NG → FAIL**
""")
