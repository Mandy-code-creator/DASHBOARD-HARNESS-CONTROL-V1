import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SPC Quality Dashboard",
    layout="wide"
)

# =========================
# STYLE – POWER BI LOOK
# =========================
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
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# GOOGLE SHEET LOADER
# =========================
@st.cache_data
def load_google_sheet(sheet_id, gid=0):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url)

# ====== CHANGE THESE IDs ======
MAIN_DATA_SHEET = "1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI"
GAUGE_MASTER_SHEET = "1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM"

df = load_google_sheet(MAIN_DATA_SHEET)
gauge_master = load_google_sheet(GAUGE_MASTER_SHEET)

# =========================
# GAUGE RANGE PARSER
# =========================
def parse_range(text):
    nums = re.findall(r"\d+\.?\d*", str(text))
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])

gauge_ranges = []
for _, r in gauge_master.iterrows():
    parsed = parse_range(r["Range_Name"])
    if parsed:
        lo, hi = parsed
        gauge_ranges.append((lo, hi, r["Range_Name"]))

def map_gauge_range(x):
    if pd.isna(x):
        return np.nan
    for lo, hi, name in gauge_ranges:
        if lo <= x < hi:
            return name
    return np.nan

df["Gauge_Range"] = df["ORDER_GAUGE"].apply(map_gauge_range)

# =========================
# QUALITY GROUP LOGIC
# =========================
def map_quality(q):
    if q in ["CQ00", "CQ06"]:
        return "CQ00/CQ06"
    return q

df["Quality_Group"] = df["QUALITY_CODE"].apply(map_quality)

# =========================
# SIDEBAR – FILTER
# =========================
st.sidebar.header("Filters")

material = st.sidebar.multiselect(
    "Classify Material",
    sorted(df["Classify material"].dropna().unique())
)

coat = st.sidebar.multiselect(
    "Metallic Coating Type",
    sorted(df["METALLIC COATING TYPE"].dropna().unique())
)

quality = st.sidebar.multiselect(
    "Quality Group",
    sorted(df["Quality_Group"].dropna().unique())
)

gauge_range = st.sidebar.multiselect(
    "Gauge Range",
    sorted(df["Gauge_Range"].dropna().unique())
)

df_f = df.copy()

if material:
    df_f = df_f[df_f["Classify material"].isin(material)]
if coat:
    df_f = df_f[df_f["METALLIC COATING TYPE"].isin(coat)]
if quality:
    df_f = df_f[df_f["Quality_Group"].isin(quality)]
if gauge_range:
    df_f = df_f[df_f["Gauge_Range"].isin(gauge_range)]

# =========================
# HEADER
# =========================
st.title("📊 SPC Quality Dashboard – Management View")

st.info("""
**PC Analysis Logic – Management Note**

• Grouped by: Classify Material, Metallic Coating Type, Quality Group, Gauge Range, HR Steel Grade  
• CQ00 & CQ06 are merged due to equivalent quality behavior  
• Thickness is analyzed by predefined gauge ranges  
• SPC condition valid only when ≥ 30 coils  
• QA logic is strict: **1 NG → FAIL**
""")

# =========================
# KPI SUMMARY
# =========================
total_coils = len(df_f)
ng_lab = (df_f["NG_LAB"] == 1).sum() if "NG_LAB" in df_f else 0
ng_line = (df_f["NG_LINE"] == 1).sum() if "NG_LINE" in df_f else 0
qa_result = "FAIL" if (ng_lab + ng_line) > 0 else "PASS"

st.markdown('<div class="section-title">Overall Summary</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Total Coils</div>
        <div class="kpi-value">{total_coils}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">NG (LAB)</div>
        <div class="kpi-value">{ng_lab}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">NG (LINE)</div>
        <div class="kpi-value">{ng_line}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">QA Result</div>
        <div class="kpi-value {'kpi-pass' if qa_result=='PASS' else 'kpi-fail'}">
            {qa_result}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATA TABLE – MANAGER FIRST
# =========================
st.markdown('<div class="section-title">Detail Data (Manager Priority)</div>', unsafe_allow_html=True)

show_cols = [
    "COIL_NO",
    "Classify material",
    "METALLIC COATING TYPE",
    "Quality_Group",
    "ORDER_GAUGE",
    "Gauge_Range",
    "Hardness_LAB",
    "Hardness_LINE",
    "NG_LAB",
    "NG_LINE"
]

exist_cols = [c for c in show_cols if c in df_f.columns]

st.dataframe(
    df_f[exist_cols],
    use_container_width=True,
    hide_index=True
)

# =========================
# SPC / ENGINEER VIEW
# =========================
st.markdown('<div class="section-title">SPC Analysis (Engineering View)</div>', unsafe_allow_html=True)

with st.expander("Show SPC Charts"):
    if len(df_f) >= 30 and "Hardness_LAB" in df_f:
        fig, ax = plt.subplots()
        ax.plot(df_f["Hardness_LAB"].values, marker="o")
        ax.set_title("Hardness Trend (LAB)")
        ax.set_ylabel("HRB")
        ax.set_xlabel("Coil Sequence")
        st.pyplot(fig)
    else:
        st.warning("SPC condition not met (≥ 30 coils required)")
