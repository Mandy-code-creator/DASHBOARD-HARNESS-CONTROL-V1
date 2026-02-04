import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="SPC Quality Dashboard",
    layout="wide"
)

st.title("📊 SPC Quality Control Dashboard")

# ================================
# NOTE FOR MANAGEMENT
# ================================
st.info("""
**SPC Analysis Logic – Management Note**

• Data is grouped by: Classify material, Metallic Coating Type, Quality Group, Gauge Range, HR Steel Grade  
• CQ00 and CQ06 are merged due to equivalent quality behavior  
• Thickness is analyzed by predefined gauge ranges (not single values)  
• SPC condition is valid only when ≥ 30 coils  
• QA logic is strict: **1 NG → FAIL**
""")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_main_data():
    return pd.read_csv("data.csv")   # ← file chính của bạn

df = load_main_data()

# ================================
# FORCE NUMERIC
# ================================
NUM_COLS = ["Hardness_LAB", "Hardness_LINE", "Order_Gauge"]
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# QUALITY CODE GROUPING
# ================================
def map_quality_group(q):
    if q in ["CQ00", "CQ06"]:
        return "CQ00/CQ06"
    return q

df["QUALITY_CODE_GROUP"] = df["Quality_Code"].apply(map_quality_group)

# ================================
# LOAD GAUGE RANGE MASTER
# ================================
GAUGE_GROUP_URL = "https://docs.google.com/spreadsheets/d/XXXX/export?format=csv"

@st.cache_data
def load_gauge_master(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

gauge_master = load_gauge_master(GAUGE_GROUP_URL)

def map_gauge_range(g):
    row = gauge_master[
        (gauge_master["Gauge_Min"] <= g) &
        (gauge_master["Gauge_Max"] > g)
    ]
    if not row.empty:
        return row.iloc[0]["Gauge_Group"]
    return "UNDEFINED"

df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge_range)

# ================================
# SIDEBAR FILTERS
# ================================
st.sidebar.header("🎛 Filters")

rolling = st.sidebar.radio(
    "Classify Material",
    sorted(df["Rolling_Type"].dropna().unique())
)
df = df[df["Rolling_Type"] == rolling]

metal = st.sidebar.radio(
    "Metallic Coating Type",
    sorted(df["Metallic_Type"].dropna().unique())
)
df = df[df["Metallic_Type"] == metal]

qc = st.sidebar.radio(
    "Quality Group",
    sorted(df["QUALITY_CODE_GROUP"].dropna().unique())
)
df = df[df["QUALITY_CODE_GROUP"] == qc]

# ================================
# CONDITION DEFINITION
# ================================
GROUP_COLS = [
    "Rolling_Type",
    "Metallic_Type",
    "QUALITY_CODE_GROUP",
    "Gauge_Range",
    "Material"
]

cond_df = (
    df.groupby(GROUP_COLS)
      .agg(N_Coils=("COIL_NO", "nunique"))
      .reset_index()
)

valid_conditions = cond_df[cond_df["N_Coils"] >= 30]

if valid_conditions.empty:
    st.warning("⚠️ No SPC condition with ≥ 30 coils")
    st.stop()

# ================================
# MAIN LOOP
# ================================
for _, cond in valid_conditions.iterrows():

    st.subheader(
        f"Material: {cond['Material']} | "
        f"Gauge: {cond['Gauge_Range']} | "
        f"Coils: {cond['N_Coils']}"
    )

    sub = df[
        (df["Rolling_Type"] == cond["Rolling_Type"]) &
        (df["Metallic_Type"] == cond["Metallic_Type"]) &
        (df["QUALITY_CODE_GROUP"] == cond["QUALITY_CODE_GROUP"]) &
        (df["Gauge_Range"] == cond["Gauge_Range"]) &
        (df["Material"] == cond["Material"])
    ].copy().sort_values("COIL_NO").reset_index(drop=True)

    # ===== QA LOGIC =====
    sub["QA_Result"] = np.where(sub["NG_Flag"] == 1, "FAIL", "PASS")

    # ===== TABLE =====
    st.dataframe(sub, use_container_width=True)

    # ===== SPC CHART =====
    fig, ax = plt.subplots(figsize=(10, 4))

    y = sub["Hardness_LAB"]
    ax.plot(y.index + 1, y, marker="o")
    ax.set_title("Hardness LAB Trend")
    ax.set_xlabel("Batch No")
    ax.set_ylabel("HRB")

    ax.axhline(y.mean(), linestyle="--")
    ax.grid(True)

    st.pyplot(fig)

    st.divider()
