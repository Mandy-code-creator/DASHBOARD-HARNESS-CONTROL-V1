# ================================
# FULL STREAMLIT APP – FINAL
# ONLY FILTER / GROUP LOGIC PATCHED
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

# ================================
# UTILITY
# ================================
def fig_to_png(fig, dpi=200):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Material-level Hardness Detail", layout="wide")
st.title("📊 Material-level Hardness & Mechanical Detail")

# ================================
# REFRESH
# ================================
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ================================
# LOAD DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"

@st.cache_data
def load_data(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_data(DATA_URL)

# ================================
# FIND METALLIC TYPE COLUMN
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
# REQUIRED COLUMNS
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
# 🔧 PATCH 1 — QUALITY CODE GROUP
# ================================
def qc_group(x):
    if x in ["CQ00", "CQ06"]:
        return "CQ00/06"
    return x

df["Quality_Group"] = df["Quality_Code"].apply(qc_group)

# ================================
# SPLIT STANDARD RANGE
# ================================
def split_std(x):
    if isinstance(x, str) and "~" in x:
        try:
            lo, hi = x.split("~")
            return pd.Series([float(lo), float(hi)])
        except:
            pass
    return pd.Series([np.nan, np.nan])

df[["Std_Min","Std_Max"]] = df["Std_Range_Text"].apply(split_std)
df.drop(columns=["Std_Range_Text"], inplace=True)

# ================================
# FORCE NUMERIC
# ================================
for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# SIDEBAR FILTERS (PATCHED)
# ================================
st.sidebar.header("🎛 FILTERS")

rolling = st.sidebar.radio(
    "Classify Material",
    sorted(df["Rolling_Type"].dropna().unique())
)
df = df[df["Rolling_Type"] == rolling]

metal = st.sidebar.radio(
    "Metallic Coating",
    sorted(df["Metallic_Type"].dropna().unique())
)
df = df[df["Metallic_Type"] == metal]

qc = st.sidebar.radio(
    "Quality Group",
    sorted(df["Quality_Group"].dropna().unique())
)
df = df[df["Quality_Group"] == qc]

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
# 🔧 PATCH 2 — GROUPING (NO PRODUCT SPEC)
# ================================
GROUP_COLS = [
    "Rolling_Type",
    "Material",
    "Metallic_Type",
    "Top_Coatmass",
    "Order_Gauge",
    "Quality_Group"
]

count_df = (
    df.groupby(GROUP_COLS)
      .agg(N_Coils=("COIL_NO","nunique"))
      .reset_index()
)

valid_conditions = count_df[count_df["N_Coils"] >= 30]

if valid_conditions.empty:
    st.warning("⚠️ No condition with ≥ 30 coils")
    st.stop()

# ================================
# MAIN LOOP
# ================================
for _, cond in valid_conditions.iterrows():

    sub = df.copy()
    for c in GROUP_COLS:
        sub = sub[sub[c] == cond[c]]

    sub = sub.sort_values("COIL_NO").reset_index(drop=True)

    lo, hi = sub[["Std_Min","Std_Max"]].iloc[0]

    # ===== QA STRICT =====
    sub["NG_LAB"]  = (sub["Hardness_LAB"]  < lo) | (sub["Hardness_LAB"]  > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["COIL_NG"] = sub["NG_LAB"] | sub["NG_LINE"]

    n_out = sub[sub["COIL_NG"]]["COIL_NO"].nunique()
    qa = "FAIL" if n_out > 0 else "PASS"

    spec_list = ", ".join(sorted(sub["Product_Spec"].dropna().unique()))

    st.markdown(
        f"""
        ## 🧱 Product Spec Group  
        **Specs**: {spec_list}  
        **Material**: {cond["Material"]} | **Gauge**: {cond["Order_Gauge"]}  
        **Quality Group**: {cond["Quality_Group"]}  
        ➡️ **n = {cond["N_Coils"]}** | ❌ **Out = {n_out}** | 🧪 **{qa}**
        """
    )

    # ================================
    # VIEW — DATA TABLE
    # ================================
    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    # ================================
    # VIEW — TREND
    # ================================
    elif view_mode == "📈 Trend (LAB / LINE)":
        sub["X"] = np.arange(1, len(sub)+1)
        c1, c2 = st.columns(2)

        for col, title, box in [
            ("Hardness_LAB","Hardness LAB",c1),
            ("Hardness_LINE","Hardness LINE",c2)
        ]:
            with box:
                d = sub[sub[col] > 0]
                fig, ax = plt.subplots(figsize=(5,3))
                ax.plot(d["X"], d[col], marker="o")
                ax.axhline(lo, linestyle="--")
                ax.axhline(hi, linestyle="--")
                ax.set_yticks(np.arange(np.floor(lo), np.ceil(hi)+0.01, 2.5))
                ax.set_title(title)
                ax.grid(alpha=0.3)
                st.pyplot(fig)

    # ================================
    # VIEW — IQR
    # ================================
    elif view_mode == "📐 Hardness Optimal Range (IQR)":
        lab = sub[sub["Hardness_LAB"] > 0]["Hardness_LAB"]
        line = sub[sub["Hardness_LINE"] > 0]["Hardness_LINE"]

        def iqr(x,k):
            q1,q3 = x.quantile([0.25,0.75])
            i = q3-q1
            return q1-k*i, q3+k*i

        L1,U1 = iqr(lab,K)
        L2,U2 = iqr(line,K)

        opt_lo, opt_hi = max(L1,L2), min(U1,U2)
        safe_lo, safe_hi = max(opt_lo,lo), min(opt_hi,hi)
        target = (safe_lo+safe_hi)/2 if safe_lo < safe_hi else np.nan

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(lab, bins=10, alpha=0.5, label="LAB")
        ax.hist(line, bins=10, alpha=0.5, label="LINE")
        ax.axvline(lo, linestyle="--", label="LSL")
        ax.axvline(hi, linestyle="--", label="USL")
        if safe_lo < safe_hi:
            ax.axvspan(safe_lo, safe_hi, alpha=0.25, label="OPTIMAL")
        if not np.isnan(target):
            ax.axvline(target, linestyle="-.", label=f"TARGET {target:.1f}")
        ax.legend(bbox_to_anchor=(1.02,0.5), loc="center left", frameon=False)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
