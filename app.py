# ================================
# FULL STREAMLIT APP – FINAL FIXED
# CQ00 + CQ06 MERGED
# PRODUCT SPEC MERGED IN SAME GAUGE RANGE
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import math

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")
st.title("📊 SPC Hardness – Material / Gauge Level Analysis")

# ================================
# REFRESH
# ================================
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ================================
# UTILS
# ================================
def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf
def spc_stats(data, lsl, usl):
    data = data.dropna()
    if len(data) < 2:
        return None
def normal_pdf(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    mean = data.mean()
    std = data.std(ddof=1)

    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    ca = (mean - (usl + lsl) / 2) / ((usl - lsl) / 2) * 100 if usl > lsl else np.nan
    cpk = min((usl - mean), (mean - lsl)) / (3 * std) if std > 0 else np.nan

    return mean, std, cp, ca, cpk

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
# METALLIC TYPE AUTO
# ================================
metal_col = next(c for c in raw.columns if "METALLIC" in c.upper())
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
    "Standard Hardness": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE",
    "TENSILE_YIELD": "YS",
    "TENSILE_TENSILE": "TS",
    "TENSILE_ELONG": "EL",
})

# ================================
# STANDARD HARDNESS
# ================================
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))

# ================================
# FORCE NUMERIC
# ================================
for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# QUALITY GROUP (CQ00 + CQ06)
# ================================
df["Quality_Group"] = df["Quality_Code"].replace({
    "CQ00": "CQ00 / CQ06",
    "CQ06": "CQ00 / CQ06"
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

def parse_range(text):
    nums = re.findall(r"\d+\.\d+|\d+", str(text))
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
st.sidebar.header("🎛 FILTER")

rolling = st.sidebar.radio("Rolling Type", sorted(df["Rolling_Type"].unique()))
metal   = st.sidebar.radio("Metallic Type", sorted(df["Metallic_Type"].unique()))
qgroup  = st.sidebar.radio("Quality Group", sorted(df["Quality_Group"].unique()))

df = df[
    (df["Rolling_Type"] == rolling) &
    (df["Metallic_Type"] == metal) &
    (df["Quality_Group"] == qgroup)
]

view_mode = st.sidebar.radio(
    "📊 View Mode",
    [
        "📋 Data Table",
        "📈 Trend (LAB / LINE)",
        "📊 Distribution + Normal"
    ]
)


# ================================
# GROUP CONDITION (NO PRODUCT SPEC)
# ================================
GROUP_COLS = [
    "Rolling_Type",
    "Metallic_Type",
    "Quality_Group",
    "Gauge_Range",
    "Material"
]

cnt = (
    df.groupby(GROUP_COLS)
      .agg(N_Coils=("COIL_NO","nunique"))
      .reset_index()
)

valid = cnt[cnt["N_Coils"] >= 30]

if valid.empty:
    st.warning("⚠️ No group with ≥30 coils")
    st.stop()

# ================================
# MAIN LOOP
# ================================
for _, g in valid.iterrows():

    sub = df[
        (df["Rolling_Type"] == g["Rolling_Type"]) &
        (df["Metallic_Type"] == g["Metallic_Type"]) &
        (df["Quality_Group"] == g["Quality_Group"]) &
        (df["Gauge_Range"] == g["Gauge_Range"]) &
        (df["Material"] == g["Material"])
    ].sort_values("COIL_NO")

    lo, hi = sub.iloc[0][["Std_Min","Std_Max"]]

    sub["NG_LAB"]  = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"]

    qa = "FAIL" if sub["NG"].any() else "PASS"
    specs = ", ".join(sorted(sub["Product_Spec"].unique()))

    st.markdown(
        f"""
### 🧱 Quality Group: {g['Quality_Group']}
**Material:** {g['Material']}  
**Gauge Range:** {g['Gauge_Range']}  
**Product Specs:** {specs}  
**Coils:** {sub['COIL_NO'].nunique()} | **QA:** 🧪 **{qa}**
"""
    )

    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    else:
        x = np.arange(1, len(sub) + 1)

        fig, ax = plt.subplots(figsize=(8,4))

        # ---- Trend lines
        ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB")
        ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE")

        # ---- Spec limits
        ax.axhline(lo, linestyle="--", linewidth=1.5, label=f"LSL = {lo}")
        ax.axhline(hi, linestyle="--", linewidth=1.5, label=f"USL = {hi}")

        # ---- Labels & style
        ax.set_title("Hardness Trend by Coil Sequence", fontsize=12, weight="bold")
        ax.set_xlabel("Coil Sequence")
        ax.set_ylabel("Hardness (HRB)")
        ax.grid(alpha=0.25)

        # ---- Legend OUTSIDE (Power BI style)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )

        plt.tight_layout()
        st.pyplot(fig)
# ================================
# DISTRIBUTION + NORMAL CURVE
# ================================
for label, col in [("LAB", "Hardness_LAB"), ("LINE", "Hardness_LINE")]:

    data = sub[col].dropna()
    if len(data) < 10:
        continue

    mean, std, cp, ca, cpk = spc_stats(data, lo, hi)

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(data, bins=10, density=True, alpha=0.35, edgecolor="black")

    x = np.linspace(min(data), max(data), 200)
   ax.plot(x, normal_pdf(x, mean, std), linewidth=2)

    ax.axvline(lo, linestyle="--", linewidth=1.5, label=f"LSL = {lo}")
    ax.axvline(hi, linestyle="--", linewidth=1.5, label=f"USL = {hi}")
    ax.axvline(mean, linestyle=":", linewidth=1.5, label=f"Mean = {mean:.2f}")

    ax.set_title(f"{label} Hardness Distribution", weight="bold")
    ax.set_xlabel("Hardness (HRB)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)

    note = (
        f"N = {len(data)}\n"
        f"Mean = {mean:.2f}\n"
        f"Std = {std:.2f}\n"
        f"Cp = {cp:.2f}\n"
        f"Ca = {ca:.1f}%\n"
        f"Cpk = {cpk:.2f}"
    )

    ax.text(
        1.02, 0.5,
        note,
        transform=ax.transAxes,
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.15)
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.9),
        frameon=False
    )

    plt.tight_layout()
    st.pyplot(fig)
