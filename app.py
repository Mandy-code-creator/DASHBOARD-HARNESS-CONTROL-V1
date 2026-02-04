# ================================
# FULL STREAMLIT APP – FINAL / CLEAN / CORRECT
# CQ00 + CQ06 MERGED
# PRODUCT SPEC MERGED BY GAUGE RANGE
# TREND + DISTRIBUTION (LAB + LINE COMBINED)
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
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")
st.title("📊 SPC Hardness – Material / Gauge Level Analysis")

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
# METALLIC TYPE
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
for c in ["Hardness_LAB","Hardness_LINE","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# QUALITY GROUP
# ================================
df["Quality_Group"] = df["Quality_Code"].replace({
    "CQ00": "CQ00 / CQ06",
    "CQ06": "CQ00 / CQ06"
})

# ================================
# LOAD GAUGE RANGE
# ================================
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_gauge():
    return pd.read_csv(GAUGE_URL)

gauge_df = load_gauge()
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
# SIDEBAR
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
        "📊 Distribution (LAB + LINE)"
    ]
)


# ================================
# GROUP CONDITION (NO PRODUCT SPEC)
# ================================
GROUP_COLS = [
    "Rolling_Type","Metallic_Type",
    "Quality_Group","Gauge_Range","Material"
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
# VIEW 1 – DATA TABLE
# ================================
if view_mode == "📋 Data Table":
    st.dataframe(sub, use_container_width=True)


# ================================
# VIEW 2 – TREND ONLY
# ================================
elif view_mode == "📈 Trend (LAB / LINE)":

    x = np.arange(1, len(sub) + 1)
    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(x, sub["Hardness_LAB"], marker="o", label="LAB")
    ax.plot(x, sub["Hardness_LINE"], marker="s", label="LINE")

    ax.axhline(lo, color="red", linestyle="--", linewidth=2, label=f"LSL = {lo}")
    ax.axhline(hi, color="red", linestyle="--", linewidth=2, label=f"USL = {hi}")

    ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
    ax.set_xlabel("Coil Sequence")
    ax.set_ylabel("Hardness (HRB)")
    ax.grid(alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    st.pyplot(fig)


# ================================
# VIEW 3 – DISTRIBUTION ONLY
# ================================
elif view_mode == "📊 Distribution (LAB + LINE)":

    lab = sub["Hardness_LAB"].dropna()
    line = sub["Hardness_LINE"].dropna()

    if len(lab) < 10 or len(line) < 10:
        st.info("⚠️ Not enough data for distribution (need ≥10 points each)")
    else:
        mean_lab, std_lab = lab.mean(), lab.std(ddof=1)
        mean_line, std_line = line.mean(), line.std(ddof=1)

        # ===== 3 SIGMA RANGE (CHUNG)
        x_min = min(mean_lab - 3*std_lab, mean_line - 3*std_line)
        x_max = max(mean_lab + 3*std_lab, mean_line + 3*std_line)

        bins = np.linspace(x_min, x_max, 25)

        fig, ax = plt.subplots(figsize=(8,4.5))

        # HIST
        ax.hist(lab, bins=bins, density=True, alpha=0.35,
                edgecolor="black", label="LAB")
        ax.hist(line, bins=bins, density=True, alpha=0.35,
                edgecolor="black", label="LINE")

        # NORMAL CURVE
        xs = np.linspace(x_min, x_max, 400)
        ax.plot(xs,
                np.exp(-0.5*((xs-mean_lab)/std_lab)**2)/(std_lab*np.sqrt(2*np.pi)),
                linewidth=2.5, label="LAB Normal (±3σ)")
        ax.plot(xs,
                np.exp(-0.5*((xs-mean_line)/std_line)**2)/(std_line*np.sqrt(2*np.pi)),
                linewidth=2.5, linestyle="--", label="LINE Normal (±3σ)")

        # SPEC LIMIT
        ax.axvline(lo, color="red", linestyle="--", linewidth=2, label=f"LSL = {lo}")
        ax.axvline(hi, color="red", linestyle="--", linewidth=2, label=f"USL = {hi}")

        # MEAN
        ax.axvline(mean_lab, linestyle=":", linewidth=2, label=f"LAB Mean {mean_lab:.2f}")
        ax.axvline(mean_line, linestyle=":", linewidth=2, label=f"LINE Mean {mean_line:.2f}")

        ax.set_title("Hardness Distribution – LAB vs LINE (3σ)", weight="bold")
        ax.set_xlabel("Hardness (HRB)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

        note = (
            f"LAB: N={len(lab)} | Mean={mean_lab:.2f} | Std={std_lab:.2f}\n"
            f"LINE: N={len(line)} | Mean={mean_line:.2f} | Std={std_line:.2f}"
        )

        ax.text(
            1.02, 0.5, note,
            transform=ax.transAxes,
            va="center",
            bbox=dict(boxstyle="round", alpha=0.15)
        )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.85),
            frameon=False
        )

        plt.tight_layout()
        st.pyplot(fig)
