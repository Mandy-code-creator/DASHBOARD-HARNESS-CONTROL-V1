# ================================
# REFACTORED STREAMLIT APP – SPC HARDNESS DASHBOARD
# Modular, clean, maintainable
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
# UTILITY FUNCTIONS
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
    mean = data.mean()
    std = data.std(ddof=1)
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    ca = (mean - (usl + lsl) / 2) / ((usl - lsl) / 2) * 100 if usl > lsl else np.nan
    cpk = min((usl - mean), (mean - lsl)) / (3 * std) if std > 0 else np.nan
    return mean, std, cp, ca, cpk


def check_ng(series, lsl, usl):
    mask = pd.Series(False, index=series.index)
    if pd.notna(lsl) and pd.notna(usl):
        mask = (series < lsl) | (series > usl)
    elif pd.notna(lsl):
        mask = series < lsl
    elif pd.notna(usl):
        mask = series > usl
    return mask


def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan


def parse_range(text):
    nums = re.findall(r"\d+\.\d+|\d+", str(text))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[-1])

# ================================
# DATA LOADING & CLEANING
# ================================

DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    df = pd.read_csv(StringIO(r.text))
    # Detect metallic type column
    metal_col = next(c for c in df.columns if "METALLIC" in c.upper())
    df["Metallic_Type"] = df[metal_col]
    # Rename columns
    df = df.rename(columns={
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
    # Standard hardness split
    df[["Std_Min", "Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
    # Force numeric
    for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Merge quality group
    df["Quality_Group"] = df["Quality_Code"].replace({"CQ00":"CQ00 / CQ06","CQ06":"CQ00 / CQ06"})
    # Filter GE* <88
    df = df[~(
        df["Quality_Code"].astype(str).str.startswith("GE") &
        ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88))
    )]
    return df

@st.cache_data
def load_gauge():
    g = pd.read_csv(GAUGE_URL)
    g.columns = g.columns.str.strip()
    gauge_col = next(c for c in g.columns if "RANGE" in c.upper())
    ranges = []
    for _, r in g.iterrows():
        lo, hi = parse_range(r[gauge_col])
        if lo is not None:
            ranges.append((lo, hi, r[gauge_col]))
    return ranges

raw_df = load_main()
gauge_ranges = load_gauge()

# Map gauge
raw_df["Gauge_Range"] = raw_df["Order_Gauge"].apply(lambda val: next((name for lo, hi, name in gauge_ranges if lo <= val < hi), None))
raw_df = raw_df.dropna(subset=["Gauge_Range"])

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🎛 FILTER")
rolling = st.sidebar.radio("Rolling Type", sorted(raw_df["Rolling_Type"].unique()))
metal = st.sidebar.radio("Metallic Type", sorted(raw_df["Metallic_Type"].unique()))
qgroup = st.sidebar.radio("Quality Group", sorted(raw_df["Quality_Group"].unique()))

filtered_df = raw_df[(raw_df["Rolling_Type"]==rolling) & (raw_df["Metallic_Type"]==metal) & (raw_df["Quality_Group"]==qgroup)]

view_mode = st.sidebar.radio(
    "📊 View Mode",
    ["📋 Data Table", "📈 Trend (LAB / LINE)", "📊 Distribution (LAB + LINE)", 
     "🛠 Hardness → TS/YS/EL", "📊 TS/YS/EL Trend & Distribution", "🧮 Predict TS/YS/EL (Custom Hardness)"]
)

with st.sidebar.expander("💡 About 95% Confidence Interval (CI)", expanded=False):
    st.markdown("""
- The shaded area around the predicted line represents the **95% Confidence Interval (CI)**.
- Approximately 95% of future observations expected to fall within this range.
- Narrow CI → high precision; wide CI → higher uncertainty.
""")

# ================================
# VALID GROUPS (≥30 coils)
# ================================
GROUP_COLS = ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"]
cnt = filtered_df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid_groups = cnt[cnt["N_Coils"] >= 30]
if valid_groups.empty:
    st.warning("⚠️ No group with ≥30 coils")
    st.stop()

# ================================
# VIEW FUNCTIONS
# ================================

def render_data_table(sub):
    st.dataframe(sub, use_container_width=True)


def render_trend(sub, lo, hi, group_name):
    x = np.arange(1, len(sub)+1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, sub["Hardness_LAB"], marker="o", label="LAB")
    ax.plot(x, sub["Hardness_LINE"], marker="s", label="LINE")
    ax.axhline(lo, linestyle="--", color="red", label=f"LSL={lo}")
    ax.axhline(hi, linestyle="--", color="red", label=f"USL={hi}")
    ax.set_title("Hardness Trend by Coil Sequence")
    ax.set_xlabel("Coil Sequence")
    ax.set_ylabel("Hardness (HRB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    st.pyplot(fig)
    buf = fig_to_png(fig)
    st.download_button(f"📥 Download Trend Chart {group_name}", buf, f"trend_{group_name}.png", "image/png")


def render_distribution(sub, lo, hi, group_name):
    lab = sub["Hardness_LAB"].dropna()
    line = sub["Hardness_LINE"].dropna()
    if len(lab) < 10 or len(line) < 10:
        st.warning("Not enough data for distribution chart")
        return
    mean_lab, std_lab = lab.mean(), lab.std(ddof=1)
    mean_line, std_line = line.mean(), line.std(ddof=1)
    x_min = min(mean_lab-3*std_lab, mean_line-3*std_line)
    x_max = max(mean_lab+3*std_lab, mean_line+3*std_line)
    bins = np.linspace(x_min, x_max, 25)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.hist(lab, bins=bins, alpha=0.4, color="#1f77b4", edgecolor="black", label="LAB", density=True)
    ax.hist(line, bins=bins, alpha=0.4, color="#ff7f0e", edgecolor="black", label="LINE", density=True)
    xs = np.linspace(x_min, x_max, 400)
    ys_lab = (1/(std_lab*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_lab)/std_lab)**2)
    ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
    ax.plot(xs, ys_lab, color="#1f77b4", label="LAB Normal (±3σ)")
    ax.plot(xs, ys_line, color="#ff7f0e", linestyle="--", label="LINE Normal (±3σ)")
    ax.axvline(lo, linestyle="--", color="red", label=f"LSL={lo}")
    ax.axvline(hi, linestyle="--", color="red", label=f"USL={hi}")
    ax.set_title("Hardness Distribution – LAB vs LINE (3σ)")
    ax.set_xlabel("Hardness (HRB)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.85), frameon=False)
    st.pyplot(fig)
    buf = fig_to_png(fig)
    st.download_button(f"📥 Download Distribution Chart {group_name}", buf, f"distribution_{group_name}.png", "image/png")

# ================================
# MAIN LOOP
# ================================

for _, g in valid_groups.iterrows():
    sub = filtered_df[
        (filtered_df["Rolling_Type"]==g["Rolling_Type"]) &
        (filtered_df["Metallic_Type"]==g["Metallic_Type"]) &
        (filtered_df["Quality_Group"]==g["Quality_Group"]) &
        (filtered_df["Gauge_Range"]==g["Gauge_Range"]) &
        (filtered_df["Material"]==g["Material"])
    ].sort_values("COIL_NO")

    lo, hi = sub.iloc[0][["Std_Min", "Std_Max"]]
    sub["NG_LAB"] = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"]
    qa = "FAIL" if sub["NG"].any() else "PASS"
    specs = ", ".join(sorted(sub["Product_Spec"].unique()))

    st.markdown(f"""
### 🧱 Quality Group: {g['Quality_Group']}
**Material:** {g['Material']}  
**Gauge Range:** {g['Gauge_Range']}  
**Product Specs:** {specs}  
**Coils:** {sub['COIL_NO'].nunique()} | **QA:** 🧪 **{qa}**  
**Hardness Limit (HRB):** {lo:.1f} ~ {hi:.1f}
""")

    group_name = f"{g['Material']}_{g['Gauge_Range']}"

    if view_mode == "📋 Data Table":
        render_data_table(sub)
    elif view_mode == "📈 Trend (LAB / LINE)":
        render_trend(sub, lo, hi, group_name)
    elif view_mode == "📊 Distribution (LAB + LINE)":
        render_distribution(sub, lo, hi, group_name)

# ================================
# TODO: Add other view renderings (Hardness → TS/YS/EL, TS/YS/EL Trend & Distribution, Predict TS/YS/EL)
# ================================
