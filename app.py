# ==========================================
# SPC HARDNESS DASHBOARD – FINAL STABLE
# DATA FROM GOOGLE SHEET
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re, math
from io import StringIO
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="SPC Hardness Dashboard",
    layout="wide"
)

st.title("📊 SPC Hardness Dashboard (Power BI Style)")

# ==========================================
# GOOGLE SHEET LINKS
# ==========================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

# ==========================================
# CACHE LOAD
# ==========================================
@st.cache_data
def load_csv(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_csv(DATA_URL)
gauge_df = load_csv(GAUGE_URL)

# ==========================================
# RENAME COLUMNS
# ==========================================
df = raw.rename(columns={
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "Claasify material": "Rolling_Type",
    "ORDER GAUGE": "Order_Gauge",
    "COIL NO": "COIL_NO",
    "QUALITY_CODE": "Quality_Code",
    "Standard Hardness": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE",
})

# ==========================================
# METALLIC TYPE AUTO
# ==========================================
metal_col = next(c for c in df.columns if "METALLIC" in c.upper())
df["Metallic_Type"] = df[metal_col]

# ==========================================
# STANDARD HARDNESS
# ==========================================
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

df[["Std_Min", "Std_Max"]] = df["Std_Text"].apply(
    lambda x: pd.Series(split_std(x))
)

# ==========================================
# FORCE NUMERIC
# ==========================================
for c in ["Hardness_LAB", "Hardness_LINE", "Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ==========================================
# QUALITY GROUP (CQ00 + CQ06)
# ==========================================
df["Quality_Group"] = df["Quality_Code"].replace({
    "CQ00": "CQ00 / CQ06",
    "CQ06": "CQ00 / CQ06"
})

# ==========================================
# GAUGE RANGE MAP
# ==========================================
gauge_df.columns = gauge_df.columns.str.strip()
range_col = next(c for c in gauge_df.columns if "RANGE" in c.upper())

def parse_range(txt):
    nums = re.findall(r"\d+\.\d+|\d+", str(txt))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[1])

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r[range_col])
    if lo is not None:
        ranges.append((lo, hi, r[range_col]))

def map_gauge(val):
    for lo, hi, name in ranges:
        if lo <= val < hi:
            return name
    return None

df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge)
df = df.dropna(subset=["Gauge_Range"])

# ==========================================
# SIDEBAR FILTER
# ==========================================
st.sidebar.header("🎛 Filters")

rolling = st.sidebar.selectbox("Rolling Type", sorted(df["Rolling_Type"].unique()))
metal = st.sidebar.selectbox("Metallic Type", sorted(df["Metallic_Type"].unique()))
qgroup = st.sidebar.selectbox("Quality Group", sorted(df["Quality_Group"].unique()))

df = df[
    (df["Rolling_Type"] == rolling) &
    (df["Metallic_Type"] == metal) &
    (df["Quality_Group"] == qgroup)
]

view_mode = st.sidebar.radio(
    "View Mode",
    ["📈 Trend + Distribution", "📋 Data Table"]
)

# ==========================================
# GROUP LOGIC (NO PRODUCT SPEC)
# ==========================================
GROUP_COLS = [
    "Rolling_Type",
    "Metallic_Type",
    "Quality_Group",
    "Gauge_Range",
    "Material"
]

cnt = (
    df.groupby(GROUP_COLS)
      .agg(N_Coils=("COIL_NO", "nunique"))
      .reset_index()
)

valid = cnt[cnt["N_Coils"] >= 30]

if valid.empty:
    st.warning("No group has ≥30 coils")
    st.stop()

# ==========================================
# NORMAL PDF (NO SCIPY)
# ==========================================
def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )

# ==========================================
# MAIN LOOP
# ==========================================
for _, g in valid.iterrows():

    sub = df[
        (df["Rolling_Type"] == g["Rolling_Type"]) &
        (df["Metallic_Type"] == g["Metallic_Type"]) &
        (df["Quality_Group"] == g["Quality_Group"]) &
        (df["Gauge_Range"] == g["Gauge_Range"]) &
        (df["Material"] == g["Material"])
    ].sort_values("COIL_NO")

    lo, hi = sub.iloc[0][["Std_Min", "Std_Max"]]

    ng = (
        (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi) |
        (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    )

    qa = "FAIL" if ng.any() else "PASS"
    specs = ", ".join(sorted(sub["Product_Spec"].unique()))

    st.markdown(f"""
### 🧱 {g['Quality_Group']} | {g['Material']} | {g['Gauge_Range']}
**Product Specs:** {specs}  
**Coils:** {sub['COIL_NO'].nunique()} | **QA:** **{qa}**
""")

    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)
        continue

    # ---------- TREND ----------
    x = np.arange(1, len(sub) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, sub["Hardness_LAB"], marker="o", label="LAB")
    ax.plot(x, sub["Hardness_LINE"], marker="s", label="LINE")
    ax.axhline(lo, linestyle="--", label=f"LSL {lo}")
    ax.axhline(hi, linestyle="--", label=f"USL {hi}")

    ax.set_title("Hardness Trend")
    ax.set_xlabel("Coil Sequence")
    ax.set_ylabel("HRB")
    ax.grid(alpha=0.3)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    st.pyplot(fig)

    # ---------- DISTRIBUTION (3-SIGMA NORMAL CURVE) ----------
for label, col in [("LAB", "Hardness_LAB"), ("LINE", "Hardness_LINE")]:
    data = sub[col].dropna()
    if len(data) < 10:
        continue

    mean = data.mean()
    std = data.std(ddof=1)

    # --- 3 sigma range
    x_min = mean - 3 * std
    x_max = mean + 3 * std

    fig, ax = plt.subplots(figsize=(7, 4))

    # --- Histogram (split OK / NG)
    bins = np.linspace(x_min, x_max, 20)
    ok_data = data[(data >= lo) & (data <= hi)]
    ng_data = data[(data < lo) | (data > hi)]

    ax.hist(ok_data, bins=bins, density=True,
            alpha=0.4, label="In Spec")
    ax.hist(ng_data, bins=bins, density=True,
            alpha=0.8, label="Out of Spec")

    # --- Normal curve (3 sigma)
    xs = np.linspace(x_min, x_max, 400)
    ys = (1 / (std * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((xs - mean) / std) ** 2
    )
    ax.plot(xs, ys, linewidth=2.5, label="Normal Curve (±3σ)")

    # --- Spec & mean lines (color-coded)
    ax.axvline(lo, linestyle="--", linewidth=2, label=f"LSL = {lo}")
    ax.axvline(hi, linestyle="--", linewidth=2, label=f"USL = {hi}")
    ax.axvline(mean, linestyle=":", linewidth=2.5, label=f"Mean = {mean:.2f}")

    # --- Style
    ax.set_title(f"{label} Hardness Distribution (3σ)", weight="bold")
    ax.set_xlabel("Hardness (HRB)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)

    # --- Stats note (outside chart)
    note = (
        f"N = {len(data)}\n"
        f"Mean = {mean:.2f}\n"
        f"Std = {std:.2f}\n"
        f"3σ Range:\n[{x_min:.2f}, {x_max:.2f}]"
    )

    ax.text(
        1.02, 0.5,
        note,
        transform=ax.transAxes,
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.15)
    )

    # --- Legend outside (Power BI style)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.85),
        frameon=False
    )

    plt.tight_layout()
    st.pyplot(fig)
