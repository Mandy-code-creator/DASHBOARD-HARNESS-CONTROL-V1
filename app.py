import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import io

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Quality SPC Dashboard",
    layout="wide"
)

st.title("📊 Quality SPC Dashboard – FINAL")

# =============================
# HELPER FUNCTIONS
# =============================
def parse_range(text):
    """
    Ví dụ:
    '56~62' → (56, 62)
    """
    if pd.isna(text):
        return None, None
    text = str(text).replace(" ", "")
    if "~" in text:
        lo, hi = text.split("~")
        return float(lo), float(hi)
    return None, None


def spc_stats(data, lo, hi):
    """
    Trả về:
    mean, std, Cp, Ca, Cpk
    """
    data = np.array(data)
    mean = data.mean()
    std = data.std(ddof=1) if len(data) > 1 else 0

    if std == 0 or lo is None or hi is None:
        return mean, std, None, None, None

    cp = (hi - lo) / (6 * std)
    ca = abs(mean - (hi + lo) / 2) / ((hi - lo) / 2)
    cpk = min((hi - mean) / (3 * std), (mean - lo) / (3 * std))

    return mean, std, cp, ca, cpk


def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def thickness_group(thk):
    """
    Gộp độ dày thành nhóm
    """
    if thk < 0.4:
        return "<0.4"
    elif thk < 0.6:
        return "0.4–0.6"
    elif thk < 0.8:
        return "0.6–0.8"
    else:
        return "≥0.8"


# =============================
# DATA UPLOAD
# =============================
st.sidebar.header("📂 Upload Data")

data_file = st.sidebar.file_uploader("Upload measurement data (Excel)", type=["xlsx"])
spec_file = st.sidebar.file_uploader("Upload spec table (Excel)", type=["xlsx"])

if data_file is None or spec_file is None:
    st.info("⬅️ Upload **both** measurement data & spec table to start")
    st.stop()

df = pd.read_excel(data_file)
spec = pd.read_excel(spec_file)

# =============================
# STANDARDIZE COLUMN NAMES
# =============================
df.columns = df.columns.str.strip()
spec.columns = spec.columns.str.strip()

# REQUIRED COLUMNS
required_cols = [
    "QUALITY_CODE",
    "PRODUCT_SPECIFICATION_CODE",
    "ORDER_GAUGE",
    "VALUE",
    "DATE"
]

for c in required_cols:
    if c not in df.columns:
        st.error(f"❌ Missing column in data file: {c}")
        st.stop()

# =============================
# BUILD SPEC DICT
# =============================
spec_dict = {}
for _, r in spec.iterrows():
    lo, hi = parse_range(r["RANGE_NAME"])
    key = (
        r["QUALITY_CODE"],
        thickness_group(r["ORDER_GAUGE"])
    )
    spec_dict[key] = (lo, hi)

# =============================
# ADD GROUP COLUMN
# =============================
df["THICKNESS_GROUP"] = df["ORDER_GAUGE"].apply(thickness_group)

# =============================
# FILTER
# =============================
qc = st.sidebar.selectbox(
    "QUALITY CODE",
    sorted(df["QUALITY_CODE"].unique())
)

tg = st.sidebar.selectbox(
    "Thickness Group",
    sorted(df["THICKNESS_GROUP"].unique())
)

df_sel = df[
    (df["QUALITY_CODE"] == qc) &
    (df["THICKNESS_GROUP"] == tg)
].sort_values("DATE")

if df_sel.empty:
    st.warning("No data for this selection")
    st.stop()

# =============================
# GET LIMIT
# =============================
lo, hi = spec_dict.get((qc, tg), (None, None))

# =============================
# SPC STATS
# =============================
mean, std, cp, ca, cpk = spc_stats(df_sel["VALUE"], lo, hi)

# =============================
# KPI DISPLAY
# =============================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean", f"{mean:.2f}")
c2.metric("Std", f"{std:.2f}")
c3.metric("Cp", "-" if cp is None else f"{cp:.2f}")
c4.metric("Ca", "-" if ca is None else f"{ca*100:.1f}%")
c5.metric("Cpk", "-" if cpk is None else f"{cpk:.2f}")

# =============================
# TREND LINE CHART
# =============================
st.subheader("📈 Trend Chart")

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(df_sel))
y = df_sel["VALUE"]

ax.plot(x, y, marker="o", label="Measured")

if lo is not None:
    ax.axhline(lo, linestyle="--", label="LSL")
if hi is not None:
    ax.axhline(hi, linestyle="--", label="USL")

ax.axhline(mean, linestyle=":", label="Mean")

ax.set_xticks(x)
ax.set_xticklabels(df_sel["DATE"].astype(str), rotation=45, ha="right")

ax.set_ylabel("Value")
ax.set_title("Trend with Control Limits")

# legend outside
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

st.pyplot(fig)

# =============================
# DISTRIBUTION + NORMAL CURVE
# =============================
st.subheader("📊 Distribution & Normal Curve")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.hist(y, bins=10, density=True, alpha=0.6)

if std > 0:
    xs = np.linspace(y.min(), y.max(), 200)
    ax2.plot(xs, normal_pdf(xs, mean, std), linewidth=2)

if lo is not None:
    ax2.axvline(lo, linestyle="--", label="LSL")
if hi is not None:
    ax2.axvline(hi, linestyle="--", label="USL")

ax2.set_title("Histogram + Normal Curve")
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

st.pyplot(fig2)

# =============================
# RAW DATA
# =============================
with st.expander("📋 View Raw Data"):
    st.dataframe(df_sel)
