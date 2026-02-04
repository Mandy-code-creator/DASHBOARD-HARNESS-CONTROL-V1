# ================================
# FULL STREAMLIT APP – GOOGLE SHEET
# SPC HARDNESS DASHBOARD
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import requests
import math

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")
st.title("📊 SPC Hardness – Google Sheet Data")

# ================================
# GOOGLE SHEET INPUT
# ================================
st.sidebar.header("🔗 Google Sheet")

sheet_link = st.sidebar.text_input(
    "Paste Google Sheet link",
    placeholder="https://docs.google.com/spreadsheets/d/XXXX/edit#gid=0"
)

@st.cache_data
def load_google_sheet(link):
    csv_url = link.replace("/edit#gid=", "/export?format=csv&gid=")
    r = requests.get(csv_url)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

if not sheet_link:
    st.info("⬅️ Paste Google Sheet link to start")
    st.stop()

df = load_google_sheet(sheet_link)

# ================================
# COLUMN STANDARDIZE (CHỈNH NẾU CẦN)
# ================================
df = df.rename(columns={
    "HR STEEL GRADE": "Material",
    "ORDER GAUGE": "Gauge",
    "COIL NO": "Coil_No",
    "QUALITY_CODE": "Quality",
    "Standard Hardness": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE",
})

# ================================
# STANDARD HARDNESS SPLIT
# ================================
def split_std(x):
    if isinstance(x, str) and "~" in x:
        a, b = x.split("~")
        return float(a), float(b)
    return np.nan, np.nan

df[["LSL", "USL"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))

# ================================
# FORCE NUMERIC
# ================================
for c in ["Gauge", "Hardness_LAB", "Hardness_LINE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Hardness_LAB", "Hardness_LINE", "LSL", "USL"])

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🎛 Filter")

material = st.sidebar.selectbox("Material", sorted(df["Material"].dropna().unique()))
quality  = st.sidebar.selectbox("Quality", sorted(df["Quality"].dropna().unique()))

df = df[
    (df["Material"] == material) &
    (df["Quality"] == quality)
].sort_values("Coil_No")

# ================================
# SPC STATS (NO SCIPY)
# ================================
def spc_stats(data, lsl, usl):
    n = len(data)
    mean = data.mean()
    std = data.std(ddof=1)

    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    ca = (mean - (usl + lsl) / 2) / ((usl - lsl) / 2) * 100
    cpk = min(usl - mean, mean - lsl) / (3 * std) if std > 0 else np.nan

    return n, mean, std, cp, ca, cpk

def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

lsl = df["LSL"].iloc[0]
usl = df["USL"].iloc[0]

# ================================
# TREND CHART
# ================================
st.subheader("📈 Hardness Trend")

x = np.arange(1, len(df) + 1)

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(x, df["Hardness_LAB"], marker="o", label="LAB")
ax.plot(x, df["Hardness_LINE"], marker="s", label="LINE")

ax.axhline(lsl, linestyle="--", linewidth=1.5, label=f"LSL = {lsl}")
ax.axhline(usl, linestyle="--", linewidth=1.5, label=f"USL = {usl}")

ax.set_xlabel("Coil Sequence")
ax.set_ylabel("Hardness (HRB)")
ax.set_title("Hardness Trend by Coil")
ax.grid(alpha=0.3)

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
st.subheader("📊 Distribution & Normal Curve")

for label, col in [("LAB", "Hardness_LAB"), ("LINE", "Hardness_LINE")]:
    data = df[col].dropna()
    if len(data) < 5:
        continue

    n, mean, std, cp, ca, cpk = spc_stats(data, lsl, usl)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(data, bins=10, density=True, alpha=0.35, edgecolor="black")

    x_pdf = np.linspace(min(data), max(data), 200)
    ax.plot(x_pdf, normal_pdf(x_pdf, mean, std), linewidth=2)

    ax.axvline(lsl, linestyle="--", label="LSL")
    ax.axvline(usl, linestyle="--", label="USL")
    ax.axvline(mean, linestyle=":", label=f"Mean = {mean:.2f}")

    note = (
        f"N = {n}\n"
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

    ax.set_title(f"{label} Hardness Distribution")
    ax.set_xlabel("Hardness (HRB)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.9),
        frameon=False
    )

    plt.tight_layout()
    st.pyplot(fig)
