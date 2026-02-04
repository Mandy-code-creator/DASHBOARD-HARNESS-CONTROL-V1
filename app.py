# ================================
# FULL STREAMLIT APP – FINAL FIXED
# CQ00 + CQ06 MERGED
# PRODUCT SPEC MERGED IN SAME GAUGE RANGE
# TREND + DISTRIBUTION VIEW SEPARATE
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
        "📊 Distribution (LAB + LINE)",
        "🛠 Hardness → TS/YS/EL"   # <-- mới
    ]
)


# ================================
# GROUP CONDITION
# ================================
GROUP_COLS = ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"]

cnt = df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
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

    # ================================
    # VIEW MODE SWITCH
    # ================================
    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    elif view_mode == "📈 Trend (LAB / LINE)":
        x = np.arange(1, len(sub)+1)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB")
        ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE")
        ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
        ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
        ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
        ax.set_xlabel("Coil Sequence")
        ax.set_ylabel("Hardness (HRB)")
        ax.grid(alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        # ==== DOWNLOAD BUTTON ====
        buf = fig_to_png(fig)
        st.download_button(
           label="📥 Download Trend Chart",
           data=buf,
           file_name=f"trend_{g['Material']}_{g['Gauge_Range']}.png",
           mime="image/png"
        )
    elif view_mode == "📊 Distribution (LAB + LINE)":
        lab = sub["Hardness_LAB"].dropna()
        line = sub["Hardness_LINE"].dropna()
        if len(lab) >= 10 and len(line) >= 10:
            mean_lab, std_lab = lab.mean(), lab.std(ddof=1)
            mean_line, std_line = line.mean(), line.std(ddof=1)
            x_min = min(mean_lab - 3*std_lab, mean_line - 3*std_line)
            x_max = max(mean_lab + 3*std_lab, mean_line + 3*std_line)
            bins = np.linspace(x_min, x_max, 25)
            fig, ax = plt.subplots(figsize=(8,4.5))
    
            # ---- histogram
            ax.hist(lab, bins=bins, density=True, alpha=0.4, color="#1f77b4", edgecolor="black", label="LAB")
            ax.hist(line, bins=bins, density=True, alpha=0.4, color="#ff7f0e", edgecolor="black", label="LINE")
    
            # ---- normal curves
            xs = np.linspace(x_min, x_max, 400)
            ys_lab = (1/(std_lab*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_lab)/std_lab)**2)
            ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
            ax.plot(xs, ys_lab, linewidth=2.5, label="LAB Normal (±3σ)", color="#1f77b4")
            ax.plot(xs, ys_line, linewidth=2.5, linestyle="--", label="LINE Normal (±3σ)", color="#ff7f0e")
    
            # ---- spec limits
            ax.axvline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
            ax.axvline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
    
            # ---- mean lines
            ax.axvline(mean_lab, linestyle=":", linewidth=2, color="#0b3d91", label=f"LAB Mean {mean_lab:.2f}")
            ax.axvline(mean_line, linestyle=":", linewidth=2, color="#b25e00", label=f"LINE Mean {mean_line:.2f}")
    
            # ---- Ca, Cp, Cpk
            target = (hi + lo)/2
            ca_lab = abs(mean_lab - target)/((hi-lo)/2)
            ca_line = abs(mean_line - target)/((hi-lo)/2)
            cp_lab = (hi - lo)/(6*std_lab)
            cp_line = (hi - lo)/(6*std_line)
            cpk_lab = min((hi-mean_lab)/(3*std_lab), (mean_lab-lo)/(3*std_lab))
            cpk_line = min((hi-mean_line)/(3*std_line), (mean_line-lo)/(3*std_line))
    
            # ---- note box
            note = (
                f"LAB:\n  N={len(lab)}  Mean={mean_lab:.2f}  Std={std_lab:.2f}\n"
                f"  Ca={ca_lab:.2f}  Cp={cp_lab:.2f}  Cpk={cpk_lab:.2f}\n\n"
                f"LINE:\n  N={len(line)}  Mean={mean_line:.2f}  Std={std_line:.2f}\n"
                f"  Ca={ca_line:.2f}  Cp={cp_line:.2f}  Cpk={cpk_line:.2f}"
            )
            ax.text(1.02, 0.4, note, transform=ax.transAxes, va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.2, edgecolor="gray"))
    
            # ---- style
            ax.set_title("Hardness Distribution – LAB vs LINE (3σ)", weight="bold")
            ax.set_xlabel("Hardness (HRB)")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.85), frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        # ==== DOWNLOAD BUTTON ====
        buf = fig_to_png(fig)
        st.download_button(
           label="📥 Download Distribution Chart",
           data=buf,
           file_name=f"distribution_{g['Material']}_{g['Gauge_Range']}.png",
           mime="image/png"
        )
    elif view_mode == "🛠 Hardness → TS/YS/EL":
        # 1️⃣ Binning Hardness
        bins = [0, 56, 58, 60, 62, 100]  
        labels = ["<56", "56-58", "58-60", "60-62", "≥62"]
        sub["HRB_bin"] = pd.cut(sub["Hardness_LAB"], bins=bins, labels=labels, right=False)
    
        # 2️⃣ Lấy giới hạn cơ tính
        mech_cols = ["Standard TS min", "Standard TS max", 
                     "Standard YS min", "Standard YS max", 
                     "Standard EL min", "Standard EL max"]
        sub = sub.dropna(subset=mech_cols)
    
        # 3️⃣ Summary thống kê
        summary = sub.groupby("HRB_bin").agg(
            N_coils=("COIL_NO","count"),
            TS_mean=("TS","mean"), TS_min=("TS","min"), TS_max=("TS","max"),
            YS_mean=("YS","mean"), YS_min=("YS","min"), YS_max=("YS","max"),
            EL_mean=("EL","mean"), EL_min=("EL","min"), EL_max=("EL","max")
        ).reset_index()
    
        # 4️⃣ Vẽ biểu đồ lớn
        x = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(16,6))  # rộng + cao
    
        # ---- TS
        ax.plot(x, summary["TS_mean"], marker="o", color="#1f77b4", linewidth=2, markersize=8, label="TS Mean")
        ax.fill_between(x, summary["TS_min"], summary["TS_max"], color="#1f77b4", alpha=0.15)
    
        # ---- YS
        ax.plot(x, summary["YS_mean"], marker="s", color="#2ca02c", linewidth=2, markersize=8, label="YS Mean")
        ax.fill_between(x, summary["YS_min"], summary["YS_max"], color="#2ca02c", alpha=0.15)
    
        # ---- EL
        ax.plot(x, summary["EL_mean"], marker="^", color="#ff7f0e", linewidth=2, markersize=8, label="EL Mean (%)")
        ax.fill_between(x, summary["EL_min"], summary["EL_max"], color="#ff7f0e", alpha=0.15)
    
        # ---- Annotation trực tiếp trên line
        for i, row in summary.iterrows():
            ax.annotate(f"{row['TS_mean']:.1f}", (x[i], row['TS_mean']), xytext=(0,12), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color="#1f77b4")
            ax.annotate(f"{row['YS_mean']:.1f}", (x[i], row['YS_mean']), xytext=(0,-18), textcoords="offset points", ha='center', va='top', fontsize=10, fontweight='bold', color="#2ca02c")
            ax.annotate(f"{row['EL_mean']:.1f}%", (x[i], row['EL_mean']), xytext=(0,20), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color="#ff7f0e")
    
        # ---- Legend ngoài chart
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=10)
    
        # ---- Trục X và style
        ax.set_xticks(x)
        ax.set_xticklabels(summary["HRB_bin"].astype(str), fontweight='bold', fontsize=12)
        ax.set_xlabel("Hardness Range (HRB)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Mechanical Properties (MPa)", fontsize=12, fontweight='bold')
        ax.set_title("Correlation: Hardness vs TS/YS/EL", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
    
        plt.tight_layout()
        st.pyplot(fig)
    
        # 5️⃣ Bảng collapsible dưới biểu đồ
        with st.expander("🔹 Mechanical Properties per Hardness Range", expanded=False):
            st.dataframe(
                summary.style.format("{:.1f}", subset=summary.columns[2:]),
                use_container_width=True,
                height=300  # scroll nếu bảng dài
            )
    
        # 6️⃣ Download chart
        buf = fig_to_png(fig)
        st.download_button(
            label="📥 Download Hardness → TS/YS/EL Chart",
            data=buf,
            file_name=f"Hardness_TS_YS_EL_{g['Material']}_{g['Gauge_Range']}.png",
            mime="image/png"
        )
