# ================================
# FULL STREAMLIT APP – FINAL FIXED
# CQ00 + CQ06 MERGED
# PRODUCT SPEC MERGED IN SAME GAUGE RANGE
# TREND + DISTRIBUTION VIEW SEPARATE
# GE* <88 FILTERED
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
# LOẠI BỎ GE* <88 NGAY TỪ ĐẦU
# ================================
if "Quality_Code" in df.columns:
    df = df[~(
        df["Quality_Code"].astype(str).str.startswith("GE") &
        ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88))
    )]

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
        "🛠 Hardness → TS/YS/EL",
        "📊 TS/YS/EL Trend & Distribution",
        "🧮 Predict TS/YS/EL (Custom Hardness)",
        "📊 Hardness → Mechanical Range"
    ]
)

with st.sidebar.expander("💡 About 95% Confidence Interval (CI)", expanded=False):
    st.markdown(
        """
        - The shaded area around the predicted line represents the **95% Confidence Interval (CI)**.
        - It means that **approximately 95% of future observations are expected to fall within this range** if the linear model is valid.
        - Narrow CI → high precision; wide CI → higher uncertainty.
        - This note is **shown once** for clarity and can be collapsed.
        """
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
**Hardness Limit (HRB):** {lo:.1f} ~ {hi:.1f}
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
            ax.hist(lab, bins=bins, density=True, alpha=0.4, color="#1f77b4", edgecolor="black", label="LAB")
            ax.hist(line, bins=bins, density=True, alpha=0.4, color="#ff7f0e", edgecolor="black", label="LINE")
            xs = np.linspace(x_min, x_max, 400)
            ys_lab = (1/(std_lab*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_lab)/std_lab)**2)
            ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
            ax.plot(xs, ys_lab, linewidth=2.5, label="LAB Normal (±3σ)", color="#1f77b4")
            ax.plot(xs, ys_line, linewidth=2.5, linestyle="--", label="LINE Normal (±3σ)", color="#ff7f0e")
            ax.axvline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
            ax.axvline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
            ax.axvline(mean_lab, linestyle=":", linewidth=2, color="#0b3d91", label=f"LAB Mean {mean_lab:.2f}")
            ax.axvline(mean_line, linestyle=":", linewidth=2, color="#b25e00", label=f"LINE Mean {mean_line:.2f}")
            note = (
                f"LAB:\n  N={len(lab)}  Mean={mean_lab:.2f}  Std={std_lab:.2f}\n"
                f"  Ca={abs(mean_lab-(hi+lo)/2)/((hi-lo)/2):.2f}  Cp={(hi-lo)/(6*std_lab):.2f}  Cpk={min((hi-mean_lab)/(3*std_lab),(mean_lab-lo)/(3*std_lab)):.2f}\n\n"
                f"LINE:\n  N={len(line)}  Mean={mean_line:.2f}  Std={std_line:.2f}\n"
                f"  Ca={abs(mean_line-(hi+lo)/2)/((hi-lo)/2):.2f}  Cp={(hi-lo)/(6*std_line):.2f}  Cpk={min((hi-mean_line)/(3*std_line),(mean_line-lo)/(3*std_line)):.2f}"
            )
            ax.text(1.02, 0.4, note, transform=ax.transAxes, va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.2, edgecolor="gray"))
            ax.set_title("Hardness Distribution – LAB vs LINE (3σ)", weight="bold")
            ax.set_xlabel("Hardness (HRB)")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.85), frameon=False)
            plt.tight_layout()
            st.pyplot(fig)
            buf = fig_to_png(fig)
            st.download_button(
               label="📥 Download Distribution Chart",
               data=buf,
               file_name=f"distribution_{g['Material']}_{g['Gauge_Range']}.png",
               mime="image/png"
            )

# ================================
# (Các view còn lại như Hardness → TS/YS/EL, TS/YS/EL Trend & Distribution, Predict TS/YS/EL)
# ================================
# Mình giữ nguyên code của bạn, chỉ fix indentation + GE<88 filter
# ================================


    elif view_mode == "🛠 Hardness → TS/YS/EL":

        # ================================
        # 1️⃣ Chuẩn bị dữ liệu
        # ================================
        sub = sub.dropna(subset=["Hardness_LAB","Hardness_LINE","TS","YS","EL"])
    
        # Loại bỏ hoàn toàn coil GE* <88
        if "Quality_Code" in sub.columns:
            sub = sub[~(
                sub["Quality_Code"].astype(str).str.startswith("GE") &
                ((sub["Hardness_LAB"] < 88) | (sub["Hardness_LINE"] < 88))
            )]
    
        # ================================
        # 2️⃣ Binning Hardness (chi tiết 62–88)
        # ================================
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub["HRB_bin"] = pd.cut(sub["Hardness_LAB"], bins=bins, labels=labels, right=False)
    
        # ================================
        # 3️⃣ Lấy giới hạn cơ tính
        # ================================
        mech_cols = ["Standard TS min","Standard TS max",
                     "Standard YS min","Standard YS max",
                     "Standard EL min","Standard EL max"]
        sub = sub.dropna(subset=mech_cols)
    
        # ================================
        # 4️⃣ Summary thống kê
        # ================================
        summary = (sub.groupby("HRB_bin").agg(
            N_coils=("COIL_NO","count"),
            TS_mean=("TS","mean"), TS_min=("TS","min"), TS_max=("TS","max"),
            YS_mean=("YS","mean"), YS_min=("YS","min"), YS_max=("YS","max"),
            EL_mean=("EL","mean"), EL_min=("EL","min"), EL_max=("EL","max"),
            EL_spec_min=("Standard EL min","min")
        ).reset_index())
        summary = summary[summary["N_coils"]>0]
    
        # ================================
        # 5️⃣ Vẽ biểu đồ
        # ================================
        x = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(16,6))
    
        # TS
        ax.plot(x, summary["TS_mean"], marker="o", linewidth=2, markersize=8, label="TS Mean")
        ax.fill_between(x, summary["TS_min"], summary["TS_max"], alpha=0.15)
    
        # YS
        ax.plot(x, summary["YS_mean"], marker="s", linewidth=2, markersize=8, label="YS Mean")
        ax.fill_between(x, summary["YS_min"], summary["YS_max"], alpha=0.15)
    
        # EL
        ax.plot(x, summary["EL_mean"], marker="^", linewidth=2, markersize=8, label="EL Mean (%)")
        ax.fill_between(x, summary["EL_min"], summary["EL_max"], alpha=0.15)
    
        # ================================
        # 6️⃣ Annotation (EL < spec → đỏ + ❌)
        # ================================
        for idx, row in enumerate(summary.itertuples()):
            # TS
            ax.annotate(f"{row.TS_mean:.1f}", (x[idx], row.TS_mean),
                        xytext=(0,12), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            # YS
            ax.annotate(f"{row.YS_mean:.1f}", (x[idx], row.YS_mean),
                        xytext=(0,-18), textcoords="offset points",
                        ha="center", va="top", fontsize=10, fontweight="bold")
            # EL
            el_fail = row.EL_mean < row.EL_spec_min
            ax.annotate(f"{row.EL_mean:.1f}%" + (" ❌" if el_fail else ""),
                        (x[idx], row.EL_mean),
                        xytext=(0,20), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color="red" if el_fail else None)
    
        # ================================
        # 7️⃣ Trục & style
        # ================================
        ax.set_xticks(x)
        ax.set_xticklabels(summary["HRB_bin"].astype(str), fontweight="bold", fontsize=12)
        ax.set_xlabel("Hardness Range (HRB)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mechanical Properties (MPa)", fontsize=12, fontweight="bold")
        ax.set_title("Correlation: Hardness vs TS/YS/EL", fontsize=14, fontweight="bold")
    
        # Đường phân cách FULL HARD
        if "88-92" in summary["HRB_bin"].astype(str).values:
            idx88 = summary.index[summary["HRB_bin"].astype(str)=="88-92"][0]
            ax.axvline(idx88-0.5, linestyle="--", alpha=0.5)
    
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
    
        # ================================
        # 8️⃣ Bảng dữ liệu
        # ================================
        with st.expander("🔹 Mechanical Properties per Hardness Range", expanded=False):
            st.dataframe(summary.style.format("{:.1f}", subset=summary.columns[2:]),
                         use_container_width=True, height=300)
    
        # ================================
        # 9️⃣ Download
        # ================================
        buf = fig_to_png(fig)
        st.download_button("📥 Download Hardness → TS/YS/EL Chart",
                           data=buf,
                           file_name=f"Hardness_TS_YS_EL_{g['Material']}_{g['Gauge_Range']}.png",
                           mime="image/png")
        
        # ================================
        # 🔹 Quick Conclusion per HRB bin (mới)
        # ================================
        st.markdown("### 📌 Quick Conclusion per HRB bin")
        
        qc_list = []
        for hrb_bin, group in sub.groupby("HRB_bin"):
            if group.empty:
                continue
            TS_min = group["TS"].min()
            TS_max = group["TS"].max()
            YS_min = group["YS"].min()
            YS_max = group["YS"].max()
            EL_min = group["EL"].min()
            EL_max = group["EL"].max()
        
            TS_flag = "⚠️" if (TS_min < group["Standard TS min"].min() or TS_max > group["Standard TS max"].max()) else "✅"
            YS_flag = "⚠️" if (YS_min < group["Standard YS min"].min() or YS_max > group["Standard YS max"].max()) else "✅"
            EL_flag = "⚠️" if (EL_min < group["Standard EL min"].min() or EL_max > group["Standard EL max"].max()) else "✅"
        
            qc_list.append(f"**{hrb_bin}**: TS={TS_flag} ({TS_min:.1f}-{TS_max:.1f}), "
                           f"YS={YS_flag} ({YS_min:.1f}-{YS_max:.1f}), "
                           f"EL={EL_flag} ({EL_min:.1f}-{EL_max:.1f})")
        
        for line in qc_list:
            st.markdown(line)

    elif view_mode == "📊 TS/YS/EL Trend & Distribution":
        import re, uuid
    
        # ===== 1️⃣ Binning Hardness
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub["HRB_bin"] = pd.cut(sub["Hardness_LAB"], bins=bins, labels=labels, right=False)
    
        mech_cols = ["Standard TS min","Standard TS max",
                     "Standard YS min","Standard YS max",
                     "Standard EL min","Standard EL max"]
        sub = sub.dropna(subset=mech_cols)
    
        hrb_bins = [b for b in labels if b in sub["HRB_bin"].unique()]
    
        # ===== 2️⃣ Safe NG check
        def check_ng(series, lsl, usl):
            series = series.fillna(np.nan)
            mask = pd.Series(False, index=series.index)
            if pd.notna(lsl) and pd.notna(usl):
                mask = (series < lsl) | (series > usl)
            elif pd.notna(lsl):
                mask = series < lsl
            elif pd.notna(usl):
                mask = series > usl
            return mask
    
        # ===== 3️⃣ Loop HRB bin
        for i, hrb in enumerate(hrb_bins):
            df_bin = sub[sub["HRB_bin"] == hrb].sort_values("COIL_NO")
            N = len(df_bin)
            if N == 0:
                continue
    
            # Giới hạn cơ tính
            TS_LSL, TS_USL = df_bin["Standard TS min"].iloc[0], df_bin["Standard TS max"].iloc[0]
            YS_LSL, YS_USL = df_bin["Standard YS min"].iloc[0], df_bin["Standard YS max"].iloc[0]
            EL_LSL, EL_USL = df_bin["Standard EL min"].iloc[0], df_bin["Standard EL max"].iloc[0]
    
            # Tạo cột NG safe
            df_bin["NG_TS"] = check_ng(df_bin["TS"], TS_LSL, TS_USL)
            df_bin["NG_YS"] = check_ng(df_bin["YS"], YS_LSL, YS_USL)
            df_bin["NG_EL"] = check_ng(df_bin["EL"], EL_LSL, EL_USL)
    
            st.markdown(f"### HRB bin: {hrb} | N_coils={N}")
    
            # ===== 4️⃣ Trend Chart
            fig, ax = plt.subplots(figsize=(14,4))
            x = np.arange(1, N+1)
            for col, color, marker in [("TS","#1f77b4","o"), ("YS","#2ca02c","s"), ("EL","#ff7f0e","^")]:
                ax.plot(x, df_bin[col], marker=marker, label=col, color=color)
                ax.fill_between(x, df_bin[col].min(), df_bin[col].max(), color=color, alpha=0.1)
                ng_idx = df_bin.index[df_bin[f"NG_{col}"]].to_list()
                ax.scatter([x[j] for j in range(N) if df_bin.index[j] in ng_idx],
                           df_bin.loc[ng_idx, col], color="red", s=50, zorder=5)
    
            # Spec lines
            for val, col in [(TS_LSL,"#1f77b4"),(TS_USL,"#1f77b4"),
                             (YS_LSL,"#2ca02c"),(YS_USL,"#2ca02c"),
                             (EL_LSL,"#ff7f0e"),(EL_USL,"#ff7f0e")]:
                if pd.notna(val):
                    ax.axhline(val, color=col, linestyle="--", alpha=0.5)
    
            ax.set_xlabel("Coil Sequence")
            ax.set_ylabel("Mechanical Properties (MPa / %)")
            ax.set_title(f"Trend: TS/YS/EL for HRB {hrb}")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="best")
            plt.tight_layout()
            st.pyplot(fig)
    
            safe_hrb = re.sub(r"[<≥]", "", str(hrb))
            buf_trend = fig_to_png(fig)
            st.download_button(label=f"📥 Download Trend HRB {hrb}", data=buf_trend,
                               file_name=f"trend_{safe_hrb}_{i}.png", mime="image/png",
                               key=str(uuid.uuid4()))
    
            # ===== 5️⃣ Distribution Chart
            fig, ax = plt.subplots(figsize=(14,4))
            for col, color in [("TS","#1f77b4"),("YS","#2ca02c"),("EL","#ff7f0e")]:
                series = df_bin[col].dropna()
                ax.hist(series, bins=10, alpha=0.4, color=color, edgecolor="black", label=col)
                # Mean ± Std
                mean, std = series.mean(), series.std(ddof=1)
                ax.axvline(mean, color=color, linestyle=":", label=f"{col} Mean {mean:.1f} ±{std:.1f}")
            ax.set_title(f"Distribution: TS/YS/EL for HRB {hrb}")
            ax.set_xlabel("Value"); ax.set_ylabel("Count"); ax.grid(alpha=0.3, linestyle="--"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
    
            buf_dist = fig_to_png(fig)
            st.download_button(label=f"📥 Download Distribution HRB {hrb}", data=buf_dist,
                               file_name=f"dist_{safe_hrb}_{i}.png", mime="image/png",
                               key=str(uuid.uuid4()))
    
            # ===== 6️⃣ Mechanical Properties Table
            summary_bin = df_bin[["COIL_NO","TS","YS","EL","HRB_bin","NG_TS","NG_YS","NG_EL"]].copy()
            with st.expander(f"📋 Mechanical Properties Table (HRB {hrb})", expanded=False):
                st.dataframe(summary_bin.style.format("{:.1f}", subset=["TS","YS","EL"]),
                             use_container_width=True)
    
            # ===== Quick Conclusion Safe & Gọn (HRB limit 1 lần)
            if "Std_Min" in sub.columns and "Std_Max" in sub.columns:
                lsl, usl = sub["Std_Min"].iloc[0], sub["Std_Max"].iloc[0]
                observed_min, observed_max = sub["Hardness_LAB"].min(), sub["Hardness_LAB"].max()
                
                conclusion = []
                for prop, ng_col in [("TS","NG_TS"), ("YS","NG_YS"), ("EL","NG_EL")]:
                    if prop not in sub.columns:
                        continue
                    n_ng = sub[ng_col].fillna(False).sum() if ng_col in sub.columns else 0
                    N = len(sub)
                    val_min, val_max = sub[prop].min(), sub[prop].max()
                    status = "✅ OK" if n_ng==0 else f"⚠️ {n_ng}/{N} out of spec"
                    conclusion.append(f"{prop}: {status} | {val_min:.1f}-{val_max:.1f}")
            
                st.markdown(
                    f"**📌 Quick Conclusion:** HRB limit={lsl:.1f}-{usl:.1f} | observed HRB={observed_min:.1f}-{observed_max:.1f} | " +
                    " | ".join(conclusion)
                )
    # ================================
    elif view_mode == "🧮 Predict TS/YS/EL (Custom Hardness)":
        import uuid
    
        st.markdown("## 🧮 Predict Mechanical Properties for Custom Hardness")
    
        # --- Tạo uuid để tránh Duplicate Key ---
        uid = str(uuid.uuid4())
    
        # --- Chọn kiểu dự báo ---
        pred_type = st.radio(
            "Select input type for prediction:",
            ["Single Value", "Range"],
            key=f"predict_type_{uid}"
        )
    
        if pred_type == "Single Value":
            user_hrb = st.number_input(
                "Enter desired LINE Hardness (HRB):",
                min_value=0.0, max_value=120.0, value=90.0, step=0.1,
                key=f"predict_hrb_single_{uid}"
            )
            hrb_values = [user_hrb]
    
        else:
            hrb_min = st.number_input(
                "Enter minimum LINE Hardness (HRB):",
                min_value=0.0, max_value=120.0, value=88.0, step=0.1,
                key=f"predict_hrb_min_{uid}"
            )
            hrb_max = st.number_input(
                "Enter maximum LINE Hardness (HRB):",
                min_value=0.0, max_value=120.0, value=92.0, step=0.1,
                key=f"predict_hrb_max_{uid}"
            )
            step = st.number_input(
                "Step for prediction:",
                min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                key=f"predict_hrb_step_{uid}"
            )
            hrb_values = list(np.arange(hrb_min, hrb_max + 0.01, step))
    
        # --- Chuẩn bị dữ liệu ---
        sub_fit = sub.dropna(subset=["Hardness_LINE","TS","YS","EL"]).copy()
        N_coils = len(sub_fit)
        if N_coils < 5:
            st.warning(f"⚠️ Not enough data to perform prediction (N={N_coils})")
            st.stop()
    
        # --- Fit linear model TS/YS/EL ---
        pred_values = {}
        for prop in ["TS","YS","EL"]:
            x = sub_fit["Hardness_LINE"].values
            y = sub_fit[prop].values
            a,b = np.polyfit(x,y,1)
            pred_values[prop] = a * np.array(hrb_values) + b
    
        # --- Vẽ trend + marker dự báo ---
        fig, ax = plt.subplots(figsize=(14,5))
        coils = np.arange(1, N_coils+1)
    
        for prop, color, marker, unit in [("TS","#1f77b4","o","MPa"),
                                           ("YS","#2ca02c","s","MPa"),
                                           ("EL","#ff7f0e","^","%")]:
            vals = sub_fit[prop].values
            ax.plot(coils, vals, marker=marker, color=color, label=f"{prop} Observed")
    
            # predicted marker
            pred = pred_values[prop]
            pred_x = [coils[-1] + 1 + i for i in range(len(pred))]
            ax.scatter(pred_x, pred, color="red", s=100, marker="X", label=f"{prop} Predicted ({unit})")
            # connect last observed to predicted
            for j in range(len(pred)):
                ax.plot([coils[-1], pred_x[j]], [vals[-1], pred[j]], linestyle=":", color="red", linewidth=2)
    
        ax.set_xlabel("Coil Sequence")
        ax.set_ylabel("Mechanical Properties (TS/YS in MPa, EL in %)")
        ax.set_title("Trend: Observed TS/YS/EL with Predicted Hardness")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02,0.5))
        plt.tight_layout()
        st.pyplot(fig)
    
        # --- Bảng dự báo thu gọn bằng expander ---
        pred_table = pd.DataFrame({"HRB": [int(round(h)) for h in hrb_values]})
        for prop in ["TS","YS","EL"]:
            pred_table[prop] = pred_values[prop]
    
        with st.expander("📋 Predicted Mechanical Properties (click to expand)", expanded=True):
            st.dataframe(pred_table.style.format("{:.1f}", subset=["TS","YS","EL"]), use_container_width=True)
    
        st.markdown("### 📌 Notes")
        st.markdown(
            "- Red 'X' markers on trend indicate predicted values for custom hardness.\n"
            "- Dashed lines connect last observed coil to predicted values.\n"
            "- EL unit is **%**, TS/YS units are **MPa**.\n"
            "- Table shows predicted values for selected LINE Hardness range."
        )
    elif view_mode == "📊 Hardness → Mechanical Range":
        st.markdown("## 📊 Hardness → Mechanical Properties Range")
    
        # 1️⃣ Chuẩn bị dữ liệu
        sub_stats = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"]).copy()
        if sub_stats.empty:
            st.info("No data available for Hardness → Mechanical Range")
            st.stop()
    
        # 2️⃣ Tạo bảng thống kê theo Hardness LINE rounded
        sub_stats["HRB_round"] = sub_stats["Hardness_LINE"].round(0).astype(int)
    
        summary_range = (
            sub_stats.groupby("HRB_round").agg(
                N_coils=("COIL_NO", "count"),
                TS_min=("TS", "min"), TS_max=("TS", "max"), TS_mean=("TS","mean"),
                YS_min=("YS", "min"), YS_max=("YS", "max"), YS_mean=("YS","mean"),
                EL_min=("EL", "min"), EL_max=("EL", "max"), EL_mean=("EL","mean")
            )
            .reset_index()
            .sort_values("HRB_round")
        )
    
        if summary_range.empty:
            st.info("No data found for current Hardness values")
        else:
            # 3️⃣ Hiển thị bảng gọn
            st.dataframe(
                summary_range.style.format({
                    "TS_min":"{:.1f}", "TS_max":"{:.1f}", "TS_mean":"{:.1f}",
                    "YS_min":"{:.1f}", "YS_max":"{:.1f}", "YS_mean":"{:.1f}",
                    "EL_min":"{:.1f}", "EL_max":"{:.1f}", "EL_mean":"{:.1f}"
                }),
                use_container_width=True,
                height=400
            )
    
            # 4️⃣ Thêm note
            st.markdown(
                "- HRB values rounded to nearest integer.\n"
                "- TS/YS in MPa, EL in %.\n"
                "- N_coils = số lượng coil trong mỗi Hardness."
            )
