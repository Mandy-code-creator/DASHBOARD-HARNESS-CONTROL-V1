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
        "📋 Data Inspection",
        "📉 Hardness Analysis (Trend & Dist)",     # <--- Đã gộp 2 cái cũ vào đây
        "🔗 Correlation: Hardness vs Mech Props", # <--- Tên mới cho Hardness -> TS/YS/EL
        "⚙️ Mech Props Analysis",                 # <--- Tên mới cho TS/YS/EL Trend
        "🔍 Lookup: Hardness Range → Actual Mech Props", # <--- Tính năng tra cứu
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
    if view_mode == "📋 Data Inspection":
        st.dataframe(sub, use_container_width=True)

# ========================================================
    # MODE: HARDNESS ANALYSIS (TREND & DISTRIBUTION COMBINED)
    # ========================================================
    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        
        st.markdown("### 📉 Hardness Analysis: Process Stability & Capability")
        
        # Tạo 2 Tabs
        tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & SPC"])

        # --- TAB 1: TREND CHART ---
        with tab_trend:
            st.markdown(f"**Material:** {g['Material']} | **Gauge:** {g['Gauge_Range']}")
            
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4.5)) # Resize cho đẹp hơn trong tab
            
            # Plot Data
            ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB", alpha=0.8)
            ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE", alpha=0.8)
            
            # Limits
            ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
            ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
            
            # Styling
            ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
            ax.set_xlabel("Coil Sequence")
            ax.set_ylabel("Hardness (HRB)")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4) # Legend xuống dưới cho gọn
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download
            buf = fig_to_png(fig)
            st.download_button(
                label="📥 Download Trend Chart",
                data=buf,
                file_name=f"trend_{g['Material']}_{g['Gauge_Range']}.png",
                mime="image/png",
                key=f"dl_trend_{_}"
            )

        # --- TAB 2: DISTRIBUTION CHART ---
        with tab_dist:
            lab = sub["Hardness_LAB"].dropna()
            line = sub["Hardness_LINE"].dropna()
            
            if len(lab) < 10 or len(line) < 10:
                st.warning("⚠️ Not enough data points (N < 10) to visualize distribution.")
            else:
                mean_lab, std_lab = lab.mean(), lab.std(ddof=1)
                mean_line, std_line = line.mean(), line.std(ddof=1)
                
                # Auto scale x-axis
                x_min = min(mean_lab - 3*std_lab, mean_line - 3*std_line, lo - 2)
                x_max = max(mean_lab + 3*std_lab, mean_line + 3*std_line, hi + 2)
                
                bins = np.linspace(x_min, x_max, 25)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Histogram
                ax.hist(lab, bins=bins, density=True, alpha=0.4, color="#1f77b4", edgecolor="black", label="LAB Hist")
                ax.hist(line, bins=bins, density=True, alpha=0.4, color="#ff7f0e", edgecolor="black", label="LINE Hist")
                
                # Normal Curves
                xs = np.linspace(x_min, x_max, 400)
                if std_lab > 0:
                    ys_lab = (1/(std_lab*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_lab)/std_lab)**2)
                    ax.plot(xs, ys_lab, linewidth=2.5, label="LAB Fit", color="#1f77b4")
                
                if std_line > 0:
                    ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
                    ax.plot(xs, ys_line, linewidth=2.5, linestyle="--", label="LINE Fit", color="#ff7f0e")
                
                # Limits & Means
                ax.axvline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
                ax.axvline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
                ax.axvline(mean_lab, linestyle=":", linewidth=1.5, color="#0b3d91")
                ax.axvline(mean_line, linestyle=":", linewidth=1.5, color="#b25e00")
                
                # SPC Stats Box (Moved to bottom to avoid covering chart)
                note = (
                    f"🔷 LAB: N={len(lab)} | Mean={mean_lab:.1f} | Std={std_lab:.2f} | Cpk={min((hi-mean_lab)/(3*std_lab),(mean_lab-lo)/(3*std_lab)):.2f}\n"
                    f"🔶 LINE: N={len(line)} | Mean={mean_line:.1f} | Std={std_line:.2f} | Cpk={min((hi-mean_line)/(3*std_line),(mean_line-lo)/(3*std_line)):.2f}"
                )
                
                ax.set_title("Hardness Distribution & SPC Capability", weight="bold")
                ax.set_xlabel("Hardness (HRB)")
                ax.set_ylabel("Probability Density")
                ax.grid(alpha=0.3)
                
                # Add text box below title
                plt.figtext(0.5, -0.05, note, ha="center", fontsize=10, 
                            bbox={"facecolor":"white", "alpha":0.5, "pad":5})
                
                ax.legend(loc="upper right", frameon=True)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download
                buf = fig_to_png(fig)
                st.download_button(
                   label="📥 Download Dist Chart",
                   data=buf,
                   file_name=f"distribution_{g['Material']}_{g['Gauge_Range']}.png",
                   mime="image/png",
                   key=f"dl_dist_{_}"
                )
# ================================
# ========================================================
    # MODE: CORRELATION (HARDNESS vs MECH PROPS)
    # ========================================================
    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":

        st.markdown("### 🔗 Correlation: Hardness vs Mechanical Properties")
        st.info("ℹ️ This chart shows how Mechanical Properties (TS/YS/EL) change as Hardness increases.")

        # ================================
        # 1️⃣ Chuẩn bị dữ liệu
        # ================================
        sub_corr = sub.dropna(subset=["Hardness_LAB","Hardness_LINE","TS","YS","EL"])
        
        # Loại bỏ hoàn toàn coil GE* <88 (Logic cũ giữ nguyên)
        if "Quality_Code" in sub_corr.columns:
            sub_corr = sub_corr[~(
                sub_corr["Quality_Code"].astype(str).str.startswith("GE") &
                ((sub_corr["Hardness_LAB"] < 88) | (sub_corr["Hardness_LINE"] < 88))
            )]
        
        # ================================
        # 2️⃣ Binning Hardness
        # ================================
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub_corr["HRB_bin"] = pd.cut(sub_corr["Hardness_LAB"], bins=bins, labels=labels, right=False)
        
        # ================================
        # 3️⃣ Aggregation (Tính toán trung bình theo nhóm)
        # ================================
        # Lấy columns spec để check
        mech_cols = ["Standard TS min","Standard TS max",
                     "Standard YS min","Standard YS max",
                     "Standard EL min","Standard EL max"]
        
        summary = (sub_corr.groupby("HRB_bin", observed=True).agg(
            N_coils=("COIL_NO","count"),
            TS_mean=("TS","mean"), TS_min=("TS","min"), TS_max=("TS","max"),
            YS_mean=("YS","mean"), YS_min=("YS","min"), YS_max=("YS","max"),
            EL_mean=("EL","mean"), EL_min=("EL","min"), EL_max=("EL","max"),
            # Lấy max spec của nhóm để so sánh (đại diện)
            Std_TS_min=("Standard TS min", "max"), Std_TS_max=("Standard TS max", "max"),
            Std_YS_min=("Standard YS min", "max"), Std_YS_max=("Standard YS max", "max"),
            Std_EL_min=("Standard EL min", "max"), Std_EL_max=("Standard EL max", "max"),
        ).reset_index())
        
        summary = summary[summary["N_coils"]>0]

        if summary.empty:
            st.warning("⚠️ No data available for correlation analysis.")
        else:
            # ================================
            # 4️⃣ Vẽ biểu đồ
            # ================================
            x = np.arange(len(summary))
            fig, ax = plt.subplots(figsize=(16,6))
            
            # Hàm vẽ helper
            def plot_prop(x_vals, y_mean, y_min, y_max, label, color, marker):
                ax.plot(x_vals, y_mean, marker=marker, linewidth=2, markersize=8, label=label, color=color)
                ax.fill_between(x_vals, y_min, y_max, alpha=0.15, color=color)

            plot_prop(x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "TS Mean", "#1f77b4", "o")
            plot_prop(x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "YS Mean", "#2ca02c", "s")
            plot_prop(x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "EL Mean (%)", "#ff7f0e", "^")

            # ================================
            # 5️⃣ Annotations (Check Spec thông minh)
            # ================================
            for idx, row in enumerate(summary.itertuples()):
                # TS Label
                ax.annotate(f"{row.TS_mean:.0f}", (x[idx], row.TS_mean), xytext=(0,10), 
                            textcoords="offset points", ha="center", fontsize=9, color="#1f77b4")
                # YS Label
                ax.annotate(f"{row.YS_mean:.0f}", (x[idx], row.YS_mean), xytext=(0,-15), 
                            textcoords="offset points", ha="center", fontsize=9, color="#2ca02c")
                
                # EL Check (Chỉ check nếu Spec > 0)
                el_spec = row.Std_EL_min
                is_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                
                label_text = f"{row.EL_mean:.1f}%" + (" ❌" if is_fail else "")
                label_color = "red" if is_fail else "#ff7f0e"
                font_weight = "bold" if is_fail else "normal"
                
                ax.annotate(label_text, (x[idx], row.EL_mean), xytext=(0,15), 
                            textcoords="offset points", ha="center", fontsize=9, 
                            color=label_color, fontweight=font_weight)

            # Style
            ax.set_xticks(x)
            ax.set_xticklabels(summary["HRB_bin"].astype(str), fontweight="bold")
            ax.set_xlabel("Hardness Range (HRB)", fontweight="bold")
            ax.set_ylabel("Mechanical Properties (MPa / %)", fontweight="bold")
            ax.set_title("Correlation: Hardness vs TS/YS/EL", fontweight="bold", fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download
            buf = fig_to_png(fig)
            st.download_button("📥 Download Chart", data=buf, file_name=f"Correlation_{g['Material']}.png", mime="image/png", key=f"dl_corr_{_}")

            # ================================
            # 6️⃣ Quick Conclusion (Đã Fix lỗi số 0)
            # ================================
            st.markdown("### 📌 Quick Conclusion per Hardness Bin")
            
            qc_list = []
            for row in summary.itertuples():
                # Helper check function
                def check_stat(val_min, val_max, spec_min, spec_max):
                    # Nếu spec = 0 hoặc NaN -> coi như Pass (True)
                    pass_min = (val_min >= spec_min) if (pd.notna(spec_min) and spec_min > 0) else True
                    pass_max = (val_max <= spec_max) if (pd.notna(spec_max) and spec_max > 0) else True
                    return "✅" if (pass_min and pass_max) else "⚠️"

                ts_flag = check_stat(row.TS_min, row.TS_max, row.Std_TS_min, row.Std_TS_max)
                ys_flag = check_stat(row.YS_min, row.YS_max, row.Std_YS_min, row.Std_YS_max)
                el_flag = check_stat(row.EL_min, row.EL_max, row.Std_EL_min, row.Std_EL_max)
                
                qc_list.append(
                    f"**{row.HRB_bin}**: "
                    f"TS={ts_flag} ({row.TS_min:.0f}-{row.TS_max:.0f}) | "
                    f"YS={ys_flag} ({row.YS_min:.0f}-{row.YS_max:.0f}) | "
                    f"EL={el_flag} ({row.EL_min:.1f}-{row.EL_max:.1f})"
                )
            
            for line in qc_list:
                st.markdown(line)

            # Table Expand
            with st.expander("🔹 View Detailed Data Table"):
                st.dataframe(summary, use_container_width=True)
# ========================================================
# ========================================================
# ========================================================
    # MODE: MECH PROPS ANALYSIS (WITH STATS TABLE)
    # ========================================================
    elif view_mode == "⚙️ Mech Props Analysis":
        import uuid # Đảm bảo có thư viện này
        
        st.markdown("### ⚙️ Mechanical Properties Analysis (TS / YS / EL)")
        
        # 1. Lọc dữ liệu
        sub_mech = sub.dropna(subset=["TS", "YS", "EL"]).sort_values("COIL_NO")
        N = len(sub_mech)

        if sub_mech.empty:
            st.warning("⚠️ No mechanical property data available.")
        else:
            # 2. Logic kiểm tra NG
            def get_ng_mask(col_val, col_min, col_max):
                lo = col_min.replace(0, np.nan)
                hi = col_max.replace(0, np.nan)
                is_fail = pd.Series(False, index=col_val.index)
                mask_lo = lo.notna(); is_fail[mask_lo] |= (col_val[mask_lo] < lo[mask_lo])
                mask_hi = hi.notna(); is_fail[mask_hi] |= (col_val[mask_hi] > hi[mask_hi])
                return is_fail

            sub_mech["NG_TS"] = get_ng_mask(sub_mech["TS"], sub_mech["Standard TS min"], sub_mech["Standard TS max"])
            sub_mech["NG_YS"] = get_ng_mask(sub_mech["YS"], sub_mech["Standard YS min"], sub_mech["Standard YS max"])
            sub_mech["NG_EL"] = get_ng_mask(sub_mech["EL"], sub_mech["Standard EL min"], sub_mech["Standard EL max"])

            # 3. Tabs
            tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & Stats"])

            # --- TAB 1: TREND CHART ---
            with tab_trend:
                st.markdown(f"**Total Coils:** {N}")
                fig, ax = plt.subplots(figsize=(12, 5))
                x = np.arange(1, N + 1)
                props = [("TS", "#1f77b4", "o"), ("YS", "#2ca02c", "s"), ("EL", "#ff7f0e", "^")]
                
                for col, color, marker in props:
                    ax.plot(x, sub_mech[col], marker=marker, color=color, label=col, alpha=0.7, linewidth=1.5)
                    ng_indices = np.where(sub_mech[f"NG_{col}"])[0]
                    if len(ng_indices) > 0:
                        ax.scatter(x[ng_indices], sub_mech[col].iloc[ng_indices], color="red", s=80, zorder=10, edgecolors="white", linewidth=1)

                ref_row = sub_mech.iloc[0]
                for col, color, _ in props:
                    lsl = ref_row.get(f"Standard {col} min", 0)
                    usl = ref_row.get(f"Standard {col} max", 0)
                    if lsl > 0: ax.axhline(lsl, color=color, linestyle="--", alpha=0.3)
                    if usl > 0: ax.axhline(usl, color=color, linestyle="--", alpha=0.3)

                ax.set_title("Mechanical Properties Trend (TS / YS / EL)", weight="bold")
                ax.set_xlabel("Coil Sequence"); ax.set_ylabel("Value (MPa / %)")
                ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
                plt.tight_layout()
                st.pyplot(fig)
                
                buf = fig_to_png(fig)
                st.download_button(label="📥 Download Mech Trend", data=buf, file_name=f"Mech_Trend_{g['Material']}.png", mime="image/png", key=f"dl_trend_{_}_{uuid.uuid4()}")

            # --- TAB 2: DISTRIBUTION & STATS (CẬP NHẬT MỚI) ---
            with tab_dist:
                # 1️⃣ TÍNH TOÁN BẢNG THỐNG KÊ
                stats_data = []
                for col in ["TS", "YS", "EL"]:
                    series = sub_mech[col].dropna()
                    if len(series) > 0:
                        stats_data.append({
                            "Property": col,
                            "Count": len(series),
                            "Mean": series.mean(),
                            "Std Dev": series.std(ddof=1),
                            "Min": series.min(),
                            "Max": series.max(),
                            "Range": series.max() - series.min()
                        })
                
                # Hiển thị bảng
                st.markdown("#### 📊 Statistical Summary Table")
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(
                    df_stats.style.format({
                        "Mean": "{:.1f}", "Std Dev": "{:.2f}", 
                        "Min": "{:.1f}", "Max": "{:.1f}", "Range": "{:.1f}"
                    }),
                    use_container_width=True
                )

                # 2️⃣ VẼ BIỂU ĐỒ (Giữ nguyên logic Normal Fit an toàn)
                st.markdown("#### 📉 Distribution Charts")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i, (col, color, _) in enumerate(props):
                    ax = axes[i]
                    data = sub_mech[col].dropna()
                    
                    # Histogram
                    ax.hist(data, bins=15, color=color, alpha=0.5, edgecolor="black", density=True)
                    
                    # Normal Curve (Numpy)
                    if len(data) > 1:
                        mean, std = data.mean(), data.std()
                        if std > 0:
                            xmin, xmax = ax.get_xlim()
                            x_plot = np.linspace(xmin, xmax, 100)
                            y_plot = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plot - mean) / std)**2)
                            ax.plot(x_plot, y_plot, color=color, linewidth=2, label="Normal Fit")
                            # Thêm đường Mean
                            ax.axvline(mean, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean:.1f}")

                    ax.set_title(f"{col} Distribution", weight="bold")
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                buf = fig_to_png(fig)
                st.download_button(label="📥 Download Mech Dist", data=buf, file_name=f"Mech_Dist_{g['Material']}.png", mime="image/png", key=f"dl_dist_{_}_{uuid.uuid4()}")
            
            # --- NG LIST (Footer) ---
            with st.expander("📋 Out-of-Spec Coils List", expanded=False):
                ng_rows = sub_mech[sub_mech["NG_TS"] | sub_mech["NG_YS"] | sub_mech["NG_EL"]]
                if not ng_rows.empty:
                    st.error(f"⚠️ Found {len(ng_rows)} Out-of-Spec Coils!")
                    st.dataframe(ng_rows[["COIL_NO", "Hardness_LINE", "TS", "YS", "EL"]].style.format("{:.1f}"), use_container_width=True)
                else:
                    st.success("✅ All coils passed Mechanical Specs.")
    # ================================
    elif view_mode == "🧮 Predict TS/YS/EL (Custom Hardness)":
            
            st.markdown("## 🧮 Predict Mechanical Properties (Auto-Update)")
            
            # ===============================
            # Prepare data
            # ===============================
            sub_fit = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"]).copy()
            N = len(sub_fit)
            
            if N < 5:
                st.warning(f"⚠️ Not enough data for prediction (N={N})")
            else:
                hrb_min_data = float(sub_fit["Hardness_LINE"].min())
                hrb_max_data = float(sub_fit["Hardness_LINE"].max())
                
                # ===============================
                # INPUT AREA (AUTO UPDATE - NO BUTTON)
                # ===============================
                # Thêm _{_} vào key để tránh lỗi Duplicate Key
                pred_type = st.radio(
                    "Select input type for prediction:",
                    ["Single Value", "Range"],
                    index=0,
                    key=f"pred_type_custom_{_}" 
                )
                
                hrb_values = []
                
                if pred_type == "Single Value":
                    val = st.number_input(
                        "Enter desired LINE Hardness (HRB):",
                        value=round((hrb_min_data + hrb_max_data) / 2, 1),
                        step=0.1,
                        key=f"hrb_single_{_}"
                    )
                    hrb_values = [val]
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        hrb_min = st.number_input(
                            "Min Hardness:",
                            value=round(hrb_min_data, 1),
                            step=0.1,
                            key=f"hrb_min_{_}"
                        )
                    with c2:
                        hrb_max = st.number_input(
                            "Max Hardness:",
                            value=round(hrb_max_data, 1),
                            step=0.1,
                            key=f"hrb_max_{_}"
                        )
                    with c3:
                        step = st.number_input(
                            "Step:",
                            value=1.0,
                            step=0.1,
                            key=f"hrb_step_{_}"
                        )
                    
                    if hrb_min > hrb_max:
                        st.error("⚠️ Min > Max! Please adjust.")
                    else:
                        hrb_values = list(np.arange(hrb_min, hrb_max + 0.001, step))
                
                # ===============================
                # CALCULATION & PLOT (ALWAYS RUN)
                # ===============================
                if len(hrb_values) > 0:
                    pred_values = {}
                    
                    # Fit & Predict Logic
                    for prop in ["TS", "YS", "EL"]:
                        try:
                            a, b = np.polyfit(
                                sub_fit["Hardness_LINE"].values,
                                sub_fit[prop].values,
                                1
                            )
                            pred_values[prop] = a * np.array(hrb_values) + b
                        except:
                            pred_values[prop] = np.zeros(len(hrb_values))
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(14, 5))
                    coils = np.arange(1, N + 1)
                    
                    for prop, color, marker, unit in [
                        ("TS", "#1f77b4", "o", "MPa"),
                        ("YS", "#2ca02c", "s", "MPa"),
                        ("EL", "#ff7f0e", "^", "%")
                    ]:
                        # Vẽ dữ liệu thực tế
                        obs = sub_fit[prop].values
                        ax.plot(coils, obs, marker=marker, color=color, alpha=0.6, label=f"{prop} Observed")
                        
                        # Vẽ dữ liệu dự báo
                        pred = pred_values[prop]
                        pred_x = coils[-1] + np.arange(1, len(pred) + 1)
                        
                        ax.scatter(
                            pred_x, pred,
                            color="red", s=100, marker="X",
                            label=f"{prop} Predicted ({unit})" if prop == "TS" else "" # Chỉ hiện label 1 lần cho gọn
                        )
                        
                        # Nối nét đứt
                        if len(pred) > 0:
                            ax.plot(
                                [coils[-1], pred_x[0]],
                                [obs[-1], pred[0]],
                                linestyle=":",
                                color="red",
                                alpha=0.5
                            )
                            if len(pred) > 1:
                                ax.plot(pred_x, pred, linestyle="--", color=color, alpha=0.8)
    
                    ax.set_xlabel("Coil Sequence")
                    ax.set_ylabel("Mechanical Properties")
                    ax.set_title("Observed vs Predicted TS / YS / EL")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
                    st.pyplot(fig)
                    
                    # Table Result
                    pred_table = pd.DataFrame({"HRB Input": hrb_values})
                    for prop in ["TS", "YS", "EL"]:
                        pred_table[prop] = pred_values[prop]
                    
                    with st.expander("📋 Predicted Values Table", expanded=True):
                        st.dataframe(
                            pred_table.style.format({
                                "HRB Input": "{:.1f}",
                                "TS": "{:.1f}",
                                "YS": "{:.1f}",
                                "EL": "{:.1f}"
                            }),
                            use_container_width=True
                        )
# ========================================================
# ========================================================
    # MODE: HARDNESS LOOKUP -> ACTUAL MECHANICAL PROPERTIES
    # ========================================================
    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        
        st.markdown("### 🔍 Actual Mechanical Properties Lookup (by Hardness Range)")
        st.info("ℹ️ This feature analyzes historical data to show how mechanical properties fluctuate within a specific hardness range.")

        # 1. Prepare Data
        # Filter out rows with missing hardness or mechanical properties
        df_lookup = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"]).copy()
        
        if df_lookup.empty:
            st.warning("⚠️ No data available for analysis.")
        else:
            # Get real min/max for input reference
            min_h_real = float(df_lookup["Hardness_LINE"].min())
            max_h_real = float(df_lookup["Hardness_LINE"].max())

            # 2. Input Area
            c1, c2 = st.columns(2)
            with c1:
                input_min = st.number_input("Min Hardness (HRB):", min_value=0.0, max_value=120.0, 
                                            value=58.0, step=0.5, key=f"lookup_min_{_}")
            with c2:
                input_max = st.number_input("Max Hardness (HRB):", min_value=0.0, max_value=120.0, 
                                            value=65.0, step=0.5, key=f"lookup_max_{_}")

            if input_min > input_max:
                st.error("⚠️ 'Min' value must be less than 'Max' value.")
            else:
                # 3. Filter Data
                mask = (df_lookup["Hardness_LINE"] >= input_min) & (df_lookup["Hardness_LINE"] <= input_max)
                df_filtered = df_lookup[mask]
                n_count = len(df_filtered)

                # 4. Display Results
                st.markdown(f"#### 📊 Results for Hardness Range: **{input_min} ~ {input_max} HRB**")
                
                if n_count == 0:
                    st.warning(f"⚠️ No coils found in hardness range {input_min}~{input_max} in history.")
                else:
                    st.success(f"✅ Found **{n_count}** historical coils.")

                    # --- SUMMARY TABLE ---
                    stats_data = []
                    for col in ["TS", "YS", "EL"]:
                        series = df_filtered[col]
                        stats_data.append({
                            "Property": col,
                            "Min": series.min(),
                            "Average": series.mean(),
                            "Max": series.max(),
                            "Std Dev": series.std(ddof=1),
                            "Range": series.max() - series.min()
                        })
                    
                    df_stats = pd.DataFrame(stats_data)
                    
                    st.markdown("##### 1. Summary Statistics")
                    st.dataframe(
                        df_stats.style.format({
                            "Min": "{:.1f}", "Average": "{:.1f}", "Max": "{:.1f}", 
                            "Std Dev": "{:.2f}", "Range": "{:.1f}"
                        }),
                        use_container_width=True
                    )

                    # --- SPEC COMPLIANCE CHECK (FIXED 0 VALUE ISSUE) ---
                    st.markdown("##### 2. Specification Compliance Rate")
                    
                    spec_res = []
                    for col, lsl_col, usl_col in [("TS", "Standard TS min", "Standard TS max"),
                                                  ("YS", "Standard YS min", "Standard YS max"),
                                                  ("EL", "Standard EL min", "Standard EL max")]:
                        
                        # FIX: Replace 0 with NaN (Treat 0 as No Limit)
                        lsl = df_filtered[lsl_col].replace(0, np.nan)
                        usl = df_filtered[usl_col].replace(0, np.nan)
                        
                        # Logic: Pass if (val >= lsl) AND (val <= usl OR usl is NaN)
                        pass_mask = pd.Series(True, index=df_filtered.index)
                        
                        # Check Min Limit (if exists)
                        if lsl.notna().any(): # If at least one row has a limit
                            # Only enforce limit where it is not NaN
                            has_limit = lsl.notna()
                            pass_mask[has_limit] &= (df_filtered.loc[has_limit, col] >= lsl[has_limit])
                            
                        # Check Max Limit (if exists)
                        if usl.notna().any():
                            has_limit = usl.notna()
                            pass_mask[has_limit] &= (df_filtered.loc[has_limit, col] <= usl[has_limit])
                        
                        pass_count = pass_mask.sum()
                        pass_rate = (pass_count / n_count) * 100
                        
                        # Add icon based on rate
                        icon = "🟢" if pass_rate >= 95 else ("🟡" if pass_rate >= 80 else "🔴")
                        
                        # Show limit used for info (optional, taking mode)
                        limit_info = ""
                        if lsl.max() > 0 or usl.max() > 0:
                           # Try to grab a representative limit to show user
                           l_val = lsl.max() if lsl.max() > 0 else "None"
                           u_val = usl.max() if usl.max() > 0 else "None"
                           limit_info = f"(Limit: {l_val:.0f}~{u_val:.0f})"
                        else:
                           limit_info = "(No Spec Limits Found in Data)"

                        spec_res.append(f"- {icon} **{col}**: {pass_rate:.1f}% coils passed ({pass_count}/{n_count}) {limit_info}")
                    
                    st.markdown("\n".join(spec_res))

                    # --- BOXPLOT DISTRIBUTION ---
                    st.markdown("##### 3. Actual Distribution Charts (Boxplot)")
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    colors = {"TS": "#1f77b4", "YS": "#2ca02c", "EL": "#ff7f0e"}
                    for i, col in enumerate(["TS", "YS", "EL"]):
                        ax = axes[i]
                        data = df_filtered[col].dropna()
                        
                        # Boxplot
                        ax.boxplot(data, patch_artist=True, 
                                   boxprops=dict(facecolor=colors[col], alpha=0.5),
                                   medianprops=dict(color="black", linewidth=1.5))
                        
                        # Jitter points
                        y = data
                        x = np.random.normal(1, 0.04, size=len(y))
                        ax.scatter(x, y, alpha=0.6, color=colors[col], s=20)
                        
                        # Mean Line
                        ax.axhline(data.mean(), color='red', linestyle='--', alpha=0.7, label=f"Mean: {data.mean():.1f}")
                        
                        ax.set_title(f"{col} Distribution", fontweight="bold")
                        ax.set_xticks([])
                        ax.set_ylabel("Value (MPa / %)")
                        ax.legend()
                        ax.grid(axis='y', linestyle='--', alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download Button
                    buf = fig_to_png(fig)
                    st.download_button("📥 Download Distribution Chart", data=buf, 
                                       file_name=f"Lookup_{input_min}_{input_max}_{g['Material']}.png",
                                       mime="image/png", key=f"dl_lookup_{_}")
