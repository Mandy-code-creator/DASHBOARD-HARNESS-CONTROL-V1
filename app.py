# ================================
# FULL STREAMLIT APP – FINAL COMPLETE VERSION
# INTEGRATED GLOBAL DASHBOARD + SMART LOGIC + BUG FIXES
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import uuid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# PRE-PROCESSING
# ================================
# 1. Metallic Type
metal_col = next(c for c in raw.columns if "METALLIC" in c.upper())
raw["Metallic_Type"] = raw[metal_col]

# 2. Rename Columns
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
    "Standard TS min": "Standard TS min",
    "Standard TS max": "Standard TS max",
    "Standard YS min": "Standard YS min",
    "Standard YS max": "Standard YS max",
    "Standard EL min": "Standard EL min",
    "Standard EL max": "Standard EL max"
})

# 3. Standard Hardness Split
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))

# 4. Force Numeric
for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 5. Quality Group Merge
df["Quality_Group"] = df["Quality_Code"].replace({
    "CQ00": "CQ00 / CQ06",
    "CQ06": "CQ00 / CQ06"
})

# 6. Filter GE* < 88
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

all_rolling = sorted(df["Rolling_Type"].unique())
all_metal = sorted(df["Metallic_Type"].unique())
all_qgroup = sorted(df["Quality_Group"].unique())

rolling = st.sidebar.radio("Rolling Type", all_rolling)
metal   = st.sidebar.radio("Metallic Type", all_metal)
qgroup  = st.sidebar.radio("Quality Group", all_qgroup)

df = df[
    (df["Rolling_Type"] == rolling) &
    (df["Metallic_Type"] == metal) &
    (df["Quality_Group"] == qgroup)
]

view_mode = st.sidebar.radio(
    "📊 View Mode",
    [
        "📋 Data Inspection",
        "🚀 Global Summary Dashboard",
        "📉 Hardness Analysis (Trend & Dist)",
        "🔗 Correlation: Hardness vs Mech Props",
        "⚙️ Mech Props Analysis",
        "🔍 Lookup: Hardness Range → Actual Mech Props",
        "🎯 Find Target Hardness (Reverse Lookup)",
        "🧮 Predict TS/YS/EL from Std Hardness",
        "🎛️ Control Limit Calculator (Compare 3 Methods)",
    ]
)

# ================================
# GROUP CONDITION
# ================================
GROUP_COLS = ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"]
cnt = df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 30]

if valid.empty:
    st.warning("⚠️ No group with ≥30 coils found.")
    st.stop()

# ==============================================================================
#  🚀 GLOBAL SUMMARY DASHBOARD (FINAL: STATS + LIMITS + SIMULATION)
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Dashboard")
    
    # Create Tabs
    tab1, tab2 = st.tabs(["📊 1. Statistical Overview (With Limits)", "🎯 2. Prediction Simulator"])

    # --- TAB 1: STATS TABLE WITH LIMITS ---
    with tab1:
        st.info("ℹ️ This table compares ACTUAL statistics (Min/Max/Avg) against STANDARD LIMITS.")
        
        stats_rows = []
        
        for _, g in valid.iterrows():
            sub_grp = df[
                (df["Rolling_Type"] == g["Rolling_Type"]) &
                (df["Metallic_Type"] == g["Metallic_Type"]) &
                (df["Quality_Group"] == g["Quality_Group"]) &
                (df["Gauge_Range"] == g["Gauge_Range"]) &
                (df["Material"] == g["Material"])
            ].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 5: continue

            # Specs List
            specs_str = ", ".join(sorted(sub_grp["Product_Spec"].astype(str).unique()))

            # --- HELPER: GET LIMIT STRING ---
            def get_limit_str(s_min_col, s_max_col):
                v_min = sub_grp[s_min_col].max() if s_min_col in sub_grp else 0 
                v_max = sub_grp[s_max_col].min() if s_max_col in sub_grp else 0 
                
                if pd.isna(v_min): v_min = 0
                if pd.isna(v_max): v_max = 0

                if v_min > 0 and v_max > 0 and v_max < 9000:
                    return f"{v_min:.0f}~{v_max:.0f}"
                elif v_min > 0:
                    return f"≥ {v_min:.0f}"
                elif v_max > 0 and v_max < 9000:
                    return f"≤ {v_max:.0f}"
                else:
                    return "-"

            # Get Limits Text
            lim_hrb = f"{sub_grp['Std_Min'].min():.0f}~{sub_grp['Std_Max'].max():.0f}"
            lim_ts = get_limit_str("Standard TS min", "Standard TS max")
            lim_ys = get_limit_str("Standard YS min", "Standard YS max")
            lim_el = get_limit_str("Standard EL min", "Standard EL max")

            stats_rows.append({
                "Quality": g["Quality_Group"],
                "Material": g["Material"],
                "Gauge": g["Gauge_Range"],
                "Specs": specs_str,
                "N": len(sub_grp),
                
                # Hardness Stats
                "HRB Limit": lim_hrb,          
                "HRB (Avg)": sub_grp["Hardness_LINE"].mean(),
                "HRB (Min)": sub_grp["Hardness_LINE"].min(),
                "HRB (Max)": sub_grp["Hardness_LINE"].max(),
                
                # TS Stats
                "TS Limit": lim_ts,            
                "TS (Avg)": sub_grp["TS"].mean(),
                "TS (Min)": sub_grp["TS"].min(),
                "TS (Max)": sub_grp["TS"].max(),

                # YS Stats
                "YS Limit": lim_ys,            
                "YS (Avg)": sub_grp["YS"].mean(),
                "YS (Min)": sub_grp["YS"].min(),
                "YS (Max)": sub_grp["YS"].max(),
                
                # EL Stats
                "EL Limit": lim_el,            
                "EL (Avg)": sub_grp["EL"].mean(),
                "EL (Min)": sub_grp["EL"].min(),
                "EL (Max)": sub_grp["EL"].max(),
            })

        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            
            # Reorder columns
            cols = [
                "Quality", "Material", "Gauge", "Specs", "N",
                "HRB Limit", "HRB (Avg)", "HRB (Min)", "HRB (Max)",
                "TS Limit", "TS (Avg)", "TS (Min)", "TS (Max)",
                "YS Limit", "YS (Avg)", "YS (Min)", "YS (Max)",
                "EL Limit", "EL (Avg)", "EL (Min)", "EL (Max)"
            ]
            cols = [c for c in cols if c in df_stats.columns]
            df_stats = df_stats[cols]

            # Format & Style
            st.dataframe(
                df_stats.style.format("{:.1f}", subset=[c for c in df_stats.columns if "(Avg)" in c or "(Min)" in c or "(Max)" in c])
                              .background_gradient(subset=["HRB (Avg)"], cmap="Blues"),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("Insufficient data for statistics.")

    # --- TAB 2: PREDICTION SIMULATOR ---
    with tab2:
        st.info("🎯 Enter your Target Hardness. The system uses AI models per group to forecast Mechanical Properties.")
        
        col_in, _ = st.columns([1, 3])
        with col_in:
            user_hrb = st.number_input("📥 Input Target Hardness (HRB):", value=60.0, step=0.5, format="%.1f")

        pred_rows = []

        for _, g in valid.iterrows():
            sub_grp = df[
                (df["Rolling_Type"] == g["Rolling_Type"]) &
                (df["Metallic_Type"] == g["Metallic_Type"]) &
                (df["Quality_Group"] == g["Quality_Group"]) &
                (df["Gauge_Range"] == g["Gauge_Range"]) &
                (df["Material"] == g["Material"])
            ].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 10: continue 

            specs_str = ", ".join(sorted(sub_grp["Product_Spec"].astype(str).unique()))
            
            # 1. Get Historical Range
            h_min, h_max = sub_grp["Hardness_LINE"].min(), sub_grp["Hardness_LINE"].max()
            
            # 2. Get Standard Control Limits (Hardness)
            std_lo = sub_grp["Std_Min"].min()
            std_hi = sub_grp["Std_Max"].max()
            if pd.isna(std_lo): std_lo = 0
            if pd.isna(std_hi): std_hi = 0
            std_txt = f"{std_lo:.1f} ~ {std_hi:.1f}"
            if std_lo == 0 and std_hi == 0: std_txt = "No Spec"

            # 3. Check Status
            status_msgs = []
            if user_hrb < h_min or user_hrb > h_max:
                status_msgs.append("⚠️ Extrapolated")
            if (std_lo > 0 and user_hrb < std_lo) or (std_hi > 0 and user_hrb > std_hi):
                 status_msgs.append("⛔ Out of Spec")
            if not status_msgs:
                status_msgs.append("✅ Safe Zone")
            status_final = " | ".join(status_msgs)

            # 4. AI Prediction
            X = sub_grp[["Hardness_LINE"]].values
            
            m_ts = LinearRegression().fit(X, sub_grp["TS"].values)
            pred_ts = m_ts.predict([[user_hrb]])[0]
            r2_ts = r2_score(sub_grp["TS"], m_ts.predict(X))

            m_ys = LinearRegression().fit(X, sub_grp["YS"].values)
            pred_ys = m_ys.predict([[user_hrb]])[0]
            
            m_el = LinearRegression().fit(X, sub_grp["EL"].values)
            pred_el = m_el.predict([[user_hrb]])[0]

            pred_rows.append({
                "Quality": g["Quality_Group"],
                "Material": g["Material"],
                "Gauge": g["Gauge_Range"],
                "Std Limit (HRB)": std_txt,
                "Hist Range (HRB)": f"{h_min:.1f}~{h_max:.1f}",
                "Status": status_final,
                "Model Trust (R2)": r2_ts,
                "Target HRB": user_hrb,
                "Pred TS": pred_ts,
                "Pred YS": pred_ys,
                "Pred EL": pred_el
            })

        if pred_rows:
            df_pred = pd.DataFrame(pred_rows)
            
            def highlight_r2(val):
                color = '#ffcccc' if val < 0.3 else ('#ccffcc' if val > 0.7 else '')
                return f'background-color: {color}'
            
            def highlight_status(val):
                if "⛔" in val: return 'color: red; font-weight: bold'
                if "⚠️" in val: return 'color: orange'
                return 'color: green'

            st.dataframe(
                df_pred.style.format({
                    "Pred TS": "{:.0f}", "Pred YS": "{:.0f}", "Pred EL": "{:.1f}",
                    "Model Trust (R2)": "{:.2f}", "Target HRB": "{:.1f}"
                })
                .applymap(highlight_r2, subset=["Model Trust (R2)"])
                .applymap(highlight_status, subset=["Status"]),
                use_container_width=True, height=600
            )
            st.caption("* Model Trust (R2): Closer to 1.0 is better. \n* Status: Checks if Target is within History and Standard Limits.")
        else:
            st.warning("Insufficient data for prediction.")

    st.stop()

# ==============================================================================
# MAIN LOOP (DETAILS)
# ==============================================================================
# Sử dụng enumerate để có thể sử dụng biến i cho key duy nhất nếu cần
for i, (_, g) in enumerate(valid.iterrows()):
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

    # Chỉ hiển thị Header nếu không phải Global view (đã có ở trên)
    if view_mode != "🚀 Global Summary Dashboard":
        st.markdown(f"### 🧱 {g['Quality_Group']} | {g['Material']} | {g['Gauge_Range']}")
        st.markdown(f"**Specs:** {specs} | **Coils:** {sub['COIL_NO'].nunique()} | **Limit:** {lo:.1f}~{hi:.1f}")

    # ================================
    # 1. DATA INSPECTION
    # ================================
    if view_mode == "📋 Data Inspection":
        st.dataframe(sub, use_container_width=True)

    # ================================
    # 2. HARDNESS ANALYSIS (FULL FINAL VERSION)
    # ================================
    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        
        st.markdown("### 📉 Hardness Analysis: Process Stability & Capability")
        
        tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & SPC"])

        # --- TAB 1: TREND CHART ---
        with tab_trend:
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            
            # Vẽ cả 2 để đối chiếu xu hướng
            ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB", alpha=0.5)
            ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE", alpha=0.9) 
            
            # Vẽ giới hạn
            ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
            ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
            
            ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
            ax.set_xlabel("Coil Sequence"); ax.set_ylabel("Hardness (HRB)")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Nút download
            buf = fig_to_png(fig)
            st.download_button("📥 Download Trend Chart", data=buf, file_name=f"trend_{g['Material']}.png", mime="image/png", key=f"dl_tr_{uuid.uuid4()}")

        # --- TAB 2: DISTRIBUTION & SPC (LINE FOCUS + LAB REF) ---
        with tab_dist:
            line = sub["Hardness_LINE"].dropna()
            lab = sub["Hardness_LAB"].dropna()
            
            if len(line) < 5:
                st.warning("⚠️ Not enough LINE data (N < 5) to calculate SPC.")
            else:
                # 1. Hàm tính toán SPC Helper
                def calc_spc_metrics(data, lsl, usl):
                    if len(data) < 2: return None
                    mean = data.mean()
                    std = data.std(ddof=1)
                    if std == 0: return None 
                    
                    # Cp: Process Potential
                    cp = (usl - lsl) / (6 * std)
                    
                    # Ca: Accuracy (%)
                    mid = (usl + lsl) / 2
                    tol = (usl - lsl)
                    ca = ((mean - mid) / (tol / 2)) * 100
                    
                    # Cpk: Process Capability
                    cpu = (usl - mean) / (3 * std)
                    cpl = (mean - lsl) / (3 * std)
                    cpk = min(cpu, cpl)
                    
                    return mean, std, cp, ca, cpk

                # CHỈ TÍNH TOÁN SPC CHO LINE (Để hiển thị bảng)
                spc_line = calc_spc_metrics(line, lo, hi)

                # 2. Chuẩn bị vẽ biểu đồ
                mean_line, std_line = line.mean(), line.std(ddof=1)
                
                # Tính toán cho LAB (chỉ để vẽ đường curve tham khảo)
                if not lab.empty:
                    mean_lab, std_lab = lab.mean(), lab.std(ddof=1)
                else:
                    mean_lab, std_lab = 0, 0
                
                # Auto scale trục X
                data_min = min(line.min(), lab.min()) if not lab.empty else line.min()
                data_max = max(line.max(), lab.max()) if not lab.empty else line.max()
                x_min = min(data_min, lo) - 2
                x_max = max(data_max, hi) + 2
                
                bins = np.linspace(x_min, x_max, 30)
                xs = np.linspace(x_min, x_max, 400) # Trục X cho đường cong chuẩn
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Histogram
                ax.hist(line, bins=bins, density=True, alpha=0.6, color="#ff7f0e", edgecolor="white", label="LINE Hist")
                if not lab.empty:
                    ax.hist(lab, bins=bins, density=True, alpha=0.3, color="#1f77b4", edgecolor="None", label="LAB Hist")
                
                # --- VẼ NORMAL CURVE (LINE - Nét liền đậm) ---
                if std_line > 0:
                    ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
                    ax.plot(xs, ys_line, linewidth=2.5, color="#b25e00", label="LINE Fit")

                # --- VẼ NORMAL CURVE (LAB - Nét đứt màu xanh) ---
                if not lab.empty and std_lab > 0:
                    ys_lab = (1/(std_lab*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_lab)/std_lab)**2)
                    ax.plot(xs, ys_lab, linewidth=2, linestyle="--", color="#1f77b4", label="LAB Fit")
                
                # Limits
                ax.axvline(lo, linestyle="--", linewidth=2, color="red", label="LSL")
                ax.axvline(hi, linestyle="--", linewidth=2, color="red", label="USL")
                
                ax.set_title(f"Hardness Distribution (LINE vs LAB)", weight="bold")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

                # 3. Hiển thị bảng chỉ số SPC (CHỈ LINE)
                st.markdown("#### 📐 SPC Capability Indices (LINE ONLY)")
                
                if spc_line:
                    mean_val, std_val, cp_val, ca_val, cpk_val = spc_line
                    
                    eval_msg = "Excellent" if cpk_val >= 1.33 else ("Good" if cpk_val >= 1.0 else "Poor")
                    color_code = "green" if cpk_val >= 1.33 else ("orange" if cpk_val >= 1.0 else "red")

                    df_spc = pd.DataFrame([{
                        "N (Coils)": len(line),
                        "Mean": mean_val, "Std Dev": std_val,
                        "Cp": cp_val, "Ca (%)": ca_val, "Cpk": cpk_val,
                        "Rating": eval_msg
                    }])

                    # Format bảng: TẤT CẢ LÀ 2 SỐ THẬP PHÂN
                    st.dataframe(
                        df_spc.style.format({
                            "Mean": "{:.2f}", 
                            "Std Dev": "{:.2f}",          
                            "Cp": "{:.2f}", 
                            "Ca (%)": "{:.2f}%", 
                            "Cpk": "{:.2f}"
                        }).applymap(lambda v: f'color: {color_code}; font-weight: bold', subset=['Rating']),
                        use_container_width=True, 
                        hide_index=True
                    )

    # ================================
    # 3. CORRELATION (FULL CHART + TABLE)
    # ================================
    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        st.markdown("### 🔗 Correlation: Hardness vs Mechanical Properties")
        
        # 1. Prepare Data
        sub_corr = sub.dropna(subset=["Hardness_LAB","TS","YS","EL"])
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub_corr["HRB_bin"] = pd.cut(sub_corr["Hardness_LAB"], bins=bins, labels=labels, right=False)
        
        summary = (sub_corr.groupby("HRB_bin", observed=True).agg(
            N_coils=("COIL_NO","count"),
            TS_mean=("TS","mean"), TS_min=("TS","min"), TS_max=("TS","max"),
            YS_mean=("YS","mean"), YS_min=("YS","min"), YS_max=("YS","max"),
            EL_mean=("EL","mean"), EL_min=("EL","min"), EL_max=("EL","max"),
            Std_TS_min=("Standard TS min", "max"), Std_TS_max=("Standard TS max", "max"),
            Std_YS_min=("Standard YS min", "max"), Std_YS_max=("Standard YS max", "max"),
            Std_EL_min=("Standard EL min", "max"), Std_EL_max=("Standard EL max", "max"),
        ).reset_index())
        summary = summary[summary["N_coils"]>0]

        if not summary.empty:
            # 2. Setup Plot
            x = np.arange(len(summary))
            fig, ax = plt.subplots(figsize=(15,6))
            
            # Helper Function to Draw Lines
            def plot_prop(x, y, ymin, ymax, c, lbl, m):
                ax.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax.fill_between(x, ymin, ymax, color=c, alpha=0.1)

            # Draw 3 Lines
            plot_prop(x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS", "o")
            plot_prop(x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS", "s")
            plot_prop(x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL", "^")

            # 3. Annotations (Gắn nhãn số lên biểu đồ)
            for j, row in enumerate(summary.itertuples()):
                # TS Label (Blue)
                ax.annotate(f"{row.TS_mean:.0f}", (x[j], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="#1f77b4")
                
                # YS Label (Green)
                ax.annotate(f"{row.YS_mean:.0f}", (x[j], row.YS_mean), xytext=(0,-15), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="#2ca02c")
                
                # EL Label (Orange/Red)
                el_spec = row.Std_EL_min
                is_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                lbl = f"{row.EL_mean:.1f}%" + ("❌" if is_fail else "")
                clr = "red" if is_fail else "#ff7f0e"
                ax.annotate(lbl, (x[j], row.EL_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, color=clr, fontweight=("bold" if is_fail else "normal"))

            # Settings
            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_title("Hardness vs Mechanical Properties (Mean & Range)"); ax.grid(True, ls="--", alpha=0.5); ax.legend()
            
            # 4. RENDER CHART
            st.pyplot(fig)
            
            # 5. Quick Conclusion Table
            st.markdown("#### 📌 Quick Conclusion per Hardness Bin (Table View)")
            conclusion_data = []

            for row in summary.itertuples():
                def get_status(val_min, val_max, spec_min, spec_max):
                    pass_min = (val_min >= spec_min) if (pd.notna(spec_min) and spec_min > 0) else True
                    pass_max = (val_max <= spec_max) if (pd.notna(spec_max) and spec_max > 0) else True
                    return "✅" if (pass_min and pass_max) else "⚠️"

                ts_stat = get_status(row.TS_min, row.TS_max, row.Std_TS_min, row.Std_TS_max)
                ys_stat = get_status(row.YS_min, row.YS_max, row.Std_YS_min, row.Std_YS_max)
                el_stat = get_status(row.EL_min, row.EL_max, row.Std_EL_min, row.Std_EL_max)

                conclusion_data.append({
                    "Hardness Range": row.HRB_bin,
                    "TS Check (Min~Max)": f"{ts_stat} ({row.TS_min:.0f}~{row.TS_max:.0f})",
                    "YS Check (Min~Max)": f"{ys_stat} ({row.YS_min:.0f}~{row.YS_max:.0f})",
                    "EL Check (Min~Max)": f"{el_stat} ({row.EL_min:.1f}~{row.EL_max:.1f})"
                })

            if conclusion_data:
                st.dataframe(pd.DataFrame(conclusion_data), use_container_width=True, hide_index=True)

    # ================================
    # 4. MECH PROPS ANALYSIS
    # ================================
    elif view_mode == "⚙️ Mech Props Analysis":
        sub_mech = sub.dropna(subset=["TS","YS","EL"])
        if not sub_mech.empty:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for j, (col, c) in enumerate([("TS","#1f77b4"),("YS","#2ca02c"),("EL","#ff7f0e")]):
                data = sub_mech[col]
                mean, std = data.mean(), data.std()
                axes[j].hist(data, bins=15, color=c, alpha=0.5, density=True)
                if std > 0:
                    x_p = np.linspace(mean-4*std, mean+4*std, 100)
                    y_p = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2)
                    axes[j].plot(x_p, y_p, color=c, lw=2)
                axes[j].set_title(f"{col} Distribution")
            st.pyplot(fig)

    # ================================
    # 5. LOOKUP
    # ================================
    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        c1, c2 = st.columns(2)
        mn = st.number_input("Min HRB", 58.0, step=0.5, key=f"lk1_{uuid.uuid4()}")
        mx = st.number_input("Max HRB", 65.0, step=0.5, key=f"lk2_{uuid.uuid4()}")
        
        filt = sub[(sub["Hardness_LINE"]>=mn) & (sub["Hardness_LINE"]<=mx)].dropna(subset=["TS","YS","EL"])
        st.success(f"Found {len(filt)} coils.")
        if not filt.empty:
            st.dataframe(filt[["TS","YS","EL"]].describe().T)

    # ================================
    # 6. REVERSE LOOKUP (SMART LIMITS RESTORED)
    # ================================
    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        st.subheader("🎯 Target Hardness Calculator (Smart Limits)")
        
        # --- SMART LIMIT FUNCTION (RESTORED) ---
        def calculate_smart_limits(name, col_val, col_spec_min, col_spec_max, step=5.0):
            try:
                series_val = pd.to_numeric(sub[col_val], errors='coerce')
                valid_data = series_val[series_val > 0.1].dropna()
                if valid_data.empty: return 0.0, 0.0
                
                mean = float(valid_data.mean())
                std = float(valid_data.std()) if len(valid_data) > 1 else 0.0
                stat_min = mean - (3 * std); stat_max = mean + (3 * std)

                spec_min = 0.0
                if col_spec_min in sub.columns:
                    s_min = pd.to_numeric(sub[col_spec_min], errors='coerce').max()
                    if not pd.isna(s_min): spec_min = float(s_min)
                
                spec_max = 9999.0
                if col_spec_max in sub.columns:
                    s_max_series = pd.to_numeric(sub[col_spec_max], errors='coerce')
                    s_max_valid = s_max_series[s_max_series > 0]
                    if not s_max_valid.empty: spec_max = float(s_max_valid.min())

                is_no_spec = (spec_min < 1.0) and (spec_max > 9000.0)

                final_min = max(stat_min, spec_min)
                if spec_max < 9000:
                    final_max = min(stat_max, spec_max)
                else:
                    final_max = stat_max + (1 * std) if is_no_spec else stat_max

                if final_min >= final_max: final_min, final_max = stat_min, stat_max + std

                rec_min = float(round(max(0.0, final_min) / step) * step)
                rec_max = float(round(final_max / step) * step)
                return rec_min, rec_max
            except:
                return 0.0, 0.0

        # Calc Limits
        d_ys_min, d_ys_max = calculate_smart_limits('YS', 'YS', 'Standard YS min', 'Standard YS max', 5.0)
        d_ts_min, d_ts_max = calculate_smart_limits('TS', 'TS', 'Standard TS min', 'Standard TS max', 5.0)
        d_el_min, d_el_max = calculate_smart_limits('EL', 'EL', 'Standard EL min', 'Standard EL max', 1.0)

        # Input UI
        c1, c2, c3 = st.columns(3)
        with c1: 
            r_ys_min = st.number_input("Min YS", value=d_ys_min, step=5.0, key=f"rys1_{uuid.uuid4()}")
            r_ys_max = st.number_input("Max YS", value=d_ys_max, step=5.0, key=f"rys2_{uuid.uuid4()}")
        with c2:
            r_ts_min = st.number_input("Min TS", value=d_ts_min, step=5.0, key=f"rts1_{uuid.uuid4()}")
            r_ts_max = st.number_input("Max TS", value=d_ts_max, step=5.0, key=f"rts2_{uuid.uuid4()}")
        with c3:
            r_el_min = st.number_input("Min EL", value=d_el_min, step=1.0, key=f"rel1_{uuid.uuid4()}")
            r_el_max = st.number_input("Max EL", value=d_el_max, step=1.0, key=f"rel2_{uuid.uuid4()}")

        filtered = sub[
            (sub['YS'] >= r_ys_min) & (sub['YS'] <= r_ys_max) &
            (sub['TS'] >= r_ts_min) & (sub['TS'] <= r_ts_max) &
            ((sub['EL'] >= r_el_min) | (r_el_min==0)) & (sub['EL'] <= r_el_max)
        ]
        
        if not filtered.empty:
            rec_min_hrb = filtered['Hardness_LINE'].min()
            rec_max_hrb = filtered['Hardness_LINE'].max()
            st.success(f"✅ Target Hardness: **{rec_min_hrb:.1f} ~ {rec_max_hrb:.1f} HRB** (N={len(filtered)})")
            st.dataframe(filtered[['COIL_NO','Hardness_LINE','YS','TS','EL']], height=300)
        else:
            st.error("❌ No coils found matching these specs.")

    # ================================
    # 7. AI PREDICTION
    # ================================
    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown("### 🚀 AI Forecast (Linear Regression)")
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        
        if len(train_df) < 5:
            st.warning("⚠️ Need at least 5 coils.")
        else:
            mean_h = train_df["Hardness_LINE"].mean()
            target_h = st.number_input("Target Hardness", value=round(mean_h, 1), step=0.1, key=f"ai_{uuid.uuid4()}")
            
            X_train = train_df[["Hardness_LINE"]].values
            preds = {}
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                preds[col] = model.predict([[target_h]])[0]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            colors = {"TS": "#004BA0", "YS": "#1B5E20", "EL": "#B71C1C"}
            idx = list(range(len(train_df))); nxt = len(train_df)

            for col in ["TS","YS","EL"]:
                sec = (col=="EL")
                fig.add_trace(go.Scatter(x=idx, y=train_df[col], mode='lines', line=dict(color=colors[col], width=1, dash='dot'), opacity=0.3, name=col), secondary_y=sec)
                fig.add_trace(go.Scatter(x=[nxt], y=[preds[col]], mode='markers+text', text=[f"{preds[col]:.0f}"], marker=dict(color=colors[col], size=15, symbol='diamond'), name=f"Pred {col}"), secondary_y=sec)
            
            fig.update_layout(height=500, title="Prediction Visualization")
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Pred TS", f"{preds['TS']:.0f}")
            c2.metric("Pred YS", f"{preds['YS']:.0f}")
            c3.metric("Pred EL", f"{preds['EL']:.1f}")

    # ================================
# ================================
    # 8. CONTROL LIMIT CALCULATOR (FIXED: REMOVE NESTED LOOP)
    # ================================
    elif view_mode == "🎛️ Control Limit Calculator (Compare 3 Methods)":
        
        # Tiêu đề (Sử dụng biến 'g' từ vòng lặp chính bên ngoài)
        st.markdown(f"### 🎛️ Limits Analysis: {g['Material']} | {g['Gauge_Range']}")

        # Lấy dữ liệu (Sử dụng biến 'sub' đã được lọc ở vòng lặp chính)
        data = sub["Hardness_LINE"].dropna()
        
        if len(data) < 10:
            st.warning(f"⚠️ {g['Material']}: 數據不足 (N={len(data)})")
        else:
            # --- 1. CẤU HÌNH THAM SỐ (Settings) ---
            # Tạo Key duy nhất dựa trên 'i' và tên vật liệu từ vòng lặp chính
            # Đảm bảo không bao giờ trùng lặp
            key_sigma = f"sigma_{i}_{g['Material']}_{g['Gauge_Range']}"
            key_iqr = f"iqr_{i}_{g['Material']}_{g['Gauge_Range']}"

            with st.expander("⚙️ 設定參數 (Settings)", expanded=False):
                col_par1, col_par2 = st.columns(2)
                with col_par1:
                    sigma_n = st.number_input(
                        "1. Sigma 倍數", 
                        min_value=1.0, max_value=6.0, value=3.0, step=0.5,
                        key=key_sigma 
                    )
                with col_par2:
                    iqr_k = st.number_input(
                        "2. IQR 靈敏度", 
                        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                        key=key_iqr
                    )

            # --- 2. TÍNH TOÁN ---
            spec_min = sub["Std_Min"].max() if "Std_Min" in sub else 0
            spec_max = sub["Std_Max"].min() if "Std_Max" in sub else 0
            if pd.isna(spec_min): spec_min = 0
            if pd.isna(spec_max): spec_max = 0
            display_max = spec_max if (spec_max > 0 and spec_max < 9000) else 0

            # Method 1: N-Sigma
            mu, sigma = data.mean(), data.std()
            m1_min, m1_max = mu - sigma_n*sigma, mu + sigma_n*sigma
            
            # Method 2: IQR Robust
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            clean_data = data[~((data < (Q1 - iqr_k * IQR)) | (data > (Q3 + iqr_k * IQR)))]
            if clean_data.empty: clean_data = data
            mu_clean, sigma_clean = clean_data.mean(), clean_data.std()
            m2_min, m2_max = mu_clean - sigma_n*sigma_clean, mu_clean + sigma_n*sigma_clean

            # Method 3: Smart Hybrid
            m3_min = max(m2_min, spec_min)
            m3_max = min(m2_max, spec_max) if (spec_max > 0 and spec_max < 9000) else m2_max
            if m3_min >= m3_max: m3_min, m3_max = m2_min, m2_max

            # --- 3. HIỂN THỊ BẢNG & BIỂU ĐỒ ---
            col_chart, col_table = st.columns([2, 1])
            
            with col_chart:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(data, bins=30, density=True, alpha=0.3, color="gray", label="Raw Data")
                
                # Limit lines
                ax.axvline(m1_min, color="red", ls=":", alpha=0.5); ax.axvline(m1_max, color="red", ls=":", alpha=0.5)
                ax.axvline(m2_min, color="blue", ls="--", alpha=0.8); ax.axvline(m2_max, color="blue", ls="--", alpha=0.8)
                ax.axvspan(m3_min, m3_max, color="green", alpha=0.2, label="Smart Hybrid")
                
                if spec_min > 0: ax.axvline(spec_min, color="black", lw=2)
                if display_max > 0: ax.axvline(display_max, color="black", lw=2)

                ax.set_title(f"Limits: {g['Material']} (σ={sigma_n}, K={iqr_k})", fontsize=10)
                st.pyplot(fig)

            with col_table:
                comp_data = [
                    {"Method": "0. Spec", "Min": spec_min, "Max": display_max, "Range": (display_max-spec_min) if display_max>0 else 0, "Note": "Ref"},
                    {"Method": f"1. {sigma_n}σ", "Min": m1_min, "Max": m1_max, "Range": m1_max-m1_min, "Note": "Loose"},
                    {"Method": f"2. IQR", "Min": m2_min, "Max": m2_max, "Range": m2_max-m2_min, "Note": "Robust"},
                    {"Method": "3. Hybrid", "Min": m3_min, "Max": m3_max, "Range": m3_max-m3_min, "Note": "✅ Best"}
                ]
                st.dataframe(pd.DataFrame(comp_data).style.format("{:.1f}", subset=["Min", "Max", "Range"]), use_container_width=True, hide_index=True)
