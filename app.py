# ================================
# FULL STREAMLIT APP – FINAL STABLE VERSION
# FIXED: NameError (fig_to_png), High Contrast Charts, Real Date Range
# ================================



import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")
st.title("📊 Hardness – Visual Analytics Dashboard")
# ================================
def add_custom_css():
    st.markdown("""
        <style>
        /* 1. Nền tổng thể: Xám nhạt hiện đại */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* 2. Sidebar: Trắng tinh + Đổ bóng nhẹ tách biệt */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
            border-right: none;
        }

        /* 3. Tiêu đề: Màu xanh đen doanh nghiệp (Corporate Blue) */
        h1, h2, h3 {
            color: #2c3e50 !important;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 600;
        }

        /* 4. Các khối dữ liệu (Metric Cards): Trắng + Bo góc + Đổ bóng */
        [data-testid="stMetricValue"] {
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: #007bff; /* Số màu xanh dương */
        }

        /* 5. Bảng dữ liệu: Header màu xám đậm */
        thead tr th:first-child {display:none}
        tbody th {display:none}
        .stDataFrame {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
# ================================
# UTILS (QUAN TRỌNG: KHÔNG ĐƯỢC XÓA)
# ================================
def fig_to_png(fig):
    """Chuyển đổi biểu đồ Matplotlib thành ảnh PNG để download"""
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
# PRE-PROCESSING & DATE HANDLING
# ================================
# 1. Calculate Data Period
data_period_str = "N/A"
if "PRODUCTION DATE" in raw.columns:
    raw["PRODUCTION DATE"] = pd.to_datetime(raw["PRODUCTION DATE"], errors='coerce')
    min_date = raw["PRODUCTION DATE"].min()
    max_date = raw["PRODUCTION DATE"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        data_period_str = f"{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}"

# --- DISPLAY HEADER WITH DATE ---
current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <strong>🕒 Report Generated:</strong> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <strong>📅 Data Period:</strong> {data_period_str}
</div>
""", unsafe_allow_html=True)

# 2. Metallic Type
metal_col = next(c for c in raw.columns if "METALLIC" in c.upper())
raw["Metallic_Type"] = raw[metal_col]

# 3. Rename Columns
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

# 4. Standard Hardness Split
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))

# 5. Force Numeric
numeric_cols = [
    "Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", "Order_Gauge",
    "Standard TS min", "Standard TS max",
    "Standard YS min", "Standard YS max",
    "Standard EL min", "Standard EL max"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 6. Quality Group Merge
df["Quality_Group"] = df["Quality_Code"].replace({
    "CQ00": "CQ00 / CQ06",
    "CQ06": "CQ00 / CQ06"
})

# 7. Filter GE* < 88
if "Quality_Code" in df.columns:
    df = df[~(
        df["Quality_Code"].astype(str).str.startswith("GE") &
        ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88))
    )]

# =========================================================
# 8. APPLY GLOBAL COMPANY RULES (COLD ROLLING LOGIC)
# =========================================================
def apply_company_rules(row):
    std_min = row["Std_Min"] if pd.notna(row["Std_Min"]) else 0
    std_max = row["Std_Max"] if pd.notna(row["Std_Max"]) else 0
    lab_min, lab_max = 0, 0
    rule_name = "Standard (Excel)"

    is_cold = "COLD" in str(row["Rolling_Type"]).upper()
    q_grp = str(row["Quality_Group"])
    target_qs = ["CQ00", "CQ06", "CQ07", "CQB0"]
    is_target_q = any(q in q_grp for q in target_qs)

    if is_cold and is_target_q:
        mat = str(row["Material"]).upper().strip()
        if mat == "A1081":
            return 56.0, 62.0, 52.0, 70.0, "Rule A1081 (Cold)"
        elif mat == "A108M":
            return 60.0, 68.0, 55.0, 72.0, "Rule A108M (Cold)"
        elif mat in ["A108", "A108G", "A108R", "A108MR", "A1081B"]:
            return 58.0, 62.0, 52.0, 65.0, "Rule A108-Gen (Cold)"

    return std_min, std_max, lab_min, lab_max, rule_name

df[['Limit_Min', 'Limit_Max', 'Lab_Min', 'Lab_Max', 'Rule_Name']] = df.apply(
    apply_company_rules, axis=1, result_type="expand"
)

# ================================
# REFRESH BUTTON
# ================================
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

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
    if len(nums) < 2: return None, None
    return float(nums[0]), float(nums[-1])

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r[gauge_col])
    if lo is not None:
        ranges.append((lo, hi, r[gauge_col]))

def map_gauge(val):
    for lo, hi, name in ranges:
        if lo <= val < hi: return name
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
# ==============================================================================
#  🚀 GLOBAL SUMMARY DASHBOARD (FINAL: FULL COLUMNS - QUALITY & SPECS ADDED)
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Dashboard")
    
    tab1, tab2 = st.tabs(["📊 1. Performance Overview", "🧠 2. Decision Support (Risk AI)"])

    # --- TAB 1: THỐNG KÊ HIỆU SUẤT ---
    with tab1:
        st.info("ℹ️ Color Guide: 🟢 High Pass Rate (>98%) | 🔴 Low Pass Rate (<90%) | 🟡 Rule Applied")
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

            specs_str = ", ".join(sorted(sub_grp["Product_Spec"].astype(str).unique()))

            l_min_val = sub_grp['Limit_Min'].min(); l_max_val = sub_grp['Limit_Max'].max()
            lim_hrb = f"{l_min_val:.0f}~{l_max_val:.0f}"
            
            def get_limit_str(s_min, s_max):
                v_min = sub_grp[s_min].max() if s_min in sub_grp else 0 
                v_max = sub_grp[s_max].min() if s_max in sub_grp else 0 
                if pd.isna(v_min): v_min = 0
                if pd.isna(v_max): v_max = 0
                if v_min > 0 and v_max > 0 and v_max < 9000: return f"{v_min:.0f}~{v_max:.0f}"
                elif v_min > 0: return f"≥ {v_min:.0f}"
                elif v_max > 0 and v_max < 9000: return f"≤ {v_max:.0f}"
                else: return "-"

            lim_ts = get_limit_str("Standard TS min", "Standard TS max")
            lim_ys = get_limit_str("Standard YS min", "Standard YS max")
            lim_el = get_limit_str("Standard EL min", "Standard EL max")

            rule_name = sub_grp['Rule_Name'].iloc[0]
            lab_min = sub_grp['Lab_Min'].iloc[0]; lab_max = sub_grp['Lab_Max'].iloc[0]
            lim_lab = f"{lab_min:.0f}~{lab_max:.0f}" if (lab_min > 0 and lab_max > 0) else "-"

            n_total = len(sub_grp)
            n_ng = sub_grp[(sub_grp["Hardness_LINE"] < sub_grp["Limit_Min"]) | (sub_grp["Hardness_LINE"] > sub_grp["Limit_Max"])].shape[0]
            pass_rate = ((n_total - n_ng) / n_total) * 100

            stats_rows.append({
                "Quality": g["Quality_Group"], "Material": g["Material"], "Gauge": g["Gauge_Range"],
                "Specs": specs_str,
                "Rule": rule_name, "Lab Limit": lim_lab, "HRB Limit": lim_hrb, "N": len(sub_grp),
                "Pass Rate": pass_rate,
                "HRB (Avg)": sub_grp["Hardness_LINE"].mean(), "TS (Avg)": sub_grp["TS"].mean(),
                "YS (Avg)": sub_grp["YS"].mean(), "EL (Avg)": sub_grp["EL"].mean(),
                "HRB (Min)": sub_grp["Hardness_LINE"].min(), "HRB (Max)": sub_grp["Hardness_LINE"].max(),
                "TS Limit": lim_ts, "YS Limit": lim_ys, "EL Limit": lim_el,            
            })

        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            cols = ["Quality", "Material", "Gauge", "Specs", "Rule", "Pass Rate", "HRB Limit", "HRB (Avg)", "TS (Avg)", "YS (Avg)", "EL (Avg)", "N"]
            cols = [c for c in cols if c in df_stats.columns]
            df_stats = df_stats[cols]

            def color_pass_rate(val):
                color = '#d4edda' if val >= 98 else ('#fff3cd' if val >= 90 else '#f8d7da')
                text_color = '#155724' if val >= 98 else ('#856404' if val >= 90 else '#721c24')
                return f'background-color: {color}; color: {text_color}; font-weight: bold'

            st.dataframe(
                df_stats.style.format("{:.1f}", subset=[c for c in df_stats.columns if "(Avg)" in c or "Pass" in c])
                .applymap(color_pass_rate, subset=["Pass Rate"])
                .background_gradient(subset=["HRB (Avg)"], cmap="Blues"),
                use_container_width=True
            )
        else: st.warning("Insufficient data.")

    # --- TAB 2: PHÂN TÍCH RỦI RO (ĐÃ THÊM QUALITY & SPECS) ---
    with tab2:
        st.markdown("#### 🧠 AI Decision Support (Risk-Based)")
        st.caption("AI Decision Support (Risk-Based) (TS / YS / EL).")

        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            user_hrb = st.number_input("1️⃣ Target HRB", value=60.0, step=0.5, format="%.1f")
        with col_in2:
            safety_k = st.selectbox("2️⃣ Sellect Safety Factor):", [1.0, 2.0, 3.0], index=1,
                                    format_func=lambda x: f"{x} Sigma (reliability {68 if x==1 else (95 if x==2 else 99.7)}%)")

        rows_ts, rows_ys, rows_el = [], [], []
        
        for _, g in valid.iterrows():
            sub_grp = df[
                (df["Rolling_Type"] == g["Rolling_Type"]) &
                (df["Metallic_Type"] == g["Metallic_Type"]) &
                (df["Quality_Group"] == g["Quality_Group"]) &
                (df["Gauge_Range"] == g["Gauge_Range"]) &
                (df["Material"] == g["Material"])
            ].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 10: continue 

            # Lấy Specs
            specs_str = ", ".join(sorted(sub_grp["Product_Spec"].astype(str).unique()))

            spec_ts_min = sub_grp["Standard TS min"].max() if "Standard TS min" in sub_grp else 0
            spec_ys_min = sub_grp["Standard YS min"].max() if "Standard YS min" in sub_grp else 0
            spec_el_min = sub_grp["Standard EL min"].max() if "Standard EL min" in sub_grp else 0
            
            X = sub_grp[["Hardness_LINE"]].values

            
            # --- TS Analysis ---
            m_ts = LinearRegression().fit(X, sub_grp["TS"].values)
            pred_ts = m_ts.predict([[user_hrb]])[0]
            err_ts = np.sqrt(np.mean((sub_grp["TS"] - m_ts.predict(X))**2))
            safe_ts = pred_ts - (safety_k * err_ts)
            risk_ts = "🔴 High Risk" if (spec_ts_min > 0 and safe_ts < spec_ts_min) else "🟢 Safe"
            
            rows_ts.append({
                "Quality": g["Quality_Group"], # Mới
                "Material": g["Material"],
                "Gauge": g["Gauge_Range"],
                "Specs": specs_str,            # Mới
                "Pred TS": f"{pred_ts:.0f}",
                "Worst Case": f"{safe_ts:.0f}",
                "Limit": f"≥ {spec_ts_min:.0f}" if spec_ts_min > 0 else "-",
                "Status": risk_ts
            })

            # --- YS Analysis ---
            m_ys = LinearRegression().fit(X, sub_grp["YS"].values)
            pred_ys = m_ys.predict([[user_hrb]])[0]
            err_ys = np.sqrt(np.mean((sub_grp["YS"] - m_ys.predict(X))**2))
            safe_ys = pred_ys - (safety_k * err_ys)
            risk_ys = "🔴 High Risk" if (spec_ys_min > 0 and safe_ys < spec_ys_min) else "🟢 Safe"

            rows_ys.append({
                "Quality": g["Quality_Group"], # Mới
                "Material": g["Material"],
                "Gauge": g["Gauge_Range"],
                "Specs": specs_str,            # Mới
                "Pred YS": f"{pred_ys:.0f}",
                "Worst Case": f"{safe_ys:.0f}",
                "Limit": f"≥ {spec_ys_min:.0f}" if spec_ys_min > 0 else "-",
                "Status": risk_ys
            })

            # --- EL Analysis ---
            m_el = LinearRegression().fit(X, sub_grp["EL"].values)
            pred_el = m_el.predict([[user_hrb]])[0]
            err_el = np.sqrt(np.mean((sub_grp["EL"] - m_el.predict(X))**2))
            safe_el = pred_el - (safety_k * err_el)
            risk_el = "🔴 High Risk" if (spec_el_min > 0 and safe_el < spec_el_min) else "🟢 Safe"

            rows_el.append({
                "Quality": g["Quality_Group"], # Mới
                "Material": g["Material"],
                "Gauge": g["Gauge_Range"],
                "Specs": specs_str,            # Mới
                "Pred EL": f"{pred_el:.1f}",
                "Worst Case": f"{safe_el:.1f}",
                "Limit": f"≥ {spec_el_min:.1f}" if spec_el_min > 0 else "-",
                "Status": risk_el
            })

        if rows_ts:
            def style_risk(val):
                return 'color: red; font-weight: bold' if "🔴" in val else 'color: green; font-weight: bold'

            # Layout: 2 Bảng trên (TS, YS)
            c_top1, c_top2 = st.columns(2)
            
            with c_top1:
                st.markdown("##### 🔹 Tensile Strength (TS)")
                # Không dùng .drop() nữa để hiện đủ cột
                st.dataframe(pd.DataFrame(rows_ts).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
            
            with c_top2:
                st.markdown("##### 🔸 Yield Strength (YS)")
                st.dataframe(pd.DataFrame(rows_ys).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
            
            # Layout: 1 Bảng dưới (EL)
            st.markdown("---")
            st.markdown("##### 🔻 Elongation (EL)")
            st.dataframe(pd.DataFrame(rows_el).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)

        else:
            st.warning("Insufficient data.")
    
    st.stop()
# ==============================================================================
# MAIN LOOP (DETAILS)
# ==============================================================================
for i, (_, g) in enumerate(valid.iterrows()):
    sub = df[
        (df["Rolling_Type"] == g["Rolling_Type"]) &
        (df["Metallic_Type"] == g["Metallic_Type"]) &
        (df["Quality_Group"] == g["Quality_Group"]) &
        (df["Gauge_Range"] == g["Gauge_Range"]) &
        (df["Material"] == g["Material"])
    ].sort_values("COIL_NO")

    lo, hi = sub.iloc[0][["Limit_Min", "Limit_Max"]] 
    rule_used = sub.iloc[0]["Rule_Name"]
    l_lo, l_hi = sub.iloc[0][["Lab_Min", "Lab_Max"]]

    sub["NG_LAB"] = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"] 

    specs = ", ".join(sorted(sub["Product_Spec"].unique()))

    if view_mode != "🚀 Global Summary Dashboard":
        st.markdown(f"### 🧱 {g['Quality_Group']} | {g['Material']} | {g['Gauge_Range']}")
        st.markdown(f"**Specs:** {specs} | **Coils:** {sub['COIL_NO'].nunique()} | **Limit:** {lo:.1f}~{hi:.1f}")
        
        if view_mode != "⚙️ Mech Props Analysis":
            if "Rule" in rule_used: st.success(f"✅ Applied: **{rule_used}** (Control: {lo:.0f} - {hi:.0f} | Lab: {l_lo:.0f} - {l_hi:.0f})")
            else: st.caption(f"ℹ️ Applied: **Standard Excel Spec**")

    # ================================
    # 1. DATA INSPECTION (CLEAN - INTEGERS ONLY)
    # ================================
    if view_mode == "📋 Data Inspection":
        st.markdown(f"### 📋 {g['Material']} | {g['Gauge_Range']}")
        def highlight_ng_rows(row): return ['background-color: #ffe6e6'] * len(row) if row['NG'] else [''] * len(row)
        
        # Lấy danh sách các cột số để làm tròn
        num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        
        st.dataframe(
            sub.style.format("{:.0f}", subset=num_cols) # <--- LÀM TRÒN TẤT CẢ CỘT SỐ
            .apply(highlight_ng_rows, axis=1), 
            use_container_width=True
        )

    # ================================
    # 2. HARDNESS ANALYSIS
    # ================================
    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        st.markdown("### 📉 Hardness Analysis: Process Stability & Capability")
        tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & SPC"])

        with tab_trend:
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB", alpha=0.5)
            ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE", alpha=0.9) 
            ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"Control LSL={lo}")
            ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"Control USL={hi}")
            if l_lo > 0 and l_hi > 0:
                ax.axhline(l_lo, linestyle="-.", linewidth=1.5, color="purple", label=f"Lab LSL={l_lo}", alpha=0.7)
                ax.axhline(l_hi, linestyle="-.", linewidth=1.5, color="purple", label=f"Lab USL={l_hi}", alpha=0.7)
            ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
            ax.set_xlabel("Coil Sequence"); ax.set_ylabel("Hardness (HRB)")
            ax.grid(alpha=0.25, linestyle="--"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4)
            plt.tight_layout(); st.pyplot(fig)
            
            # --- [FIX] GỌI HÀM fig_to_png ĐÃ KHAI BÁO ---
            st.download_button("📥 Download Trend Chart", data=fig_to_png(fig), file_name=f"trend_{g['Material']}.png", mime="image/png", key=f"dl_tr_{uuid.uuid4()}")

        with tab_dist:
            line = sub["Hardness_LINE"].dropna(); lab = sub["Hardness_LAB"].dropna()
            if len(line) < 5: st.warning("⚠️ Not enough LINE data (N < 5).")
            else:
                def calc_spc_metrics(data, lsl, usl):
                    if len(data) < 2: return None
                    mean = data.mean(); std = data.std(ddof=1)
                    if std == 0: return None 
                    cp = (usl - lsl) / (6 * std)
                    mid = (usl + lsl) / 2; tol = (usl - lsl); ca = ((mean - mid) / (tol / 2)) * 100
                    cpu = (usl - mean) / (3 * std); cpl = (mean - lsl) / (3 * std)
                    return mean, std, cp, ca, min(cpu, cpl)

                spc_line = calc_spc_metrics(line, lo, hi)
                mean_line, std_line = line.mean(), line.std(ddof=1)
                
                vals = [line.min(), line.max(), lo, hi]
                if l_lo > 0: vals.extend([l_lo, l_hi])
                if not lab.empty: vals.extend([lab.min(), lab.max()])
                x_min = min(vals) - 2; x_max = max(vals) + 2
                bins = np.linspace(x_min, x_max, 30)
                
                range_curve = max(5 * std_line, (x_max - x_min)/2)
                xs = np.linspace(mean_line - range_curve, mean_line + range_curve, 400)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(line, bins=bins, density=True, alpha=0.6, color="#ff7f0e", edgecolor="white", label="LINE Hist")
                if not lab.empty: ax.hist(lab, bins=bins, density=True, alpha=0.3, color="#1f77b4", edgecolor="None", label="LAB Hist")
                
                if std_line > 0:
                    ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
                    ax.plot(xs, ys_line, linewidth=2.5, color="#b25e00", label="LINE Fit")
                
                ax.axvline(lo, linestyle="--", linewidth=2, color="red", label="Control LSL")
                ax.axvline(hi, linestyle="--", linewidth=2, color="red", label="Control USL")
                if l_lo > 0 and l_hi > 0:
                    ax.axvline(l_lo, linestyle="-.", linewidth=2, color="purple", label="Lab LSL")
                    ax.axvline(l_hi, linestyle="-.", linewidth=2, color="purple", label="Lab USL")
                
                ax.set_xlim(x_min, x_max); ax.set_title(f"Hardness Distribution (LINE vs LAB)", weight="bold")
                ax.legend(); ax.grid(alpha=0.3); st.pyplot(fig)

                st.markdown("#### 📐 SPC Capability Indices (LINE ONLY)")
                if spc_line:
                    mean_val, std_val, cp_val, ca_val, cpk_val = spc_line
                    eval_msg = "Excellent" if cpk_val >= 1.33 else ("Good" if cpk_val >= 1.0 else "Poor")
                    color_code = "green" if cpk_val >= 1.33 else ("orange" if cpk_val >= 1.0 else "red")
                    df_spc = pd.DataFrame([{"N": len(line), "Mean": mean_val, "Std": std_val, "Cp": cp_val, "Ca (%)": ca_val, "Cpk": cpk_val, "Rating": eval_msg}])
                    st.dataframe(df_spc.style.format("{:.2f}", subset=["Mean", "Std", "Cp", "Ca (%)", "Cpk"]).applymap(lambda v: f'color: {color_code}; font-weight: bold', subset=['Rating']), hide_index=True)

    # ================================
    # 3. CORRELATION
    # ================================
    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        
        # --- 1. KHỞI TẠO DANH SÁCH TỔNG HỢP Ở VÒNG LẶP ĐẦU TIÊN ---
        if i == 0:
            corr_bin_summary = []

        st.markdown("### 🔗 Correlation: Hardness vs Mechanical Properties")
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
            x = np.arange(len(summary))
            fig, ax = plt.subplots(figsize=(15,6))
            def plot_prop(x, y, ymin, ymax, c, lbl, m):
                ax.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax.fill_between(x, ymin, ymax, color=c, alpha=0.1)
            
            plot_prop(x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS", "o")
            plot_prop(x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS", "s")
            plot_prop(x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL", "^")

            for j, row in enumerate(summary.itertuples()):
                ax.annotate(f"{row.TS_mean:.0f}", (x[j], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="#1f77b4")
                ax.annotate(f"{row.YS_mean:.0f}", (x[j], row.YS_mean), xytext=(0,-15), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="#2ca02c")
                el_spec = row.Std_EL_min
                is_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                lbl = f"{row.EL_mean:.1f}%" + ("❌" if is_fail else "")
                clr = "red" if is_fail else "#ff7f0e"
                ax.annotate(lbl, (x[j], row.EL_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, color=clr, fontweight=("bold" if is_fail else "normal"))

            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_title("Hardness vs Mechanical Properties"); ax.grid(True, ls="--", alpha=0.5); ax.legend(); st.pyplot(fig)

            # --- 2. THU THẬP DỮ LIỆU TỔNG HỢP ---
            col_spec = "Product_Spec"
            specs_str = f"Specs: {', '.join(str(x) for x in sub[col_spec].dropna().unique())}" if col_spec in sub.columns else "Specs: N/A"

            for row in summary.itertuples():
                # Tính Std Dev cho từng Bin để sếp bạn theo dõi độ ổn định theo dải độ cứng
                bin_data = sub_corr[sub_corr["HRB_bin"] == row.HRB_bin]
                
                corr_bin_summary.append({
                    "Specification List": specs_str,
                    "Material": g["Material"],
                    "Gauge": g["Gauge_Range"],
                    "Hardness Bin": row.HRB_bin,
                    "N": row.N_coils,
                    # TS Data
                    "TS Spec": f"{row.Std_TS_min:.0f}~{row.Std_TS_max:.0f}" if row.Std_TS_max < 9000 else f"≥{row.Std_TS_min:.0f}",
                    "TS Actual": f"{row.TS_min:.0f}~{row.TS_max:.0f}",
                    "TS Mean": f"{row.TS_mean:.1f}",
                    "TS Std": f"{bin_data['TS'].std():.1f}",
                    # YS Data
                    "YS Spec": f"{row.Std_YS_min:.0f}~{row.Std_YS_max:.0f}" if row.Std_YS_max < 9000 else f"≥{row.Std_YS_min:.0f}",
                    "YS Actual": f"{row.YS_min:.0f}~{row.YS_max:.0f}",
                    "YS Mean": f"{row.YS_mean:.1f}",
                    "YS Std": f"{bin_data['YS'].std():.1f}",
                    # EL Data
                    "EL Spec": f"≥{row.Std_EL_min:.0f}",
                    "EL Actual": f"{row.EL_min:.1f}~{row.EL_max:.1f}",
                    "EL Mean": f"{row.EL_mean:.1f}",
                    "EL Std": f"{bin_data['EL'].std():.1f}"
                })

        # --- 3. HIỂN THỊ CÁC BẢNG TỔNG HỢP RIÊNG BIỆT Ở CUỐI TRANG ---
        if i == len(valid) - 1 and 'corr_bin_summary' in locals() and len(corr_bin_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 📊 Hardness Binning Comprehensive Report: {qgroup}")
            
            df_full = pd.DataFrame(corr_bin_summary)
            
            def display_bin_table(title, cols, color_code):
                st.markdown(f"#### {title}")
                # Lọc ra các cột chung + các cột thông số cụ thể
                base_cols = ["Specification List", "Material", "Gauge", "Hardness Bin", "N"]
                target_df = df_full[base_cols + cols]
                
                # Highlight cột Std để theo dõi biến động theo từng Bin độ cứng
                std_col = [c for c in target_df.columns if "Std" in c]
                styled = target_df.style.set_properties(**{'background-color': color_code, 'font-weight': 'bold'}, subset=std_col)
                st.dataframe(styled, use_container_width=True, hide_index=True)

            display_bin_table("📉 TS Analysis by Hardness Bin", ["TS Spec", "TS Actual", "TS Mean", "TS Std"], "#e6f2ff")
            display_bin_table("📉 YS Analysis by Hardness Bin", ["YS Spec", "YS Actual", "YS Mean", "YS Std"], "#f2fff2")
            display_bin_table("📉 EL Analysis by Hardness Bin", ["EL Spec", "EL Actual", "EL Mean", "EL Std"], "#fff5e6")
            
            # Xuất file CSV tổng hợp
            import datetime
            csv_name = f"Hardness_Bin_Report_{str(qgroup).replace(' ','')}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            st.download_button("📥 Export Binning Report CSV", df_full.to_csv(index=False).encode('utf-8-sig'), csv_name)
    # ================================
  # 4. MECH PROPS ANALYSIS
    # ================================
    elif view_mode == "⚙️ Mech Props Analysis":
        
        # --- 1. KHỞI TẠO 3 DANH SÁCH TỔNG HỢP RIÊNG BIỆT ---
        if i == 0:
            ts_summary, ys_summary, el_summary = [], [], []

        st.markdown(f"### ⚙️ Mechanical Properties Analysis: {g['Material']} | {g['Gauge_Range']}")
        # Lấy dữ liệu cơ tính và giữ lại Hardness_LINE để tính dải độ cứng
        sub_mech = sub.dropna(subset=["TS","YS","EL"])
        
        if sub_mech.empty: 
            st.warning("⚠️ No Mech Data.")
        else:
            props_config = [
                {"col": "TS", "name": "Tensile Strength (TS)", "color": "#1f77b4", "min_c": "Standard TS min", "max_c": "Standard TS max"},
                {"col": "YS", "name": "Yield Strength (YS)", "color": "#2ca02c", "min_c": "Standard YS min", "max_c": "Standard YS max"},
                {"col": "EL", "name": "Elongation (EL)", "color": "#ff7f0e", "min_c": "Standard EL min", "max_c": "Standard EL max"}
            ]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Xử lý trích xuất Specs từ cột Product_Spec giống View 6
            col_spec = "Product_Spec"
            specs_str = f"Specs: {', '.join(str(x) for x in sub[col_spec].dropna().unique())}" if col_spec in sub.columns else "Specs: N/A"

            # --- TÍNH TOÁN DẢI ĐỘ CỨNG THỰC TẾ ---
            if "Hardness_LINE" in sub_mech.columns:
                h_data = sub_mech["Hardness_LINE"].dropna()
                if not h_data.empty:
                    hardness_range_str = f"{h_data.min():.1f} ~ {h_data.max():.1f}"
                else:
                    hardness_range_str = "N/A"
            else:
                hardness_range_str = "N/A"

            for j, cfg in enumerate(props_config):
                col = cfg["col"]; data = sub_mech[col]; mean, std = data.mean(), data.std()
                spec_min = sub_mech[cfg["min_c"]].max() if cfg["min_c"] in sub_mech else 0
                spec_max = sub_mech[cfg["max_c"]].min() if cfg["max_c"] in sub_mech else 0
                if pd.isna(spec_min): spec_min = 0
                if pd.isna(spec_max): spec_max = 0
                
                # Tính toán giới hạn 3-Sigma
                lcl_3s = mean - 3 * std
                ucl_3s = mean + 3 * std
                
                # Vẽ biểu đồ
                axes[j].hist(data, bins=20, color=cfg["color"], alpha=0.5, density=True)
                if std > 0:
                    x_p = np.linspace(mean - 5 * std, mean + 5 * std, 200)
                    y_p = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2)
                    axes[j].plot(x_p, y_p, color=cfg["color"], lw=2)
                
                if spec_min > 0: axes[j].axvline(spec_min, color="red", linestyle="--", linewidth=2)
                if spec_max > 0 and spec_max < 9000: axes[j].axvline(spec_max, color="red", linestyle="--", linewidth=2)
                
                # Vẽ thêm đường 3-Sigma trên biểu đồ để trực quan hóa
                axes[j].axvline(lcl_3s, color="blue", linestyle=":", linewidth=1.5)
                axes[j].axvline(ucl_3s, color="blue", linestyle=":", linewidth=1.5)
                
                axes[j].set_title(f"{cfg['name']}\n(Mean={mean:.1f}, Std={std:.1f})", fontweight="bold")
                axes[j].grid(alpha=0.3, linestyle="--")

                # --- PHÂN LOẠI DỮ LIỆU VÀO 3 BẢNG RIÊNG VỚI CỘT 3-SIGMA VÀ HARDNESS RANGE ---
                row_data = {
                    "Specification List": specs_str,
                    "Material": g["Material"],
                    "Gauge": g["Gauge_Range"],
                    "N": len(sub_mech),
                    "Hardness Range (HRB)": hardness_range_str, # <--- CỘT MỚI: DẢI ĐỘ CỨNG THỰC TẾ
                    "Limit (Spec)": f"{spec_min:.0f}~{spec_max:.0f}" if (spec_max > 0 and spec_max < 9000) else f"≥ {spec_min:.0f}",
                    "Actual Range": f"{data.min():.1f}~{data.max():.1f}",
                    "Mean": f"{mean:.1f}",
                    "Std Dev": f"{std:.1f}",
                    "LCL (-3σ)": f"{lcl_3s:.1f}", 
                    "UCL (+3σ)": f"{ucl_3s:.1f}"  
                }
                
                if col == "TS": ts_summary.append(row_data)
                elif col == "YS": ys_summary.append(row_data)
                elif col == "EL": el_summary.append(row_data)
            
            st.pyplot(fig)

        # --- 2. HIỂN THỊ 3 BẢNG TỔNG HỢP RIÊNG BIỆT Ở CUỐI VÒNG LẶP ---
        if i == len(valid) - 1:
            st.markdown("---")
            st.markdown(f"## 📊 Mechanical Properties Comprehensive Report: {qgroup}")
            
            def display_summary_table(title, data_list, color_code):
                if data_list:
                    st.markdown(f"#### {title}")
                    df = pd.DataFrame(data_list)
                    # Định dạng in đậm cột Mean, Hardness Range và highlight cụm cột 3-Sigma
                    styled_df = df.style.set_properties(**{'font-weight': 'bold'}, subset=['Mean']) \
                                        .set_properties(**{'background-color': '#f0f8ff', 'font-weight': 'bold', 'color': '#0056b3'}, subset=['Hardness Range (HRB)']) \
                                        .set_properties(**{'background-color': color_code, 'color': '#004085'}, subset=['LCL (-3σ)', 'UCL (+3σ)'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

            display_summary_table("1️⃣ Tensile Strength (TS) Summary", ts_summary, "#e6f2ff") 
            display_summary_table("2️⃣ Yield Strength (YS) Summary", ys_summary, "#f2fff2")   
            display_summary_table("3️⃣ Elongation (EL) Summary", el_summary, "#fff5e6")        

            import datetime
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            full_df = pd.concat([pd.DataFrame(ts_summary), pd.DataFrame(ys_summary), pd.DataFrame(el_summary)], keys=['TS','YS','EL'])
            st.download_button("📥 Export Full Mech Report CSV", full_df.to_csv(index=True).encode('utf-8-sig'), f"Full_Mech_Report_{today_str}.csv")
   # ================================
    # 5. LOOKUP (UPDATED: DYNAMIC DEFAULTS)
    # ================================
    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        c1, c2 = st.columns(2)
        
        # Lấy min/max thực tế từ dữ liệu đang hiển thị
        actual_min = float(sub["Hardness_LINE"].min())
        actual_max = float(sub["Hardness_LINE"].max())
        
        # Thiết lập giá trị mặc định linh hoạt thay vì con số 58 và 65 cố định
        mn = c1.number_input("Min HRB", value=actual_min, step=0.5, key=f"lk1_{uuid.uuid4()}")
        mx = c2.number_input("Max HRB", value=actual_max, step=0.5, key=f"lk2_{uuid.uuid4()}")
        
        filt = sub[(sub["Hardness_LINE"]>=mn) & (sub["Hardness_LINE"]<=mx)].dropna(subset=["TS","YS","EL"])
        st.success(f"Found {len(filt)} coils.")
        
        if not filt.empty: 
            st.dataframe(filt[["TS","YS","EL"]].describe().T)

    # ================================
 # 6. REVERSE LOOKUP
    # ================================
    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        
        # --- 1. Initialize summary list at the first iteration ---
        if i == 0:
            reverse_lookup_summary = []

        st.subheader(f"🎯 Target Hardness Calculator: {g['Material']} | {g['Gauge_Range']}")
        
        # --- PRESERVED LOGIC FOR SMART LIMITS ---
        def calculate_smart_limits(name, col_val, col_spec_min, col_spec_max, step=5.0):
            try:
                series_val = pd.to_numeric(sub[col_val], errors='coerce')
                valid_data = series_val[series_val > 0.1].dropna()
                if valid_data.empty: return 0.0, 0.0
                mean = float(valid_data.mean()); std = float(valid_data.std()) if len(valid_data) > 1 else 0.0
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
                final_max = min(stat_max, spec_max) if spec_max < 9000 else (stat_max + (1 * std) if is_no_spec else stat_max)
                if final_min >= final_max: final_min, final_max = stat_min, stat_max + std
                return float(round(max(0.0, final_min) / step) * step), float(round(final_max / step) * step)
            except: return 0.0, 0.0

        d_ys_min, d_ys_max = calculate_smart_limits('YS', 'YS', 'Standard YS min', 'Standard YS max', 5.0)
        d_ts_min, d_ts_max = calculate_smart_limits('TS', 'TS', 'Standard TS min', 'Standard TS max', 5.0)
        d_el_min, d_el_max = calculate_smart_limits('EL', 'EL', 'Standard EL min', 'Standard EL max', 1.0)

        c1, c2, c3 = st.columns(3)
        
        # Keys added to prevent duplicate widget errors
        r_ys_min = c1.number_input("Min YS", value=d_ys_min, step=5.0, key=f"ymin_{i}")
        r_ys_max = c1.number_input("Max YS", value=d_ys_max, step=5.0, key=f"ymax_{i}")
        r_ts_min = c2.number_input("Min TS", value=d_ts_min, step=5.0, key=f"tmin_{i}")
        r_ts_max = c2.number_input("Max TS", value=d_ts_max, step=5.0, key=f"tmax_{i}")
        r_el_min = c3.number_input("Min EL", value=d_el_min, step=1.0, key=f"emin_{i}")
        r_el_max = c3.number_input("Max EL", value=d_el_max, step=1.0, key=f"emax_{i}")

        filtered = sub[
            (sub['YS'] >= r_ys_min) & (sub['YS'] <= r_ys_max) &
            (sub['TS'] >= r_ts_min) & (sub['TS'] <= r_ts_max) &
            ((sub['EL'] >= r_el_min) | (r_el_min==0)) & (sub['EL'] <= r_el_max)
        ]
        
        if not filtered.empty:
            target_min = filtered['Hardness_LINE'].min()
            target_max = filtered['Hardness_LINE'].max()
            n_coils = len(filtered)
            target_text = f"{target_min:.1f} ~ {target_max:.1f}"
            st.success(f"✅ Target Hardness: **{target_text} HRB** (N={n_coils})")
            st.dataframe(filtered[['COIL_NO','Hardness_LINE','YS','TS','EL']], height=300)
        else: 
            target_text = "❌ No Coils Found"
            n_coils = 0
            st.error("❌ No coils found matching these specs.")

        # --- 2. XỬ LÝ CHUỖI TIÊU CHUẨN (SPECS) TỪ CỘT Product_Spec ---
        col_name = "Product_Spec"  # Tên cột chính xác tuyệt đối từ danh sách của bạn
        
        if col_name in sub.columns:
            unique_specs = sub[col_name].dropna().unique()
            if len(unique_specs) > 0:
                specs_str = f"Specs: {', '.join(str(x) for x in unique_specs)}"
            else:
                specs_str = "Specs: N/A"
        else:
            specs_str = "Specs: N/A"

        # LƯU VÀO DANH SÁCH TỔNG HỢP
        reverse_lookup_summary.append({
            "Specification List": specs_str,
            "Material": g["Material"],
            "Gauge": g["Gauge_Range"],
            "YS Setup": f"{r_ys_min:.0f} ~ {r_ys_max:.0f}",
            "TS Setup": f"{r_ts_min:.0f} ~ {r_ts_max:.0f}",
            "EL Setup": f"{r_el_min:.0f} ~ {r_el_max:.0f}",
            "Target Hardness (HRB)": target_text,
            "Matching Coils": n_coils
        })
        # --- 3. DISPLAY THE SUMMARY TABLE AT THE LAST ITERATION ---
        if i == len(valid) - 1 and 'reverse_lookup_summary' in locals() and len(reverse_lookup_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 🎯 Comprehensive Target Hardness Summary for {qgroup}")
            
            df_target = pd.DataFrame(reverse_lookup_summary)
            
            # Apply styling for better visualization
            def style_target(val):
                if isinstance(val, str) and "❌" in val:
                    return 'color: red; font-weight: bold'
                elif isinstance(val, str) and "~" in val:
                    return 'color: #0056b3; font-weight: bold; background-color: #e6f2ff'
                return ''

            st.dataframe(
                df_target.style.applymap(style_target, subset=['Target Hardness (HRB)']),
                use_container_width=True,
                hide_index=True
            )
            
            # Export to CSV with UTF-8-SIG to support Vietnamese characters in Excel
            import datetime
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            safe_qgroup = str(qgroup).replace(" / ", "_").replace("/", "_").replace(" ", "")
            csv_filename = f"Target_Hardness_{safe_qgroup}_{today_str}.csv"
            
            csv_data = df_target.to_csv(index=False).encode('utf-8-sig')
            st.download_button(f"📥 Export Target Hardness CSV ({today_str})", csv_data, csv_filename, "text/csv")
    # ================================
   # ================================
    # 7. AI PREDICTION (ULTIMATE FIX: STABLE INPUT + PRO TOOLTIP)
    # ================================
    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown(f"### 🧮 AI Prediction: {g['Material']}") # Hiển thị tên vật liệu trên tiêu đề
        
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        
        if len(train_df) < 5:
            st.warning("⚠️ Need at least 5 coils.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                mean_h = train_df["Hardness_LINE"].mean()
                # [FIX QUAN TRỌNG] Dùng key theo biến 'i' để cố định, không bị reset khi nhập
                target_h = st.number_input(
                    "🎯 Target Hardness", 
                    value=float(round(mean_h, 1)), 
                    step=0.1, 
                    key=f"ai_fix_{i}" 
                )
            
            X_train = train_df[["Hardness_LINE"]].values
            preds = {}
            
            # Tính toán dự báo ngay lập tức theo target_h mới
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                val = model.predict([[target_h]])[0]
                preds[col] = val 

            # --- VẼ BIỂU ĐỒ ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            colors = {"TS": "#2980b9", "YS": "#27ae60", "EL": "#c0392b"} 
            idx = list(range(len(train_df)))
            nxt = len(train_df)

            for col in ["TS", "YS", "EL"]:
                sec = (col == "EL")
                
                # 1. Đường lịch sử
                fig.add_trace(go.Scatter(
                    x=idx, y=train_df[col], 
                    mode='lines', 
                    line=dict(color=colors[col], width=2, shape='spline'), 
                    name=f"{col} (History)",
                    opacity=0.6,
                    hoverinfo='y' 
                ), secondary_y=sec)
                
                # Lấy giá trị cuộn cuối cùng 
                last_val_raw = train_df[col].iloc[-1]
                
                # Làm sạch số liệu (Clean Numbers)
                pred_clean = round(preds[col], 1) if col == "EL" else int(round(preds[col]))
                last_clean = round(last_val_raw, 1) if col == "EL" else int(round(last_val_raw))
                
                # 2. Đường nối (Connector)
                fig.add_trace(go.Scatter(
                    x=[idx[-1], nxt], y=[last_val_raw, preds[col]],
                    mode='lines',
                    line=dict(color=colors[col], width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ), secondary_y=sec)

                # 3. Điểm Dự Báo (Tooltip Đầy Đủ)
                fig.add_trace(go.Scatter(
                    x=[nxt], y=[preds[col]], 
                    mode='markers+text', 
                    text=[f"<b>{pred_clean}</b>"], 
                    textposition="middle right" if nxt < 10 else "top center",
                    marker=dict(color=colors[col], size=14, symbol='diamond', line=dict(width=2, color='white')), 
                    name=f"Pred {col}",
                    # Tooltip thông minh: Hiện cả Pred và Last để so sánh
                    hovertemplate=(
                        f"<b>🎯 Pred {col}: {pred_clean}</b><br>"
                        f"🔙 Last {col}: {last_clean}<br>"
                        f"📈 Change: {pred_clean - last_clean:.1f}"
                        "<extra></extra>"
                    )
                ), secondary_y=sec)

            # Trang trí
            fig.add_vline(x=nxt - 0.5, line_width=1, line_dash="dash", line_color="gray")
            fig.add_annotation(x=nxt - 0.5, y=1.05, yref="paper", text="Forecast Zone ➔", showarrow=False, font=dict(color="gray"))

            fig.update_layout(
                height=500,
                title=dict(text=f"📈 Prediction at Hardness = {target_h}", font=dict(size=18)),
                plot_bgcolor="white",
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=80, b=20)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', title="Coil Sequence")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', secondary_y=False, title="Strength (MPa)")
            fig.update_yaxes(showgrid=False, secondary_y=True, title="Elongation (%)")

            st.plotly_chart(fig, use_container_width=True)
            
            # Cards summary
            st.markdown("#### 🏁 Forecast Summary")
            c1, c2, c3 = st.columns(3)
            
            def get_delta(p, l): return round(p - l, 1)
            
            last_ts = train_df["TS"].iloc[-1]
            last_ys = train_df["YS"].iloc[-1]
            last_el = train_df["EL"].iloc[-1]

            c1.metric("Tensile Strength (TS)", f"{int(round(preds['TS']))} MPa", f"{get_delta(preds['TS'], last_ts)} vs Last")
            c2.metric("Yield Strength (YS)", f"{int(round(preds['YS']))} MPa", f"{get_delta(preds['YS'], last_ys)} vs Last")
            c3.metric("Elongation (EL)", f"{round(preds['EL'], 1)} %", f"{get_delta(preds['EL'], last_el)} vs Last")
    # ================================
  # # 8. CONTROL LIMIT CALCULATOR
    # ================================
    elif view_mode == "🎛️ Control Limit Calculator (Compare 3 Methods)":
        
        # --- 1. KHỞI TẠO DANH SÁCH TỔNG HỢP Ở VÒNG LẶP ĐẦU TIÊN ---
        if i == 0:
            all_groups_summary = []

        st.markdown(f"### 🎛️ Control Limits Analysis: {g['Material']} | {g['Gauge_Range']}")
        data = sub["Hardness_LINE"].dropna()
        data_lab = sub["Hardness_LAB"].dropna()
        
        if len(data) < 10: 
            st.warning(f"⚠️ {g['Material']}: 數據不足 (N={len(data)})")
        else:
            with st.expander("⚙️ 設定參數 (Settings)", expanded=False):
                c1, c2 = st.columns(2)
                sigma_n = c1.number_input("1. Sigma Multiplier (K)", 1.0, 6.0, 3.0, 0.5, key=f"sig_{i}")
                iqr_k = c2.number_input("2. IQR Sensitivity", 0.5, 3.0, 0.7, 0.1, key=f"iqr_{i}")

            # --- LẤY GIỚI HẠN CONTROL VÀ LAB ---
            spec_min = sub["Limit_Min"].max(); spec_max = sub["Limit_Max"].min()
            lab_min = sub["Lab_Min"].max(); lab_max = sub["Lab_Max"].min()
            rule_name = sub["Rule_Name"].iloc[0] 
            
            if pd.isna(spec_min): spec_min = 0
            if pd.isna(spec_max): spec_max = 0
            if pd.isna(lab_min): lab_min = 0
            if pd.isna(lab_max): lab_max = 0
            
            display_max = spec_max if (spec_max > 0 and spec_max < 9000) else 0
            display_lab_max = lab_max if (lab_max > 0 and lab_max < 9000) else 0
            
            mu = data.mean(); std_dev = data.std()
            
            # M1: Standard
            m1_min, m1_max = mu - sigma_n*std_dev, mu + sigma_n*std_dev
            
            # M2: IQR Robust
            Q1 = data.quantile(0.25); Q3 = data.quantile(0.75); IQR = Q3 - Q1
            clean_data = data[~((data < (Q1 - iqr_k * IQR)) | (data > (Q3 + iqr_k * IQR)))]
            if clean_data.empty: clean_data = data
            mu_clean, sigma_clean = clean_data.mean(), clean_data.std()
            m2_min, m2_max = mu_clean - sigma_n*sigma_clean, mu_clean + sigma_n*sigma_clean
            
            # M3: Smart Hybrid
            m3_min = max(m2_min, spec_min)
            m3_max = min(m2_max, spec_max) if (spec_max > 0 and spec_max < 9000) else m2_max
            if m3_min >= m3_max: m3_min, m3_max = m2_min, m2_max
            
            # M4: I-MR (SPC)
            mrs = np.abs(np.diff(data)); mr_bar = np.mean(mrs); sigma_imr = mr_bar / 1.128
            m4_min, m4_max = mu - sigma_n * sigma_imr, mu + sigma_n * sigma_imr

            # --- TẠO CHUỖI HIỂN THỊ CHO CỘT SPEC ---
            if display_lab_max > 0:
                spec_str = f"Ctrl: {spec_min:.0f}~{display_max:.0f} | Lab: {lab_min:.0f}~{display_lab_max:.0f}"
            else:
                spec_str = f"{spec_min:.0f} ~ {display_max:.0f}"

            # --- 2. XỬ LÝ CHUỖI TIÊU CHUẨN (SPECS) ---
            col_spec = "Product_Spec"
            if col_spec in sub.columns:
                unique_specs = sub[col_spec].dropna().unique()
                specs_val = f"Specs: {', '.join(str(x) for x in unique_specs)}" if len(unique_specs) > 0 else "Specs: N/A"
            else:
                specs_val = "Specs: N/A"

            # --- LƯU DỮ LIỆU VÀO DANH SÁCH TỔNG HỢP ---
            all_groups_summary.append({
                "Specification List": specs_val, # <--- ĐÃ ĐỔI TỪ QUALITY SANG SPEC LIST
                "Material": g["Material"],
                "Gauge": g["Gauge_Range"],
                "N": len(data),
                "Rule Applied": rule_name,
                "Current Spec": spec_str,
                "M1: Standard": f"{m1_min:.1f} ~ {m1_max:.1f}",
                "M2: IQR (Robust)": f"{m2_min:.1f} ~ {m2_max:.1f}",
                "M3: Smart Hybrid": f"{m3_min:.1f} ~ {m3_max:.1f}", 
                "M4: I-MR (Optimal)": f"{m4_min:.1f} ~ {m4_max:.1f}",
                "Status": "✅ Stable" if (display_max > 0 and m4_max <= display_max) else "⚠️ Narrow Spec"
            })

            # --- PHẦN VẼ BIỂU ĐỒ (GIỮ NGUYÊN THIẾT KẾ CỦA BẠN) ---
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(data, bins=30, density=True, alpha=0.6, color="#1f77b4", label="LINE (Production)")
                if not data_lab.empty: ax.hist(data_lab, bins=30, density=True, alpha=0.4, color="#ff7f0e", label="LAB (Ref)")
                ax.axvline(m1_min, c="red", ls=":", alpha=0.4); ax.axvline(m1_max, c="red", ls=":", alpha=0.4, label="M1: Standard")
                ax.axvline(m2_min, c="blue", ls="--", alpha=0.5); ax.axvline(m2_max, c="blue", ls="--", alpha=0.5, label="M2: IQR")
                ax.axvline(m4_min, c="purple", ls="-.", lw=2); ax.axvline(m4_max, c="purple", ls="-.", lw=2, label="M4: I-MR (SPC)")
                ax.axvspan(m3_min, m3_max, color="green", alpha=0.15, label="M3: Hybrid Zone")
                if spec_min > 0: ax.axvline(spec_min, c="black", lw=2)
                if display_max > 0: ax.axvline(display_max, c="black", lw=2)
                ax.set_title(f"Limits Comparison (σ={sigma_n})", fontsize=10, fontweight="bold")
                ax.legend(loc="upper right", fontsize="small"); st.pyplot(fig)

            with col_table:
                comp_data = [
                    {"Method": "0. Spec (Rule)", "Min": spec_min, "Max": display_max, "Range": display_max-spec_min if display_max>0 else 0, "Note": rule_name},
                    {"Method": "1. Standard", "Min": m1_min, "Max": m1_max, "Range": m1_max-m1_min, "Note": "Basic Stats"},
                    {"Method": "2. IQR Robust", "Min": m2_min, "Max": m2_max, "Range": m2_max-m2_min, "Note": "Filtered"},
                    {"Method": "3. Smart Hybrid", "Min": m3_min, "Max": m3_max, "Range": m3_max-m3_min, "Note": "Configurable"},
                    {"Method": "4. I-MR (SPC)", "Min": m4_min, "Max": m4_max, "Range": m4_max-m4_min, "Note": "✅ Professional"}
                ]
                st.dataframe(pd.DataFrame(comp_data).style.format("{:.1f}", subset=["Min", "Max", "Range"]), use_container_width=True, hide_index=True)

        # --- HIỂN THỊ BẢNG TỔNG HỢP Ở VÒNG LẶP CUỐI CÙNG ---
        if i == len(valid) - 1 and 'all_groups_summary' in locals() and len(all_groups_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 📊 Summary of Control Limits for {qgroup}")
            
            df_total = pd.DataFrame(all_groups_summary)
            
            def style_status(val):
                color = 'red' if 'Narrow' in val else 'green'
                return f'color: {color}; font-weight: bold'

            styled_df = (
                df_total.style
                .applymap(style_status, subset=['Status'])
                .set_properties(**{'background-color': '#e6f2ff', 'color': '#004085', 'font-weight': 'bold', 'border': '2px solid #0056b3'}, subset=['M4: I-MR (Optimal)'])
            )

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Export CSV (Hỗ trợ tiếng Việt cho Specs)
            st.download_button("📥 Export Summary CSV", df_total.to_csv(index=False).encode('utf-8-sig'), f"SPC_Summary_{str(qgroup).replace(' ','')}.csv")
