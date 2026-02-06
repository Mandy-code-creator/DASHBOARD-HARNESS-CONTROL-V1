# ================================
# FULL STREAMLIT APP – FINAL FIXED
# Features: Auto-Switch Prediction (Line/Lab), Unique Keys, Correct Mapping
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

# ================================
# 1. PAGE CONFIG & UTILS
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")
st.title("📊 SPC Hardness – Material / Gauge Level Analysis")

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# 2. LOAD DATA & PRE-PROCESSING
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_data():
    # Load Main Data
    r1 = requests.get(DATA_URL); r1.encoding = "utf-8"
    df = pd.read_csv(StringIO(r1.text))
    
    # Load Gauge Mapping
    r2 = requests.get(GAUGE_URL); r2.encoding = "utf-8"
    g_df = pd.read_csv(StringIO(r2.text))
    return df, g_df

raw, gauge_df = load_data()

# --- A. RENAME COLUMNS ---
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
    "TENSILE_TENSILE": "TS", 
    "TENSILE_YIELD": "YS", 
    "TENSILE_ELONG": "EL",
    "Standard TS min": "Standard TS min", "Standard TS max": "Standard TS max",
    "Standard YS min": "Standard YS min", "Standard YS max": "Standard YS max",
    "Standard EL min": "Standard EL min", "Standard EL max": "Standard EL max"
})

# --- B. FORCE NUMERIC ---
for c in ["Hardness_LAB","Hardness_LINE","TS","YS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- C. PARSE STANDARD HARDNESS ---
def split_std(x):
    if isinstance(x, str) and "~" in x:
        try:
            lo, hi = x.split("~")
            return float(lo), float(hi)
        except:
            pass
    return np.nan, np.nan

df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
df["Std_Hardness_Range"] = df["Std_Min"].astype(str) + " ~ " + df["Std_Max"].astype(str)

# --- D. MAPPING GAUGE RANGE (CRITICAL FIX) ---
gauge_df.columns = gauge_df.columns.str.strip()
gauge_col = next(c for c in gauge_df.columns if "RANGE" in c.upper())

def parse_range_text(text):
    nums = re.findall(r"\d+\.\d+|\d+", str(text))
    return (float(nums[0]), float(nums[-1])) if len(nums) >= 2 else (None, None)

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range_text(r[gauge_col])
    if lo is not None: 
        ranges.append((lo, hi, r[gauge_col]))

def map_gauge(val):
    for lo, hi, name in ranges:
        if lo <= val < hi: # Logic: 0.28 <= T < 0.35
            return name
    return "Other Groups"

df["Gauge_Range_Group"] = df["Order_Gauge"].apply(map_gauge)

# --- E. FILTER GE* < 88 ---
if "Quality_Code" in df.columns:
    df = df[~(
        df["Quality_Code"].astype(str).str.startswith("GE") &
        ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88))
    )]

# ================================
# 3. SIDEBAR & FILTERS
# ================================
st.sidebar.header("🎛 FILTER")

# Filter logic
rolling_opts = sorted(df["Rolling_Type"].dropna().unique())
rolling = st.sidebar.radio("Rolling Type", rolling_opts)

# Quality Group logic (CQ00/06 merged for filtering)
df["Quality_Short"] = df["Quality_Code"].astype(str).str[:4]
q_opts = sorted(df["Quality_Short"].unique())
qgroup = st.sidebar.multiselect("Quality Group", q_opts, default=q_opts)

# Apply filters
df_filtered = df[
    (df["Rolling_Type"] == rolling) & 
    (df["Quality_Short"].isin(qgroup))
]

view_mode = st.sidebar.radio("📊 View Mode", [
    "📋 Data Table", 
    "📈 Trend (LAB / LINE)", 
    "🛠 Hardness → TS/YS/EL", 
    "🧮 Predict TS/YS/EL (Custom Hardness)", 
    "📊 Hardness → Mechanical Range"
])

# ================================
# 4. MAIN LOOP (FIXED INDENTATION & KEYS)
# ================================
GROUP_COLS = ["Rolling_Type", "Gauge_Range_Group", "Material"]

# Chỉ lấy các nhóm có dữ liệu >= 5 cuộn để hiển thị
valid_groups = df_filtered.groupby(GROUP_COLS).size().reset_index(name='N')
valid_groups = valid_groups[valid_groups['N'] >= 5] 

if valid_groups.empty:
    st.warning("⚠️ Không tìm thấy nhóm dữ liệu nào thỏa mãn điều kiện lọc.")
    st.stop()

for idx, g in valid_groups.iterrows():
    # --- UNIQUE ID GENERATION ---
    # Tạo ID duy nhất cho mỗi nhóm để widget không bị trùng
    uid = f"{g['Material']}_{g['Gauge_Range_Group']}_{idx}".replace(".", "_").replace(" ", "")
    
    sub = df_filtered[
        (df_filtered["Rolling_Type"] == g["Rolling_Type"]) &
        (df_filtered["Gauge_Range_Group"] == g["Gauge_Range_Group"]) &
        (df_filtered["Material"] == g["Material"])
    ].sort_values("COIL_NO")
    
    N_sub = len(sub)
    
    st.markdown("---")
    st.markdown(f"### 🧱 {g['Material']} | Gauge: {g['Gauge_Range_Group']} (n={N_sub})")

    # ==========================
    # VIEW 1: DATA TABLE
    # ==========================
    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    # ==========================
    # VIEW 2: TREND LAB/LINE
    # ==========================
    elif view_mode == "📈 Trend (LAB / LINE)":
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(sub))
        ax.plot(x, sub["Hardness_LAB"], marker="o", label="LAB")
        ax.plot(x, sub["Hardness_LINE"], marker="s", label="LINE")
        
        # Vẽ giới hạn Spec nếu có
        std_min = sub["Std_Min"].iloc[0] if "Std_Min" in sub.columns else np.nan
        std_max = sub["Std_Max"].iloc[0] if "Std_Max" in sub.columns else np.nan
        if pd.notna(std_min): ax.axhline(std_min, color='red', linestyle='--', label=f'Min {std_min}')
        if pd.notna(std_max): ax.axhline(std_max, color='red', linestyle='--', label=f'Max {std_max}')
        
        ax.set_title("Hardness Trend")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ==========================
    # VIEW 3: CORRELATION
    # ==========================
    elif view_mode == "🛠 Hardness → TS/YS/EL":
        # Binning
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub["HRB_bin"] = pd.cut(sub["Hardness_LAB"], bins=bins, labels=labels)
        
        summary = sub.groupby("HRB_bin", observed=True).agg(
            TS=("TS","mean"), YS=("YS","mean"), EL=("EL","mean"), Count=("COIL_NO","count")
        ).dropna()
        
        if summary.empty:
            st.info("Không đủ dữ liệu để vẽ biểu đồ tương quan.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        x_plot = np.arange(len(summary))
        
        ax.plot(x_plot, summary["TS"], marker="o", color="#1f77b4", label="TS")
        ax.plot(x_plot, summary["YS"], marker="s", color="#2ca02c", label="YS")
        
        # Annotations không trùng nhau
        for i, val in enumerate(summary["TS"]):
            ax.annotate(f"{val:.1f}", (i, val), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom', color="#1f77b4", fontweight='bold')
        for i, val in enumerate(summary["YS"]):
            ax.annotate(f"{val:.1f}", (i, val), xytext=(0, -15), textcoords="offset points", ha='center', va='top', color="#2ca02c", fontweight='bold')
            
        ax2 = ax.twinx()
        ax2.plot(x_plot, summary["EL"], marker="^", color="#ff7f0e", label="EL", linestyle='--')
        
        ax.set_xticks(x_plot)
        ax.set_xticklabels(summary.index)
        ax.set_title("Correlation: Hardness vs Mechanical Properties")
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left'); ax2.legend(loc='upper right')
        st.pyplot(fig)

    # ==========================
    # VIEW 4: PREDICT (AUTO-SWITCH FIXED)
    # ==========================
    elif view_mode == "🧮 Predict TS/YS/EL (Custom Hardness)":
        st.write("##### 🔮 Dự báo cơ tính")

        # 1. AUTO-SWITCH LOGIC
        count_line = sub["Hardness_LINE"].count()
        count_lab = sub["Hardness_LAB"].count()
        
        if count_line >= 5:
            x_col = "Hardness_LINE"; x_name = "LINE Hardness"
        elif count_lab >= 5:
            x_col = "Hardness_LAB"; x_name = "LAB Hardness"
            st.info(f"💡 Dữ liệu Line bị thiếu (Line={count_line}). Đang dùng **{x_name}** để dự báo.")
        else:
            st.warning(f"⚠️ Không đủ dữ liệu độ cứng (Line={count_line}, Lab={count_lab}) để dự báo.")
            continue

        # 2. Prepare Data
        sub_fit = sub.dropna(subset=[x_col, "TS", "YS", "EL"]).copy()
        if len(sub_fit) < 5:
            st.warning("⚠️ Không đủ dữ liệu hợp lệ (Cần >5 cuộn có đủ độ cứng & cơ tính).")
            continue
            
        hrb_min_data = float(sub_fit[x_col].min())
        hrb_max_data = float(sub_fit[x_col].max())

        # 3. Input with Unique Keys
        pred_type = st.radio("Input Type", ["Single", "Range"], key=f"rad_{uid}", horizontal=True)
        
        if pred_type == "Single":
            val = st.number_input(f"Nhập {x_name} (HRB)", value=round((hrb_min_data+hrb_max_data)/2, 1), step=0.1, key=f"num_{uid}")
            hrb_values = [val]
        else:
            c1, c2 = st.columns(2)
            with c1: 
                h_min = st.number_input(f"Min {x_name}", value=hrb_min_data, key=f"min_{uid}")
            with c2: 
                h_max = st.number_input(f"Max {x_name}", value=hrb_max_data, key=f"max_{uid}")
            hrb_values = list(np.arange(h_min, h_max + 0.1, 1.0))

        # 4. Predict Button
        if st.button("🚀 Chạy dự báo", key=f"btn_{uid}"):
            pred_res = {}
            for prop in ["TS","YS","EL"]:
                a, b = np.polyfit(sub_fit[x_col], sub_fit[prop], 1)
                pred_res[prop] = a * np.array(hrb_values) + b
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            coils_idx = np.arange(len(sub_fit))
            
            for prop, col, mark in [("TS","blue","o"), ("YS","green","s")]:
                # Vẽ dữ liệu thật
                ax.plot(coils_idx, sub_fit[prop], marker=mark, color=col, alpha=0.3, label=f"{prop} Obs")
                # Vẽ điểm dự báo (nối tiếp)
                pred_vals = pred_res[prop]
                x_pred = coils_idx[-1] + np.arange(1, len(pred_vals)+1)
                ax.scatter(x_pred, pred_vals, color="red", marker="x", s=50)
                # Nối dây
                for i in range(len(pred_vals)):
                    ax.plot([coils_idx[-1], x_pred[i]], [sub_fit[prop].iloc[-1], pred_vals[i]], 'r:', alpha=0.5)

            ax.set_title(f"Prediction based on {x_name}")
            st.pyplot(fig)
            
            # Table Result
            res_df = pd.DataFrame({f"{x_name}": hrb_values})
            for p in ["TS","YS","EL"]: res_df[p] = pred_res[p]
            st.dataframe(res_df.style.format("{:.1f}"), use_container_width=True)

    # ==========================
    # VIEW 5: SUMMARY TABLE
    # ==========================
    elif view_mode == "📊 Hardness → Mechanical Range":
        # Sửa lỗi KeyError & Indentation
        st.write("##### 📋 Hard Bin Mapping Summary")
        
        # Đảm bảo cột tồn tại
        if "Std_Hardness_Range" not in sub.columns:
            sub["Std_Hardness_Range"] = sub["Std_Min"].astype(str) + "~" + sub["Std_Max"].astype(str)
            
        summary_range = sub.groupby(["Product_Spec", "Gauge_Range_Group", "Std_Hardness_Range"]).agg(
            N=("COIL_NO", "count"),
            TS_min=("TS", "min"), TS_max=("TS", "max"), TS_avg=("TS", "mean"),
            YS_min=("YS", "min"), YS_max=("YS", "max"), YS_avg=("YS", "mean"),
            EL_min=("EL", "min"), EL_max=("EL", "max"), EL_avg=("EL", "mean")
        ).reset_index()
        
        st.dataframe(
            summary_range.style.format("{:.1f}", subset=summary_range.columns[4:]), 
            use_container_width=True
        )
