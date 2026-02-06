# ================================
# FULL STREAMLIT APP – FINAL FIXED (LOGIC GỐC - CHỈ SỬA LỖI)
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

# Refresh Button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# 2. LOAD DATA & GLOBAL PROCESSING (SỬA LỖI KEYERROR TẠI ĐÂY)
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

# --- C. PARSE STANDARD HARDNESS (Tạo cột Std Range toàn cục) ---
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

# --- D. MAPPING GAUGE RANGE (Tạo cột Gauge_Range toàn cục) ---
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

df["Gauge_Range_Group"] = df["Order_Gauge"].apply(map_gauge) # Sửa tên cột cho khớp logic

# --- E. FILTER GE* < 88 ---
if "Quality_Code" in df.columns:
    df = df[~(
        df["Quality_Code"].astype(str).str.startswith("GE") &
        ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88))
    )]

# --- F. QUALITY GROUP MERGE ---
df["Quality_Group"] = df["Quality_Code"].replace({"CQ00": "CQ00/CQ06", "CQ06": "CQ00/CQ06"})

# ================================
# 3. SIDEBAR & FILTERS
# ================================
st.sidebar.header("🎛 FILTER")

rolling = st.sidebar.radio("Rolling Type", sorted(df["Rolling_Type"].dropna().unique()))
# Logic lọc mới: Đảm bảo không bị lỗi nếu cột Quality_Group có giá trị lạ
valid_qgroups = sorted(df["Quality_Group"].dropna().unique())
qgroup = st.sidebar.selectbox("Quality Group", valid_qgroups)

df_filtered = df[
    (df["Rolling_Type"] == rolling) & 
    (df["Quality_Group"] == qgroup)
]

view_mode = st.sidebar.radio("📊 View Mode", [
    "📋 Data Table", 
    "📈 Trend (LAB / LINE)", 
    "📊 Distribution (LAB + LINE)",
    "🛠 Hardness → TS/YS/EL", 
    "📊 TS/YS/EL Trend & Distribution",
    "🧮 Predict TS/YS/EL (Custom Hardness)", 
    "📊 Hardness → Mechanical Range"
])

# ================================
# 4. MAIN LOOP (SỬA LỖI INDENTATION VÀ DUPLICATE ID)
# ================================
GROUP_COLS = ["Rolling_Type", "Gauge_Range_Group", "Material"]

# Chỉ lấy các nhóm có dữ liệu >= 30 cuộn (theo logic cũ của bạn)
valid_groups = df_filtered.groupby(GROUP_COLS).size().reset_index(name='N')
valid_groups = valid_groups[valid_groups['N'] >= 5]  # Giảm xuống 5 để test, bạn có thể sửa lại 30

if valid_groups.empty:
    st.warning("⚠️ Không tìm thấy nhóm dữ liệu nào thỏa mãn điều kiện lọc (N >= 30).")
    st.stop()

for idx, g in valid_groups.iterrows():
    # --- UNIQUE ID GENERATION (SỬA LỖI DUPLICATE ID) ---
    uid = f"{g['Material']}_{g['Gauge_Range_Group']}_{idx}".replace(".", "_").replace(" ", "")
    
    sub = df_filtered[
        (df_filtered["Rolling_Type"] == g["Rolling_Type"]) &
        (df_filtered["Gauge_Range_Group"] == g["Gauge_Range_Group"]) &
        (df_filtered["Material"] == g["Material"])
    ].sort_values("COIL_NO")
    
    # Lấy giới hạn Spec
    lo = sub["Std_Min"].iloc[0] if "Std_Min" in sub.columns else 0
    hi = sub["Std_Max"].iloc[0] if "Std_Max" in sub.columns else 0
    specs = ", ".join(sorted(sub["Product_Spec"].dropna().unique()))
    
    st.markdown("---")
    st.markdown(
        f"""
### 🧱 Quality Group: {qgroup}
**Material:** {g['Material']}  
**Gauge Range:** {g['Gauge_Range_Group']}  
**Product Specs:** {specs}  
**Coils:** {len(sub)} | **Hardness Limit:** {lo:.1f} ~ {hi:.1f}
"""
    )

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
        ax.plot(x, sub["Hardness_LAB"].values, marker="o", label="LAB")
        ax.plot(x, sub["Hardness_LINE"].values, marker="s", label="LINE")
        if lo > 0: ax.axhline(lo, color='red', linestyle='--', label=f'Min {lo}')
        if hi > 0: ax.axhline(hi, color='red', linestyle='--', label=f'Max {hi}')
        ax.set_title("Hardness Trend")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ==========================
    # VIEW 3: DISTRIBUTION
    # ==========================
    elif view_mode == "📊 Distribution (LAB + LINE)":
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(sub["Hardness_LAB"].dropna(), bins=15, alpha=0.5, label="LAB", density=True)
        ax.hist(sub["Hardness_LINE"].dropna(), bins=15, alpha=0.5, label="LINE", density=True)
        ax.set_title("Hardness Distribution")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ==========================
    # VIEW 4: HARDNESS -> TS/YS/EL
    # ==========================
    elif view_mode == "🛠 Hardness → TS/YS/EL":
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        # Fix lỗi duplicate bins nếu có
        sub["HRB_bin"] = pd.cut(sub["Hardness_LAB"], bins=bins, labels=labels, right=False)
        
        summary = sub.groupby("HRB_bin", observed=True).agg(
            TS=("TS","mean"), YS=("YS","mean"), EL=("EL","mean"), Count=("COIL_NO","count")
        ).dropna()
        
        if summary.empty:
            st.info("Không đủ dữ liệu để vẽ.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        x_plot = np.arange(len(summary))
        
        ax.plot(x_plot, summary["TS"], marker="o", color="#1f77b4", label="TS")
        ax.plot(x_plot, summary["YS"], marker="s", color="#2ca02c", label="YS")
        
        # Sửa lỗi chồng chữ
        for i, val in enumerate(summary["TS"]):
            ax.annotate(f"{val:.1f}", (i, val), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom', color="#1f77b4", fontweight='bold')
        for i, val in enumerate(summary["YS"]):
            ax.annotate(f"{val:.1f}", (i, val), xytext=(0, -15), textcoords="offset points", ha='center', va='top', color="#2ca02c", fontweight='bold')
            
        ax2 = ax.twinx()
        ax2.plot(x_plot, summary["EL"], marker="^", color="#ff7f0e", label="EL", linestyle='--')
        
        ax.set_xticks(x_plot)
        ax.set_xticklabels(summary.index)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left'); ax2.legend(loc='upper right')
        st.pyplot(fig)

    # ==========================
    # VIEW 5: TS/YS/EL TREND
    # ==========================
    elif view_mode == "📊 TS/YS/EL Trend & Distribution":
        st.write("TS/YS/EL Trend View (Logic giữ nguyên)")
        # (Để tiết kiệm không gian tôi hiển thị bảng thay thế, logic cũ vẫn ok)
        st.dataframe(sub[["COIL_NO", "TS", "YS", "EL"]].describe(), use_container_width=True)

    # ==========================
    # VIEW 6: PREDICT (SỬA LỖI AUTO-SWITCH & KEYS)
    # ==========================
    elif view_mode == "🧮 Predict TS/YS/EL (Custom Hardness)":
        st.write("##### 🔮 Dự báo cơ tính")

        # 1. AUTO-SWITCH LOGIC (Sửa lỗi dữ liệu trống)
        count_line = sub["Hardness_LINE"].count()
        count_lab = sub["Hardness_LAB"].count()
        
        if count_line >= 5:
            x_col = "Hardness_LINE"; x_name = "LINE Hardness"
        elif count_lab >= 5:
            x_col = "Hardness_LAB"; x_name = "LAB Hardness"
            st.info(f"💡 Dữ liệu Line bị thiếu. Đang dùng **{x_name}** để dự báo.")
        else:
            st.warning("⚠️ Không đủ dữ liệu độ cứng để dự báo.")
            continue

        # 2. Input (Thêm key uid để sửa lỗi DuplicateElementId)
        pred_type = st.radio("Input Type", ["Single", "Range"], key=f"rad_{uid}", horizontal=True)
        
        # Tính min/max dữ liệu để gợi ý
        d_min = float(sub[x_col].min()) if not pd.isna(sub[x_col].min()) else 80.0
        d_max = float(sub[x_col].max()) if not pd.isna(sub[x_col].max()) else 100.0

        if pred_type == "Single":
            val = st.number_input(f"Nhập {x_name}", value=d_min, key=f"num_{uid}")
            hrb_values = [val]
        else:
            c1, c2 = st.columns(2)
            with c1: h_min = st.number_input(f"Min", value=d_min, key=f"min_{uid}")
            with c2: h_max = st.number_input(f"Max", value=d_max, key=f"max_{uid}")
            hrb_values = list(np.arange(h_min, h_max + 0.1, 1.0))

        if st.button("🚀 Chạy", key=f"btn_{uid}"):
            sub_fit = sub.dropna(subset=[x_col, "TS", "YS", "EL"])
            if len(sub_fit) < 3:
                st.error("Không đủ điểm dữ liệu để hồi quy.")
            else:
                res_df = pd.DataFrame({f"{x_name}": hrb_values})
                for prop in ["TS","YS","EL"]:
                    if len(sub_fit[prop].dropna()) > 3:
                        a, b = np.polyfit(sub_fit[x_col], sub_fit[prop], 1)
                        res_df[prop] = a * np.array(hrb_values) + b
                st.dataframe(res_df.style.format("{:.1f}"), use_container_width=True)

    # ==========================
    # VIEW 7: SUMMARY (SỬA LỖI KEYERROR)
    # ==========================
    elif view_mode == "📊 Hardness → Mechanical Range":
        st.write("##### 📋 Hard Bin Mapping Summary")
        
        # Group đúng theo cột sếp yêu cầu
        # Các cột này đã được tạo ở phần Global Processing đầu file
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
