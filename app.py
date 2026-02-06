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
# 2. LOAD DATA & MAPPING LOGIC
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_all_data():
    r1 = requests.get(DATA_URL); r1.encoding = "utf-8"
    raw_df = pd.read_csv(StringIO(r1.text))
    r2 = requests.get(GAUGE_URL); r2.encoding = "utf-8"
    g_df = pd.read_csv(StringIO(r2.text))
    return raw_df, g_df

raw, gauge_df = load_all_data()

# --- Pre-processing & Renaming ---
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
    "TENSILE_TENSILE": "TS", "TENSILE_YIELD": "YS", "TENSILE_ELONG": "EL",
    "Standard TS min": "Standard TS min", "Standard TS max": "Standard TS max",
    "Standard YS min": "Standard YS min", "Standard YS max": "Standard YS max",
    "Standard EL min": "Standard EL min", "Standard EL max": "Standard EL max"
})

# Ép kiểu số
for c in ["Hardness_LAB","Hardness_LINE","TS","YS","EL","Order_Gauge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Tách Standard Hardness
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan
df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
df["Std_Hardness_Range"] = df["Std_Min"].astype(str) + " ~ " + df["Std_Max"].astype(str)

# --- Mapping Gauge Range (Sửa lỗi KeyError: Gauge_Range_Group) ---
gauge_df.columns = gauge_df.columns.str.strip()
gauge_col = next(c for c in gauge_df.columns if "RANGE" in c.upper())

def parse_range(text):
    nums = re.findall(r"\d+\.\d+|\d+", str(text))
    return (float(nums[0]), float(nums[-1])) if len(nums) >= 2 else (None, None)

ranges = []
for _, r in gauge_df.iterrows():
    lo, hi = parse_range(r[gauge_col])
    if lo is not None: ranges.append((lo, hi, r[gauge_col]))

def map_gauge(val):
    for lo, hi, name in ranges:
        if lo <= val < hi: return name # Logic 0.28 <= T < 0.35
    return "Other Groups"

df["Gauge_Range_Group"] = df["Order_Gauge"].apply(map_gauge)

# Filter GE* < 88
df = df[~((df["Quality_Code"].astype(str).str.startswith("GE")) & 
          ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88)))]

# ================================
# 3. SIDEBAR & FILTERS
# ================================
st.sidebar.header("🎛 FILTER")
rolling = st.sidebar.radio("Rolling Type", sorted(df["Rolling_Type"].dropna().unique()))
qgroup = st.sidebar.radio("Quality Group (CQ00/06)", ["CQ00", "CQ06"]) # Có thể gộp nếu muốn

df_filtered = df[(df["Rolling_Type"] == rolling) & (df["Quality_Code"].str.contains(qgroup))]

view_mode = st.sidebar.radio("📊 View Mode", [
    "📋 Data Table", "📈 Trend (LAB / LINE)", "🛠 Hardness → TS/YS/EL", 
    "🧮 Predict TS/YS/EL", "📊 Hardness → Mechanical Range"
])

# ================================
# 4. MAIN LOOP (Sửa lỗi Indentation & Duplicate ID)
# ================================
GROUP_COLS = ["Rolling_Type", "Gauge_Range_Group", "Material"]
valid_groups = df_filtered.groupby(GROUP_COLS).size().reset_index(name='N')
valid_groups = valid_groups[valid_groups['N'] >= 5] # Giảm xuống 5 để test

for idx, g in valid_groups.iterrows():
    # Tạo ID duy nhất cho widget để tránh DuplicateElementId
    uid = f"{g['Material']}_{g['Gauge_Range_Group']}_{idx}".replace(".", "_")
    
    sub = df_filtered[
        (df_filtered["Rolling_Type"] == g["Rolling_Type"]) &
        (df_filtered["Gauge_Range_Group"] == g["Gauge_Range_Group"]) &
        (df_filtered["Material"] == g["Material"])
    ].sort_values("COIL_NO")

    st.subheader(f"🧱 {g['Material']} | Gauge: {g['Gauge_Range_Group']} (n={len(sub)})")

    if view_mode == "📋 Data Table":
        st.dataframe(sub, use_container_width=True)

    elif view_mode == "📈 Trend (LAB / LINE)":
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub["Hardness_LAB"].values, marker="o", label="LAB")
        ax.plot(sub["Hardness_LINE"].values, marker="s", label="LINE")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    elif view_mode == "🛠 Hardness → TS/YS/EL":
        # Annotation logic (Sửa lỗi trùng nhãn)
        summary = sub.groupby(pd.cut(sub["Hardness_LAB"], bins=5)).agg({"TS":"mean", "YS":"mean", "EL":"mean"}).dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(summary))
        ax.plot(x, summary["TS"], marker="o", color="blue", label="TS")
        ax.plot(x, summary["YS"], marker="s", color="green", label="YS")
        for i, v in enumerate(summary["TS"]):
            ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom', color="blue")
        for i, v in enumerate(summary["YS"]):
            ax.annotate(f"{v:.1f}", (i, v), xytext=(0, -15), textcoords="offset points", ha='center', va='top', color="green")
        ax.legend(); st.pyplot(fig)

    elif view_mode == "🧮 Predict TS/YS/EL":
        # Sửa lỗi DuplicateElementId bằng cách thêm key=uid
        pred_type = st.radio("Input Type", ["Single", "Range"], key=f"radio_{uid}")
        if pred_type == "Single":
            val = st.number_input("Enter HRB", value=90.0, key=f"num_{uid}")
            st.write(f"Dự báo cho {val} HRB...")
        # ... (Logic predict của bạn) ...

    elif view_mode == "📊 Hardness → Mechanical Range":
        # Sửa lỗi KeyError và Indentation cho bảng Summary
        st.markdown("#### 📋 Hard Bin Mapping Summary")
        summary_range = sub.groupby(["Product_Spec", "Gauge_Range_Group", "Std_Hardness_Range"]).agg(
            N_coils=("COIL_NO", "count"),
            TS_mean=("TS", "mean"), YS_mean=("YS", "mean"), EL_mean=("EL", "mean")
        ).reset_index()
        st.dataframe(summary_range, use_container_width=True)
