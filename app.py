# ================================
# FULL STREAMLIT APP – FINAL CLEAN VERSION
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import uuid

# ================================
# 1. PAGE CONFIG & UTILS
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# 2. LOAD DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_data():
    # A. Main Data
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    df = pd.read_csv(StringIO(r.text))
    
    # Auto Metadata
    metal_col = next(c for c in df.columns if "METALLIC" in c.upper())
    df["Metallic_Type"] = df[metal_col]
    
    # Rename
    df = df.rename(columns={
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
    
    # Parse Standards
    def split_std(x):
        if isinstance(x, str) and "~" in x:
            try:
                lo, hi = x.split("~")
                return float(lo), float(hi)
            except: pass
        return np.nan, np.nan

    df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
    
    # Numeric Force
    for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    # Quality Group Fix
    df["Quality_Group"] = df["Quality_Code"].replace({"CQ00": "CQ00 / CQ06", "CQ06": "CQ00 / CQ06"})
    
    # Remove GE* < 88
    if "Quality_Code" in df.columns:
        df = df[~(df["Quality_Code"].astype(str).str.startswith("GE") & ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88)))]
        
    # B. Gauge Range Map
    gauge_df = pd.read_csv(GAUGE_URL)
    gauge_df.columns = gauge_df.columns.str.strip()
    gauge_col = next(c for c in gauge_df.columns if "RANGE" in c.upper())
    
    ranges = []
    for _, row in gauge_df.iterrows():
        nums = re.findall(r"\d+\.\d+|\d+", str(row[gauge_col]))
        if len(nums) >= 2:
            ranges.append((float(nums[0]), float(nums[-1]), row[gauge_col]))
            
    def map_gauge(val):
        for lo, hi, name in ranges:
            if lo <= val < hi: return name
        return None

    df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge)
    df = df.dropna(subset=["Gauge_Range"])
    
    return df

# Load Data Once
df = load_data()

# ================================
# 3. SIDEBAR (ĐẶT Ở ĐÂY LÀ ĐÚNG)
# ================================
with st.sidebar:
    try:
        st.image("image_4.png", use_container_width=True)
    except: pass
    
    st.title("🎛️ Control Panel")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    st.divider()
    st.header("Filters")
    
    # Filters
    col_rolling = "Rolling_Type" if "Rolling_Type" in df.columns else "Claasify material"
    rolling = st.radio("Rolling Type", sorted(df[col_rolling].dropna().unique()))
    
    col_metal = "Metallic_Type" if "Metallic_Type" in df.columns else "METALLIC COATING TYPE"
    metal = st.radio("Metallic Type", sorted(df[col_metal].dropna().unique()))
    
    col_group = "Quality_Group" if "Quality_Group" in df.columns else "Quality Group"
    qgroup = st.radio("Quality Group", sorted(df[col_group].dropna().unique()))
    
    # --- CREATE 'SUB' DATAFRAME HERE ---
    sub = df[
        (df[col_rolling] == rolling) & 
        (df[col_metal] == metal) & 
        (df[col_group] == qgroup)
    ]
    
    st.divider()
    
    # --- DEFINE 'VIEW_MODE' HERE (BEFORE IT IS USED) ---
    view_mode = st.radio(
        "📊 View Mode",
        [
            "📋 Data Inspection",
            "📉 Hardness Analysis (Trend & Dist)",
            "🔗 Correlation: Hardness vs Mech Props",
            "⚙️ Mech Props Analysis",
            "🔍 Lookup: Hardness Range → Actual Mech Props",
            "🎯 Find Target Hardness (Reverse Lookup)",
        ]
    )
    
    st.info(f"Found: {len(sub)} coils")

# ================================
# 4. MAIN CONTENT LOGIC
# ================================
st.title("📊 SPC Hardness – Material / Gauge Level Analysis")

# Check Data
GROUP_COLS = ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"]
cnt = sub.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 30]

if valid.empty:
    st.warning("⚠️ No group with ≥30 coils found for the selected filters.")
    st.stop()

# Helper for Analysis Text
def analyze_distribution(series, name, unit="MPa"):
    try:
        mean = series.mean()
        median = series.median()
        std = series.std()
        skew = series.skew()
        
        if skew > 0.5: skew_text, skew_icon = "Right Skewed (High values dominant)", "↗️"
        elif skew < -0.5: skew_text, skew_icon = "Left Skewed (Low values dominant)", "↙️"
        else: skew_text, skew_icon = "Symmetrical (Normal Distribution)", "↔️"

        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = len(series[(series < lower) | (series > upper)])
        range_val = series.max() - series.min()

        return f"""
        **{name} Statistics:**
        - **Mean:** {mean:.1f} {unit} | **Median:** {median:.1f} {unit}
        - **Distribution:** {skew_icon} {skew_text}
        - **Stability (StdDev):** {std:.2f} (Range: {range_val:.1f})
        - **Outliers:** {outliers} coils (outside {lower:.0f}~{upper:.0f})
        """
    except: return "Insufficient data."

# --- MAIN LOOP ---
for _, g in valid.iterrows():
    # Filter group data
    g_sub = sub[
        (sub["Gauge_Range"] == g["Gauge_Range"]) &
        (sub["Material"] == g["Material"])
    ].sort_values("COIL_NO")
    
    lo, hi = g_sub.iloc[0][["Std_Min","Std_Max"]]
    g_sub["NG"] = (g_sub["Hardness_LAB"] < lo) | (g_sub["Hardness_LAB"] > hi) | (g_sub["Hardness_LINE"] < lo) | (g_sub["Hardness_LINE"] > hi)
    qa = "FAIL" if g_sub["NG"].any() else "PASS"
    specs = ", ".join(sorted(g_sub["Product_Spec"].unique()))

    st.markdown(f"""
    ### 🧱 Quality Group: {g['Quality_Group']}
    **Material:** {g['Material']} | **Gauge:** {g['Gauge_Range']} | **Specs:** {specs}  
    **Coils:** {len(g_sub)} | **QA:** 🧪 **{qa}** | **Limit:** {lo:.1f} ~ {hi:.1f} HRB
    """)

    # ========================================================
    # VIEW MODES
    # ========================================================
    
    # 1. DATA INSPECTION
    if view_mode == "📋 Data Inspection":
        st.dataframe(g_sub, use_container_width=True)

    # 2. HARDNESS ANALYSIS
    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        t1, t2 = st.tabs(["📈 Trend", "📊 Distribution"])
        
        with t1: # Trend
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(1, len(g_sub)+1)
            ax.plot(x, g_sub["Hardness_LAB"], marker="o", label="LAB", alpha=0.7)
            ax.plot(x, g_sub["Hardness_LINE"], marker="s", label="LINE", alpha=0.7)
            ax.axhline(lo, color="red", ls="--", label=f"LSL {lo}")
            ax.axhline(hi, color="red", ls="--", label=f"USL {hi}")
            ax.legend(ncol=4)
            ax.set_title("Hardness Trend")
            st.pyplot(fig)
            
        with t2: # Distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(g_sub["Hardness_LAB"].dropna(), bins=20, alpha=0.5, label="LAB", density=True)
            ax.hist(g_sub["Hardness_LINE"].dropna(), bins=20, alpha=0.5, label="LINE", density=True)
            ax.axvline(lo, color="red", ls="--"); ax.axvline(hi, color="red", ls="--")
            ax.legend()
            ax.set_title("Hardness Distribution")
            st.pyplot(fig)

    # 3. CORRELATION
    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        st.info("Correlation Analysis - Shows average properties per Hardness bin.")
        # Binning
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        g_sub["HRB_bin"] = pd.cut(g_sub["Hardness_LAB"], bins=bins, labels=labels)
        
        summ = g_sub.groupby("HRB_bin", observed=True)[["TS","YS","EL"]].mean().dropna()
        
        if not summ.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(summ.index.astype(str), summ["TS"], marker="o", label="TS")
            ax.plot(summ.index.astype(str), summ["YS"], marker="s", label="YS")
            ax.plot(summ.index.astype(str), summ["EL"], marker="^", label="EL")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Correlation: Hardness vs Properties")
            st.pyplot(fig)
        else:
            st.warning("Not enough data.")

    # 4. MECH PROPS ANALYSIS
    elif view_mode == "⚙️ Mech Props Analysis":
        st.info("Mechanical Properties SPC & Distribution (±5 Sigma View)")
        
        c_ts, c_ys, c_el = st.columns(3)
        configs = [("TS", "#1f77b4", c_ts), ("YS", "#2ca02c", c_ys), ("EL", "#ff7f0e", c_el)]
        
        for col, color, col_ui in configs:
            with col_ui:
                data = g_sub[col].dropna()
                if len(data) > 1:
                    mean, std = data.mean(), data.std()
                    # Plot
                    fig, ax = plt.subplots(figsize=(4, 5))
                    ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color, alpha=0.5))
                    # Jitter
                    y = data
                    x = np.random.normal(1, 0.04, size=len(y))
                    ax.scatter(x, y, alpha=0.6, color=color, s=20)
                    ax.set_title(f"{col} Dist")
                    st.pyplot(fig)
                    # Text Analysis
                    st.info(analyze_distribution(data, col, "%" if col=="EL" else "MPa"))

    # 5. LOOKUP: HARDNESS -> MECH PROPS
    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        st.markdown("#### 🔍 Historical Lookup")
        c1, c2 = st.columns(2)
        h_min = st.number_input("Min HRB", value=58.0, step=0.5, key=f"hmin_{_}")
        h_max = st.number_input("Max HRB", value=65.0, step=0.5, key=f"hmax_{_}")
        
        matches = g_sub[(g_sub["Hardness_LINE"] >= h_min) & (g_sub["Hardness_LINE"] <= h_max)]
        
        if not matches.empty:
            st.success(f"Found {len(matches)} coils.")
            st.dataframe(matches[["COIL_NO", "Hardness_LINE", "TS", "YS", "EL"]], height=200)
            
            # Mini charts
            cols = st.columns(3)
            for i, col in enumerate(["TS", "YS", "EL"]):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.hist(matches[col].dropna(), bins=10, alpha=0.7)
                    ax.set_title(col)
                    st.pyplot(fig)
        else:
            st.warning("No coils found in this range.")

    # 6. REVERSE LOOKUP: TARGET HARDNESS
    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        st.markdown("#### 🎯 Find Target Hardness")
        c1, c2, c3 = st.columns(3)
        req_ys_min = c1.number_input("Min YS", value=250.0, step=5.0, key=f"rymin_{_}")
        req_ts_min = c2.number_input("Min TS", value=350.0, step=5.0, key=f"rtmin_{_}")
        req_el_min = c3.number_input("Min EL", value=0.0, step=1.0, key=f"remin_{_}")
        
        safe = g_sub[
            (g_sub["YS"] >= req_ys_min) &
            (g_sub["TS"] >= req_ts_min) &
            (g_sub["EL"] >= req_el_min)
        ]
        
        if not safe.empty:
            rec_min = safe["Hardness_LINE"].min()
            rec_max = safe["Hardness_LINE"].max()
            st.success(f"✅ Safe Hardness Window: **{rec_min:.1f} - {rec_max:.1f} HRB** (Based on {len(safe)} coils)")
            st.dataframe(safe[["COIL_NO", "Hardness_LINE", "YS", "TS", "EL"]])
        else:
            st.error("No coils meet these requirements.")

    st.divider() # Separator between groups
