# ==============================================================================
# FULL APP V1.3 - ADDED SMART PREDICTION MODULE
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import uuid
# Thêm thư viện Machine Learning đơn giản
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="SPC Dashboard", layout="wide")

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# 2. LOAD DỮ LIỆU
@st.cache_data
def load_data():
    DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"
    GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"
    
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    df = pd.read_csv(StringIO(r.text))
    
    metal_col = next(c for c in df.columns if "METALLIC" in c.upper())
    df = df.rename(columns={
        metal_col: "Metallic_Type",
        "PRODUCT SPECIFICATION CODE": "Product_Spec",
        "HR STEEL GRADE": "Material",
        "Claasify material": "Rolling_Type",
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
    
    for c in ["Hardness_LAB","Hardness_LINE","YS","TS","EL","Order_Gauge"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    def split_std(x):
        try: return float(x.split("~")[0]), float(x.split("~")[1])
        except: return np.nan, np.nan
    df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
    
    df["Quality_Group"] = df["Quality_Code"].replace({"CQ00": "CQ00 / CQ06", "CQ06": "CQ00 / CQ06"})
    
    g_df = pd.read_csv(GAUGE_URL)
    g_col = next(c for c in g_df.columns if "RANGE" in c.upper())
    ranges = []
    for _, row in g_df.iterrows():
        nums = re.findall(r"\d+\.\d+|\d+", str(row[g_col]))
        if len(nums) >= 2: ranges.append((float(nums[0]), float(nums[-1]), row[g_col]))
        
    def map_gauge(val):
        for lo, hi, name in ranges:
            if lo <= val < hi: return name
        return None
    df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge)
    
    return df.dropna(subset=["Gauge_Range"])

df = load_data()

# 3. SIDEBAR
with st.sidebar:
    try: st.image("image_4.png", use_container_width=True)
    except: pass
    
    st.title("🎛️ Control Panel")
    if st.button("🔄 Refresh Data"): st.cache_data.clear(); st.rerun()
    
    st.header("Filters")
    col_roll = "Rolling_Type" if "Rolling_Type" in df.columns else "Claasify material"
    rolling = st.radio("Rolling Type", sorted(df[col_roll].dropna().unique()))
    
    col_metal = "Metallic_Type" if "Metallic_Type" in df.columns else "METALLIC COATING TYPE"
    metal = st.radio("Metallic Type", sorted(df[col_metal].dropna().unique()))
    
    col_group = "Quality_Group" if "Quality_Group" in df.columns else "Quality Group"
    qgroup = st.radio("Quality Group", sorted(df[col_group].dropna().unique()))
    
    sub = df[
        (df[col_roll] == rolling) &
        (df[col_metal] == metal) &
        (df[col_group] == qgroup)
    ]
    
    st.divider()
    
    view_mode = st.radio(
        "📊 View Mode",
        [
            "📋 Data Inspection",
            "📉 Hardness Analysis (Trend & Dist)",
            "🔗 Correlation: Hardness vs Mech Props",
            "⚙️ Mech Props Analysis",
            "🔍 Lookup: Hardness Range → Actual Mech Props",
            "🎯 Find Target Hardness (Reverse Lookup)",
            "🧮 Predict TS/YS/EL from Std Hardness" # <--- TÍNH NĂNG MỚI
        ]
    )
    st.info(f"Found: {len(sub)} coils")

# 4. MAIN CONTENT
st.title("📊 SPC Hardness Dashboard")

GROUP_COLS = ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"]
valid_groups = sub.groupby(GROUP_COLS).agg(N=("COIL_NO","nunique")).reset_index()
valid_groups = valid_groups[valid_groups["N"] >= 30]

if valid_groups.empty:
    st.warning("⚠️ No data groups with >30 coils found.")
    st.stop()

def analyze_dist(series, name, unit=""):
    try:
        m, med, s = series.mean(), series.median(), series.std()
        skew = series.skew()
        icon = "↗️" if skew > 0.5 else ("↙️" if skew < -0.5 else "↔️")
        return f"""
        **{name}**: Mean={m:.1f} | Median={med:.1f} | Std={s:.2f}
        Dist: {icon} (Skew: {skew:.2f})
        """
    except: return "No Data"

for _, g in valid_groups.iterrows():
    g_sub = sub[
        (sub["Gauge_Range"] == g["Gauge_Range"]) &
        (sub["Material"] == g["Material"])
    ].sort_values("COIL_NO")
    
    lo, hi = g_sub.iloc[0][["Std_Min","Std_Max"]]
    qa_status = "FAIL" if ((g_sub["Hardness_LAB"] < lo) | (g_sub["Hardness_LAB"] > hi)).any() else "PASS"

    st.markdown(f"""
    ### 🧱 {g['Quality_Group']} | {g['Material']} | {g['Gauge_Range']}
    **Limit:** {lo}~{hi} HRB | **QA:** {qa_status} | **Coils:** {len(g_sub)}
    """)

    # ---------------- VIEW MODES ----------------
    
    if view_mode == "📋 Data Inspection":
        st.dataframe(g_sub)

    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Trend")
            st.line_chart(g_sub[["Hardness_LAB","Hardness_LINE"]].reset_index(drop=True))
        with c2:
            st.caption("Distribution")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(g_sub["Hardness_LAB"].dropna(), bins=15, alpha=0.5, label="LAB")
            ax.hist(g_sub["Hardness_LINE"].dropna(), bins=15, alpha=0.5, label="LINE")
            ax.legend()
            st.pyplot(fig)

    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        st.info("Correlation: Hardness Bins vs Average Properties")
        g_sub["Bin"] = pd.cut(g_sub["Hardness_LAB"], bins=np.arange(50, 100, 2))
        summ = g_sub.groupby("Bin", observed=True)[["TS","YS","EL"]].mean()
        st.line_chart(summ)

    elif view_mode == "⚙️ Mech Props Analysis":
        cols = st.columns(3)
        for i, col in enumerate(["TS","YS","EL"]):
            with cols[i]:
                data = g_sub[col].dropna()
                if not data.empty:
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.boxplot(data, patch_artist=True)
                    ax.set_title(col)
                    st.pyplot(fig)
                    st.info(analyze_dist(data, col))

    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        c1, c2 = st.columns(2)
        h_min = c1.number_input(f"Min HRB {uuid.uuid4()}", value=58.0)
        h_max = c2.number_input(f"Max HRB {uuid.uuid4()}", value=65.0)
        matches = g_sub[(g_sub["Hardness_LINE"] >= h_min) & (g_sub["Hardness_LINE"] <= h_max)]
        if not matches.empty:
            st.success(f"Found {len(matches)} coils")
            st.dataframe(matches[["COIL_NO","Hardness_LINE","TS","YS","EL"]])
        else:
            st.warning("No data found")

    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        c1, c2 = st.columns(3)
        req_ys = c1.number_input(f"Min YS {uuid.uuid4()}", value=300.0)
        req_ts = c2.number_input(f"Min TS {uuid.uuid4()}", value=400.0)
        req_el = c3.number_input(f"Min EL {uuid.uuid4()}", value=20.0)
        
        safe = g_sub[(g_sub["YS"] >= req_ys) & (g_sub["TS"] >= req_ts) & (g_sub["EL"] >= req_el)]
        if not safe.empty:
            rec_min, rec_max = safe["Hardness_LINE"].min(), safe["Hardness_LINE"].max()
            st.success(f"✅ Target Hardness: {rec_min:.1f} - {rec_max:.1f} HRB (Based on {len(safe)} coils)")
            st.dataframe(safe[["COIL_NO", "Hardness_LINE", "YS", "TS", "EL"]])
        else:
            st.error("No coils match these specs.")

   "
