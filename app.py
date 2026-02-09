# ================================
# FULL STREAMLIT APP – FINAL COMPLETE VERSION
# INTEGRATED GLOBAL DASHBOARD + SMART LOGIC
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
#  🚀 GLOBAL SUMMARY DASHBOARD
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Summary & Forecast Dashboard")
    st.info("Bảng tổng hợp toàn bộ các nhóm vật liệu (Material/Gauge). Cột **'Specs List'** hiển thị các tiêu chuẩn chi tiết.")

    summary_rows = []

    for _, g in valid.iterrows():
        sub_grp = df[
            (df["Rolling_Type"] == g["Rolling_Type"]) &
            (df["Metallic_Type"] == g["Metallic_Type"]) &
            (df["Quality_Group"] == g["Quality_Group"]) &
            (df["Gauge_Range"] == g["Gauge_Range"]) &
            (df["Material"] == g["Material"])
        ].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

        if len(sub_grp) < 10: continue

        specs_list = sorted(sub_grp["Product_Spec"].astype(str).unique())
        specs_str = ", ".join(specs_list)

        avg_h = sub_grp["Hardness_LINE"].mean()
        avg_ts = sub_grp["TS"].mean()
        avg_ys = sub_grp["YS"].mean()
        avg_el = sub_grp["EL"].mean()

        X = sub_grp[["Hardness_LINE"]].values
        
        m_ts = LinearRegression().fit(X, sub_grp["TS"].values)
        pred_ts = m_ts.predict([[avg_h]])[0]
        r2_ts = r2_score(sub_grp["TS"], m_ts.predict(X))

        m_ys = LinearRegression().fit(X, sub_grp["YS"].values)
        pred_ys = m_ys.predict([[avg_h]])[0]

        m_el = LinearRegression().fit(X, sub_grp["EL"].values)
        pred_el = m_el.predict([[avg_h]])[0]

        summary_rows.append({
            "Quality Group": g["Quality_Group"],
            "Material": g["Material"],
            "Gauge Range": g["Gauge_Range"],
            "Specs List": specs_str,
            "Coils": len(sub_grp),
            "HRB Avg": float(f"{avg_h:.1f}"),
            "TS (Act)": float(f"{avg_ts:.0f}"),
            "TS (Model)": float(f"{pred_ts:.0f}"),
            "TS Diff": float(f"{avg_ts - pred_ts:.0f}"),
            "TS R2": float(f"{r2_ts:.2f}"),
            "YS (Act)": float(f"{avg_ys:.0f}"),
            "YS (Model)": float(f"{pred_ys:.0f}"),
            "YS Diff": float(f"{avg_ys - pred_ys:.0f}"),
            "EL (Act)": float(f"{avg_el:.1f}"),
            "EL (Model)": float(f"{pred_el:.1f}"),
            "EL Diff": float(f"{avg_el - pred_el:.1f}"),
        })

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        cols_order = [
            "Quality Group", "Material", "Gauge Range", "Specs List", "Coils", "HRB Avg",
            "TS (Act)", "TS (Model)", "TS Diff", "TS R2",
            "YS (Act)", "YS (Model)", "YS Diff",
            "EL (Act)", "EL (Model)", "EL Diff"
        ]
        df_sum = df_sum[cols_order]

        st.dataframe(
            df_sum.style.background_gradient(subset=["TS Diff", "YS Diff"], cmap="coolwarm", vmin=-20, vmax=20)
                  .format("{:.1f}", subset=["EL (Act)", "EL (Model)", "EL Diff"])
                  .format("{:.0f}", subset=["TS (Act)", "TS (Model)", "TS Diff", "YS (Act)", "YS (Model)", "YS Diff"]),
            use_container_width=True, height=800
        )
        st.success(f"✅ Đã phân tích xong {len(df_sum)} nhóm điều kiện.")
    else:
        st.warning("⚠️ Không có đủ dữ liệu để tạo bảng tổng hợp.")
    st.stop()

# ==============================================================================
# MAIN LOOP (DETAILS)
# ==============================================================================
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

    st.markdown(f"### 🧱 {g['Quality_Group']} | {g['Material']} | {g['Gauge_Range']}")
    st.markdown(f"**Specs:** {specs} | **Coils:** {sub['COIL_NO'].nunique()} | **Limit:** {lo:.1f}~{hi:.1f}")

    # ================================
    # 1. DATA INSPECTION
    # ================================
    if view_mode == "📋 Data Inspection":
        st.dataframe(sub, use_container_width=True)

    # ================================
    # 2. HARDNESS ANALYSIS
    # ================================
    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        tab_trend, tab_dist = st.tabs(["📈 Trend", "📊 Distribution"])
        
        with tab_trend:
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, sub["Hardness_LAB"], marker="o", label="LAB")
            ax.plot(x, sub["Hardness_LINE"], marker="s", label="LINE")
            ax.axhline(lo, color="red", ls="--"); ax.axhline(hi, color="red", ls="--")
            ax.set_title("Hardness Trend"); ax.legend()
            st.pyplot(fig)
            
        with tab_dist:
            lab = sub["Hardness_LAB"].dropna()
            line = sub["Hardness_LINE"].dropna()
            if len(lab) > 5:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(lab, alpha=0.5, density=True, label="LAB")
                ax.hist(line, alpha=0.5, density=True, label="LINE")
                ax.axvline(lo, color="red", ls="--"); ax.axvline(hi, color="red", ls="--")
                ax.legend(); ax.set_title("Hardness Distribution")
                st.pyplot(fig)

    # ================================
    # 3. CORRELATION (FULL VERSION)
    # ================================
    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
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
            
            # Plot Helper
            def plot_prop(x, y, ymin, ymax, c, lbl, m):
                ax.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax.fill_between(x, ymin, ymax, color=c, alpha=0.1)

            plot_prop(x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS", "o")
            plot_prop(x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS", "s")
            plot_prop(x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL", "^")

            # Annotations
            for i, row in enumerate(summary.itertuples()):
                ax.annotate(f"{row.TS_mean:.0f}", (x[i], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", color="#1f77b4")
                
                # EL Check
                el_spec = row.Std_EL_min
                is_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                lbl = f"{row.EL_mean:.1f}%" + ("❌" if is_fail else "")
                clr = "red" if is_fail else "#ff7f0e"
                ax.annotate(lbl, (x[i], row.EL_mean), xytext=(0,-15), textcoords="offset points", ha="center", color=clr, fontweight="bold" if is_fail else "normal")

            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_title("Hardness vs Mechanical Properties"); ax.grid(True, ls="--", alpha=0.5); ax.legend()
            st.pyplot(fig)
            
            # Quick Conclusion Text (RESTORED)
            st.markdown("#### 📌 Quick Conclusion per Hardness Bin")
            for row in summary.itertuples():
                def check(val_min, val_max, spec_min, spec_max):
                    p_min = (val_min >= spec_min) if (pd.notna(spec_min) and spec_min > 0) else True
                    p_max = (val_max <= spec_max) if (pd.notna(spec_max) and spec_max > 0) else True
                    return "✅" if (p_min and p_max) else "⚠️"
                
                ts_f = check(row.TS_min, row.TS_max, row.Std_TS_min, row.Std_TS_max)
                el_f = check(row.EL_min, row.EL_max, row.Std_EL_min, row.Std_EL_max)
                st.write(f"**{row.HRB_bin}**: TS={ts_f} ({row.TS_min:.0f}-{row.TS_max:.0f}) | EL={el_f} ({row.EL_min:.1f}-{row.EL_max:.1f})")

    # ================================
    # 4. MECH PROPS ANALYSIS
    # ================================
    elif view_mode == "⚙️ Mech Props Analysis":
        sub_mech = sub.dropna(subset=["TS","YS","EL"])
        if not sub_mech.empty:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, (col, c) in enumerate([("TS","#1f77b4"),("YS","#2ca02c"),("EL","#ff7f0e")]):
                data = sub_mech[col]
                mean, std = data.mean(), data.std()
                axes[i].hist(data, bins=15, color=c, alpha=0.5, density=True)
                if std > 0:
                    x_p = np.linspace(mean-4*std, mean+4*std, 100)
                    y_p = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2)
                    axes[i].plot(x_p, y_p, color=c, lw=2)
                axes[i].set_title(f"{col} Distribution")
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
