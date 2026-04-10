# ==============================================================================
# FULL STREAMLIT APP – ULTIMATE FINAL VERSION
# INCLUDES: All Views, Master Dictionary, Auto-Mapping, Zero/NaN Filter
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid

# ================================
# PAGE CONFIG & CSS
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard", layout="wide")
st.title("📊 Hardness – Visual Analytics Dashboard")

def add_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #ffffff; box-shadow: 2px 0 5px rgba(0,0,0,0.05); border-right: none; }
        h1, h2, h3 { color: #2c3e50 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 600; }
        [data-testid="stMetricValue"] { background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #007bff; }
        thead tr th:first-child {display:none}
        tbody th {display:none}
        .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)
add_custom_css()

# ================================
# UTILS
# ================================
def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# LOAD DATA & PRE-PROCESSING
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1GdnY09hJ2qVHuEBAIJ-eU6B5z8ZdgcGf4P7ZjlAt4JI/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# 1. Date Handling
data_period_str = "N/A"
if "PRODUCTION DATE" in raw.columns:
    raw["PRODUCTION DATE"] = pd.to_datetime(raw["PRODUCTION DATE"], errors='coerce')
    min_date = raw["PRODUCTION DATE"].min()
    max_date = raw["PRODUCTION DATE"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        data_period_str = f"{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}"

current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <strong>🕒 Report Generated:</strong> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <strong>📅 Data Period:</strong> {data_period_str}
</div>
""", unsafe_allow_html=True)

# 2. Rename & Auto-Mapping (Xử lý tên cột tiếng Hoa/tiếng Anh dài)
if any("METALLIC" in str(c).upper() for c in raw.columns):
    metal_col = next(c for c in raw.columns if "METALLIC" in str(c).upper())
    raw["Metallic_Type"] = raw[metal_col]

rename_mapping = {
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "Claasify material": "Rolling_Type",
    "TOP COATMASS": "Top_Coatmass",
    "ORDER GAUGE": "Order_Gauge",
    "COIL NO": "COIL_NO",
    "QUALITY_CODE": "Quality_Code",
    "Standard Hardness": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "TENSILE_YIELD": "YS",
    "TENSILE_TENSILE": "TS",
    "TENSILE_ELONG": "EL",
    "Standard TS min": "Standard TS min",
    "Standard TS max": "Standard TS max",
    "Standard YS min": "Standard YS min",
    "Standard YS max": "Standard YS max",
    "Standard EL min": "Standard EL min",
    "Standard EL max": "Standard EL max"
}
df = raw.rename(columns=rename_mapping)

# Xử lý riêng độ cứng LINE: Ưu tiên N -> C -> S để tránh bị trùng lặp cột (Duplicate columns)
if "HARDNESS 鍍鋅線 N" in df.columns:
    df["Hardness_LINE"] = df["HARDNESS 鍍鋅線 N"]
elif "HARDNESS 鍍鋅線 C" in df.columns:
    df["Hardness_LINE"] = df["HARDNESS 鍍鋅線 C"]
elif "HARDNESS 鍍鋅線 S" in df.columns:
    df["Hardness_LINE"] = df["HARDNESS 鍍鋅線 S"]

# Loại bỏ các cột trùng tên (nếu file Excel gốc lỡ có cột bị trùng) để không bị lỗi TypeError
df = df.loc[:, ~df.columns.duplicated()]

# 3. Standard Hardness Split
def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

if "Std_Text" in df.columns:
    df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))

# 4. Force Numeric
numeric_cols = [
    "Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", "Order_Gauge",
    "Standard TS min", "Standard TS max",
    "Standard YS min", "Standard YS max",
    "Standard EL min", "Standard EL max"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 5. Quality Group Merge
if "Quality_Code" in df.columns:
    df["Quality_Group"] = df["Quality_Code"].replace({
        "CQ00": "CQ00 / CQ06",
        "CQ06": "CQ00 / CQ06"
    })

# 6. Filter GE* < 88
if "Quality_Code" in df.columns and "Hardness_LAB" in df.columns and "Hardness_LINE" in df.columns:
    df = df[~(
        df["Quality_Code"].astype(str).str.startswith("GE") &
        ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88))
    )]

# 7. Apply Company Rules
def apply_company_rules(row):
    std_min = row["Std_Min"] if "Std_Min" in row and pd.notna(row["Std_Min"]) else 0
    std_max = row["Std_Max"] if "Std_Max" in row and pd.notna(row["Std_Max"]) else 0
    lab_min, lab_max = 0, 0
    rule_name = "Standard (Excel)"

    is_cold = "COLD" in str(row.get("Rolling_Type", "")).upper()
    q_grp = str(row.get("Quality_Group", ""))
    target_qs = ["CQ00", "CQ06", "CQ07", "CQB0"]
    is_target_q = any(q in q_grp for q in target_qs)

    if is_cold and is_target_q:
        mat = str(row.get("Material", "")).upper().strip()
        if mat in ["A1081","A1081B"]: return 56.0, 62.0, 52.0, 70.0, "Rule A1081 (Cold)"
        elif mat in ["A108M","A108MR"]: return 60.0, 68.0, 55.0, 72.0, "Rule A108M (Cold)"
        elif mat in ["A108", "A108G", "A108R"]: return 58.0, 62.0, 52.0, 65.0, "Rule A108 (Cold)"

    return std_min, std_max, lab_min, lab_max, rule_name

df[['Limit_Min', 'Limit_Max', 'Lab_Min', 'Lab_Max', 'Rule_Name']] = df.apply(
    apply_company_rules, axis=1, result_type="expand"
)

# ================================
# LOAD GAUGE RANGE
# ================================
GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"

@st.cache_data
def load_gauge(): return pd.read_csv(GAUGE_URL)

gauge_df = load_gauge()
gauge_df.columns = gauge_df.columns.str.strip()
gauge_col = next((c for c in gauge_df.columns if "RANGE" in c.upper()), None)

if gauge_col and "Order_Gauge" in df.columns:
    def parse_range(text):
        nums = re.findall(r"\d+\.\d+|\d+", str(text))
        if len(nums) < 2: return None, None
        return float(nums[0]), float(nums[-1])

    ranges = []
    for _, r in gauge_df.iterrows():
        lo, hi = parse_range(r[gauge_col])
        if lo is not None: ranges.append((lo, hi, r[gauge_col]))

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

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

all_rolling = sorted(df["Rolling_Type"].astype(str).unique()) if "Rolling_Type" in df else []
all_metal = sorted(df["Metallic_Type"].astype(str).unique()) if "Metallic_Type" in df else []
all_qgroup = sorted(df["Quality_Group"].astype(str).unique()) if "Quality_Group" in df else []

rolling = st.sidebar.radio("Rolling Type", all_rolling) if all_rolling else None
metal   = st.sidebar.radio("Metallic Type", all_metal) if all_metal else None
qgroup  = st.sidebar.radio("Quality Group", all_qgroup) if all_qgroup else None

# ---> MASTER DATA SAFE (Dùng cho Master Export & KPI) <---
df_master_full = df.copy() 

# Lọc dữ liệu hiển thị theo Sidebar
if rolling and metal and qgroup:
    df = df[
        (df["Rolling_Type"] == rolling) &
        (df["Metallic_Type"] == metal) &
        (df["Quality_Group"] == qgroup)
    ]

view_mode = st.sidebar.radio(
    "📊 View Mode",
    [
        "📋 Data Inspection",
        "📊 Executive KPI Dashboard",
        "🚀 Global Summary Dashboard",
        "📉 Hardness Analysis (Trend & Dist)",
        "🔗 Correlation: Hardness vs Mech Props",
        "⚙️ Mech Props Analysis",
        "🔍 Lookup: Hardness Range → Actual Mech Props",
        "🎯 Find Target Hardness (Reverse Lookup)",
        "🧮 Predict TS/YS/EL from Std Hardness",
        "🎛️ Control Limit Calculator (Compare 3 Methods)",
        "👑 Master Dictionary Export",
    ]
)

GROUP_COLS = ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"]
cnt = df.groupby(GROUP_COLS, observed=True).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 30]

# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# 9. MASTER DICTIONARY EXPORT (ULTIMATE FIX - REMOVE GF + GI/GM MERGE + GL SEPARATE)
# ==============================================================================
if view_mode == "👑 Master Dictionary Export":
    import datetime as dt
    from io import BytesIO
    import pandas as pd
    import numpy as np
    import streamlit as st

    st.markdown("---")
    st.header("👑 Master Mechanical Properties Dictionary")
    st.caption("Control limits summary: Smart grouping by Quality, Material, and Metallic Category (GI/GM grouped together, GL separated). GF is excluded from production data.")
    
    col_sig1, col_sig2, col_sig3 = st.columns(3)
    with col_sig1:
        target_k = st.number_input("🎯 Target Zone (σ)", value=1.0, step=0.1, key="k_target")
    with col_sig2:
        control_k = st.number_input("🚧 Proposed Control Limit (σ)", value=2.0, step=0.5, key="k_control")
    with col_sig3:
        min_coils_req = st.number_input("📦 Min Coils Required", value=30, step=1, key="min_coils")

    if st.button("🚀 Generate Comprehensive Dictionary", type="primary"):
        source_df = df_master_full.copy()
        
        # --- 1. REMOVE GF (Out of production) ---
        source_df = source_df[source_df['Metallic_Type'].astype(str).str.strip().str.upper() != 'GF']
        
        # --- 2. MERGE QUALITY GROUPS: GE00 & GE01 ---
        # Chuyển GE01 thành GE00 / GE01 để chúng về cùng 1 nhóm khi groupby
        source_df['Quality_Group'] = source_df['Quality_Group'].astype(str).str.strip()
        source_df['Quality_Group'] = source_df['Quality_Group'].replace({
            'GE00': 'GE00 / GE01',
            'GE01': 'GE00 / GE01'
        })

        # --- 3. MERGE MATERIALS: A108 & A108G ---
        source_df['Material'] = source_df['Material'].astype(str).str.strip()
        source_df['Material'] = source_df['Material'].replace({
            'A108': 'A108 / A108G',
            'A108G': 'A108 / A108G'
        })
        
        # --- 4. CREATE METAL GROUPING LOGIC (GI/GM vs GL) ---
        def map_metallic_group(m):
            m_str = str(m).strip().upper()
            if 'GL' in m_str: return 'GL'
            if any(x in m_str for x in ['GI', 'GM']): return 'GI / GM'
            return m_str

        source_df['Metal_Category'] = source_df['Metallic_Type'].apply(map_metallic_group)

        # --- 3. ULTIMATE FUNCTION 2.0: Range Display & Smart Prefix ---
        def get_mapped_spec_range(group_df, min_col, max_col):
            temp_df = group_df[['Metallic_Type']].copy()
            
            # Extract Min
            if min_col in group_df.columns:
                temp_df['min_val'] = group_df[min_col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                temp_df['min_val'] = pd.to_numeric(temp_df['min_val'], errors='coerce').fillna(0)
            else:
                temp_df['min_val'] = 0
                
            # Extract Max
            if max_col in group_df.columns:
                temp_df['max_val'] = group_df[max_col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                temp_df['max_val'] = pd.to_numeric(temp_df['max_val'], errors='coerce').fillna(0)
            else:
                temp_df['max_val'] = 0

            valid_df = temp_df[(temp_df['min_val'] > 0) | (temp_df['max_val'] > 0)]
            if valid_df.empty:
                return "-"
                
            mapping = {}
            for _, row in valid_df.iterrows():
                metal = str(row['Metallic_Type']).strip()
                if metal not in mapping:
                    mapping[metal] = {'mins': set(), 'maxs': set()}
                if row['min_val'] > 0: mapping[metal]['mins'].add(row['min_val'])
                if row['max_val'] > 0: mapping[metal]['maxs'].add(row['max_val'])

            spec_to_metals = {}
            for metal, vals in mapping.items():
                mins = sorted(list(vals['mins']))
                maxs = sorted(list(vals['maxs']))
                
                # Format smartly
                spec_str = ""
                if mins and not maxs:
                    spec_str = f"≥{mins[0]:.0f}" if len(mins) == 1 else f"≥{mins[0]:.0f}~{mins[-1]:.0f}"
                elif maxs and not mins:
                    spec_str = f"≤{maxs[0]:.0f}" if len(maxs) == 1 else f"≤{maxs[0]:.0f}~{maxs[-1]:.0f}"
                elif mins and maxs:
                    if len(mins) == 1 and len(maxs) == 1:
                        spec_str = f"{mins[0]:.0f}~{maxs[0]:.0f}"
                    else:
                        min_disp = f"{mins[0]:.0f}" if len(mins)==1 else f"{mins[0]:.0f}~{mins[-1]:.0f}"
                        max_disp = f"{maxs[0]:.0f}" if len(maxs)==1 else f"{maxs[0]:.0f}~{maxs[-1]:.0f}"
                        spec_str = f"Min {min_disp} | Max {max_disp}"
                        
                if spec_str not in spec_to_metals:
                    spec_to_metals[spec_str] = []
                spec_to_metals[spec_str].append(metal)
                
            if len(spec_to_metals) == 1:
                return list(spec_to_metals.keys())[0]
                
            parts = []
            for spec_val, metals in spec_to_metals.items():
                metal_str = "/".join(sorted(metals))
                parts.append(f"{metal_str}: {spec_val}")
                
            return " | ".join(parts)
            
        # --- 4. DATA PRE-PROCESSING ---
        req_cols = ['Hardness_LINE', 'TS', 'YS', 'EL']
        for c in req_cols:
            source_df[c] = pd.to_numeric(source_df[c], errors='coerce')
        
        clean_master_df = source_df.dropna(subset=req_cols).copy()
        clean_master_df = clean_master_df[clean_master_df['Hardness_LINE'] > 0]
        
        clean_master_df['Quality_Group'] = clean_master_df['Quality_Group'].astype(str).str.strip()
        clean_master_df['Material'] = clean_master_df['Material'].astype(str).str.strip()
        clean_master_df['Gauge_Range'] = clean_master_df['Gauge_Range'].astype(str).str.strip()
        
        clean_master_df['Material'] = clean_master_df['Material'].replace({
            'A108': 'A108 / A108G',
            'A108G': 'A108 / A108G'
        })
        
        # --- 5. GROUPING LOGIC (Added Metal_Category) ---
        master_group_cols = ['Quality_Group', 'Material', 'Gauge_Range', 'Metal_Category']
        
        master_data = []

        def format_hrb(val):
            return str(int(float(val) + 0.5)) if pd.notna(val) and val > 0 else ""

        def get_safe_max(series):
            s = pd.to_numeric(series, errors='coerce').dropna()
            return s.max() if not s.empty else 0

        with st.spinner("Grouping and calculating statistical limits..."):
            for keys, group in clean_master_df.groupby(master_group_cols, observed=True):
                if len(group) < min_coils_req: continue 
                
                # Display category (e.g., "GI / GM" or "GL")
                metals_included = keys[3]
                
                # Calculate SPC
                hrb = group["Hardness_LINE"]
                mu = hrb.mean()
                mrs = np.abs(np.diff(hrb.values))
                sigma_imr = np.mean(mrs) / 1.128 if len(mrs) > 0 else hrb.std()
                
                p_ctrl_min, p_ctrl_max = mu - control_k * sigma_imr, mu + control_k * sigma_imr
                p_target_min, p_target_max = mu - target_k * sigma_imr, mu + target_k * sigma_imr

                # Hardness Specs
                cur_t_min = get_safe_max(group['Limit_Min'])
                cur_t_max = group['Limit_Max'].min() if 'Limit_Max' in group.columns else 0
                cur_c_min = get_safe_max(group['Lab_Min'])
                cur_c_max = group['Lab_Max'].min() if 'Lab_Max' in group.columns else 0

                # Actual properties in target zone
                actual_in = group[(group['Hardness_LINE'] >= p_target_min) & (group['Hardness_LINE'] <= p_target_max)]
                
                # Mapped Specs
                s_ys_spec = get_mapped_spec_range(group, "Standard YS min", "Standard YS max")
                s_ts_spec = get_mapped_spec_range(group, "Standard TS min", "Standard TS max")
                s_el_spec = get_mapped_spec_range(group, "Standard EL min", "Standard EL max")

                # Assemble row
                master_data.append({
                    "Quality Group": keys[0],
                    "Metallic Type": metals_included, 
                    "Material": keys[1],
                    "Gauge Range": keys[2],
                    "Current Target Spec": f"{format_hrb(cur_t_min)}~{format_hrb(cur_t_max)}" if cur_t_max < 9000 else f"≥{format_hrb(cur_t_min)}",
                    "Current Control Spec": f"{format_hrb(cur_c_min)}~{format_hrb(cur_c_max)}" if cur_c_max > 0 else "N/A",
                    f"Proposed Control Limit ({control_k}σ)": f"{format_hrb(p_ctrl_min)}~{format_hrb(p_ctrl_max)}",
                    f"Target ({target_k}σ)": f"{format_hrb(p_target_min)}~{format_hrb(p_target_max)}",
                    "YS Spec": s_ys_spec,
                    "Actual YS": f"{actual_in['YS'].min():.0f}~{actual_in['YS'].max():.0f}" if not actual_in.empty else "N/A",
                    "TS Spec": s_ts_spec,
                    "Actual TS": f"{actual_in['TS'].min():.0f}~{actual_in['TS'].max():.0f}" if not actual_in.empty else "N/A",
                    "EL Spec": s_el_spec,
                    "Actual EL": f"{actual_in['EL'].min():.1f}~{actual_in['EL'].max():.1f}" if not actual_in.empty else "N/A"
                })

        if master_data:
            df_out = pd.DataFrame(master_data)
            df_out = df_out.sort_values(by=["Quality Group", "Metallic Type", "Material", "Gauge Range"]).reset_index(drop=True)
            df_out.insert(0, "No.", range(1, len(df_out) + 1))
            
            # --- DISPLAY TABLE ---
            styled_df = df_out.style \
                .set_properties(**{'background-color': '#f8f9fa', 'color': '#333', 'border': '1px solid #ddd', 'text-align': 'center'}) \
                .set_properties(**{'background-color': '#FFF2CC', 'font-weight': 'bold'}, subset=["Current Target Spec", "Current Control Spec"]) \
                .set_properties(**{'background-color': '#D9EAD3', 'color': '#155724'}, subset=[f"Target ({target_k}σ)"]) \
                .set_properties(**{'background-color': '#CFE2F3', 'color': '#004085'}, subset=[f"Proposed Control Limit ({control_k}σ)"])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # --- EXPORT EXCEL ---
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, index=False, sheet_name='Master_Dictionary')
                workbook = writer.book
                worksheet = writer.sheets['Master_Dictionary']
                
                header_fmt = workbook.add_format({'bold': True, 'bg_color': '#EFEFEF', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
                for col_num, value in enumerate(df_out.columns.values):
                    worksheet.write(0, col_num, value, header_fmt)
                worksheet.set_column('A:A', 5)
                worksheet.set_column('B:O', 22) 

            st.download_button("📥 Download Master Dictionary (Excel)", output.getvalue(), "Master_Dictionary_SPC_Grouped.xlsx")
        else:
            st.warning("⚠️ No data found. Please reduce the 'Min Coils Required' value.")

    st.stop()
# ==============================================================================
# 1. EXECUTIVE KPI DASHBOARD (OVERVIEW) - STANDALONE BLOCK
# ==============================================================================
if view_mode == "📊 Executive KPI Dashboard":
    st.markdown("## 📊 Executive KPI Dashboard (Overall Quality Overview)")
    
    df_kpi = df_master_full.dropna(subset=['TS', 'YS', 'EL', 'Hardness_LINE']).copy()
    
    if df_kpi.empty:
        st.warning("⚠️ No sufficient data to generate KPIs.")
    else:
        total_coils = len(df_kpi)
        
        def clean_num(val, is_pct=False):
            if pd.isna(val): return "0%" if is_pct else "0"
            v = round(float(val), 2)
            res = str(int(v)) if v.is_integer() else str(v)
            return f"{res}%" if is_pct else res

        def check_pass(val, min_col, max_col):
            s_min = df_kpi[min_col].fillna(0) if min_col in df_kpi.columns else 0
            s_max = df_kpi[max_col].fillna(9999).replace(0, 9999) if max_col in df_kpi.columns else 9999
            return (val >= s_min) & (val <= s_max)
        
        df_kpi['TS_Pass'] = check_pass(df_kpi['TS'], 'Standard TS min', 'Standard TS max')
        df_kpi['YS_Pass'] = check_pass(df_kpi['YS'], 'Standard YS min', 'Standard YS max')
        df_kpi['EL_Pass'] = df_kpi['EL'] >= (df_kpi['Standard EL min'].fillna(0) if 'Standard EL min' in df_kpi.columns else 0)
        df_kpi['All_Pass'] = df_kpi['TS_Pass'] & df_kpi['YS_Pass'] & df_kpi['EL_Pass']
        
        df_kpi['HRB_Pass'] = (df_kpi['Hardness_LINE'] >= df_kpi['Limit_Min']) & (df_kpi['Hardness_LINE'] <= df_kpi['Limit_Max'])
        
        yield_rate = df_kpi['All_Pass'].mean() * 100
        hrb_yield = df_kpi['HRB_Pass'].mean() * 100 
        ts_yield = df_kpi['TS_Pass'].mean() * 100
        ys_yield = df_kpi['YS_Pass'].mean() * 100
        el_yield = df_kpi['EL_Pass'].mean() * 100
        
        scrap_col = next((c for c in df_kpi.columns if 'SCRAP' in str(c).upper() or 'CUT' in str(c).upper()), None)
        if scrap_col:
            total_scrap = df_kpi[scrap_col].sum()
            scrap_label = "Total Cut Scrap"
        else:
            total_scrap = total_coils - df_kpi['All_Pass'].sum()
            scrap_label = "Total Scrap / NG Coils"

        st.markdown("### 🏆 Overall Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Total Coils Tested", f"{total_coils:,}")
        
        delta_mech = clean_num(yield_rate - 100, True) if yield_rate < 100 else "Perfect"
        col2.metric("✅ Mech Yield Rate", clean_num(yield_rate, True), delta_mech, delta_color="normal" if yield_rate == 100 else "inverse")
        
        delta_hrb = clean_num(hrb_yield - 100, True) if hrb_yield < 100 else "In Control"
        col3.metric("🎯 HRB Yield Rate", clean_num(hrb_yield, True), delta_hrb, delta_color="normal" if hrb_yield == 100 else "inverse")
        
        col4.metric(f"✂️ {scrap_label}", f"{total_scrap:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col5, col6, col7 = st.columns(3)
        col5.metric("🔹 TS Pass Rate", clean_num(ts_yield, True))
        col6.metric("🔸 YS Pass Rate", clean_num(ys_yield, True))
        col7.metric("🔻 EL Pass Rate", clean_num(el_yield, True))
        
        st.markdown("---")
        
        st.markdown("### ⚠️ High-Risk Specs Watchlist")
        st.caption("Top list of standard codes with the lowest mechanical pass rates or out-of-control hardness, requiring priority review.")
        
        col_spec = "Product_Spec" if "Product_Spec" in df_kpi.columns else "Rule_Name"
        
        group_cols = [col_spec, "Quality_Group", "Material", "Gauge_Range"]
        valid_group_cols = [c for c in group_cols if c in df_kpi.columns]
        
        risk_summary = df_kpi.groupby(valid_group_cols, observed=True).agg(
            Total_Coils=('COIL_NO', 'count'),
            Mech_Pass_Coils=('All_Pass', 'sum'),
            HRB_Pass_Coils=('HRB_Pass', 'sum'), 
            Hardness_Mean=('Hardness_LINE', 'mean'),
            Hardness_Std=('Hardness_LINE', 'std'),
            LSL=('Limit_Min', 'first'),
            USL=('Limit_Max', 'first')
        ).reset_index()
        
        risk_summary['Mech Yield (%)'] = (risk_summary['Mech_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        risk_summary['HRB Yield (%)'] = (risk_summary['HRB_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        
        risk_top = risk_summary[risk_summary['Total_Coils'] >= 3].sort_values(['Mech Yield (%)', 'HRB Yield (%)']).head(10)
        
        if not risk_top.empty:
            rename_dict = {
                col_spec: "Specification",
                "Quality_Group": "Quality",
                "Material": "Material",
                "Gauge_Range": "Gauge",
                "Total_Coils": "Tested Coils",
                "Hardness_Mean": "Avg Hardness",
                "Hardness_Std": "Hardness Std Dev"
            }
            risk_top = risk_top.rename(columns=rename_dict)
            
            cols_order = ["Specification", "Quality", "Material", "Gauge", "Tested Coils", "Mech Yield (%)", "HRB Yield (%)", "Avg Hardness", "Hardness Std Dev"]
            cols_order = [c for c in cols_order if c in risk_top.columns]
            risk_top_display = risk_top[cols_order].copy()
            
            risk_top_display['Mech Yield (%)'] = risk_top_display['Mech Yield (%)'].apply(lambda x: clean_num(x, True))
            risk_top_display['HRB Yield (%)'] = risk_top_display['HRB Yield (%)'].apply(lambda x: clean_num(x, True)) if 'HRB Yield (%)' in risk_top_display else risk_top_display['HRB_Yield (%)'].apply(lambda x: clean_num(x, True))
            risk_top_display['Avg Hardness'] = risk_top_display['Avg Hardness'].apply(lambda x: clean_num(x))
            risk_top_display['Hardness Std Dev'] = risk_top_display['Hardness Std Dev'].apply(lambda x: clean_num(x))
            
            def style_risk(val):
                try:
                    num = float(str(val).replace('%', '').strip())
                    if num < 100: return 'color: #d32f2f; font-weight: bold; background-color: #ffebee'
                    if num >= 100: return 'color: #388e3c; font-weight: bold'
                except: pass
                return ''
            
            def style_std(val):
                try:
                    num = float(str(val).strip())
                    if num > 3.0: return 'color: #f57c00; font-weight: bold'
                except: pass
                return ''

            styled_risk = risk_top_display.style
            if hasattr(styled_risk, "map"):
                styled_risk = (styled_risk
                               .map(style_risk, subset=['Mech Yield (%)', 'HRB Yield (%)'])
                               .map(style_std, subset=['Hardness Std Dev']))
            else:
                styled_risk = (styled_risk
                               .applymap(style_risk, subset=['Mech Yield (%)', 'HRB Yield (%)'])
                               .applymap(style_std, subset=['Hardness Std Dev']))
            
            st.dataframe(styled_risk, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🔔 Visual Deep Dive: Top 5 Risk Distributions (HRB & Mech Props)")
            st.caption("Visualizing the 'bell curve' of the top 5 most critical specifications (Filtered for N ≥ 30 coils).")
            
            top_risks_for_chart = risk_summary[risk_summary['Total_Coils'] >= 30].sort_values(['Mech Yield (%)', 'HRB Yield (%)']).head(5).to_dict('records')
            
            if not top_risks_for_chart:
                st.info("💡 Hiện không có mã hàng nào gặp rủi ro mà đạt đủ điều kiện số lượng (≥ 30 cuộn) để vẽ biểu đồ phân phối.")
            else:
                for idx, risk_item in enumerate(top_risks_for_chart):
                    spec_name = risk_item[col_spec]
                    mat_name = risk_item["Material"]
                    gauge_val = risk_item["Gauge_Range"]
                    
                    target_data = df_kpi[
                        (df_kpi[col_spec] == spec_name) & 
                        (df_kpi["Material"] == mat_name) & 
                        (df_kpi["Gauge_Range"] == gauge_val)
                    ]
                    
                    if not target_data.empty:
                        st.markdown(f"**🚨 Top {idx+1}: {spec_name} | Material: {mat_name} | Gauge: {gauge_val} (N={len(target_data)})**")
                        chart_cols = st.columns(4)
                        
                        def plot_mini(ax_data, col_name, title, color, l_min, l_max):
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            
                            if col_name not in ax_data.columns:
                                data = pd.Series(dtype=float)
                            else:
                                data = pd.to_numeric(ax_data[col_name], errors='coerce').dropna()
                                
                            if data.empty or len(data) < 2:
                                ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
                                ax.axis('off')
                            else:
                                ax.hist(data, bins=15, color=color, edgecolor="white", density=True, alpha=0.8)
                                mean_val, std_val = data.mean(), data.std()
                                if std_val > 0:
                                    x_axis = np.linspace(data.min() - 2*std_val, data.max() + 2*std_val, 100)
                                    y_axis = (1/(std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - mean_val) / std_val)**2)
                                    ax.plot(x_axis, y_axis, color="black", lw=1.5, alpha=0.5)
                                
                                safe_l_min = float(l_min) if pd.notna(l_min) else 0.0
                                safe_l_max = float(l_max) if pd.notna(l_max) else 0.0
                                
                                if safe_l_min > 0: ax.axvline(safe_l_min, color="red", linestyle="--", lw=1.5)
                                if 0 < safe_l_max < 9000: ax.axvline(safe_l_max, color="red", linestyle="--", lw=1.5)
                            
                            ax.set_title(title, fontsize=11, fontweight="bold")
                            ax.tick_params(axis='both', which='major', labelsize=8)
                            ax.grid(alpha=0.3, linestyle=":")
                            return fig
                        
                        def safe_max(series): return series.max() if not series.empty and not pd.isna(series.max()) else 0
                        def safe_min(series): return series.min() if not series.empty and not pd.isna(series.min()) else 0

                        l_hrb_min = target_data["Limit_Min"].iloc[0] if "Limit_Min" in target_data.columns and not target_data.empty else 0
                        l_hrb_max = target_data["Limit_Max"].iloc[0] if "Limit_Max" in target_data.columns and not target_data.empty else 0
                        l_ts_min = safe_max(target_data["Standard TS min"]) if "Standard TS min" in target_data.columns else 0
                        l_ts_max = safe_min(target_data["Standard TS max"]) if "Standard TS max" in target_data.columns else 0
                        l_ys_min = safe_max(target_data["Standard YS min"]) if "Standard YS min" in target_data.columns else 0
                        l_ys_max = safe_min(target_data["Standard YS max"]) if "Standard YS max" in target_data.columns else 0
                        l_el_min = safe_max(target_data["Standard EL min"]) if "Standard EL min" in target_data.columns else 0
                        
                        with chart_cols[0]: 
                            fig1 = plot_mini(target_data, "Hardness_LINE", "Hardness (HRB)", "#ff9999", l_hrb_min, l_hrb_max)
                            st.pyplot(fig1)
                            plt.close(fig1)
                        with chart_cols[1]: 
                            fig2 = plot_mini(target_data, "TS", "Tensile (TS)", "#6baed6", l_ts_min, l_ts_max)
                            st.pyplot(fig2)
                            plt.close(fig2)
                        with chart_cols[2]: 
                            fig3 = plot_mini(target_data, "YS", "Yield (YS)", "#74c476", l_ys_min, l_ys_max)
                            st.pyplot(fig3)
                            plt.close(fig3)
                        with chart_cols[3]: 
                            fig4 = plot_mini(target_data, "EL", "Elongation (EL)", "#fd8d3c", l_el_min, 0)
                            st.pyplot(fig4)
                            plt.close(fig4)
                        
                        st.write("")
            
            st.markdown("---")
            st.markdown("#### 📑 Export Actionable Report")
            import streamlit.components.v1 as components
            col_csv, col_pdf, _ = st.columns([2, 2, 6])
            
            with col_csv:
                csv_data = risk_top.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 Download Watchlist (CSV)", data=csv_data, file_name="High_Risk_Watchlist.csv", mime="text/csv", use_container_width=True)
                
            with col_pdf:
                if st.button("🖨️ Save as PDF Report", use_container_width=True):
                    components.html("<script>window.parent.print();</script>", height=0)
            
            st.markdown("""<style>@media print {[data-testid="stSidebar"] { display: none !important; } header { display: none !important; } .stButton, .stDownloadButton { display: none !important; } @page { size: A4 landscape; margin: 10mm; } .stApp { background-color: white !important; }}</style>""", unsafe_allow_html=True)
        else:
            st.success("🎉 Excellent! All products are stable with no significant risks.")
    st.stop()


# ==============================================================================
# 2. GLOBAL SUMMARY DASHBOARD (FINAL)
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Dashboard")
    
    tab1, tab2 = st.tabs(["📊 1. Performance Overview", "🧠 2. Decision Support (Risk AI)"])

    # --- TAB 1: THỐNG KÊ HIỆU SUẤT ---
    with tab1:
        st.info("ℹ️ Color Guide: 🟢 High Pass Rate (>98%) | 🔴 Low Pass Rate (<90%)")
        stats_rows = []
        for _, g in valid.iterrows():
            sub_grp = df[
                (df["Rolling_Type"] == g["Rolling_Type"]) &
                (df["Metallic_Type"] == g["Metallic_Type"]) &
                (df["Quality_Group"] == g["Quality_Group"]) &
                (df["Gauge_Range"] == g["Gauge_Range"]) &
                (df["Material"] == g["Material"])
            ].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 3: continue

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

            n_total = len(sub_grp)
            n_ng = sub_grp[(sub_grp["Hardness_LINE"] < sub_grp["Limit_Min"]) | (sub_grp["Hardness_LINE"] > sub_grp["Limit_Max"])].shape[0]
            pass_rate = ((n_total - n_ng) / n_total) * 100

            stats_rows.append({
                "Quality": g["Quality_Group"], "Material": g["Material"], "Gauge": g["Gauge_Range"],
                "Specs": specs_str, "N": len(sub_grp),
                "Pass Rate": pass_rate,
                "HRB Limit": lim_hrb, "HRB (Avg)": sub_grp["Hardness_LINE"].mean(), 
                "TS Limit": lim_ts, "TS (Avg)": sub_grp["TS"].mean(),
                "YS Limit": lim_ys, "YS (Avg)": sub_grp["YS"].mean(), 
                "EL Limit": lim_el, "EL (Avg)": sub_grp["EL"].mean(),
            })

        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            
            cols = ["Quality", "Material", "Gauge", "Specs", "N", "Pass Rate", 
                    "HRB Limit", "HRB (Avg)", "TS Limit", "TS (Avg)", 
                    "YS Limit", "YS (Avg)", "EL Limit", "EL (Avg)"]
            cols = [c for c in cols if c in df_stats.columns]
            df_stats = df_stats[cols]

            def color_pass_rate(val):
                color = '#d4edda' if val >= 98 else ('#fff3cd' if val >= 90 else '#f8d7da')
                text_color = '#155724' if val >= 98 else ('#856404' if val >= 90 else '#721c24')
                return f'background-color: {color}; color: {text_color}; font-weight: bold'

            styled_df = df_stats.style.format("{:.1f}", subset=[c for c in df_stats.columns if "(Avg)" in c or "Pass" in c]) \
                .applymap(color_pass_rate, subset=["Pass Rate"]) \
                .set_properties(**{'background-color': '#FFF2CC', 'color': '#856404'}, subset=[c for c in cols if "Limit" in c]) \
                .set_properties(**{'font-weight': 'bold'}, subset=[c for c in cols if "(Avg)" in c])
                
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else: 
            st.warning("Insufficient data.")

    # --- TAB 2: PHÂN TÍCH RỦI RO (AI DECISION SUPPORT) ---
    with tab2:
        st.markdown("#### 🧠 AI Decision Support (Risk-Based)")
        st.caption("💡 **Note:** The **HRB Spec** column is the original hardness standard (to compare with Target HRB). The **Est. Range** is automatically compared against the **Mech Spec** to trigger bidirectional risk warnings (Low/High).")

        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            user_hrb = st.number_input("1️⃣ Target HRB", value=60.0, step=0.5, format="%.1f")
        with col_in2:
            safety_k = st.selectbox("2️⃣ Select Safety Factor (Sigma):", [1.0, 2.0, 3.0], index=1)

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

            specs_str = ", ".join(sorted(sub_grp["Product_Spec"].astype(str).unique()))
            spec_ts_min = sub_grp["Standard TS min"].max() if "Standard TS min" in sub_grp else 0
            spec_ys_min = sub_grp["Standard YS min"].max() if "Standard YS min" in sub_grp else 0
            spec_el_min = sub_grp["Standard EL min"].max() if "Standard EL min" in sub_grp else 0
            
            X = sub_grp[["Hardness_LINE"]].values

            # --- TS Analysis ---
            m_ts = LinearRegression().fit(X, sub_grp["TS"].values)
            pred_ts = m_ts.predict([[user_hrb]])[0]
            err_ts = np.sqrt(mean_squared_error(sub_grp["TS"], m_ts.predict(X)))
            safe_ts = pred_ts - (safety_k * err_ts)
            risk_ts = "🔴 High Risk" if (spec_ts_min > 0 and safe_ts < spec_ts_min) else "🟢 Safe"
            
            rows_ts.append({
                "Quality": g["Quality_Group"], "Material": g["Material"], "Gauge": g["Gauge_Range"], "Specs": specs_str,
                "Pred TS": f"{pred_ts:.0f}", "Worst Case": f"{safe_ts:.0f}", 
                "Limit": f"≥ {spec_ts_min:.0f}" if spec_ts_min > 0 else "-", "Status": risk_ts
            })

            # --- YS Analysis ---
            m_ys = LinearRegression().fit(X, sub_grp["YS"].values)
            pred_ys = m_ys.predict([[user_hrb]])[0]
            err_ys = np.sqrt(mean_squared_error(sub_grp["YS"], m_ys.predict(X)))
            safe_ys = pred_ys - (safety_k * err_ys)
            risk_ys = "🔴 High Risk" if (spec_ys_min > 0 and safe_ys < spec_ys_min) else "🟢 Safe"

            rows_ys.append({
                "Quality": g["Quality_Group"], "Material": g["Material"], "Gauge": g["Gauge_Range"], "Specs": specs_str,
                "Pred YS": f"{pred_ys:.0f}", "Worst Case": f"{safe_ys:.0f}", 
                "Limit": f"≥ {spec_ys_min:.0f}" if spec_ys_min > 0 else "-", "Status": risk_ys
            })

            # --- EL Analysis ---
            m_el = LinearRegression().fit(X, sub_grp["EL"].values)
            pred_el = m_el.predict([[user_hrb]])[0]
            err_el = np.sqrt(mean_squared_error(sub_grp["EL"], m_el.predict(X)))
            safe_el = pred_el - (safety_k * err_el)
            risk_el = "🔴 High Risk" if (spec_el_min > 0 and safe_el < spec_el_min) else "🟢 Safe"

            rows_el.append({
                "Quality": g["Quality_Group"], "Material": g["Material"], "Gauge": g["Gauge_Range"], "Specs": specs_str,
                "Pred EL": f"{pred_el:.1f}", "Worst Case": f"{safe_el:.1f}", 
                "Limit": f"≥ {spec_el_min:.1f}" if spec_el_min > 0 else "-", "Status": risk_el
            })

        if rows_ts:
            def style_risk(val):
                return 'color: #721c24; font-weight: bold; background-color: #f8d7da' if "🔴" in str(val) else 'color: #155724; font-weight: bold'

            c_top1, c_top2 = st.columns(2)
            with c_top1:
                st.markdown("##### 🔹 Tensile Strength (TS)")
                st.dataframe(pd.DataFrame(rows_ts).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
            with c_top2:
                st.markdown("##### 🔸 Yield Strength (YS)")
                st.dataframe(pd.DataFrame(rows_ys).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("##### 🔻 Elongation (EL)")
            st.dataframe(pd.DataFrame(rows_el).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
        else:
            st.warning("Không đủ dữ liệu để chạy mô hình AI (Cần nhóm có N ≥ 10).")
    st.stop()


# ==============================================================================
# MAIN LOOP (DETAILS)
# ==============================================================================
if valid.empty:
    st.warning("⚠️ No valid data found for detailed views.")
    st.stop()

for i, (_, g) in enumerate(valid.iterrows()):
    sub = df[
        (df["Rolling_Type"] == g["Rolling_Type"]) &
        (df["Metallic_Type"] == g["Metallic_Type"]) &
        (df["Quality_Group"] == g["Quality_Group"]) &
        (df["Gauge_Range"] == g["Gauge_Range"]) &
        (df["Material"] == g["Material"])
    ].sort_values("COIL_NO")

    if sub.empty: continue

    lo = sub["Limit_Min"].iloc[0] if "Limit_Min" in sub.columns else 0
    hi = sub["Limit_Max"].iloc[0] if "Limit_Max" in sub.columns else 0
    rule_used = sub["Rule_Name"].iloc[0] if "Rule_Name" in sub.columns else "Unknown"
    l_lo = sub["Lab_Min"].iloc[0] if "Lab_Min" in sub.columns else 0
    l_hi = sub["Lab_Max"].iloc[0] if "Lab_Max" in sub.columns else 0

    sub["NG_LAB"] = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"] 

    specs = ", ".join(sorted(sub["Product_Spec"].dropna().unique()))

    st.markdown(f"### 🧱 {g['Quality_Group']} | {g['Material']} | {g['Gauge_Range']}")
    st.markdown(f"**Specs:** {specs} | **Coils:** {sub['COIL_NO'].nunique()} | **Limit:** {lo:.1f}~{hi:.1f}")
    
    if view_mode != "⚙️ Mech Props Analysis":
        if "Rule" in str(rule_used): st.success(f"✅ Applied: **{rule_used}** (Control: {lo:.0f} - {hi:.0f} | Lab: {l_lo:.0f} - {l_hi:.0f})")
        else: st.caption(f"ℹ️ Applied: **Standard Excel Spec**")

    # ==============================================================================
    # 3. DATA INSPECTION
    # ==============================================================================
    if view_mode == "📋 Data Inspection":
        
        df_display = sub.copy()
        
        if "PRODUCTION DATE" in df_display.columns:
            df_display["PRODUCTION DATE"] = pd.to_datetime(df_display["PRODUCTION DATE"], errors='coerce').dt.strftime('%Y-%m-%d')
            
        cols_to_hide = ["Rolling_Type", "Rule_Name"]
        df_display = df_display.drop(columns=[c for c in cols_to_hide if c in df_display.columns])
        
        def highlight_ng_rows(row): 
            return ['background-color: #ffe6e6; color: #a00000'] * len(row) if row.get('NG', False) else [''] * len(row)
        
        num_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
        
        hard_cols = [c for c in num_cols if "Hardness" in c or "Limit" in c or "Lab" in c]
        other_num_cols = [c for c in num_cols if c not in hard_cols]
        
        st.dataframe(
            df_display.style.format("{:.1f}", subset=hard_cols)
                            .format("{:.0f}", subset=other_num_cols)
                            .apply(highlight_ng_rows, axis=1), 
            use_container_width=True
        )

    # ==============================================================================
    # ==============================================================================
    # 4. HARDNESS ANALYSIS
    # ==============================================================================
    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        
        df_valid = sub[(sub["Hardness_LINE"].notna()) & (sub["Hardness_LINE"] > 0)].copy()
        
        if "Hardness_LAB" in df_valid.columns:
            df_valid["Hardness_LAB"] = df_valid["Hardness_LAB"].replace(0, np.nan)

        if df_valid.empty:
            st.warning("⚠️ Không có dữ liệu hợp lệ (tất cả các cuộn đều bị rỗng hoặc bằng 0).")
        else:
            tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & SPC"])

            with tab_trend:
                x = np.arange(1, len(df_valid)+1)
                fig, ax = plt.subplots(figsize=(12, 5))
                
                if l_lo > 0 and l_hi > 0:
                    ax.fill_between(x, l_lo, l_hi, color="green", alpha=0.1, label="Target Zone")
                    ax.axhline(l_lo, linestyle="--", linewidth=2, color="green", label=f"Target LSL={l_lo:.1f}")
                    ax.axhline(l_hi, linestyle="--", linewidth=2, color="green", label=f"Target USL={l_hi:.1f}")
                
                ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"Std LSL={lo:.1f}")
                ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"Std USL={hi:.1f}")
                
                # ĐỒNG NHẤT MÀU: LAB màu Cam, LINE màu Xanh
                ax.plot(x, df_valid["Hardness_LAB"], marker="o", linewidth=2, color="#ff7f0e", markersize=6, label="LAB")
                ax.plot(x, df_valid["Hardness_LINE"], marker="s", linewidth=2, color="#1f77b4", markersize=6, label="LINE") 
                
                ng_line_mask = (df_valid["Hardness_LINE"] < lo) | (df_valid["Hardness_LINE"] > hi)
                if ng_line_mask.any():
                    ax.scatter(x[ng_line_mask], df_valid["Hardness_LINE"][ng_line_mask], color='red', s=100, zorder=5, label="Out of Control")

                ax.set_title(f"Hardness Trend by Coil Sequence - {g['Material']}", weight="bold", fontsize=14)
                ax.grid(False) 
                
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=5, fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.download_button("📥 Download Trend Chart", data=fig_to_png(fig), file_name=f"trend_{g['Material']}.png", mime="image/png", key=f"dl_trend_{i}")
                plt.close(fig) 

            with tab_dist:
                line = df_valid["Hardness_LINE"].dropna()
                lab = df_valid["Hardness_LAB"].dropna()
                
                if len(line) < 5: 
                    st.warning("⚠️ Not enough LINE data (N < 5) to plot distribution.")
                else:
                    def calc_spc_metrics(data, lsl, usl):
                        if len(data) < 2: return None
                        mean = data.mean()
                        std = data.std(ddof=1)
                        if std == 0: return None 
                        cp = (usl - lsl) / (6 * std)
                        mid = (usl + lsl) / 2
                        tol = (usl - lsl)
                        ca = ((mean - mid) / (tol / 2)) * 100 if tol > 0 else 0
                        cpu = (usl - mean) / (3 * std)
                        cpl = (mean - lsl) / (3 * std)
                        return mean, std, cp, ca, min(cpu, cpl)

                    spc_line = calc_spc_metrics(line, lo, hi)
                    mean_line, std_line = line.mean(), line.std(ddof=1)
                    
                    # TÍNH TOÁN TRỤC X ĐỒNG BỘ CHO CẢ 2 VIEW
                    x_min_sync = mean_line - 5 * std_line
                    x_max_sync = mean_line + 5 * std_line
                    if lo > 0: x_min_sync = min(x_min_sync, lo - 2)
                    if hi > 0: x_max_sync = max(x_max_sync, hi + 2)
                    
                    bins = np.linspace(x_min_sync, x_max_sync, 30)
                    bin_width = bins[1] - bins[0]
                    
                    range_curve = max(5 * std_line, (x_max_sync - x_min_sync)/2)
                    xs = np.linspace(mean_line - range_curve, mean_line + range_curve, 400)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    
                    # ĐỒNG NHẤT MÀU & HIỂN THỊ SỐ LƯỢNG (COUNT)
                    ax2.hist(line, bins=bins, density=False, alpha=0.6, color="#1f77b4", edgecolor="white", label="LINE Hist")
                    if not lab.empty: 
                        ax2.hist(lab, bins=bins, density=False, alpha=0.4, color="#ff7f0e", edgecolor="white", label="LAB Hist")
                    
                    if std_line > 0:
                        ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)
                        ys_line_scaled = ys_line * len(line) * bin_width
                        ax2.plot(xs, ys_line_scaled, linewidth=2.5, color="#004085", label="LINE Fit") 
                    
                    ax2.axvline(lo, linestyle="--", linewidth=2, color="red", label="Target LSL")
                    ax2.axvline(hi, linestyle="--", linewidth=2, color="red", label="Target USL")
                    if l_lo > 0 and l_hi > 0:
                        ax2.axvline(l_lo, linestyle="-.", linewidth=2, color="purple", label="Lab LSL")
                        ax2.axvline(l_hi, linestyle="-.", linewidth=2, color="purple", label="Lab USL")
                    
                    ax2.set_xlim(x_min_sync, x_max_sync)
                    ax2.set_title(f"Hardness Distribution (LINE vs LAB) - {g['Material']}", weight="bold")
                    ax2.set_ylabel("Number of Coils", fontweight="bold")
                    ax2.legend(loc='upper right') # Cố định legend ở góc phải để nhường góc trái cho TextBox
                    ax2.grid(alpha=0.3)
                    
                    # ==========================================
                    # THÊM TEXT BOX CHỨA CHỈ SỐ SPC LÊN BIỂU ĐỒ
                    # ==========================================
                    if spc_line:
                        mean_val, std_val, cp_val, ca_val, cpk_val = spc_line
                        eval_msg = "Excellent" if cpk_val >= 1.33 else ("Good" if cpk_val >= 1.0 else "Poor")
                        
                        # Định dạng nội dung chuỗi Text
                        spc_text = (
                            f"SPC Indices (LINE):\n"
                            f"N = {len(line)}\n"
                            f"Mean = {mean_val:.2f}\n"
                            f"Std = {std_val:.2f}\n"
                            f"Cp = {cp_val:.2f}\n"
                            f"Ca = {ca_val:.2f}%\n"
                            f"Cpk = {cpk_val:.2f}\n"
                            f"Rating: {eval_msg}"
                        )
                        
                        # Cấu hình khung viền (bbox) cho Text
                        props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da')
                        
                        # Đặt Text ở tọa độ góc trên bên trái (x=0.02, y=0.95 so với khung hình)
                        ax2.text(0.02, 0.96, spc_text, transform=ax2.transAxes, fontsize=10,
                                 verticalalignment='top', bbox=props, family='monospace')
                    
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # BẢNG DỮ LIỆU CHI TIẾT BÊN DƯỚI (Vẫn giữ lại để dễ copy/nhìn)
                    st.markdown("#### 📐 SPC Capability Indices (LINE ONLY)")
                    if spc_line:
                        color_code = "green" if cpk_val >= 1.33 else ("orange" if cpk_val >= 1.0 else "red")
                        df_spc = pd.DataFrame([{"N": len(line), "Mean": mean_val, "Std": std_val, "Cp": cp_val, "Ca (%)": ca_val, "Cpk": cpk_val, "Rating": eval_msg}])
                        
                        def style_rating(val):
                            return f'color: {color_code}; font-weight: bold'
                        
                        styled_spc = df_spc.style.format("{:.2f}", subset=["Mean", "Std", "Cp", "Ca (%)", "Cpk"])
                        if hasattr(styled_spc, "map"):
                            styled_spc = styled_spc.map(style_rating, subset=['Rating'])
                        else:
                            styled_spc = styled_spc.applymap(style_rating, subset=['Rating'])
                            
                        st.dataframe(styled_spc, hide_index=True)
    # ==============================================================================
    # 5. CORRELATION
    # ==============================================================================
    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        
        if i == 0: corr_bin_summary = []

        st.markdown("### 🔗 Correlation: Hardness vs Mechanical Properties")
        sub_corr = sub.dropna(subset=["Hardness_LAB","TS","YS","EL"]).copy()
        
        # Tìm giá trị nhỏ nhất và lớn nhất thực tế trong dữ liệu để đặt mốc
        h_min_act = sub_corr["Hardness_LAB"].min()
        h_max_act = sub_corr["Hardness_LAB"].max()
        
        # Tạo danh sách các mốc dải độ cứng
        # Chúng ta giữ các mốc tiêu chuẩn nhưng chèn giá trị Min thực tế vào đầu
        standard_thresholds = [56, 58, 60, 62, 65, 70, 75, 80, 85, 88, 92, 97]
        bins = sorted(list(set([h_min_act] + standard_thresholds + [h_max_act + 0.1])))
        
        # Tự động tạo nhãn (Labels) chính xác theo từng dải
        labels = [f"{bins[j]:.1f}-{bins[j+1]:.1f}" for j in range(len(bins)-1)]
        
        sub_corr["HRB_bin"] = pd.cut(sub_corr["Hardness_LAB"], bins=bins, labels=labels, right=False)
        sub_corr["HRB_bin"] = pd.cut(sub_corr["Hardness_LAB"], bins=bins, labels=labels, right=False)
        
        summary = (sub_corr.groupby("HRB_bin", observed=True).agg(
            N_coils=("COIL_NO","count"),
            TS_mean=("TS","mean"), TS_min=("TS","min"), TS_max=("TS","max"),
            YS_mean=("YS","mean"), YS_min=("YS","min"), YS_max=("YS","max"),
            EL_mean=("EL","mean"), EL_min=("EL","min"), EL_max=("EL","max"),
            Std_TS_min=("Standard TS min", "max"), Std_TS_max=("Standard TS max", "max"),
            Std_YS_min=("Standard YS min", "max"), Std_YS_max=("Standard YS max", "max"),
            Std_EL_min=("Standard EL min", "max")
        ).reset_index())
        summary = summary[summary["N_coils"]>0]

        if not summary.empty:
            x = np.arange(len(summary))
            fig, ax = plt.subplots(figsize=(15,6))
            ax2 = ax.twinx() 
            
            def plot_prop(ax_obj, x, y, ymin, ymax, c, lbl, m):
                ax_obj.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax_obj.fill_between(x, ymin, ymax, color=c, alpha=0.1)
            
            plot_prop(ax, x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS Actual", "o")
            plot_prop(ax, x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS Actual", "s")
            plot_prop(ax2, x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL Actual", "^")

            g_ts_min = summary["Std_TS_min"].max()
            g_ts_max = summary["Std_TS_max"].min()
            g_ys_min = summary["Std_YS_min"].max()
            g_ys_max = summary["Std_YS_max"].min()
            g_el_min = summary["Std_EL_min"].max()

            if pd.notna(g_ts_min) and g_ts_min > 0: ax.axhline(g_ts_min, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.5, label=f"TS LSL ({g_ts_min:.0f})")
            if pd.notna(g_ts_max) and 0 < g_ts_max < 9000: ax.axhline(g_ts_max, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.5, label=f"TS USL ({g_ts_max:.0f})")
            if pd.notna(g_ys_min) and g_ys_min > 0: ax.axhline(g_ys_min, color="#2ca02c", linestyle="-.", lw=1.5, alpha=0.5, label=f"YS LSL ({g_ys_min:.0f})")
            if pd.notna(g_ys_max) and 0 < g_ys_max < 9000: ax.axhline(g_ys_max, color="#2ca02c", linestyle="-.", lw=1.5, alpha=0.5, label=f"YS USL ({g_ys_max:.0f})")
            if pd.notna(g_el_min) and g_el_min > 0: ax2.axhline(g_el_min, color="#ff7f0e", linestyle=":", lw=2, alpha=0.6, label=f"EL LSL ({g_el_min:.0f})")

            for j, row in enumerate(summary.itertuples()):
                ts_min, ts_max = row.Std_TS_min, row.Std_TS_max
                ys_min, ys_max = row.Std_YS_min, row.Std_YS_max
                el_spec = row.Std_EL_min if pd.notna(row.Std_EL_min) else 0
                
                ts_fail = (pd.notna(ts_min) and ts_min > 0 and row.TS_mean < ts_min) or (pd.notna(ts_max) and 0 < ts_max < 9000 and row.TS_mean > ts_max)
                ys_fail = (pd.notna(ys_min) and ys_min > 0 and row.YS_mean < ys_min) or (pd.notna(ys_max) and 0 < ys_max < 9000 and row.YS_mean > ys_max)
                el_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                
                ax.annotate(f"{row.TS_mean:.0f}" + (" ❌" if ts_fail else ""), (x[j], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="red" if ts_fail else "#1f77b4")
                ax.annotate(f"{row.YS_mean:.0f}" + (" ❌" if ys_fail else ""), (x[j], row.YS_mean), xytext=(0,-15), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="red" if ys_fail else "#2ca02c")
                ax2.annotate(f"{row.EL_mean:.1f}%" + (" ❌" if el_fail else ""), (x[j], row.EL_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, color="red" if el_fail else "#ff7f0e", fontweight=("bold" if el_fail else "normal"))

            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_title("Hardness vs Mechanical Properties", fontweight="bold", fontsize=14)
            ax.set_ylabel("Strength (MPa)", fontweight="bold")
            ax2.set_ylabel("Elongation (%)", fontweight="bold", color="#ff7f0e")
            
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left", bbox_to_anchor=(1.05, 0.5))
            
            ax.grid(True, ls="--", alpha=0.5); fig.tight_layout(); st.pyplot(fig)
            plt.close(fig) 

            col_spec = "Product_Spec"
            specs_str = f"Specs: {', '.join(str(x) for x in sub[col_spec].dropna().unique())}" if col_spec in sub.columns else "Specs: N/A"

            for row in summary.itertuples():
                bin_data = sub_corr[sub_corr["HRB_bin"] == row.HRB_bin]
                corr_bin_summary.append({
                    "Specification List": specs_str, "Material": g["Material"], "Gauge": g["Gauge_Range"],
                    "Hardness Bin": row.HRB_bin, "N": row.N_coils,
                    "TS Spec": f"{row.Std_TS_min:.0f}~{row.Std_TS_max:.0f}" if row.Std_TS_max < 9000 else f"≥{row.Std_TS_min:.0f}",
                    "TS Actual": f"{row.TS_min:.0f}~{row.TS_max:.0f}", "TS Mean": f"{row.TS_mean:.1f}", "TS Std": f"{bin_data['TS'].std():.1f}",
                    "YS Spec": f"{row.Std_YS_min:.0f}~{row.Std_YS_max:.0f}" if row.Std_YS_max < 9000 else f"≥{row.Std_YS_min:.0f}",
                    "YS Actual": f"{row.YS_min:.0f}~{row.YS_max:.0f}", "YS Mean": f"{row.YS_mean:.1f}", "YS Std": f"{bin_data['YS'].std():.1f}",
                    "EL Spec": f"≥{row.Std_EL_min:.0f}", "EL Actual": f"{row.EL_min:.1f}~{row.EL_max:.1f}", 
                    "EL Mean": f"{row.EL_mean:.1f}", "EL Std": f"{bin_data['EL'].std():.1f}"
                })

        if i == len(valid) - 1 and 'corr_bin_summary' in locals() and len(corr_bin_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 📊 Hardness Binning Comprehensive Report: {qgroup}")
            
            df_full = pd.DataFrame(corr_bin_summary)
            
            def display_bin_table(title, cols, color_code):
                st.markdown(f"#### {title}")
                base_cols = ["Specification List", "Material", "Gauge", "Hardness Bin", "N"]
                target_df = df_full[base_cols + cols]
                std_col = [c for c in target_df.columns if "Std" in c]
                styled = target_df.style.set_properties(**{'background-color': color_code, 'font-weight': 'bold'}, subset=std_col)
                st.dataframe(styled, use_container_width=True, hide_index=True)

            display_bin_table("📉 TS Analysis by Hardness Bin", ["TS Spec", "TS Actual", "TS Mean", "TS Std"], "#e6f2ff")
            display_bin_table("📉 YS Analysis by Hardness Bin", ["YS Spec", "YS Actual", "YS Mean", "YS Std"], "#f2fff2")
            display_bin_table("📉 EL Analysis by Hardness Bin", ["EL Spec", "EL Actual", "EL Mean", "EL Std"], "#fff5e6")
            
            import datetime
            from io import BytesIO
            excel_name = f"Hardness_Bin_Report_{str(qgroup).replace(' ','')}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_full.to_excel(writer, sheet_name='All_Data', index=False)
                df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N", "TS Spec", "TS Actual", "TS Mean", "TS Std"]].to_excel(writer, sheet_name='TS_Only', index=False)
                df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N", "YS Spec", "YS Actual", "YS Mean", "YS Std"]].to_excel(writer, sheet_name='YS_Only', index=False)
                df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N", "EL Spec", "EL Actual", "EL Mean", "EL Std"]].to_excel(writer, sheet_name='EL_Only', index=False)

                workbook = writer.book
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column('A:A', 25) 
                    worksheet.set_column('B:C', 15) 
                    worksheet.set_column('D:Z', 12) 
            
            st.download_button("📥 Export Binning Report (Excel)", output.getvalue(), file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")                  

    # ==============================================================================
    # 6. MECH PROPS ANALYSIS
    # ==============================================================================
    elif view_mode == "⚙️ Mech Props Analysis":
        
        if i == 0: ts_summary, ys_summary, el_summary = [], [], []

        lab_str = f" | Target Zone: {l_lo:.1f} ~ {l_hi:.1f}" if l_lo > 0 else ""
        st.markdown(f"### ⚙️ Mechanical Properties Analysis: {g['Material']} | {g['Gauge_Range']} 🎯 (Control: {lo:.1f} ~ {hi:.1f}{lab_str})")
        
        sub_mech = sub.dropna(subset=["TS", "YS", "EL", "Hardness_LINE"]).copy()
        sub_mech = sub_mech[sub_mech["Hardness_LINE"] > 0]
        
        if sub_mech.empty: 
            st.warning("⚠️ No valid Mech Data (hoặc các cuộn đều bị rỗng / bằng 0 độ cứng).")
        else:
            props_config = [
                {"col": "TS", "name": "Tensile Strength (TS)", "color": "#1f77b4", "min_c": "Standard TS min", "max_c": "Standard TS max"},
                {"col": "YS", "name": "Yield Strength (YS)", "color": "#2ca02c", "min_c": "Standard YS min", "max_c": "Standard YS max"},
                {"col": "EL", "name": "Elongation (EL)", "color": "#ff7f0e", "min_c": "Standard EL min", "max_c": "Standard EL max"}
            ]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            col_spec = "Product_Spec"
            specs_str = f"Specs: {', '.join(str(x) for x in sub_mech[col_spec].dropna().unique())}" if col_spec in sub_mech.columns else "Specs: N/A"

            h_data = sub_mech["Hardness_LINE"]
            hardness_range_str = f"{h_data.min():.1f} ~ {h_data.max():.1f}"

            for j, cfg in enumerate(props_config):
                col = cfg["col"]; data = sub_mech[col]; mean, std = data.mean(), data.std()
                spec_min = sub_mech[cfg["min_c"]].max() if cfg["min_c"] in sub_mech else 0
                spec_max = sub_mech[cfg["max_c"]].min() if cfg["max_c"] in sub_mech else 0
                if pd.isna(spec_min): spec_min = 0
                if pd.isna(spec_max): spec_max = 0
                
                lcl_3s, ucl_3s = mean - 3 * std, mean + 3 * std
                
                axes[j].hist(data, bins=20, color=cfg["color"], alpha=0.5, density=True)
                if std > 0:
                    x_p = np.linspace(mean - 5 * std, mean + 5 * std, 200)
                    y_p = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2)
                    axes[j].plot(x_p, y_p, color=cfg["color"], lw=2)
                
                if spec_min > 0: axes[j].axvline(spec_min, color="red", linestyle="--", linewidth=2)
                if spec_max > 0 and spec_max < 9000: axes[j].axvline(spec_max, color="red", linestyle="--", linewidth=2)
                
                axes[j].axvline(lcl_3s, color="blue", linestyle=":", linewidth=1.5)
                axes[j].axvline(ucl_3s, color="blue", linestyle=":", linewidth=1.5)
                
                axes[j].set_title(f"{cfg['name']}\n(Mean={mean:.1f}, Std={std:.1f})", fontweight="bold")
                axes[j].grid(alpha=0.3, linestyle="--")

                row_data = {
                    "Specification List": specs_str, "Material": g["Material"], "Gauge": g["Gauge_Range"],
                    "N": len(sub_mech), "Control Limit (HRB)": f"{lo:.1f} ~ {hi:.1f}",
                    "Target Zone (HRB)": f"{l_lo:.1f} ~ {l_hi:.1f}" if (l_lo > 0 and l_hi > 0) else "N/A", 
                    "Actual Range (HRB)": hardness_range_str,
                    "Limit (Spec)": f"{spec_min:.0f}~{spec_max:.0f}" if (spec_max > 0 and spec_max < 9000) else (f"≥ {spec_min:.0f}" if spec_min > 0 else "-"),
                    "Actual Range": f"{data.min():.1f}~{data.max():.1f}",
                    "Mean": f"{mean:.1f}", "Std Dev": f"{std:.1f}", "LCL (-3σ)": f"{lcl_3s:.1f}", "UCL (+3σ)": f"{ucl_3s:.1f}"  
                }
                
                if col == "TS": ts_summary.append(row_data)
                elif col == "YS": ys_summary.append(row_data)
                elif col == "EL": el_summary.append(row_data)
            
            st.pyplot(fig)
            plt.close(fig) 

        if i == len(valid) - 1:
            st.markdown("---")
            st.markdown(f"## 📊 Mechanical Properties Comprehensive Report: {qgroup}")
            
            def display_summary_table(title, data_list, color_code):
                if data_list:
                    st.markdown(f"#### {title}")
                    df = pd.DataFrame(data_list)
                    styled_df = df.style.set_properties(**{'font-weight': 'bold'}, subset=['Mean']) \
                                        .set_properties(**{'background-color': '#FFF2CC', 'font-weight': 'bold', 'color': '#856404'}, subset=['Control Limit (HRB)', 'Target Zone (HRB)']) \
                                        .set_properties(**{'background-color': '#f0f8ff', 'font-weight': 'bold', 'color': '#0056b3'}, subset=['Actual Range (HRB)']) \
                                        .set_properties(**{'background-color': color_code, 'color': '#004085'}, subset=['LCL (-3σ)', 'UCL (+3σ)'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

            display_summary_table("1️⃣ Tensile Strength (TS) Summary", ts_summary, "#e6f2ff") 
            display_summary_table("2️⃣ Yield Strength (YS) Summary", ys_summary, "#f2fff2")   
            display_summary_table("3️⃣ Elongation (EL) Summary", el_summary, "#fff5e6")        

            import datetime
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            
            if ts_summary or ys_summary or el_summary:
                dfs, keys = [], []
                if ts_summary: dfs.append(pd.DataFrame(ts_summary)); keys.append('TS')
                if ys_summary: dfs.append(pd.DataFrame(ys_summary)); keys.append('YS')
                if el_summary: dfs.append(pd.DataFrame(el_summary)); keys.append('EL')
                
                full_df = pd.concat(dfs, keys=keys)
                st.download_button("📥 Export Full Mech Report CSV", full_df.to_csv(index=True).encode('utf-8-sig'), f"Full_Mech_Report_{today_str}.csv")

    # ==============================================================================
    # 7. LOOKUP
    # ==============================================================================
    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        
        st.markdown(f"### 🔍 Lookup: {g['Material']} | {g['Gauge_Range']}")
        c1, c2 = st.columns(2)
        
        actual_min = float(sub["Hardness_LINE"].min()) if not sub["Hardness_LINE"].empty else 0.0
        actual_max = float(sub["Hardness_LINE"].max()) if not sub["Hardness_LINE"].empty else 100.0
        
        if actual_min == 0.0 and actual_max == 0.0:
            st.warning("⚠️ Không có dữ liệu hợp lệ để tra cứu.")
        else:
            mn = c1.number_input("Min HRB", value=actual_min, step=0.5, key=f"lk1_lookup_{i}")
            mx = c2.number_input("Max HRB", value=actual_max, step=0.5, key=f"lk2_lookup_{i}")
            
            filt = sub[(sub["Hardness_LINE"] >= mn) & (sub["Hardness_LINE"] <= mx)].dropna(subset=["TS", "YS", "EL"])
            
            if not filt.empty: 
                st.success(f"✅ Found {len(filt)} coils matching HRB from {mn} to {mx}.")
                styled_describe = filt[["TS", "YS", "EL"]].describe().T.style.format("{:.1f}")\
                                    .set_properties(**{'background-color': '#f0f8ff', 'font-weight': 'bold', 'color': '#0056b3'}, subset=['mean', 'min', 'max'])
                st.dataframe(styled_describe, use_container_width=True)
            else:
                st.error(f"❌ No coils found in the range {mn} ~ {mx} HRB.")

    # ==============================================================================
    # 8. REVERSE LOOKUP (TARGET HARDNESS)
    # ==============================================================================
    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        
        if i == 0: reverse_lookup_summary = []

        st.markdown(f"### 🎯 Target Hardness Finder: {g['Material']} | {g['Gauge_Range']}")
        
        sub_clean = sub.dropna(subset=["TS", "YS", "EL", "Hardness_LINE"]).copy()
        sub_clean = sub_clean[sub_clean["Hardness_LINE"] > 0]

        if sub_clean.empty:
            st.warning("⚠️ No historical production data available for this group.")
        else:
            s_ts_min = sub_clean["Standard TS min"].max()
            s_ts_max = sub_clean["Standard TS max"].min()
            s_ys_min = sub_clean["Standard YS min"].max()
            s_ys_max = sub_clean["Standard YS max"].min()
            s_el_min = sub_clean["Standard EL min"].max()

            def format_ref(min_val, max_val, unit=""):
                if 0 < max_val < 9000: return f"{min_val:.0f} (Min) ~ {max_val:.0f} (Max){unit}"
                return f"≥ {min_val:.0f}{unit} (Min)"

            ts_ref = format_ref(s_ts_min, s_ts_max)
            ys_ref = format_ref(s_ys_min, s_ys_max)
            el_ref = f"≥ {s_el_min:.0f}% (Min)"

            st.caption(f"📌 **Mechanical Spec Reference:** TS: {ts_ref} | YS: {ys_ref} | EL: {el_ref}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Tensile (TS)**")
                r_ts_min = st.number_input("Desired TS (Min)", value=float(s_ts_min), step=5.0, key=f"rev_tsmin_{i}")
                default_ts_max = float(s_ts_max) if (0 < s_ts_max < 9000) else float(s_ts_min + 200)
                r_ts_max = st.number_input("Desired TS (Max)", value=default_ts_max, step=5.0, key=f"rev_tsmax_{i}")
            with c2:
                st.markdown("**Yield (YS)**")
                r_ys_min = st.number_input("Desired YS (Min)", value=float(s_ys_min), step=5.0, key=f"rev_ysmin_{i}")
                default_ys_max = float(s_ys_max) if (0 < s_ys_max < 9000) else float(s_ys_min + 200)
                r_ys_max = st.number_input("Desired YS (Max)", value=default_ys_max, step=5.0, key=f"rev_ysmax_{i}")
            with c3:
                st.markdown("**Elongation (EL)**")
                r_el_min = st.number_input("Desired EL % (Min)", value=float(s_el_min), step=1.0, key=f"rev_elmin_{i}")
                r_el_max = st.number_input("Desired EL % (Max)", value=100.0, step=1.0, key=f"rev_elmax_{i}")

            matched_coils = sub_clean[
                (sub_clean['TS'] >= r_ts_min) & (sub_clean['TS'] <= r_ts_max) &
                (sub_clean['YS'] >= r_ys_min) & (sub_clean['YS'] <= r_ys_max) &
                (sub_clean['EL'] >= r_el_min) & (sub_clean['EL'] <= r_el_max)
            ]
            
            if not matched_coils.empty:
                found_min, found_max = matched_coils['Hardness_LINE'].min(), matched_coils['Hardness_LINE'].max()
                n_found = len(matched_coils)
                st.success(f"✅ Found **{n_found}** produced coils. Their actual hardness was: **{found_min:.1f} ~ {found_max:.1f} HRB**")
                
                with st.expander("See matched coils list"):
                    st.dataframe(matched_coils[['COIL_NO', 'Hardness_LINE', 'TS', 'YS', 'EL']], hide_index=True)
                
                target_text = f"{found_min:.1f} ~ {found_max:.1f}"
            else: 
                st.error("❌ No historical coils found matching these mechanical criteria.")
                target_text = "Not Found"
                n_found = 0

            col_spec = "Product_Spec"
            specs_str = f"{', '.join(str(x) for x in sub[col_spec].dropna().unique())}" if col_spec in sub.columns else "N/A"
            
            reverse_lookup_summary.append({
                "Specification": specs_str, "Material": g["Material"], "Gauge": g["Gauge_Range"],
                "Desired Mech Range": f"TS:{r_ts_min:.0f}-{r_ts_max:.0f} | YS:{r_ys_min:.0f}-{r_ys_max:.0f} | EL:≥{r_el_min:.0f}%",
                "Historical Hardness (HRB)": target_text, "Matched Coils": n_found
            })

        if i == len(valid) - 1 and len(reverse_lookup_summary) > 0:
            st.markdown("---")
            st.markdown("#### 📊 Target Hardness Summary Table (Based on Historical Production)")
            st.dataframe(pd.DataFrame(reverse_lookup_summary), use_container_width=True, hide_index=True)

    # ==============================================================================
    # 9. AI PREDICTION
    # ==============================================================================
    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown(f"### 🧮 AI Prediction: {g['Material']} | {g['Gauge_Range']}")
        
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"]).copy()
        train_df = train_df[train_df["Hardness_LINE"] > 0]
        
        if len(train_df) < 5:
            st.warning("⚠️ Cần ít nhất 5 cuộn hợp lệ để huấn luyện mô hình AI.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                mean_h = train_df["Hardness_LINE"].mean()
                target_h = st.number_input("🎯 Target Hardness (HRB)", value=float(round(mean_h, 1)), step=0.1, key=f"ai_fix_{i}")
            
            s_ts_min = train_df["Standard TS min"].max() if "Standard TS min" in train_df.columns else 0
            s_ts_max = train_df["Standard TS max"].min() if "Standard TS max" in train_df.columns else 0
            s_ys_min = train_df["Standard YS min"].max() if "Standard YS min" in train_df.columns else 0
            s_ys_max = train_df["Standard YS max"].min() if "Standard YS max" in train_df.columns else 0
            s_el_min = train_df["Standard EL min"].max() if "Standard EL min" in train_df.columns else 0
            
            def fmt_spec(vmin, vmax):
                if pd.isna(vmin): vmin = 0
                if pd.isna(vmax): vmax = 0
                if vmax > 0 and vmax < 9000: return f"{vmin:.0f} ~ {vmax:.0f}"
                if vmin > 0: return f"≥ {vmin:.0f}"
                return "N/A"
                
            spec_ts_str = fmt_spec(s_ts_min, s_ts_max)
            spec_ys_str = fmt_spec(s_ys_min, s_ys_max)
            spec_el_str = f"≥ {s_el_min:.0f}" if pd.notna(s_el_min) and s_el_min > 0 else "N/A"

            X_train = train_df[["Hardness_LINE"]].values
            preds, model_metrics = {}, {}
            
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                val = model.predict([[target_h]])[0]
                preds[col] = val 
                
                y_true = train_df[col].values
                y_pred = model.predict(X_train)
                model_metrics[col] = {"r2": r2_score(y_true, y_pred), "rmse": np.sqrt(mean_squared_error(y_true, y_pred))}

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            colors = {"TS": "#2980b9", "YS": "#27ae60", "EL": "#c0392b"} 
            idx = list(range(len(train_df)))
            nxt = len(train_df)

            for col in ["TS", "YS", "EL"]:
                sec = (col == "EL")
                
                fig.add_trace(go.Scatter(
                    x=idx, y=train_df[col], mode='lines', 
                    line=dict(color=colors[col], width=2, shape='spline'), 
                    name=f"{col} (History)", opacity=0.6, hoverinfo='y' 
                ), secondary_y=sec)
                
                last_val_raw = train_df[col].iloc[-1]
                pred_clean = round(preds[col], 1) if col == "EL" else int(round(preds[col]))
                last_clean = round(last_val_raw, 1) if col == "EL" else int(round(last_val_raw))
                
                fig.add_trace(go.Scatter(
                    x=[idx[-1], nxt], y=[last_val_raw, preds[col]],
                    mode='lines', line=dict(color=colors[col], width=2, dash='dot'),
                    showlegend=False, hoverinfo='skip'
                ), secondary_y=sec)

                is_pass = True
                if col == "TS":
                    if pd.notna(s_ts_min) and s_ts_min > 0 and pred_clean < s_ts_min: is_pass = False
                    if pd.notna(s_ts_max) and 0 < s_ts_max < 9000 and pred_clean > s_ts_max: is_pass = False
                elif col == "YS":
                    if pd.notna(s_ys_min) and s_ys_min > 0 and pred_clean < s_ys_min: is_pass = False
                    if pd.notna(s_ys_max) and 0 < s_ys_max < 9000 and pred_clean > s_ys_max: is_pass = False
                elif col == "EL":
                    if pd.notna(s_el_min) and s_el_min > 0 and pred_clean < s_el_min: is_pass = False

                status_icon = "✅" if is_pass else "❌"

                fig.add_trace(go.Scatter(
                    x=[nxt], y=[preds[col]], mode='markers+text', text=[f"<b>{status_icon} {pred_clean}</b>"], 
                    textposition="middle right" if nxt < 10 else "top center",
                    marker=dict(color=colors[col], size=14, symbol='diamond', line=dict(width=2, color='white')), 
                    name=f"Pred {col}",
                    hovertemplate=(f"<b>🎯 Pred {col}: {pred_clean}</b><br>🔙 Last {col}: {last_clean}<br>📈 Change: {pred_clean - last_clean:.1f}<extra></extra>")
                ), secondary_y=sec)

            def add_spec_lines(vmin, vmax, color, name, is_sec):
                if pd.notna(vmin) and vmin > 0: fig.add_hline(y=vmin, line_dash="dash", line_color=color, opacity=0.3, annotation_text=f"{name} Min", secondary_y=is_sec)
                if pd.notna(vmax) and 0 < vmax < 9000: fig.add_hline(y=vmax, line_dash="dash", line_color=color, opacity=0.3, annotation_text=f"{name} Max", secondary_y=is_sec)

            add_spec_lines(s_ts_min, s_ts_max, colors["TS"], "TS", False)
            add_spec_lines(s_ys_min, s_ys_max, colors["YS"], "YS", False)
            add_spec_lines(s_el_min, 0, colors["EL"], "EL", True)

            fig.add_vline(x=nxt - 0.5, line_width=1, line_dash="dash", line_color="gray")
            fig.add_annotation(x=nxt - 0.5, y=1.05, yref="paper", text="Forecast Zone ➔", showarrow=False, font=dict(color="gray"))

            fig.update_layout(height=500, title=dict(text=f"📈 Prediction at Target HRB = {target_h:.1f}", font=dict(size=18)), plot_bgcolor="white", hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=80, b=20))
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', title="Coil Sequence")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', secondary_y=False, title="Strength (MPa)")
            fig.update_yaxes(showgrid=False, secondary_y=True, title="Elongation (%)")

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🏁 Forecast Summary & Spec Verification")
            c1, c2, c3 = st.columns(3)
            
            def get_delta(p, l): return round(p - l, 1)
            
            last_ts, last_ys, last_el = train_df["TS"].iloc[-1], train_df["YS"].iloc[-1], train_df["EL"].iloc[-1]

            ts_pred_clean = int(round(preds['TS']))
            ts_pass = not ((pd.notna(s_ts_min) and s_ts_min > 0 and ts_pred_clean < s_ts_min) or (pd.notna(s_ts_max) and 0 < s_ts_max < 9000 and ts_pred_clean > s_ts_max))
            c1.metric(f"Tensile Strength (TS)", f"{ts_pred_clean} MPa", f"{get_delta(preds['TS'], last_ts)} vs Last Coil")
            c1.markdown(f"**Spec:** `{spec_ts_str}` ➔ {'✅ **PASS**' if ts_pass else '❌ **FAIL**'}")
            c1.caption(f"🎯 **R² Score:** {model_metrics['TS']['r2']:.2f} | **RMSE:** ±{model_metrics['TS']['rmse']:.1f}")

            ys_pred_clean = int(round(preds['YS']))
            ys_pass = not ((pd.notna(s_ys_min) and s_ys_min > 0 and ys_pred_clean < s_ys_min) or (pd.notna(s_ys_max) and 0 < s_ys_max < 9000 and ys_pred_clean > s_ys_max))
            c2.metric(f"Yield Strength (YS)", f"{ys_pred_clean} MPa", f"{get_delta(preds['YS'], last_ys)} vs Last Coil")
            c2.markdown(f"**Spec:** `{spec_ys_str}` ➔ {'✅ **PASS**' if ys_pass else '❌ **FAIL**'}")
            c2.caption(f"🎯 **R² Score:** {model_metrics['YS']['r2']:.2f} | **RMSE:** ±{model_metrics['YS']['rmse']:.1f}")

            el_pred_clean = round(preds['EL'], 1)
            el_pass = not (pd.notna(s_el_min) and s_el_min > 0 and el_pred_clean < s_el_min)
            c3.metric(f"Elongation (EL)", f"{el_pred_clean} %", f"{get_delta(preds['EL'], last_el)} vs Last Coil")
            c3.markdown(f"**Spec:** `{spec_el_str}` ➔ {'✅ **PASS**' if el_pass else '❌ **FAIL**'}")
            c3.caption(f"🎯 **R² Score:** {model_metrics['EL']['r2']:.2f} | **RMSE:** ±{model_metrics['EL']['rmse']:.1f}")

    # ==============================================================================
# ================================
# FULL STREAMLIT APP – A118T ULTIMATE STABLE VERSION
# (Order_Gauge grouping removed per manager request)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ================================
# PAGE CONFIG & CSS
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard - A118T", layout="wide")
st.title("📊 Hardness – Visual Analytics Dashboard (A118T)")

def add_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #ffffff; box-shadow: 2px 0 5px rgba(0,0,0,0.05); border-right: none; }
        h1, h2, h3 { color: #2c3e50 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 600; }
        
        [data-testid="stMetricValue"] { 
            background-color: white; 
            padding: 10px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
            color: #007bff; 
            font-size: 1.6rem !important; 
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            white-space: normal !important; 
        }
        
        thead tr th:first-child {display:none}
        tbody th {display:none}
        .stDataFrame { border: none !important; }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ================================
# DYNAMIC TARGET SETTINGS (SIDEBAR)
# ================================
st.sidebar.header("⚙️ GLOBAL SETTINGS")
st.sidebar.markdown("**🎯 Target Hardness (HRB)**")
c_t1, c_t2 = st.sidebar.columns(2)

TARGET_MIN = c_t1.number_input("Target Min", value=85.0, step=0.5, format="%.1f", key="global_target_min")
TARGET_MAX = c_t2.number_input("Target Max", value=90.0, step=0.5, format="%.1f", key="global_target_max")
st.sidebar.markdown("---")

# ================================
# LOAD & CLEAN DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1hC5nnxqDLjF8-wUm8gtj11_5HFMxBlogY84Z0cRCj2s/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    if "<!doctype html>" in r.text[:50].lower() or "<html" in r.text[:50].lower():
        st.error("🚨 SECURITY ERROR: Restricted Google Sheet link. Please go to Google Sheet -> Share -> Select 'Anyone with the link'.")
        st.stop()
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# Date/Time Processing
data_period_str = "N/A"
date_col = next((c for c in raw.columns if 'DATE' in str(c).upper()), None)
if date_col:
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    min_date = raw[date_col].min()
    max_date = raw[date_col].max()
    if pd.notna(min_date) and pd.notna(max_date):
        data_period_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

import datetime as dt
tz_tw = dt.timezone(dt.timedelta(hours=8))
current_time = dt.datetime.now(tz_tw).strftime("%Y-%m-%d %H:%M")

st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <strong>🕒 Report Generated:</strong> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <strong>📅 Data Period:</strong> {data_period_str} &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>🎯 Target Hardness:</strong> {TARGET_MIN} ~ {TARGET_MAX}
</div>
""", unsafe_allow_html=True)

# Absolute Column Header Scanner
col_mapping = {}
for col in raw.columns:
    clean_name = re.sub(r'[\s_]+', '', str(col)).upper()
    if 'PRODUCTSPEC' in clean_name: col_mapping[col] = 'Product_Spec'
    elif 'HRSTEELGRADE' in clean_name or 'STEELGRADE' in clean_name: col_mapping[col] = 'Material'
    elif 'CLASSIFY' in clean_name: col_mapping[col] = 'Rolling_Type'
    elif 'QUALITYCODE' in clean_name: col_mapping[col] = 'Quality_Code'
    elif 'ORDERGAUGE' in clean_name: col_mapping[col] = 'Order_Gauge'
    elif 'METALLICCOATING' in clean_name: col_mapping[col] = 'Metallic_Type'
    elif 'COILNO' in clean_name: col_mapping[col] = 'COIL_NO'
    elif 'STANDARDHARDNESS' in clean_name: col_mapping[col] = 'Std_Text'
    elif 'HARDNESS冶金' in clean_name or '冶金' in clean_name: col_mapping[col] = 'Hardness_LAB'
    elif '鍍鋅線C' in clean_name or ('HARDNESS' in clean_name and 'C' in clean_name and 'N' not in clean_name and 'S' not in clean_name): col_mapping[col] = 'Hardness_LINE'
    elif 'TENSILEYIELD' in clean_name: col_mapping[col] = 'YS'
    elif 'TENSILETENSILE' in clean_name: col_mapping[col] = 'TS'
    elif 'TENSILEELONG' in clean_name: col_mapping[col] = 'EL'
    elif 'STANDARDTSMIN' in clean_name: col_mapping[col] = 'Standard TS min'
    elif 'STANDARDTSMAX' in clean_name: col_mapping[col] = 'Standard TS max'
    elif 'STANDARDYSMIN' in clean_name: col_mapping[col] = 'Standard YS min'
    elif 'STANDARDYSMAX' in clean_name: col_mapping[col] = 'Standard YS max'
    elif 'STANDARDELMIN' in clean_name: col_mapping[col] = 'Standard EL min'
    elif 'STANDARDELMAX' in clean_name: col_mapping[col] = 'Standard EL max'

df = raw.rename(columns=col_mapping)

if "Hardness_LINE" not in df.columns:
    st.error(f"⚠️ Column error: Hardness_LINE missing. Available columns: {list(raw.columns)}")
    st.stop()

if "COIL_NO" not in df.columns: df["COIL_NO"] = df.index 
if "Material" not in df.columns: df["Material"] = "A118T"
if "Product_Spec" not in df.columns: df["Product_Spec"] = "N/A"

if df.empty:
    st.error("⚠️ No valid data found in the Google Sheet.")
    st.stop()

# Data processing & remove 0/NA
df["Limit_Min"] = 80.0
df["Limit_Max"] = 93.0

def split_std(x):
    if isinstance(x, str) and "~" in x:
        try:
            lo, hi = x.split("~")
            return float(lo.strip()), float(hi.strip())
        except: pass
    return 80.0, 93.0

if "Std_Text" in df.columns:
    limits = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
    if not limits.empty and limits.shape[1] == 2:
        df[["Limit_Min", "Limit_Max"]] = limits

df["Lab_Min"] = df["Limit_Min"]
df["Lab_Max"] = df["Limit_Max"]
df["Rule_Name"] = "Direct Spec"

test_cols = ["Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", 
             "Standard TS min", "Standard TS max", "Standard YS min", "Standard YS max", 
             "Standard EL min", "Standard EL max"]

for c in test_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == 0, c] = np.nan 

# Order_Gauge kept for display/filter only, NOT in grouping
if "Order_Gauge" in df.columns:
    df["Order_Gauge"] = pd.to_numeric(df["Order_Gauge"], errors="coerce")
    df["Order_Gauge"] = df["Order_Gauge"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

df = df.dropna(subset=["Hardness_LINE"])

# ================================
# SIDEBAR FILTER & SETTINGS
# ================================
st.sidebar.header("🔍 DATA FILTER")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

all_specs = sorted(df["Product_Spec"].dropna().astype(str).unique()) if "Product_Spec" in df else []
all_rolling = sorted(df["Rolling_Type"].dropna().astype(str).unique()) if "Rolling_Type" in df else []
all_metal = sorted(df["Metallic_Type"].dropna().astype(str).unique()) if "Metallic_Type" in df else []
all_qgroup = sorted(df["Quality_Group"].dropna().astype(str).unique()) if "Quality_Group" in df else []

# Order_Gauge still available as optional filter (display only, not grouping)
all_gauge = []
if "Order_Gauge" in df.columns:
    valid_gauges = pd.to_numeric(df["Order_Gauge"], errors="coerce").dropna()
    if not valid_gauges.empty:
        all_gauge = sorted(valid_gauges.unique())

specs_filter = st.sidebar.selectbox("1. Product Specs", ["All"] + list(all_specs))
rolling = st.sidebar.selectbox("2. Rolling Type", ["All"] + list(all_rolling))
metal = st.sidebar.selectbox("3. Metallic Type", ["All"] + list(all_metal))

gauge_input = st.sidebar.text_input("4. Order Gauge (ex: 1.5 or 1.5~1.8) [Filter only]", value="")
if all_gauge:
    gauge_hint = ", ".join([f"{g:.2f}" for g in all_gauge[:10]]) + ("..." if len(all_gauge) > 10 else "")
    st.sidebar.caption(f"💡 **Available:** {gauge_hint}")

qgroup = st.sidebar.selectbox("5. Quality Group", ["All"] + list(all_qgroup))

# Apply filtering logic
df_master_full = df.copy() 

if specs_filter != "All": df = df[df["Product_Spec"].astype(str) == specs_filter]
if rolling != "All": df = df[df["Rolling_Type"].astype(str) == rolling]
if metal != "All": df = df[df["Metallic_Type"].astype(str) == metal]

if gauge_input.strip() != "" and "Order_Gauge" in df:
    df["temp_gauge_num"] = pd.to_numeric(df["Order_Gauge"], errors="coerce")
    if "~" in gauge_input or "-" in gauge_input:
        sep = "~" if "~" in gauge_input else "-"
        try:
            parts = gauge_input.split(sep)
            df = df[(df["temp_gauge_num"] >= float(parts[0])) & (df["temp_gauge_num"] <= float(parts[1]))]
        except: st.sidebar.error("⚠️ Invalid gauge format")
    else:
        try: df = df[(df["temp_gauge_num"] >= float(gauge_input) - 0.001) & (df["temp_gauge_num"] <= float(gauge_input) + 0.001)]
        except: st.sidebar.error("⚠️ Invalid gauge format")
    df = df.drop(columns=["temp_gauge_num"])
    
if qgroup != "All": df = df[df["Quality_Group"].astype(str) == qgroup]

# ============================================================
# GROUP_COLS: Order_Gauge REMOVED — only spec/type/material
# ============================================================
GROUP_COLS = [c for c in ["Product_Spec", "Rolling_Type", "Metallic_Type", "Quality_Group", "Material"] if c in df.columns]
if not GROUP_COLS: GROUP_COLS = ["Material"]

cnt = df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 1]

# ================================
# NAVIGATION MENU (GROUPED)
# ================================
st.sidebar.markdown("---")
st.sidebar.header("🧭 NAVIGATION")

menu_category = st.sidebar.selectbox("📂 Select Category", ["📊 Dashboards & KPIs", "🔬 Deep Analytics", "🛠️ Tools & AI Models"])

if menu_category == "📊 Dashboards & KPIs":
    view_mode = st.sidebar.radio("📍 Select View", ["📊 Executive KPI Dashboard", "🚀 Global Summary Dashboard", "📋 Data Inspection"])
elif menu_category == "🔬 Deep Analytics":
    view_mode = st.sidebar.radio("📍 Select View", ["📉 Hardness Analysis (Trend & Dist)", "🔗 Correlation: Hardness vs Mech Props", "⚙️ Mech Props Analysis"])
else:
    view_mode = st.sidebar.radio("📍 Select View", ["🔍 Lookup: Hardness Range → Actual Mech Props", "🎯 Find Target Hardness (Reverse Lookup)", "🧮 Predict TS/YS/EL from Std Hardness", "🎛️ Control Limit Calculator (Compare 3 Methods)", "👑 Master Dictionary Export"])

if valid.empty:
    st.warning("⚠️ No valid coils found for the current filter. Please adjust the sidebar.")
    st.stop()

# ==============================================================================
# 0. EXECUTIVE KPI DASHBOARD (OVERVIEW)
# ==============================================================================
if view_mode == "📊 Executive KPI Dashboard":
    st.markdown("## 📊 Executive KPI Dashboard (Overall Quality Overview)")
    df_kpi = df.dropna(subset=['TS', 'YS', 'EL', 'Hardness_LINE']).copy()
    
    if df_kpi.empty:
        st.warning("⚠️ Insufficient Mechanical Properties data for the selected filters.")
    else:
        total_coils = len(df_kpi)
        
        def clean_num(val, is_pct=False):
            if pd.isna(val): return "0%" if is_pct else "0"
            v = round(float(val), 1) 
            res = str(int(v)) if v.is_integer() else str(v)
            return f"{res}%" if is_pct else res

        def check_pass(val, min_col, max_col):
            s_min = df_kpi[min_col].fillna(0) if min_col in df_kpi.columns else 0
            s_max = df_kpi[max_col].fillna(9999).replace(0, 9999) if max_col in df_kpi.columns else 9999
            return (val >= s_min) & (val <= s_max)
        
        df_kpi['TS_Pass'] = check_pass(df_kpi['TS'], 'Standard TS min', 'Standard TS max')
        df_kpi['YS_Pass'] = check_pass(df_kpi['YS'], 'Standard YS min', 'Standard YS max')
        df_kpi['EL_Pass'] = df_kpi['EL'] >= (df_kpi['Standard EL min'].fillna(0) if 'Standard EL min' in df_kpi.columns else 0)
        df_kpi['All_Pass'] = df_kpi['TS_Pass'] & df_kpi['YS_Pass'] & df_kpi['EL_Pass']
        df_kpi['HRB_Pass'] = (df_kpi['Hardness_LINE'] >= df_kpi['Limit_Min']) & (df_kpi['Hardness_LINE'] <= df_kpi['Limit_Max'])
        df_kpi['Target_Pass'] = (df_kpi['Hardness_LINE'] >= TARGET_MIN) & (df_kpi['Hardness_LINE'] <= TARGET_MAX)
        
        yield_rate = df_kpi['All_Pass'].mean() * 100
        hrb_yield = df_kpi['HRB_Pass'].mean() * 100 
        target_yield = df_kpi['Target_Pass'].mean() * 100

        st.markdown("### 🏆 Overall Quality Metrics")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        col1.metric("📦 Total Coils", f"{total_coils:,}")
        col2.metric("✅ Mech Yield", clean_num(yield_rate, True), clean_num(yield_rate - 100, True) if yield_rate < 100 else "Perfect")
        col3.metric("🎯 HRB Yield", clean_num(hrb_yield, True), clean_num(hrb_yield - 100, True) if hrb_yield < 100 else "Control")
        col4.metric(f"🌟 Target Yield", clean_num(target_yield, True))
        col5.metric("TS Pass", clean_num(df_kpi['TS_Pass'].mean() * 100, True))
        col6.metric("YS Pass", clean_num(df_kpi['YS_Pass'].mean() * 100, True))
        col7.metric("EL Pass", clean_num(df_kpi['EL_Pass'].mean() * 100, True))
        
        st.markdown("---")

        st.markdown("### ⚠️ High-Risk Specs Watchlist")
        # Group cols for KPI — no Order_Gauge
        group_cols = ["Product_Spec", "Quality_Group", "Material"]
        valid_group_cols = [c for c in group_cols if c in df_kpi.columns]
        
        risk_summary = df_kpi.groupby(valid_group_cols).agg(
            Total_Coils=('COIL_NO', 'count'),
            Mech_Pass_Coils=('All_Pass', 'sum'),
            HRB_Pass_Coils=('HRB_Pass', 'sum'), 
            Target_Pass_Coils=('Target_Pass', 'sum'),
            Hardness_Mean=('Hardness_LINE', 'mean'),
            Hardness_Std=('Hardness_LINE', 'std'),
            LSL=('Limit_Min', 'first'),
            USL=('Limit_Max', 'first')
        ).reset_index()
        
        risk_summary['Mech Yield (%)'] = (risk_summary['Mech_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        risk_summary['HRB Yield (%)'] = (risk_summary['HRB_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        risk_summary['Target Yield (%)'] = (risk_summary['Target_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        
        risk_top = risk_summary[risk_summary['Total_Coils'] >= 3].sort_values(['Mech Yield (%)', 'HRB Yield (%)']).head(10)
        
        if not risk_top.empty:
            rename_dict = {"Product_Spec": "Specification", "Quality_Group": "Quality", "Material": "Material", "Total_Coils": "Tested Coils", "Hardness_Mean": "Avg Hardness", "Hardness_Std": "Hardness Std Dev"}
            risk_top = risk_top.rename(columns=rename_dict)
            
            cols_order = ["Specification", "Quality", "Material", "Tested Coils", "Mech Yield (%)", "HRB Yield (%)", "Target Yield (%)", "Avg Hardness", "Hardness Std Dev"]
            risk_top_display = risk_top[[c for c in cols_order if c in risk_top.columns]].copy()
            
            for col in ['Mech Yield (%)', 'HRB Yield (%)', 'Target Yield (%)']:
                if col in risk_top_display.columns: risk_top_display[col] = risk_top_display[col].apply(lambda x: clean_num(x, True))
            risk_top_display['Avg Hardness'] = risk_top_display['Avg Hardness'].apply(lambda x: clean_num(x))
            risk_top_display['Hardness Std Dev'] = risk_top_display['Hardness Std Dev'].apply(lambda x: clean_num(x))
            
            def style_risk(val):
                try:
                    num = float(str(val).replace('%', '').strip())
                    if num < 100: return 'color: #d32f2f; font-weight: bold; background-color: #ffebee'
                    if num >= 100: return 'color: #388e3c; font-weight: bold'
                except: pass
                return ''

            styled_risk = risk_top_display.style
            styled_risk = styled_risk.map(style_risk, subset=['Mech Yield (%)', 'HRB Yield (%)'])
            
            st.dataframe(styled_risk, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🔔 Visual Deep Dive: Top 10 Risk Distributions (Hardness & Mechanical)")
            top_10_risks = risk_top.head(10).to_dict('records')
            
            if len(top_10_risks) > 0:
                chart_cols = st.columns(2)
                for idx, item in enumerate(top_10_risks):
                    spec = item.get("Specification", "N/A")
                    mat = item.get("Material", "N/A")
                    tdf = df_kpi[(df_kpi.get("Product_Spec", pd.Series(dtype=str)) == spec) & (df_kpi.get("Material", pd.Series(dtype=str)) == mat)]
                    
                    if not tdf.empty:
                        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
                        fig.suptitle(f"TOP {idx+1}: {spec} | Mat: {mat}", fontsize=12, fontweight="bold")
                        
                        metrics = [
                            {"col": "Hardness_LINE", "name": "Hardness (HRB)", "min": tdf["Limit_Min"].iloc[0] if "Limit_Min" in tdf.columns else 0, "max": tdf["Limit_Max"].iloc[0] if "Limit_Max" in tdf.columns else 0, "ax": axes[0, 0], "color": "#1f77b4"},
                            {"col": "TS", "name": "Tensile Strength (TS)", "min": tdf["Standard TS min"].max() if "Standard TS min" in tdf.columns else 0, "max": tdf["Standard TS max"].min() if "Standard TS max" in tdf.columns else 0, "ax": axes[0, 1], "color": "#2ca02c"},
                            {"col": "YS", "name": "Yield Strength (YS)", "min": tdf["Standard YS min"].max() if "Standard YS min" in tdf.columns else 0, "max": tdf["Standard YS max"].min() if "Standard YS max" in tdf.columns else 0, "ax": axes[1, 0], "color": "#ff7f0e"},
                            {"col": "EL", "name": "Elongation (EL)", "min": tdf["Standard EL min"].max() if "Standard EL min" in tdf.columns else 0, "max": 0, "ax": axes[1, 1], "color": "#9467bd"}
                        ]
                        
                        for m in metrics:
                            ax = m["ax"]
                            d = tdf[m["col"]].dropna()
                            if not d.empty:
                                ax.hist(d, bins=15, color=m["color"], edgecolor="white", density=True, alpha=0.6)
                                m_val, s_val = d.mean(), d.std()
                                if s_val > 0:
                                    x_ax = np.linspace(d.min() - 2*s_val, d.max() + 2*s_val, 100)
                                    ax.plot(x_ax, (1/(s_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ax - m_val) / s_val)**2), color="#cc0000", lw=2)
                                
                                if pd.notna(m["min"]) and m["min"] > 0:
                                    ax.axvline(m["min"], color="black", linestyle="--", lw=1.5, label=f"LSL ({m['min']:.0f})")
                                if pd.notna(m["max"]) and 0 < m["max"] < 9000:
                                    ax.axvline(m["max"], color="black", linestyle="--", lw=1.5, label=f"USL ({m['max']:.0f})")
                                
                                if m["col"] == "Hardness_LINE":
                                    ax.axvline(TARGET_MIN, color="green", linestyle=":", lw=2, label="Target")
                                    ax.axvline(TARGET_MAX, color="green", linestyle=":", lw=2)
                                    ax.axvspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.1)

                                ax.set_title(f"{m['name']} (N={len(d)})", fontsize=10, fontweight="bold")
                                ax.legend(fontsize=7, loc="upper right")
                                ax.grid(alpha=0.3, linestyle=":")
                            else:
                                ax.set_title(f"{m['name']} (No Data)", fontsize=10)
                                ax.axis('off')
                                
                        plt.tight_layout()
                        chart_cols[idx % 2].pyplot(fig)
            
            st.markdown("#### 📑 Export Actionable Report")
            import streamlit.components.v1 as components
            col_csv, col_pdf, _ = st.columns([2, 2, 6])
            with col_csv: st.download_button("📥 Download Watchlist (CSV)", data=risk_top.to_csv(index=False).encode('utf-8-sig'), file_name="High_Risk_Watchlist.csv", mime="text/csv", use_container_width=True)
            with col_pdf: 
                if st.button("🖨️ Save as PDF", use_container_width=True): components.html("<script>window.parent.print();</script>", height=0)
    st.stop()

# ==============================================================================
# 🚀 GLOBAL SUMMARY DASHBOARD
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Dashboard (A118T)")
    tab1, tab2 = st.tabs(["📊 1. Performance Overview", "🧠 2. Decision Support (Risk AI)"])

    with tab1:
        stats_rows = []
        for _, g in valid.iterrows():
            conditions = [df[col] == g[col] for col in GROUP_COLS]
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 1: continue

            l_min_val, l_max_val = sub_grp['Limit_Min'].min(), sub_grp['Limit_Max'].max()
            
            def get_l_str(s_min, s_max):
                v_min = sub_grp[s_min].max() if s_min in sub_grp else 0 
                v_max = sub_grp[s_max].min() if s_max in sub_grp else 0 
                if pd.isna(v_min): v_min = 0
                if pd.isna(v_max): v_max = 0
                if v_min > 0 and v_max > 0 and v_max < 9000: return f"{v_min:.0f}~{v_max:.0f}"
                elif v_min > 0: return f"≥ {v_min:.0f}"
                elif v_max > 0 and v_max < 9000: return f"≤ {v_max:.0f}"
                else: return "-"

            n_total = len(sub_grp)
            n_ng = sub_grp[(sub_grp["Hardness_LINE"] < l_min_val) | (sub_grp["Hardness_LINE"] > l_max_val)].shape[0]

            row_data = {col: g[col] for col in GROUP_COLS}
            row_data.update({
                "HRB Limit": f"{l_min_val:.0f}~{l_max_val:.0f}", "N": n_total, "Pass Rate": ((n_total - n_ng) / n_total) * 100,
                "HRB (Avg)": sub_grp["Hardness_LINE"].mean(), "TS (Avg)": sub_grp["TS"].mean(),
                "YS (Avg)": sub_grp["YS"].mean(), "EL (Avg)": sub_grp["EL"].mean(),
                "TS Limit": get_l_str("Standard TS min", "Standard TS max"), 
                "YS Limit": get_l_str("Standard YS min", "Standard YS max"), 
                "EL Limit": get_l_str("Standard EL min", "Standard EL max")
            })
            stats_rows.append(row_data)

        if stats_rows:
            def c_pass(val): return f"background-color: {'#d4edda' if val >= 98 else ('#fff3cd' if val >= 90 else '#f8d7da')}; color: {'#155724' if val >= 98 else ('#856404' if val >= 90 else '#721c24')}; font-weight: bold"
            st.dataframe(pd.DataFrame(stats_rows).style.format("{:.1f}", subset=[c for c in pd.DataFrame(stats_rows).columns if "(Avg)" in c or "Pass" in c]).map(c_pass, subset=["Pass Rate"]).background_gradient(subset=["HRB (Avg)"], cmap="Blues"), use_container_width=True)
        else: st.warning("⚠️ Insufficient data.")

    with tab2:
        col_in1, col_in2 = st.columns([1, 1])
        with col_in1: user_hrb = st.number_input("1️⃣ Target HRB (Simulated)", value=85.0, step=0.5, format="%.1f")
        with col_in2: safety_k = st.selectbox("2️⃣ Select Safety Factor (σ):", [1.0, 2.0, 3.0], index=1)

        rows_ts, rows_ys, rows_el = [], [], []
        for _, g in valid.iterrows():
            conditions = [df[col] == g[col] for col in GROUP_COLS]
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 3: continue 

            l_min_val = sub_grp['Limit_Min'].min() if 'Limit_Min' in sub_grp.columns else 0
            l_max_val = sub_grp['Limit_Max'].max() if 'Limit_Max' in sub_grp.columns else 0
            hrb_spec_str = f"{l_min_val:.0f}~{l_max_val:.0f}" if l_max_val > 0 else "-"

            X = sub_grp[["Hardness_LINE"]].values
            
            def eval_risk(col, sp_min, sp_max, is_el=False):
                m = LinearRegression().fit(X, sub_grp[col].values)
                pred = m.predict([[user_hrb]])[0]
                rmse = np.sqrt(mean_squared_error(sub_grp[col], m.predict(X)))
                worst = pred - (safety_k * rmse) 
                best = pred + (safety_k * rmse)  
                
                sp_min = sp_min if pd.notna(sp_min) else 0
                sp_max = sp_max if pd.notna(sp_max) else 0
                
                status = "🟢 Safe"
                if sp_min > 0 and worst < sp_min: status = "🔴 Risk (Low)"
                if not is_el and 0 < sp_max < 9000 and best > sp_max: status = "🔴 Risk (High)"
                
                if is_el:
                    lim_str = f"≥ {sp_min:.1f}" if sp_min > 0 else "-"
                else:
                    if sp_min > 0 and 0 < sp_max < 9000: lim_str = f"{sp_min:.0f}~{sp_max:.0f}"
                    elif sp_min > 0: lim_str = f"≥ {sp_min:.0f}"
                    elif 0 < sp_max < 9000: lim_str = f"≤ {sp_max:.0f}"
                    else: lim_str = "-"
                    
                return pred, worst, best, lim_str, status

            try:
                b_dict = {}
                for col in GROUP_COLS:
                    if col not in ["Rolling_Type", "Metallic_Type", "Material", "Quality_Group"]:
                        b_dict[col] = g[col]
                b_dict["HRB Spec"] = hrb_spec_str
                
                ts_m_min = sub_grp["Standard TS min"].max() if "Standard TS min" in sub_grp else 0
                ts_m_max = sub_grp["Standard TS max"].min() if "Standard TS max" in sub_grp else 0
                p_ts, w_ts, b_ts, l_ts, st_ts = eval_risk("TS", ts_m_min, ts_m_max)
                dt_row = b_dict.copy()
                dt_row.update({"Pred TS": f"{p_ts:.0f}", "Est. Range": f"{w_ts:.0f}~{b_ts:.0f}", "Mech Spec": l_ts, "Status": st_ts})
                rows_ts.append(dt_row)
                
                ys_m_min = sub_grp["Standard YS min"].max() if "Standard YS min" in sub_grp else 0
                ys_m_max = sub_grp["Standard YS max"].min() if "Standard YS max" in sub_grp else 0
                p_ys, w_ys, b_ys, l_ys, st_ys = eval_risk("YS", ys_m_min, ys_m_max)
                dy = b_dict.copy()
                dy.update({"Pred YS": f"{p_ys:.0f}", "Est. Range": f"{w_ys:.0f}~{b_ys:.0f}", "Mech Spec": l_ys, "Status": st_ys})
                rows_ys.append(dy)

                el_m_min = sub_grp["Standard EL min"].max() if "Standard EL min" in sub_grp else 0
                p_el, w_el, b_el, l_el, st_el = eval_risk("EL", el_m_min, 0, is_el=True)
                de = b_dict.copy()
                de.update({"Pred EL": f"{p_el:.1f}", "Est. Range": f"{w_el:.1f}~{b_el:.1f}", "Mech Spec": l_el, "Status": st_el})
                rows_el.append(de)
            except: pass

        if rows_ts:
            def sr(val): 
                if isinstance(val, str):
                    if "🔴" in val: return 'color: #721c24; font-weight: bold; background-color: #f8d7da'
                    if "🟢" in val: return 'color: #155724; font-weight: bold; background-color: #d4edda'
                return ''
            
            st.info("💡 **Note:** The **HRB Spec** column is the original hardness standard (to compare with Target HRB). The **Est. Range** is automatically compared against the **Mech Spec** to trigger bidirectional risk warnings (Low/High).")
            
            c_top1, c_top2 = st.columns(2)
            with c_top1: 
                st.markdown("##### 🔹 Tensile Strength (TS)")
                st.dataframe(pd.DataFrame(rows_ts).style.map(sr, subset=["Status"]), use_container_width=True, hide_index=True)
            with c_top2: 
                st.markdown("##### 🔸 Yield Strength (YS)")
                st.dataframe(pd.DataFrame(rows_ys).style.map(sr, subset=["Status"]), use_container_width=True, hide_index=True)
            
            st.markdown("##### 🔻 Elongation (EL)")
            st.dataframe(pd.DataFrame(rows_el).style.map(sr, subset=["Status"]), use_container_width=True, hide_index=True)
    st.stop()

# ==============================================================================
# 👑 MASTER DICTIONARY EXPORT
# ==============================================================================
if view_mode == "👑 Master Dictionary Export":
    st.markdown("---")
    st.header("👑 Master Mechanical Properties Dictionary (A118T)")
    st.info("💡 **Interactive View & Export:** Review the logically grouped limits directly on the screen below, then download the formatted Excel file for your records.")
    
    if st.button("🚀 Generate Comprehensive Dictionary", type="primary"):
        master_data = []
        clean_master_df = df_master_full.dropna(subset=['Hardness_LINE', 'TS', 'YS', 'EL'])
        
        sigma_n = 2.0
        target_k = 1.0

        for keys, group in clean_master_df.groupby(GROUP_COLS):
            if len(group) < 3: continue 
            
            data = group["Hardness_LINE"]
            mu = data.mean()
            mrs = np.abs(np.diff(data.values))
            sigma_imr = np.mean(mrs) / 1.128 if len(mrs) > 0 else data.std()
            
            c_min, c_max = mu - sigma_n * sigma_imr, mu + sigma_n * sigma_imr
            t_min, t_max = mu - target_k * sigma_imr, mu + target_k * sigma_imr
            
            X_train = group[["Hardness_LINE"]].values
            m_ts = LinearRegression().fit(X_train, group["TS"].values)
            m_ys = LinearRegression().fit(X_train, group["YS"].values)
            m_el = LinearRegression().fit(X_train, group["EL"].values)
            
            s_ts_min = group["Standard TS min"].max()
            s_ts_max = group["Standard TS max"].min()
            s_ys_min = group["Standard YS min"].max()
            s_ys_max = group["Standard YS max"].min()
            s_el_min = group["Standard EL min"].max()
            
            def fmt_s(mi, ma):
                if pd.isna(mi): mi = 0
                if pd.isna(ma): ma = 0
                if mi > 0 and 0 < ma < 9000: return f"{mi:.0f}~{ma:.0f}"
                elif mi > 0: return f"≥ {mi:.0f}"
                elif 0 < ma < 9000: return f"≤ {ma:.0f}"
                return "-"

            master_dict = {col: (keys[idx] if isinstance(keys, tuple) else keys) for idx, col in enumerate(GROUP_COLS)}
            master_dict.update({
                "N Coils": len(group),
                "Current Hardness Spec": f"{group['Limit_Min'].max():.1f}~{group['Limit_Max'].min():.1f}",
                "Proposed Control Limit (2σ)": f"{c_min:.1f} ~ {c_max:.1f}",
                "🎯 Proposed Target Zone (1σ)": f"{t_min:.1f} ~ {t_max:.1f}",
                "Spec: TS": fmt_s(s_ts_min, s_ts_max),
                "Exp. TS (at Target)": f"{int(m_ts.predict([[t_min]])[0])}~{int(m_ts.predict([[t_max]])[0])}",
                "Spec: YS": fmt_s(s_ys_min, s_ys_max),
                "Exp. YS (at Target)": f"{int(m_ys.predict([[t_min]])[0])}~{int(m_ys.predict([[t_max]])[0])}",
                "Spec: EL": f"≥ {s_el_min:.1f}%" if s_el_min > 0 else "-",
                "Exp. EL (at Target)": f"{min(m_el.predict([[t_min]])[0], m_el.predict([[t_max]])[0]):.1f}% ~ {max(m_el.predict([[t_min]])[0], m_el.predict([[t_max]])[0]):.1f}%"
            })
            master_data.append(master_dict)
        
        if master_data:
            df_out = pd.DataFrame(master_data)
            
            ordered_cols = GROUP_COLS + [
                "N Coils", 
                "Current Hardness Spec", "Proposed Control Limit (2σ)", "🎯 Proposed Target Zone (1σ)",
                "Spec: TS", "Exp. TS (at Target)",
                "Spec: YS", "Exp. YS (at Target)",
                "Spec: EL", "Exp. EL (at Target)"
            ]
            final_cols = [c for c in ordered_cols if c in df_out.columns]
            df_out = df_out[final_cols]
            
            st.markdown("### 👁️ Preview Master Dictionary")
            
            styled_df = df_out.style.set_properties(**{'background-color': '#FFF2CC', 'color': '#856404'}, subset=[c for c in final_cols if "Spec:" in c or "Current Hardness Spec" in c]) \
                                    .set_properties(**{'background-color': '#D9EAD3', 'color': '#155724', 'font-weight': 'bold'}, subset=[c for c in final_cols if "Target" in c or "Exp." in c]) \
                                    .set_properties(**{'background-color': '#CFE2F3', 'color': '#004085'}, subset=["Proposed Control Limit (2σ)"])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            import datetime as dt
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, sheet_name='Master_Specs', index=False)
                workbook = writer.book
                worksheet = writer.sheets['Master_Specs']
                
                header_fmt = workbook.add_format({'bold': True, 'bg_color': '#CFE2F3', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
                target_fmt = workbook.add_format({'bg_color': '#D9EAD3', 'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
                spec_fmt = workbook.add_format({'bg_color': '#FFF2CC', 'italic': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
                
                worksheet.set_row(0, 30) 
                
                for col_num, value in enumerate(df_out.columns.values):
                    fmt = header_fmt
                    if "Target" in value or "Exp." in value: fmt = target_fmt
                    if "Spec:" in value or "Current Hardness Spec" in value: fmt = spec_fmt
                    
                    worksheet.write(0, col_num, value, fmt)
                    worksheet.set_column(col_num, col_num, max(12, len(value) * 0.8))
            
            st.markdown("### 📥 Download Report")
            st.success(f"✅ Full Master Dictionary created for {len(master_data)} groups!")
            st.download_button("📥 Download Full Dictionary (Excel)", output.getvalue(), f"Full_Master_Dictionary_{dt.datetime.now().strftime('%Y%m%d')}.xlsx")
    st.stop()
    
# ==============================================================================
# MAIN LOOP FOR ALL OTHER VIEWS 
# ==============================================================================
for i, (_, g) in enumerate(valid.iterrows()):
    conditions = [df[col] == g[col] for col in GROUP_COLS]
    sub = df[np.logical_and.reduce(conditions)].sort_values("COIL_NO")

    lo, hi = sub["Limit_Min"].iloc[0], sub["Limit_Max"].iloc[0]

    sub["NG_LAB"] = (sub.get("Hardness_LAB", sub["Hardness_LINE"]) < lo) | (sub.get("Hardness_LAB", sub["Hardness_LINE"]) > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"] 

    group_title = " | ".join([f"{str(g[c])}" for c in GROUP_COLS])
    st.markdown("---")
    st.markdown(f"### 🧱 {group_title}")
    st.markdown(f"**Coils:** {sub['COIL_NO'].nunique()} | **Std Limit:** {lo:.1f} ~ {hi:.1f}")
        
    if view_mode == "📋 Data Inspection":
        display_df = sub.drop(columns=["Rolling_Type"]) if "Rolling_Type" in sub.columns else sub.copy()
        
        date_cols = display_df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        for d_col in date_cols:
            display_df[d_col] = display_df[d_col].dt.strftime('%Y-%m-%d')
            
        for col in display_df.columns:
            if 'DATE' in str(col).upper() and col not in date_cols:
                try:
                    display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass
        
        def highlight_ng_rows(row): 
            return ['background-color: #ffe6e6'] * len(row) if row.get('NG', False) else [''] * len(row)
        
        num_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        
        fmt = {}
        for c in num_cols:
            if c in ["Hardness_LINE", "Hardness_LAB"]:
                fmt[c] = "{:.1f}"
            else:
                fmt[c] = "{:.0f}"
                
        st.dataframe(display_df.style.format(fmt).apply(highlight_ng_rows, axis=1), use_container_width=True)

    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & SPC"])
        
        with tab_trend:
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4.5))

            # --- LAB series ---
            lab_ng_mask = np.zeros(len(sub), dtype=bool)
            if "Hardness_LAB" in sub.columns and not sub["Hardness_LAB"].isna().all():
                lab_vals = sub["Hardness_LAB"].values.astype(float)
                ax.plot(x, lab_vals, linewidth=1.8, color="#6baed6", alpha=0.55, zorder=2)
                lab_valid   = ~np.isnan(lab_vals)
                lab_ok_mask = lab_valid & (lab_vals >= lo) & (lab_vals <= hi)
                lab_ng_mask = lab_valid & ((lab_vals < lo) | (lab_vals > hi))
                ax.scatter(x[lab_ok_mask], lab_vals[lab_ok_mask], s=40, color="#6baed6", zorder=3, label="LAB")
                if lab_ng_mask.any():
                    ax.scatter(x[lab_ng_mask], lab_vals[lab_ng_mask], s=110, color="red", zorder=5,
                               marker="o", edgecolors="darkred", linewidths=1.4, label="LAB NG ⚠️")
                    for xi, yi in zip(x[lab_ng_mask], lab_vals[lab_ng_mask]):
                        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, -14),
                                    ha="center", fontsize=7.5, color="darkred", fontweight="bold")

            # --- LINE series ---
            line_vals = sub["Hardness_LINE"].values.astype(float)
            ax.plot(x, line_vals, linewidth=1.8, color="#fd8d3c", alpha=0.75, zorder=2)
            line_valid    = ~np.isnan(line_vals)
            line_ok_mask  = line_valid & (line_vals >= lo) & (line_vals <= hi)
            line_ng_mask  = line_valid & ((line_vals < lo) | (line_vals > hi))
            ax.scatter(x[line_ok_mask], line_vals[line_ok_mask], s=45, marker="s", color="#fd8d3c", zorder=3, label="LINE")
            if line_ng_mask.any():
                ax.scatter(x[line_ng_mask], line_vals[line_ng_mask], s=120, marker="s", color="red", zorder=5,
                           edgecolors="darkred", linewidths=1.5, label="LINE NG ⚠️")
                for xi, yi in zip(x[line_ng_mask], line_vals[line_ng_mask]):
                    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=7.5, color="darkred", fontweight="bold")

            # --- Spec & target reference lines ---
            ax.axhline(lo,         linestyle="--", linewidth=2, color="red",   label=f"Std LSL={lo}")
            ax.axhline(hi,         linestyle="--", linewidth=2, color="red",   label=f"Std USL={hi}")
            ax.axhline(TARGET_MIN, linestyle="--", linewidth=2, color="green", label=f"Target LSL={TARGET_MIN}")
            ax.axhline(TARGET_MAX, linestyle="--", linewidth=2, color="green", label=f"Target USL={TARGET_MAX}")
            ax.fill_between(x, TARGET_MIN, TARGET_MAX, color="green", alpha=0.1, label="Target Zone")

            # --- Title with NG summary ---
            total_ng = int(line_ng_mask.sum()) + int(lab_ng_mask.sum())
            if total_ng > 0:
                ax.set_title(
                    f"Hardness Trend by Coil Sequence  |  🔴 {total_ng} point(s) out of Std Limit",
                    weight="bold", color="#8B0000")
            else:
                ax.set_title(
                    "Hardness Trend by Coil Sequence  |  ✅ All points within Std Limit",
                    weight="bold")

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=6)
            ax.grid(alpha=0.3, linestyle=":")
            plt.tight_layout()
            st.pyplot(fig)
            
        with tab_dist:
            line = sub["Hardness_LINE"].dropna()
            lab = sub["Hardness_LAB"].dropna() if "Hardness_LAB" in sub.columns else pd.Series(dtype=float)
            
            if len(line) < 5: st.warning("⚠️ At least 5 coils are required for distribution analysis.")
            else:
                spc_line = None
                if len(line) >= 2 and line.std(ddof=1) > 0:
                    mean, std = line.mean(), line.std(ddof=1)
                    cp = (hi - lo) / (6 * std)
                    ca = ((mean - (hi + lo) / 2) / ((hi - lo) / 2)) * 100
                    cpu, cpl = (hi - mean) / (3 * std), (mean - lo) / (3 * std)
                    spc_line = (mean, std, cp, ca, min(cpu, cpl))

                mean_line, std_line = line.mean(), line.std(ddof=1)
                
                vals = [line.min(), line.max(), lo, hi, TARGET_MIN, TARGET_MAX]
                if not lab.empty: vals.extend([lab.min(), lab.max()])
                x_min, x_max = min(vals) - 2, max(vals) + 2
                
                fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                ax_dist.hist(line, bins=np.linspace(x_min, x_max, 30), density=True, alpha=0.6, color="#ff7f0e", edgecolor="white", label="LINE Hist")
                if not lab.empty: ax_dist.hist(lab, bins=np.linspace(x_min, x_max, 30), density=True, alpha=0.3, color="#1f77b4", edgecolor="None", label="LAB Hist")
                
                if std_line > 0:
                    xs = np.linspace(x_min, x_max, 400)
                    ax_dist.plot(xs, (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2), linewidth=2.5, color="#b25e00", label="LINE Fit")
                
                ax_dist.axvline(lo, linestyle="--", linewidth=2, color="red", label="Std LSL")
                ax_dist.axvline(hi, linestyle="--", linewidth=2, color="red", label="Std USL")
                ax_dist.axvline(TARGET_MIN, linestyle=":", linewidth=2, color="green", label="Target LSL")
                ax_dist.axvline(TARGET_MAX, linestyle=":", linewidth=2, color="green", label="Target USL")
                ax_dist.axvspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.1)

                ax_dist.set_xlim(x_min, x_max)
                ax_dist.set_title("Hardness Distribution (LINE vs LAB)", weight="bold")
                ax_dist.legend(); ax_dist.grid(alpha=0.3)
                st.pyplot(fig_dist)

                if spc_line:
                    mean_val, std_val, cp_val, ca_val, cpk_val = spc_line
                    eval_msg = "Excellent" if cpk_val >= 1.33 else ("Good" if cpk_val >= 1.0 else "Poor")
                    color_code = "green" if cpk_val >= 1.33 else ("orange" if cpk_val >= 1.0 else "red")
                    df_spc = pd.DataFrame([{"N": len(line), "Mean": mean_val, "Std": std_val, "Cp": cp_val, "Ca (%)": ca_val, "Cpk": cpk_val, "Rating": eval_msg}])
                    st.dataframe(df_spc.style.format("{:.2f}", subset=["Mean", "Std", "Cp", "Ca (%)", "Cpk"]).map(lambda v: f'color: {color_code}; font-weight: bold', subset=['Rating']), hide_index=True)

    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        if i == 0: corr_bin_summary = []
        sub_corr = sub.dropna(subset=["Hardness_LINE","TS","YS","EL"]).copy()
        
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub_corr["HRB_bin"] = pd.cut(sub_corr["Hardness_LINE"], bins=bins, labels=labels, right=False)
        
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
            ax2 = ax.twinx()
            
            def p_prop(ax_obj, x, y, ymin, ymax, c, lbl, m):
                ax_obj.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax_obj.fill_between(x, ymin, ymax, color=c, alpha=0.1)
            
            p_prop(ax, x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS Actual", "o")
            p_prop(ax, x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS Actual", "s")
            p_prop(ax2, x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL Actual", "^")

            g_ts_min = summary["Std_TS_min"].max()
            g_ts_max = summary["Std_TS_max"].min()
            g_ys_min = summary["Std_YS_min"].max()
            g_ys_max = summary["Std_YS_max"].min()
            g_el_min = summary["Std_EL_min"].max()

            if pd.notna(g_ts_min) and g_ts_min > 0: ax.axhline(g_ts_min, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.5, label=f"TS LSL ({g_ts_min:.0f})")
            if pd.notna(g_ts_max) and 0 < g_ts_max < 9000: ax.axhline(g_ts_max, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.5, label=f"TS USL ({g_ts_max:.0f})")
            if pd.notna(g_ys_min) and g_ys_min > 0: ax.axhline(g_ys_min, color="#2ca02c", linestyle="-.", lw=1.5, alpha=0.5, label=f"YS LSL ({g_ys_min:.0f})")
            if pd.notna(g_ys_max) and 0 < g_ys_max < 9000: ax.axhline(g_ys_max, color="#2ca02c", linestyle="-.", lw=1.5, alpha=0.5, label=f"YS USL ({g_ys_max:.0f})")
            if pd.notna(g_el_min) and g_el_min > 0: ax2.axhline(g_el_min, color="#ff7f0e", linestyle=":", lw=2, alpha=0.6, label=f"EL LSL ({g_el_min:.0f})")

            for j, row in enumerate(summary.itertuples()):
                ts_min, ts_max = row.Std_TS_min, row.Std_TS_max
                ys_min, ys_max = row.Std_YS_min, row.Std_YS_max
                el_spec = row.Std_EL_min if pd.notna(row.Std_EL_min) else 0
                
                ts_fail = (pd.notna(ts_min) and ts_min > 0 and row.TS_mean < ts_min) or (pd.notna(ts_max) and 0 < ts_max < 9000 and row.TS_mean > ts_max)
                ys_fail = (pd.notna(ys_min) and ys_min > 0 and row.YS_mean < ys_min) or (pd.notna(ys_max) and 0 < ys_max < 9000 and row.YS_mean > ys_max)
                el_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                
                ax.annotate(f"{row.TS_mean:.0f}" + (" ❌" if ts_fail else ""), (x[j], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="red" if ts_fail else "#1f77b4")
                ax.annotate(f"{row.YS_mean:.0f}" + (" ❌" if ys_fail else ""), (x[j], row.YS_mean), xytext=(0,-15), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="red" if ys_fail else "#2ca02c")
                ax2.annotate(f"{row.EL_mean:.1f}%" + (" ❌" if el_fail else ""), (x[j], row.EL_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, color="red" if el_fail else "#ff7f0e", fontweight=("bold" if el_fail else "normal"))

            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_ylabel("Strength (MPa)", fontweight="bold")
            ax2.set_ylabel("Elongation (%)", fontweight="bold", color="#ff7f0e")
            ax.set_title("Hardness vs Mechanical Properties with Spec Limits", fontweight="bold")
            
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left", bbox_to_anchor=(1.08, 0.5))
            
            ax.grid(True, ls="--", alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)

            specs_str = f"Specs: {', '.join(str(x) for x in sub['Product_Spec'].dropna().unique())}" if 'Product_Spec' in sub.columns else "Specs: N/A"

            def check_limit(act_min, act_max, sp_min, sp_max, is_el=False):
                fail = False
                if pd.notna(sp_min) and sp_min > 0 and act_min < sp_min: fail = True
                if not is_el and pd.notna(sp_max) and 0 < sp_max < 9000 and act_max > sp_max: fail = True
                res = f"{act_min:.1f}~{act_max:.1f}" if is_el else f"{act_min:.0f}~{act_max:.0f}"
                return res + (" ❌" if fail else " ✅")

            for row in summary.itertuples():
                bin_data = sub_corr[sub_corr["HRB_bin"] == row.HRB_bin]
                
                ts_act_str = check_limit(row.TS_min, row.TS_max, row.Std_TS_min, row.Std_TS_max)
                ys_act_str = check_limit(row.YS_min, row.YS_max, row.Std_YS_min, row.Std_YS_max)
                el_act_str = check_limit(row.EL_min, row.EL_max, row.Std_EL_min, 0, is_el=True)

                ts_std = bin_data['TS'].std()
                ys_std = bin_data['YS'].std()
                el_std = bin_data['EL'].std()

                corr_bin_summary.append({
                    "Specification List": specs_str, "Material": g.get("Material", "N/A"),
                    "Hardness Bin": row.HRB_bin, "N": row.N_coils,
                    "TS Spec": f"{row.Std_TS_min:.0f}~{row.Std_TS_max:.0f}" if pd.notna(row.Std_TS_max) and row.Std_TS_max < 9000 else (f"≥{row.Std_TS_min:.0f}" if pd.notna(row.Std_TS_min) else "-"),
                    "TS Actual": ts_act_str, "TS Mean": f"{row.TS_mean:.1f}", "TS Std": f"{ts_std:.1f}" if pd.notna(ts_std) else "-",
                    "YS Spec": f"{row.Std_YS_min:.0f}~{row.Std_YS_max:.0f}" if pd.notna(row.Std_YS_max) and row.Std_YS_max < 9000 else (f"≥{row.Std_YS_min:.0f}" if pd.notna(row.Std_YS_min) else "-"),
                    "YS Actual": ys_act_str, "YS Mean": f"{row.YS_mean:.1f}", "YS Std": f"{ys_std:.1f}" if pd.notna(ys_std) else "-",
                    "EL Spec": f"≥{row.Std_EL_min:.0f}" if pd.notna(row.Std_EL_min) else "-",
                    "EL Actual": el_act_str, "EL Mean": f"{row.EL_mean:.1f}", "EL Std": f"{el_std:.1f}" if pd.notna(el_std) else "-"
                })

        if i == len(valid) - 1 and 'corr_bin_summary' in locals() and len(corr_bin_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 📊 Hardness Binning Comprehensive Report")
            df_full = pd.DataFrame(corr_bin_summary)
            
            def d_bin(title, cols, c_code):
                st.markdown(f"#### {title}")
                target_df = df_full[["Specification List", "Material", "Hardness Bin", "N"] + cols]
                
                def hl_status(val):
                    if isinstance(val, str):
                        if '❌' in val: return 'color: #721c24; font-weight: bold; background-color: #f8d7da'
                        if '✅' in val: return 'color: #155724; font-weight: bold'
                    return ''
                
                styled = target_df.style.set_properties(**{'background-color': c_code, 'font-weight': 'bold'}, subset=[c for c in cols if "Std" in c])
                styled = styled.map(hl_status, subset=[c for c in cols if "Actual" in c])
                    
                st.dataframe(styled, use_container_width=True, hide_index=True)

            d_bin("📉 TS Analysis by Hardness Bin", ["TS Spec", "TS Actual", "TS Mean", "TS Std"], "#e6f2ff")
            d_bin("📉 YS Analysis by Hardness Bin", ["YS Spec", "YS Actual", "YS Mean", "YS Std"], "#f2fff2")
            d_bin("📉 EL Analysis by Hardness Bin", ["EL Spec", "EL Actual", "EL Mean", "EL Std"], "#fff5e6")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_full.to_excel(writer, sheet_name='All_Data', index=False)
                df_full[["Specification List", "Material", "Hardness Bin", "N", "TS Spec", "TS Actual", "TS Mean", "TS Std"]].to_excel(writer, sheet_name='TS_Only', index=False)
                df_full[["Specification List", "Material", "Hardness Bin", "N", "YS Spec", "YS Actual", "YS Mean", "YS Std"]].to_excel(writer, sheet_name='YS_Only', index=False)
                df_full[["Specification List", "Material", "Hardness Bin", "N", "EL Spec", "EL Actual", "EL Mean", "EL Std"]].to_excel(writer, sheet_name='EL_Only', index=False)
                for s in writer.sheets:
                    writer.sheets[s].set_column('A:A', 25); writer.sheets[s].set_column('B:C', 15); writer.sheets[s].set_column('D:Z', 12) 
            
            st.download_button("📥 Export Binning Report (Excel)", data=output.getvalue(), file_name=f"Hardness_Bin_Report_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif view_mode == "⚙️ Mech Props Analysis":
        if i == 0: ts_summary, ys_summary, el_summary = [], [], []

        props_config = [
            {"col": "TS", "name": "Tensile Strength (TS)", "color": "#1f77b4", "min_c": "Standard TS min", "max_c": "Standard TS max"},
            {"col": "YS", "name": "Yield Strength (YS)", "color": "#2ca02c", "min_c": "Standard YS min", "max_c": "Standard YS max"},
            {"col": "EL", "name": "Elongation (EL)", "color": "#ff7f0e", "min_c": "Standard EL min", "max_c": "Standard EL max"}
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        has_data = False
        
        h_data = sub["Hardness_LINE"].dropna()
        hrb_rng = f"{h_data.min():.1f} ~ {h_data.max():.1f}" if not h_data.empty else "N/A"
        
        for j, cfg in enumerate(props_config):
            col = cfg["col"]
            data = sub[col].dropna()
            
            if not data.empty:
                has_data = True
                mean, std = data.mean(), data.std() if len(data) > 1 else 0
                
                spec_min = sub[cfg["min_c"]].max() if cfg["min_c"] in sub.columns else 0
                spec_max = sub[cfg["max_c"]].min() if cfg["max_c"] in sub.columns else 0
                if pd.isna(spec_min): spec_min = 0
                if pd.isna(spec_max): spec_max = 0
                
                lcl_3s, ucl_3s = mean - 3 * std, mean + 3 * std
                
                axes[j].hist(data, bins=15, color=cfg["color"], alpha=0.5, density=True, label="Actual Data")
                if std > 0:
                    x_p = np.linspace(data.min() - 3*std, data.max() + 3*std, 200)
                    axes[j].plot(x_p, (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2), color=cfg["color"], lw=2, label="Normal Dist")
                
                if spec_min > 0: axes[j].axvline(spec_min, color="red", linestyle="--", linewidth=2, label=f"Spec Min ({spec_min:.0f})")
                if spec_max > 0 and spec_max < 9000: axes[j].axvline(spec_max, color="red", linestyle="--", linewidth=2, label=f"Spec Max ({spec_max:.0f})")
                axes[j].axvline(lcl_3s, color="blue", linestyle=":", linewidth=2, label=f"-3σ LCL ({lcl_3s:.1f})")
                axes[j].axvline(ucl_3s, color="blue", linestyle=":", linewidth=2, label=f"+3σ UCL ({ucl_3s:.1f})")
                
                axes[j].set_title(f"{cfg['name']}\n(Mean={mean:.1f}, Std={std:.1f})", fontweight="bold")
                axes[j].legend(fontsize=9, loc="upper right")
                
                cpk = None
                if std > 0 and (spec_min > 0 or (spec_max > 0 and spec_max < 9000)):
                    if spec_min > 0 and spec_max > 0 and spec_max < 9000:
                        cpk = min((spec_max - mean) / (3 * std), (mean - spec_min) / (3 * std))
                    elif spec_min > 0:
                        cpk = (mean - spec_min) / (3 * std)
                    elif spec_max > 0 and spec_max < 9000:
                        cpk = (spec_max - mean) / (3 * std)

                row_data = {
                    "Group": group_title, "N": len(data), "Hardness Range (HRB)": hrb_rng,
                    "Limit (Spec)": f"{spec_min:.0f}~{spec_max:.0f}" if (spec_max > 0 and spec_max < 9000) else (f"≥ {spec_min:.0f}" if spec_min > 0 else "N/A"),
                    "Actual Range": f"{data.min():.1f}~{data.max():.1f}",
                    "Mean": f"{mean:.1f}", "Std Dev": f"{std:.1f}", 
                    "LCL (-3σ)": f"{lcl_3s:.1f}", "UCL (+3σ)": f"{ucl_3s:.1f}",
                    "Cpk": f"{cpk:.2f}" if cpk is not None else "-"
                }
                if col == "TS": ts_summary.append(row_data)
                elif col == "YS": ys_summary.append(row_data)
                elif col == "EL": el_summary.append(row_data)
            else: axes[j].set_title(f"{cfg['name']}\n(No Data)")
            axes[j].grid(alpha=0.3, linestyle="--")

        if has_data: st.pyplot(fig)
        else: st.warning("⚠️ Insufficient Mechanical Properties data for the selected group.")

        if i == len(valid) - 1:
            st.markdown("---")
            st.markdown("## 📊 Mechanical Properties Comprehensive Report")
            
            def d_sum(title, data_list, c_code):
                if data_list:
                    st.markdown(f"#### {title}")
                    
                    def cpk_color(val):
                        try:
                            v = float(val)
                            if v >= 1.33: return 'color: green; font-weight: bold'
                            elif v >= 1.0: return 'color: orange; font-weight: bold'
                            else: return 'color: red; font-weight: bold'
                        except: return ''
                        
                    styled_df = pd.DataFrame(data_list).style.set_properties(**{'font-weight': 'bold'}, subset=['Mean']) \
                                        .set_properties(**{'background-color': '#f0f8ff', 'font-weight': 'bold', 'color': '#0056b3'}, subset=['Hardness Range (HRB)']) \
                                        .set_properties(**{'background-color': c_code, 'color': '#004085'}, subset=['LCL (-3σ)', 'UCL (+3σ)']) \
                                        .map(cpk_color, subset=['Cpk'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

            d_sum("1️⃣ Tensile Strength (TS)", ts_summary, "#e6f2ff") 
            d_sum("2️⃣ Yield Strength (YS)", ys_summary, "#f2fff2")   
            d_sum("3️⃣ Elongation (EL)", el_summary, "#fff5e6")        

            if ts_summary or ys_summary or el_summary:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    if ts_summary: pd.DataFrame(ts_summary).to_excel(writer, sheet_name='TS_Summary', index=False)
                    if ys_summary: pd.DataFrame(ys_summary).to_excel(writer, sheet_name='YS_Summary', index=False)
                    if el_summary: pd.DataFrame(el_summary).to_excel(writer, sheet_name='EL_Summary', index=False)
                st.download_button("📥 Export Full Mech Report (Excel)", data=output.getvalue(), file_name=f"Mech_Report_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        c1, c2 = st.columns(2)
        actual_min = float(sub["Hardness_LINE"].min()) if not sub["Hardness_LINE"].empty else 0.0
        actual_max = float(sub["Hardness_LINE"].max()) if not sub["Hardness_LINE"].empty else 100.0
        mn = c1.number_input("Min HRB", value=actual_min, step=0.5, key=f"lk1_{i}")
        mx = c2.number_input("Max HRB", value=actual_max, step=0.5, key=f"lk2_{i}")
        filt = sub[(sub["Hardness_LINE"] >= mn) & (sub["Hardness_LINE"] <= mx)].dropna(subset=["TS", "YS", "EL"])
        if not filt.empty: 
            st.success(f"✅ Found {len(filt)} coils.")
            st.dataframe(filt[["TS", "YS", "EL"]].describe().T.style.format("{:.1f}"), use_container_width=True)
        else: st.error("No coils found.")

    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        spec_ts_min = sub["Standard TS min"].max() if "Standard TS min" in sub.columns else 0
        spec_ts_max = sub["Standard TS max"].min() if "Standard TS max" in sub.columns else 0
        spec_ys_min = sub["Standard YS min"].max() if "Standard YS min" in sub.columns else 0
        spec_ys_max = sub["Standard YS max"].min() if "Standard YS max" in sub.columns else 0
        spec_el_min = sub["Standard EL min"].max() if "Standard EL min" in sub.columns else 0
        
        def fmt_spec(s_min, s_max):
            if pd.isna(s_min): s_min = 0
            if pd.isna(s_max): s_max = 0
            if s_min > 0 and 0 < s_max < 9000: return f"{s_min:.0f} ~ {s_max:.0f}"
            elif s_min > 0: return f"≥ {s_min:.0f}"
            elif 0 < s_max < 9000: return f"≤ {s_max:.0f}"
            return "N/A"

        ts_spec_str = fmt_spec(spec_ts_min, spec_ts_max)
        ys_spec_str = fmt_spec(spec_ys_min, spec_ys_max)
        el_spec_str = fmt_spec(spec_el_min, 0)
        
        st.info(f"📋 **Current Standard Specs:** &nbsp;&nbsp;&nbsp; TS: **{ts_spec_str}** &nbsp;&nbsp;|&nbsp;&nbsp; YS: **{ys_spec_str}** &nbsp;&nbsp;|&nbsp;&nbsp; EL: **{el_spec_str}**")
        
        c1, c2, c3 = st.columns(3)
        r_ts_min = c1.number_input("Min TS", value=float(sub['TS'].min()) if not sub['TS'].isna().all() else 0.0, step=5.0, key=f"tmin_{i}")
        r_ts_max = c1.number_input("Max TS", value=float(sub['TS'].max()) if not sub['TS'].isna().all() else 1000.0, step=5.0, key=f"tmax_{i}")
        
        r_ys_min = c2.number_input("Min YS", value=float(sub['YS'].min()) if not sub['YS'].isna().all() else 0.0, step=5.0, key=f"ymin_{i}")
        r_ys_max = c2.number_input("Max YS", value=float(sub['YS'].max()) if not sub['YS'].isna().all() else 1000.0, step=5.0, key=f"ymax_{i}")
        
        r_el_min = c3.number_input("Min EL", value=float(sub['EL'].min()) if not sub['EL'].isna().all() else 0.0, step=1.0, key=f"emin_{i}")
        r_el_max = c3.number_input("Max EL", value=float(sub['EL'].max()) if not sub['EL'].isna().all() else 100.0, step=1.0, key=f"emax_{i}")

        filtered = sub[(sub['YS'] >= r_ys_min) & (sub['YS'] <= r_ys_max) & (sub['TS'] >= r_ts_min) & (sub['TS'] <= r_ts_max) & (sub['EL'] >= r_el_min) & (sub['EL'] <= r_el_max)]
        
        if not filtered.empty:
            st.success(f"✅ Found **{len(filtered)}** coils. Optimal Hardness Range: **{filtered['Hardness_LINE'].min():.1f} ~ {filtered['Hardness_LINE'].max():.1f} HRB**")
            st.dataframe(filtered[['COIL_NO','Hardness_LINE','TS','YS','EL']].style.format("{:.1f}", subset=['Hardness_LINE', 'TS', 'YS', 'EL']), use_container_width=True, hide_index=True)
        else: 
            st.error("❌ No coils found matching these target parameters.")

    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown(f"#### 🧮 AI Prediction Engine: {group_title}")
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        
        if len(train_df) < 3:
            st.warning("⚠️ At least 3 coils are required to activate AI Prediction.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                mean_h = train_df["Hardness_LINE"].mean()
                target_h = st.number_input("🎯 Target Hardness", value=float(round(mean_h, 1)), step=0.1, key=f"ai_{i}")
            
            X_train = train_df[["Hardness_LINE"]].values
            preds = {}
            model_metrics = {}
            
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                val = model.predict([[target_h]])[0]
                preds[col] = val 
                y_true = train_df[col].values
                y_pred = model.predict(X_train)
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                model_metrics[col] = {"r2": r2, "rmse": rmse}

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            colors = {"TS": "#2980b9", "YS": "#27ae60", "EL": "#c0392b"} 
            idx = list(range(len(train_df)))
            nxt = len(train_df)

            for col in ["TS", "YS", "EL"]:
                sec = (col == "EL")
                fig.add_trace(go.Scatter(
                    x=idx, y=train_df[col], mode='lines', line=dict(color=colors[col], width=2, shape='spline'), 
                    name=f"{col} (History)", opacity=0.6, hoverinfo='y' 
                ), secondary_y=sec)
                
                last_val_raw = train_df[col].iloc[-1]
                pred_clean = round(preds[col], 1) if col == "EL" else int(round(preds[col]))
                last_clean = round(last_val_raw, 1) if col == "EL" else int(round(last_val_raw))
                
                fig.add_trace(go.Scatter(
                    x=[idx[-1], nxt], y=[last_val_raw, preds[col]], mode='lines',
                    line=dict(color=colors[col], width=2, dash='dot'), showlegend=False, hoverinfo='skip'
                ), secondary_y=sec)

                fig.add_trace(go.Scatter(
                    x=[nxt], y=[preds[col]], mode='markers+text', text=[f"<b>{pred_clean}</b>"], 
                    textposition="middle right" if nxt < 10 else "top center",
                    marker=dict(color=colors[col], size=14, symbol='diamond', line=dict(width=2, color='white')), 
                    name=f"Pred {col}",
                    hovertemplate=(f"<b>🎯 Pred {col}: {pred_clean}</b><br>🔙 Last {col}: {last_clean}<br>📈 Change: {pred_clean - last_clean:.1f}<extra></extra>")
                ), secondary_y=sec)

            fig.add_vline(x=nxt - 0.5, line_width=1, line_dash="dash", line_color="gray")
            fig.add_annotation(x=nxt - 0.5, y=1.05, yref="paper", text="Forecast Zone ➔", showarrow=False, font=dict(color="gray"))

            fig.update_layout(
                height=500, title=dict(text=f"📈 AI Prediction at Target Hardness = {target_h}", font=dict(size=18)),
                plot_bgcolor="white", hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=80, b=20)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', title="Coil Sequence")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', secondary_y=False, title="Strength (MPa)")
            fig.update_yaxes(showgrid=False, secondary_y=True, title="Elongation (%)")

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### 🏁 Forecast Summary & Spec Evaluation")
            c1, c2, c3 = st.columns(3)
            def get_delta(p, l): return round(p - l, 1)
            last_ts = train_df["TS"].iloc[-1]; last_ys = train_df["YS"].iloc[-1]; last_el = train_df["EL"].iloc[-1]

            ts_m_min = sub["Standard TS min"].max() if "Standard TS min" in sub.columns else 0
            ts_m_max = sub["Standard TS max"].min() if "Standard TS max" in sub.columns else 0
            ys_m_min = sub["Standard YS min"].max() if "Standard YS min" in sub.columns else 0
            ys_m_max = sub["Standard YS max"].min() if "Standard YS max" in sub.columns else 0
            el_m_min = sub["Standard EL min"].max() if "Standard EL min" in sub.columns else 0
            
            def check_sp(val, s_min, s_max, is_el=False):
                s_min = s_min if pd.notna(s_min) else 0
                s_max = s_max if pd.notna(s_max) else 0
                lim_str = f"{s_min:.0f}~{s_max:.0f}" if (0 < s_max < 9000) else (f"≥ {s_min:.0f}" if s_min > 0 else "-")
                if is_el: lim_str = f"≥ {s_min:.1f}" if s_min > 0 else "-"
                
                if s_min > 0 and val < s_min: return "❌ FAIL", lim_str
                if not is_el and 0 < s_max < 9000 and val > s_max: return "❌ FAIL", lim_str
                return "✅ PASS", lim_str

            ts_stat, ts_spec = check_sp(preds['TS'], ts_m_min, ts_m_max)
            ys_stat, ys_spec = check_sp(preds['YS'], ys_m_min, ys_m_max)
            el_stat, el_spec = check_sp(preds['EL'], el_m_min, 0, is_el=True)

            c1.metric(f"Tensile (TS) - {ts_stat}", f"{int(round(preds['TS']))} MPa", f"{get_delta(preds['TS'], last_ts)} vs Last")
            c1.caption(f"**Spec:** {ts_spec} | **R²:** {model_metrics['TS']['r2']:.2f}")

            c2.metric(f"Yield (YS) - {ys_stat}", f"{int(round(preds['YS']))} MPa", f"{get_delta(preds['YS'], last_ys)} vs Last")
            c2.caption(f"**Spec:** {ys_spec} | **R²:** {model_metrics['YS']['r2']:.2f}")

            c3.metric(f"Elongation (EL) - {el_stat}", f"{round(preds['EL'], 1)} %", f"{get_delta(preds['EL'], last_el)} vs Last")
            c3.caption(f"**Spec:** {el_spec} | **R²:** {model_metrics['EL']['r2']:.2f}")

    elif view_mode == "🎛️ Control Limit Calculator (Compare 3 Methods)":
        import io
        from scipy.stats import norm
        import seaborn as sns

        if i == 0:
            all_groups_summary = []
            st.markdown("### 📘 Control Limit Calculation Methods")
            with st.expander("🔍 Click to view method details", expanded=True):
                st.markdown("""
                | Method | Name | Description |
                | :--- | :--- | :--- |
                | **M1: Standard** | Standard Stat | Calculated based on all data. Limits can be over-stretched if extreme outliers exist. |
                | **M2: IQR Robust** | Interquartile Range | Automatically filters out extreme values, making limits more aligned with actual distribution. |
                | **M3: Hybrid** | Smart Hybrid | Combines statistical trends and customer specifications to ensure limits stay in safe zones. |
                | **M4: I-MR (SPC)** | Process Control | **Optimal approach:** Monitors variation between adjacent coils; highly scientific for process stability. |
                """)

        st.markdown(f"### 🎛️ Control Limits Analysis: {group_title}")
        sub_clean = sub[(sub["Hardness_LINE"].notna()) & (sub["Hardness_LINE"] > 0)].copy()
        data = sub_clean["Hardness_LINE"]
        data_lab = sub_clean["Hardness_LAB"].dropna() if "Hardness_LAB" in sub_clean.columns else pd.Series(dtype=float)

        if len(data) < 5:
            st.warning(f"⚠️ Not enough data for analysis (N={len(data)}). Minimum 5 coils required.")
        else:
            with st.expander("⚙️ Parameter Settings", expanded=False):
                c1, c2 = st.columns(2)
                sigma_n = c1.number_input("1. Sigma Multiplier (K)", 1.0, 6.0, 2.0, 0.5, key=f"sig_{i}")
                iqr_k = c2.number_input("2. IQR Sensitivity", 0.1, 3.0, 0.5, 0.1, key=f"iqr_{i}")

            spec_min = lo
            spec_max = hi
            display_max = spec_max if (spec_max > 0 and spec_max < 9000) else 0

            mu = data.mean()
            std_dev = data.std() if len(data) > 1 else 1.0

            # M1: Standard
            m1_min, m1_max = mu - sigma_n * std_dev, mu + sigma_n * std_dev

            # M2: IQR Robust
            Q1 = data.quantile(0.25); Q3 = data.quantile(0.75); IQR = Q3 - Q1
            clean_data = data[~((data < (Q1 - iqr_k * IQR)) | (data > (Q3 + iqr_k * IQR)))]
            if clean_data.empty or len(clean_data) < 2: clean_data = data
            mu_clean, sigma_clean = clean_data.mean(), clean_data.std()
            if pd.isna(sigma_clean) or sigma_clean == 0: sigma_clean = std_dev
            m2_min, m2_max = mu_clean - sigma_n * sigma_clean, mu_clean + sigma_n * sigma_clean

            # M3: Smart Hybrid
            m3_min = max(m2_min, spec_min)
            m3_max = min(m2_max, spec_max) if (spec_max > 0 and spec_max < 9000) else m2_max
            if m3_min >= m3_max: m3_min, m3_max = m2_min, m2_max

            # M4: I-MR (SPC)
            mrs = np.abs(np.diff(data.values))
            mr_bar = np.mean(mrs) if len(mrs) > 0 else 0
            sigma_imr = mr_bar / 1.128 if mr_bar > 0 else std_dev
            m4_min, m4_max = mu - sigma_n * sigma_imr, mu + sigma_n * sigma_imr

            # New Core Target
            target_k = 1.0
            new_target_min = mu - target_k * sigma_imr
            new_target_max = mu + target_k * sigma_imr

            # ── Mechanical specs ──────────────────────────────────────────────
            sub_mech = sub_clean.dropna(subset=["TS", "YS", "EL"]).copy()
            s_ts_min = sub_mech["Standard TS min"].max() if "Standard TS min" in sub_mech.columns else 0
            s_ts_max = sub_mech["Standard TS max"].min() if "Standard TS max" in sub_mech.columns else 0
            s_ys_min = sub_mech["Standard YS min"].max() if "Standard YS min" in sub_mech.columns else 0
            s_ys_max = sub_mech["Standard YS max"].min() if "Standard YS max" in sub_mech.columns else 0
            s_el_min = sub_mech["Standard EL min"].max()  if "Standard EL min" in sub_mech.columns else 0
            for v in [s_ts_min, s_ts_max, s_ys_min, s_ys_max, s_el_min]:
                if pd.isna(v): v = 0

            def fmt_spec_cl(mi, ma):
                if pd.isna(mi): mi = 0
                if pd.isna(ma): ma = 0
                if mi > 0 and 0 < ma < 9000: return f"{mi:.0f}~{ma:.0f}"
                elif mi > 0: return f"≥ {mi:.0f}"
                elif 0 < ma < 9000: return f"≤ {ma:.0f}"
                return "-"

            st.info(f"🎯 **Mechanical Specs Target:** TS: **{fmt_spec_cl(s_ts_min, s_ts_max)}** | YS: **{fmt_spec_cl(s_ys_min, s_ys_max)}** | EL: **{fmt_spec_cl(s_el_min, 0)}**")

            # ── AI regression models ──────────────────────────────────────────
            has_model = False
            if len(sub_mech) >= 3:
                has_model = True
                X_tr = sub_mech[["Hardness_LINE"]].values
                m_ts = LinearRegression().fit(X_tr, sub_mech["TS"].values)
                m_ys = LinearRegression().fit(X_tr, sub_mech["YS"].values)
                m_el = LinearRegression().fit(X_tr, sub_mech["EL"].values)

            def get_theo(l_min, l_max):
                if not has_model: return ("-","-"), ("-","-"), ("-","-")
                calc_max = l_max if l_max > 0 else sub_mech["Hardness_LINE"].max()
                ts_p = sorted([m_ts.predict([[l_min]])[0], m_ts.predict([[calc_max]])[0]])
                ys_p = sorted([m_ys.predict([[l_min]])[0], m_ys.predict([[calc_max]])[0]])
                el_p = sorted([m_el.predict([[l_min]])[0], m_el.predict([[calc_max]])[0]])
                return ts_p, ys_p, el_p

            def get_actual(l_min, l_max):
                mask = sub_mech["Hardness_LINE"] >= l_min
                if l_max > 0: mask = mask & (sub_mech["Hardness_LINE"] <= l_max)
                ac = sub_mech[mask]
                if ac.empty: return "No Data", "No Data", "No Data"
                ts_s = f"{ac['TS'].min():.0f}~{ac['TS'].max():.0f} (Avg:{ac['TS'].mean():.0f})"
                ys_s = f"{ac['YS'].min():.0f}~{ac['YS'].max():.0f} (Avg:{ac['YS'].mean():.0f})"
                el_s = f"{ac['EL'].min():.1f}~{ac['EL'].max():.1f} (Avg:{ac['EL'].mean():.1f})"
                return ts_s, ys_s, el_s

            def eval_prop_cl(p_lo, p_hi, sp_min, sp_max, is_el=False):
                if not has_model: return "N/A"
                sp_min = sp_min if pd.notna(sp_min) else 0
                sp_max = sp_max if pd.notna(sp_max) else 0
                if sp_min > 0 and p_lo < sp_min: return "❌ Fail"
                if not is_el and 0 < sp_max < 9000 and p_hi > sp_max: return "❌ Fail"
                return "✅ Pass"

            configs_cl = [
                ("🎯 Old Target Goal",           spec_min,        display_max,     "-"),
                ("🔴 M1: Standard (Historical)",  m1_min,          m1_max,          f"σ={std_dev:.2f}"),
                ("🔵 M2: IQR (Robust)",           m2_min,          m2_max,          f"σ={sigma_clean:.2f}"),
                ("🟢 M3: Smart Hybrid",           m3_min,          m3_max,          "-"),
                ("🟣 M4: I-MR (Control Limits)",  m4_min,          m4_max,          f"σ={sigma_imr:.2f}"),
                (f"🌟 New Core Target (±{target_k}σ)", new_target_min, new_target_max, "-"),
            ]

            rows_cl = []
            passed_methods = []
            for cat, l_min, l_max, sig in configs_cl:
                ts_p, ys_p, el_p = get_theo(l_min, l_max)
                act_ts, act_ys, act_el = get_actual(l_min, l_max)
                lim_str = f"{l_min:.1f} ~ {l_max:.1f}" if l_max > 0 else f"≥ {l_min:.1f}"

                if has_model:
                    ts_eval = eval_prop_cl(ts_p[0], ts_p[1], s_ts_min, s_ts_max)
                    ys_eval = eval_prop_cl(ys_p[0], ys_p[1], s_ys_min, s_ys_max)
                    el_eval = eval_prop_cl(el_p[0], el_p[1], s_el_min, 0, is_el=True)
                    theo_ts = f"{ts_p[0]:.0f} ~ {ts_p[1]:.0f}"
                    theo_ys = f"{ys_p[0]:.0f} ~ {ys_p[1]:.0f}"
                    theo_el = f"{el_p[0]:.1f} ~ {el_p[1]:.1f}"
                    overall = "✅ Optimal" if all(e == "✅ Pass" for e in [ts_eval, ys_eval, el_eval]) else "⚠️ Warning"
                else:
                    ts_eval = ys_eval = el_eval = "N/A"
                    theo_ts = theo_ys = theo_el = "-"
                    overall = "N/A"

                if overall == "✅ Optimal" and "M" in cat: passed_methods.append(cat)

                rows_cl.append({
                    "Limit Type":          cat,
                    "Hardness Limits":     lim_str,
                    "Variation (σ)":       sig,
                    "Theoretical TS":      theo_ts,
                    "Actual TS":           act_ts,
                    "TS Eval":             ts_eval,
                    "Theoretical YS":      theo_ys,
                    "Actual YS":           act_ys,
                    "YS Eval":             ys_eval,
                    "Theoretical EL (%)":  theo_el,
                    "Actual EL (%)":       act_el,
                    "EL Eval":             el_eval,
                    "Overall":             overall,
                })

            # Best recommendation logic
            if any("M4" in m for m in passed_methods): best_rec = "🟣 M4: I-MR (Recommended)"
            elif any("M3" in m for m in passed_methods): best_rec = "🟢 M3: Smart Hybrid"
            elif any("M2" in m for m in passed_methods): best_rec = "🔵 M2: IQR Robust"
            elif any("M1" in m for m in passed_methods): best_rec = "🔴 M1: Standard"
            else: best_rec = "❌ High Risk – Manual Review Required"

            st.success(f"🏆 **Best Recommended Control Limit:** {best_rec}")

            df_cl = pd.DataFrame(rows_cl)

            eval_cols = ["TS Eval", "YS Eval", "EL Eval", "Overall"]

            def hl_cl_row(s):
                if "🌟 New Core Target" in str(s["Limit Type"]):
                    return ["background-color: #e6f4ea"] * len(s)
                return [""] * len(s)

            def hl_cl_val(val):
                if isinstance(val, str):
                    if "✅" in val: return "color: #155724; font-weight: bold"
                    if "❌" in val: return "color: #721c24; font-weight: bold; background-color: #f8d7da"
                    if "⚠️" in val: return "color: #856404; font-weight: bold"
                return ""

            styled_cl = (
                df_cl.style
                .apply(hl_cl_row, axis=1)
                .map(hl_cl_val, subset=eval_cols)
                .set_properties(**{"background-color": "#f8f9fa"}, subset=["Actual TS", "Actual YS", "Actual EL (%)"])
            )
            st.dataframe(styled_cl, use_container_width=True, hide_index=True)

            if not has_model:
                st.caption("⚠️ Not enough mech data (N<3) for AI estimation.")
            else:
                st.caption("(**) Theoretical = AI Linear Regression estimate. Actual = observed min/max of coils within that hardness range.")

            # ── Excel export per group ────────────────────────────────────────
            buf_cl = io.BytesIO()
            with pd.ExcelWriter(buf_cl, engine="xlsxwriter") as writer:
                df_cl.to_excel(writer, sheet_name="Control_Limits", index=False)
                wb = writer.book; ws = writer.sheets["Control_Limits"]
                hfmt = wb.add_format({"bold": True, "bg_color": "#CFE2F3", "border": 1, "align": "center"})
                pfmt = wb.add_format({"font_color": "#155724", "bg_color": "#D4EDDA", "bold": True, "border": 1})
                ffmt = wb.add_format({"font_color": "#721c24", "bg_color": "#F8D7DA", "bold": True, "border": 1})
                for ci, cn in enumerate(df_cl.columns):
                    ws.write(0, ci, cn, hfmt)
                    ws.set_column(ci, ci, max(14, len(cn)))
                for ri in range(len(df_cl)):
                    for ci in range(len(df_cl.columns)):
                        v = str(df_cl.iloc[ri, ci])
                        fmt = pfmt if ("Pass" in v or "Optimal" in v) else (ffmt if ("Fail" in v or "Warning" in v) else None)
                        if fmt: ws.write(ri + 1, ci, v, fmt)

            safe_grp = re.sub(r"[^a-zA-Z0-9_\-]", "_", group_title)
            st.download_button(
                label="📥 Download Control Limit Report (Excel)",
                data=buf_cl.getvalue(),
                file_name=f"ControlLimit_{safe_grp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_cl_{i}"
            )

            # ── Distribution charts ───────────────────────────────────────────
            st.markdown(f"#### 📈 Distribution Analysis")
            tab_all, tab_m1m4 = st.tabs(["📊 All 4 Methods", "📊 M1 vs M4"])

            x_min_sync = mu - 5 * std_dev
            x_max_sync = mu + 5 * std_dev
            if spec_min > 0: x_min_sync = min(x_min_sync, spec_min - 2)
            if display_max > 0: x_max_sync = max(x_max_sync, display_max + 2)
            x_axis_cl = np.linspace(x_min_sync, x_max_sync, 500)
            shared_bins = np.linspace(x_min_sync, x_max_sync, 30)
            bin_w = shared_bins[1] - shared_bins[0]

            def _draw_dist(ax_d, show_m2m3=True):
                sns.histplot(data, stat="count", bins=shared_bins, color="#1f77b4", edgecolor="white", alpha=0.6, label="LINE (Production)", ax=ax_d)
                if not data_lab.empty:
                    sns.histplot(data_lab, stat="count", bins=shared_bins, color="#ff7f0e", edgecolor="white", alpha=0.4, label="LAB (QC)", ax=ax_d)
                ax_d.plot(x_axis_cl, norm.pdf(x_axis_cl, mu, std_dev) * len(data) * bin_w, color="#004085", lw=2.5, label="Normal Curve (LINE)")
                if not data_lab.empty and len(data_lab) > 1:
                    mu_l, std_l = data_lab.mean(), data_lab.std()
                    ax_d.plot(x_axis_cl, norm.pdf(x_axis_cl, mu_l, std_l) * len(data_lab) * bin_w, color="#b33c00", lw=2.5, ls="-.", label="Normal Curve (LAB)")
                if spec_min > 0:
                    ax_d.axvline(spec_min, color="black", lw=2.5, label=f"Old Spec ({spec_min:.1f}~{display_max:.1f})")
                    if display_max > 0: ax_d.axvline(display_max, color="black", lw=2.5)
                ax_d.axvline(m1_min, color="#d62728", ls="--", lw=2, label=f"M1 ({m1_min:.1f}~{m1_max:.1f})")
                ax_d.axvline(m1_max, color="#d62728", ls="--", lw=2)
                ax_d.axvline(m4_min, color="#9467bd", ls="-", lw=2.5, label=f"M4 ({m4_min:.1f}~{m4_max:.1f})")
                ax_d.axvline(m4_max, color="#9467bd", ls="-", lw=2.5)
                if show_m2m3:
                    ax_d.axvline(m2_min, color="#007acc", ls="-.", lw=2, label=f"M2 ({m2_min:.1f}~{m2_max:.1f})")
                    ax_d.axvline(m2_max, color="#007acc", ls="-.", lw=2)
                    ax_d.axvline(m3_min, color="#28a745", ls=":", lw=2.5, label=f"M3 ({m3_min:.1f}~{m3_max:.1f})")
                    ax_d.axvline(m3_max, color="#28a745", ls=":", lw=2.5)
                ax_d.set_xlim(x_min_sync, x_max_sync)
                ax_d.set_xlabel("Hardness (HRB)", fontweight="bold")
                ax_d.set_ylabel("Number of Coils", fontweight="bold")
                ax_d.grid(axis="y", ls=":", alpha=0.5)
                ax_d.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

            with tab_all:
                fig_all, ax_all = plt.subplots(figsize=(11, 5))
                _draw_dist(ax_all, show_m2m3=True)
                ax_all.set_title(f"All 4 Methods – {group_title}", fontsize=12, fontweight="bold")
                plt.tight_layout(); st.pyplot(fig_all); plt.close(fig_all)

            with tab_m1m4:
                fig_m4, ax_m4 = plt.subplots(figsize=(11, 5))
                _draw_dist(ax_m4, show_m2m3=False)
                ax_m4.set_title(f"M1 vs M4 – {group_title}", fontsize=12, fontweight="bold")
                plt.tight_layout(); st.pyplot(fig_m4); plt.close(fig_m4)

            # ── Accumulate for end-of-loop summary ───────────────────────────
            all_groups_summary.append({
                "Group":                    group_title,
                "N":                        len(data),
                "Current Spec":             f"{spec_min:.0f}~{display_max:.0f}",
                "M1: Standard":             f"{m1_min:.1f}~{m1_max:.1f}",
                "M2: IQR (Robust)":         f"{m2_min:.1f}~{m2_max:.1f}",
                "M3: Smart Hybrid":         f"{m3_min:.1f}~{m3_max:.1f}",
                "M4: I-MR (Optimal)":       f"{m4_min:.1f}~{m4_max:.1f}",
                "New Core Target (±1σ)":    f"{new_target_min:.1f}~{new_target_max:.1f}",
                "Best Recommendation":      best_rec,
                "Status":                   "✅ Stable" if (display_max > 0 and m4_max <= display_max) else "⚠️ Narrow Spec",
            })

        # ── END-OF-LOOP: Grand Summary Table (all groups, no gauge split) ────
        if i == len(valid) - 1 and "all_groups_summary" in locals() and len(all_groups_summary) > 0:
            st.markdown("---")
            st.markdown("## 📊 Overall Control Limit Summary (All Groups)")
            df_total = pd.DataFrame(all_groups_summary)

            def hl_total(s):
                styles = [""] * len(s)
                status_val = str(s.get("Status", ""))
                if "Narrow" in status_val or "⚠️" in status_val:
                    styles = ["background-color: #fff3cd"] * len(s)
                return styles

            def hl_total_val(val):
                if isinstance(val, str):
                    if "✅" in val: return "color: #155724; font-weight: bold"
                    if "⚠️" in val or "❌" in val: return "color: #856404; font-weight: bold"
                return ""

            styled_total = (
                df_total.style
                .apply(hl_total, axis=1)
                .map(hl_total_val, subset=["Best Recommendation", "Status"])
                .set_properties(**{"background-color": "#e6f2ff", "color": "#004085", "font-weight": "bold"}, subset=["M4: I-MR (Optimal)"])
                .set_properties(**{"background-color": "#e2efda", "color": "#155724", "font-weight": "bold"}, subset=["New Core Target (±1σ)"])
            )
            st.dataframe(styled_total, use_container_width=True, hide_index=True)

            buf_total = io.BytesIO()
            with pd.ExcelWriter(buf_total, engine="xlsxwriter") as writer:
                df_total.to_excel(writer, sheet_name="SPC_Summary", index=False)
                wb2 = writer.book; ws2 = writer.sheets["SPC_Summary"]
                hfmt2 = wb2.add_format({"bold": True, "bg_color": "#CFE2F3", "border": 1, "align": "center", "text_wrap": True})
                m4fmt = wb2.add_format({"bg_color": "#e6f2ff", "bold": True, "border": 1, "align": "center"})
                tgfmt = wb2.add_format({"bg_color": "#e2efda", "bold": True, "border": 1, "align": "center"})
                ws2.set_row(0, 30)
                for ci, cn in enumerate(df_total.columns):
                    fmt2 = m4fmt if "M4" in cn else (tgfmt if "Target" in cn else hfmt2)
                    ws2.write(0, ci, cn, fmt2)
                    ws2.set_column(ci, ci, max(14, len(cn) + 2))

            st.download_button(
                label="📥 Export Complete SPC Summary (Excel)",
                data=buf_total.getvalue(),
                file_name=f"SPC_Summary_A118T_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        # ==============================================================================
        # DISPLAY FACTORY-WIDE SUMMARY (OUTSIDE INDIVIDUAL LOOP)
        # ==============================================================================
        if i == len(valid) - 1 and 'all_groups_summary' in locals() and len(all_groups_summary) > 0:
            st.markdown(f"## 📊 Factory-wide Operation & Control Limits Summary: {qgroup}")
            df_total = pd.DataFrame(all_groups_summary)
            
            def style_recommendation(val):
                if 'M4' in str(val) or 'M3' in str(val) or 'M2' in str(val) or 'M1' in str(val): return 'color: #155724; background-color: #d4edda; font-weight: bold'
                elif 'Keep' in str(val): return 'color: #856404; background-color: #fff3cd; font-weight: bold'
                else: return 'color: #721c24; background-color: #f8d7da; font-weight: bold'

            styled_df_total = df_total.style
            if hasattr(styled_df_total, "map"):
                styled_df_total = styled_df_total.map(style_recommendation, subset=['🚧 Control Limit Rec.'])\
                                                 .set_properties(**{'background-color': '#e6f4ea', 'font-weight': 'bold', 'color': '#0d5302'}, subset=['🎯 Core Target (±1.0σ)'])
            else:
                styled_df_total = styled_df_total.applymap(style_recommendation, subset=['🚧 Control Limit Rec.'])\
                                                 .set_properties(**{'background-color': '#e6f4ea', 'font-weight': 'bold', 'color': '#0d5302'}, subset=['🎯 Core Target (±1.0σ)'])
            
            st.dataframe(styled_df_total, use_container_width=True, hide_index=True)

            total_buffer = io.BytesIO()
            with pd.ExcelWriter(total_buffer, engine='xlsxwriter') as writer:
                df_total.to_excel(writer, index=False, sheet_name='Factory_Summary')
                workbook = writer.book
                worksheet = writer.sheets['Factory_Summary']
                
                header_fmt = workbook.add_format({'bold': True, 'bg_color': '#CFE2F3', 'border': 1, 'align': 'center'})
                for col_num, value in enumerate(df_total.columns.values):
                    worksheet.write(0, col_num, value, header_fmt)
                    worksheet.set_column(col_num, col_num, 18)
            
            st.download_button(
                label="📥 Download Factory-wide Summary (Excel)",
                data=total_buffer.getvalue(),
                file_name=f"Factory_Summary_{qgroup}_{datetime.now().strftime('%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_total_summary_final"
            )
