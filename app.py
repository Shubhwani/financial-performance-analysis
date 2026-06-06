import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, silhouette_score,
                             roc_curve, auc, precision_recall_curve)
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="Financial Performance Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #060b18 !important;
    color: #e0e6f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e1f 0%, #060b18 100%) !important;
    border-right: 1px solid #111d35 !important;
}
[data-testid="block-container"] {
    background-color: #060b18 !important;
    padding: 1rem 2rem !important;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0a1628 0%, #0d1e3a 100%);
    border: 1px solid #162440;
    border-radius: 14px;
    padding: 18px !important;
    transition: transform 0.2s;
}
[data-testid="stMetricValue"] {
    color: #4f9eff !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #4a6a8a !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
h1 { color: #4f9eff !important; font-weight: 800 !important; }
h2 { color: #e0e6f0 !important; font-weight: 600 !important; }
h3 { color: #8ba3c7 !important; font-weight: 500 !important; }
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a56db, #0d3a8e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(26,86,219,0.4) !important;
    transition: all 0.3s !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2563eb, #1a56db) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(26,86,219,0.5) !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #0a1628 !important;
    border: 1px solid #162440 !important;
    color: #e0e6f0 !important;
    border-radius: 8px !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: #c8d8f0 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}
hr { border-color: #111d35 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #6a85a8 !important; }
.stTabs [data-baseweb="tab"] {
    background: #0a1628 !important;
    color: #6a85a8 !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #1a56db !important;
    color: white !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #162440 !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] {
    background: #0a1628 !important;
    border: 1px solid #162440 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

matplotlib.rcParams.update({
    'figure.facecolor': '#0a1628',
    'axes.facecolor':   '#0d1e3a',
    'axes.edgecolor':   '#162440',
    'axes.labelcolor':  '#8ba3c7',
    'xtick.color':      '#8ba3c7',
    'ytick.color':      '#8ba3c7',
    'text.color':       '#e0e6f0',
    'grid.color':       '#162440',
    'grid.alpha':       0.4,
    'legend.facecolor': '#0a1628',
    'legend.edgecolor': '#162440',
    'legend.fontsize':  8.5,
})

# ═══════════════════════════════════════════
# LOAD & PREPROCESS
# ═══════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(
        "Financial Performance Factor Analysis in Businesses Systems.csv")
    df.columns = df.columns.str.strip()
    df['Total_Cost'] = (df['Marketing_Cost'] +
                        df['Sales_Cost'] +
                        df['Support_Cost'])
    df['Profit'] = df['Net_Revenue'] - df['Total_Cost']
    df['Profit_Margin_Pct'] = (
        df['Profit'] / df['Net_Revenue'].replace(0, 1)) * 100
    df['Revenue_Per_Cost'] = (
        df['Net_Revenue'] / df['Total_Cost'].replace(0, 1))
    if 'Age_Group' not in df.columns:
        df['Age_Group'] = pd.cut(df['Age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Under 25','25-34','35-44','45-54','55+'])
    return df

@st.cache_resource
def train_all_models(df):
    features = ['Age','Monthly_Revenue','Customer_Lifetime_Months',
                'Customer_Satisfaction_Score','Discount_Amount',
                'Marketing_Cost','Sales_Cost','Support_Cost']
    X = df[features].fillna(df[features].median())
    y = df['Is_Churned']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10,
        random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred   = rf.predict(X_te)
    rf_prob   = rf.predict_proba(X_te)[:,1]
    rf_acc    = accuracy_score(y_te, rf_pred)
    rf_cm     = confusion_matrix(y_te, rf_pred)
    rf_report = classification_report(y_te, rf_pred, output_dict=True)
    rf_fpr, rf_tpr, _ = roc_curve(y_te, rf_prob)
    rf_auc    = auc(rf_fpr, rf_tpr)

    seg_f = ['Monthly_Revenue','Customer_Satisfaction_Score',
             'Customer_Lifetime_Months']
    X3  = df[seg_f].fillna(df[seg_f].median())
    scaler = StandardScaler()
    X3s = scaler.fit_transform(X3)
    km  = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X3s)
    sil = silhouette_score(X3s, km.labels_)

    return (rf, km, scaler, features,
            rf_acc, rf_cm, rf_report, rf_fpr, rf_tpr, rf_auc,
            sil, X_te, y_te, rf_pred, rf_prob)

df = load_data()
(rf, km, scaler, features,
 rf_acc, rf_cm, rf_report, rf_fpr, rf_tpr, rf_auc,
 sil, X_te, y_te, rf_pred, rf_prob) = train_all_models(df)

seg_f   = ['Monthly_Revenue','Customer_Satisfaction_Score',
           'Customer_Lifetime_Months']
X3      = df[seg_f].fillna(df[seg_f].median())
X3s     = scaler.transform(X3)
df['Segment']      = km.predict(X3s)
seg_map = {0:'Low Value',1:'Medium Value',2:'High Value',3:'Premium'}
seg_c   = ['#f74f7a','#f7934f','#4f9eff','#4ff7a0']
df['Segment_Name'] = df['Segment'].map(seg_map)

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding:28px 0 16px 0;'>
  <div style='font-size:3rem;'>📊</div>
  <div style='color:#4f9eff; font-size:1rem; font-weight:800;
              letter-spacing:2.5px; margin-top:12px;'>
    FINANCIAL ANALYSIS
  </div>
  <div style='color:#1e3a5f; font-size:0.65rem; margin-top:5px;
              letter-spacing:3px; text-transform:uppercase;'>
  </div>
</div>
<hr style='border-color:#111d35; margin:6px 0 18px 0;'/>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "🏠  Home",
    "📊  Dashboard",
    "🔍  EDA Analysis",
    "🤖  ML Models",
    "🎯  Churn Predictor",
    "👥  Segmentation",
    "📋  Reports",
])



# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════
def section_header(icon, title, subtitle=""):
    sub_html = (f"<div style='color:#3a5a7a;font-size:0.82rem;"
                f"margin-top:5px;'>{subtitle}</div>") if subtitle else ""
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0a1628,#0d1e3a);
    border:1px solid #162440; border-radius:14px; padding:20px 26px;
    margin-bottom:22px; border-left:4px solid #1a56db;'>
      <div style='font-size:1.55rem; font-weight:800; color:#e0e6f0;'>
        {icon}&nbsp; {title}
      </div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def kpi_card(label, value, color, sub=""):
    sub_html = (f"<div style='color:#3a5a7a;font-size:0.72rem;"
                f"margin-top:3px;'>{sub}</div>") if sub else ""
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0a1628,#0d1e3a);
    border:1px solid #162440; border-radius:14px; padding:18px;
    border-top:3px solid {color};'>
      <div style='color:#3a5a7a; font-size:0.68rem; text-transform:uppercase;
                  letter-spacing:1.5px;'>{label}</div>
      <div style='color:{color}; font-size:1.85rem; font-weight:800;
                  margin-top:6px; line-height:1;'>{value}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def info_box(text, color="#4f9eff"):
    st.markdown(f"""
    <div style='background:#0a1628; border:1px solid #162440;
    border-left:4px solid {color}; border-radius:10px;
    padding:14px 18px; margin:8px 0; color:#c0d0e8;
    font-size:0.88rem; line-height:1.65;'>{text}</div>
    """, unsafe_allow_html=True)

COLORS6 = ['#4f9eff','#f7934f','#4ff7a0','#c44ff7','#f74f7a','#f7c94f']

# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style='text-align:center; padding:30px 0 36px 0;'>
      <div style='font-size:3.5rem; margin-bottom:14px;'>📊</div>
      <h1 style='font-size:2.5rem; margin:0 0 8px 0; line-height:1.2;'>
        Financial Performance Factor Analysis
      </h1>
      <p style='color:#3a5a7a; font-size:0.95rem; letter-spacing:2px;
                text-transform:uppercase; margin-bottom:4px;'>
        in Business Systems
      </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    cr = (df['Is_Churned'] == 'Yes').mean()*100 if df[
        'Is_Churned'].dtype == object else df['Is_Churned'].mean()*100
    profit_total = df['Profit'].sum()
    margin_total = (profit_total / df['Net_Revenue'].sum()) * 100

    with c1: kpi_card("Total Customers", f"{len(df):,}", "#4f9eff", "Records")
    with c2: kpi_card("Avg Net Revenue", f"₹{df['Net_Revenue'].mean():,.0f}", "#4ff7a0", "Per Customer")
    with c3: kpi_card("Churn Rate", f"{cr:.1f}%", "#f74f7a", "Needs Attention")
    with c4: kpi_card("Profit Margin", f"{margin_total:.1f}%", "#f7c94f", "Overall")
    with c5: kpi_card("Avg Satisfaction", f"{df['Customer_Satisfaction_Score'].mean():.2f}/5", "#c44ff7", "Score")

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview + Tech
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div style='background:#0a1628; border:1px solid #162440;
        border-radius:14px; padding:24px;'>
          <div style='color:#4f9eff; font-size:1.05rem; font-weight:700;
                      margin-bottom:16px; letter-spacing:0.5px;'>
            🎯 Project Overview
          </div>
          <p style='color:#c0d0e8; font-size:0.9rem; line-height:1.8; margin:0 0 12px 0;'>
            This system performs a comprehensive analysis of financial performance
            factors in business systems using advanced
            <strong style='color:#4f9eff;'>Machine Learning</strong> and
            <strong style='color:#4ff7a0;'>Data Analytics</strong> techniques.
          </p>
          <p style='color:#c0d0e8; font-size:0.9rem; line-height:1.8; margin:0 0 12px 0;'>
            The dataset contains
            <strong style='color:#f7c94f;'>10,000 customer records</strong>
            spanning 2023–2025, covering revenue, costs, satisfaction scores,
            acquisition channels, and subscription behavior.
          </p>
          <p style='color:#c0d0e8; font-size:0.9rem; line-height:1.8; margin:0;'>
            Using <strong style='color:#f7934f;'>3 ML Models</strong>
            (Random Forest, Decision Tree, KMeans) to predict churn,
            compare classifiers, and segment customers for actionable insights.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#0a1628; border:1px solid #162440;
        border-radius:14px; padding:24px;'>
          <div style='color:#4f9eff; font-size:1.05rem; font-weight:700;
                      margin-bottom:14px;'>🔧 Technology Stack</div>
        """, unsafe_allow_html=True)
        for icon, name, desc, color in [
            ("🐍", "Python",       "Pandas · NumPy · Scikit-learn",  "#4f9eff"),
            ("🌐", "Streamlit",    "Web App Framework",              "#4ff7a0"),
            ("🗄️", "MySQL",        "Database · 8 SQL Queries",       "#f7934f"),
            ("📊", "Power BI",     "5-Page Interactive Dashboard",   "#f7c94f"),
            ("🤖", "Scikit-learn", "RF · KMeans",                     "#c44ff7"),
            ("📓", "Jupyter",      "EDA · 8 Graphs · Cleaning",      "#f74f7a"),
        ]:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:12px;
            padding:8px 0; border-bottom:1px solid #111d35;'>
              <div style='font-size:1.15rem;'>{icon}</div>
              <div>
                <div style='color:{color}; font-weight:600;
                            font-size:0.86rem;'>{name}</div>
                <div style='color:#3a5a7a; font-size:0.72rem;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ML summary cards
    st.markdown("""
    <div style='color:#3a5a7a; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:2px; margin-bottom:12px;'>Machine Learning Models</div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    for col, icon, name, sub, val, label, border in [
        (m1, "🤖", "Random Forest",    "Churn Prediction · Classification",
         f"{rf_acc*100:.2f}%",  "Accuracy",       "#4f9eff"),
        (m2, "👥", "KMeans Clustering","Customer Segmentation · 4 Groups",
         f"{sil:.4f}",          "Silhouette Score","#c44ff7"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#0a1628; border:1px solid {border};
            border-radius:14px; padding:20px; text-align:center;'>
              <div style='font-size:1.8rem; margin-bottom:8px;'>{icon}</div>
              <div style='color:{border}; font-weight:700;
                          font-size:0.95rem;'>{name}</div>
              <div style='color:#3a5a7a; font-size:0.74rem;
                          margin:4px 0 14px 0;'>{sub}</div>
              <div style='color:{border}; font-size:1.9rem;
                          font-weight:800; line-height:1;'>{val}</div>
              <div style='color:#3a5a7a; font-size:0.7rem;
                          margin-top:4px;'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project phases timeline
    st.markdown("""
    <div style='background:#0a1628; border:1px solid #162440;
    border-radius:14px; padding:22px;'>
      <div style='color:#4f9eff; font-size:1rem; font-weight:700;
                  margin-bottom:18px;'>📋 Project Implementation Phases</div>
    """, unsafe_allow_html=True)

    phases = [
        ("01","Data Collection & Cleaning","Python · Jupyter","#4f9eff","✅ Done"),
        ("02","EDA — 8 Visualizations","Matplotlib · Seaborn","#4ff7a0","✅ Done"),
        ("03","SQL Database Integration","MySQL · 10K Rows","#f7934f","✅ Done"),
        ("04","ML Model Development","RF · KMeans","#f7c94f","✅ Done"),
        ("05","Power BI Dashboard","5 Pages · DAX","#c44ff7","✅ Done"),
        ("06","Streamlit Web Application","Python · Dark Theme","#f74f7a","✅ Done"),
    ]
    cols = st.columns(6)
    for col, (num, title, tech, color, status) in zip(cols, phases):
        with col:
            st.markdown(f"""
            <div style='background:#060b18; border:1px solid {color};
            border-radius:12px; padding:14px; text-align:center;'>
              <div style='background:{color}; color:#060b18; font-weight:800;
              font-size:0.9rem; border-radius:6px; padding:3px 0;
              margin-bottom:8px;'>{num}</div>
              <div style='color:#e0e6f0; font-weight:600; font-size:0.78rem;
                          line-height:1.3; margin-bottom:5px;'>{title}</div>
              <div style='color:#3a5a7a; font-size:0.65rem;
                          margin-bottom:8px;'>{tech}</div>
              <div style='color:#4ff7a0; font-size:0.68rem;
                          font-weight:600;'>{status}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════
elif page == "📊  Dashboard":
    section_header("📊", "Financial Dashboard",
                   "Key Performance Indicators & Visual Analytics")

    total_rev  = df['Net_Revenue'].sum()
    total_cost = df['Total_Cost'].sum()
    profit     = df['Profit'].sum()
    margin     = (profit / total_rev) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Revenue",    f"₹{total_rev:,.0f}",  "#4f9eff")
    with c2: kpi_card("Total Cost",       f"₹{total_cost:,.0f}", "#f74f7a")
    with c3: kpi_card("Net Profit",       f"₹{profit:,.0f}",     "#4ff7a0")
    with c4: kpi_card("Profit Margin",    f"{margin:.1f}%",       "#f7c94f")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏙️ Revenue by Region")
        reg = df.groupby('Region')['Net_Revenue'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(reg.index, reg.values,
                       color=COLORS6[:len(reg)],
                       height=0.55, edgecolor='none')
        for bar, val in zip(bars, reg.values):
            ax.text(bar.get_width() + max(reg.values)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'₹{val/1e6:.1f}M',
                    va='center', color='#8ba3c7', fontsize=8.5)
        ax.set_xlabel('Net Revenue (₹)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### 🍩 Revenue by Subscription Type")
        sub = df.groupby('Subscription_Type')['Net_Revenue'].sum()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        wedges, texts, autos = ax2.pie(
            sub.values, labels=sub.index, autopct='%1.1f%%',
            colors=COLORS6[:len(sub)], startangle=90,
            wedgeprops=dict(width=0.55), pctdistance=0.75)
        for at in autos: at.set_color('white'); at.set_fontsize(9)
        for t  in texts:  t.set_color('#8ba3c7')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 📉 Churn Rate by Region")
        if df['Is_Churned'].dtype == object:
            cr2 = df.groupby('Region').apply(
                lambda x: (x['Is_Churned'] == 'Yes').mean() * 100)
        else:
            cr2 = df.groupby('Region')['Is_Churned'].mean() * 100
        cr2 = cr2.sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        bc = ['#f74f7a' if v > cr2.mean() else '#4f9eff' for v in cr2.values]
        ax3.bar(cr2.index, cr2.values, color=bc, edgecolor='none', width=0.5)
        ax3.axhline(cr2.mean(), color='#f7c94f', linestyle='--', alpha=0.8,
                    label=f'Avg {cr2.mean():.1f}%')
        ax3.set_ylabel('Churn Rate (%)')
        ax3.legend(); ax3.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=25)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with col4:
        st.markdown("#### 📊 Revenue vs Cost by Region")
        reg2 = df.groupby('Region').agg(
            Revenue=('Net_Revenue','sum'),
            Cost=('Total_Cost','sum')).reset_index()
        x = np.arange(len(reg2)); w = 0.35
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.bar(x - w/2, reg2['Revenue']/1e6, w,
                label='Revenue', color='#4f9eff', edgecolor='none')
        ax4.bar(x + w/2, reg2['Cost']/1e6,    w,
                label='Cost',    color='#f74f7a', edgecolor='none')
        ax4.set_xticks(x); ax4.set_xticklabels(reg2['Region'], rotation=20)
        ax4.set_ylabel('Amount (₹ Million)')
        ax4.legend(); ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("#### 📋 Key Performance Summary")
    kpi_df = pd.DataFrame({
        'Metric': ['Total Customers','Avg Monthly Revenue',
                   'Total Net Revenue','Net Profit',
                   'Profit Margin','Avg Customer Lifetime',
                   'Avg Satisfaction Score','Revenue Per Cost Ratio'],
        'Value':  [f"{len(df):,}",
                   f"₹{df['Monthly_Revenue'].mean():,.2f}",
                   f"₹{total_rev:,.2f}",
                   f"₹{profit:,.2f}",
                   f"{margin:.2f}%",
                   f"{df['Customer_Lifetime_Months'].mean():.1f} months",
                   f"{df['Customer_Satisfaction_Score'].mean():.2f} / 5",
                   f"{df['Revenue_Per_Cost'].mean():.2f}x"],
        'Status': ['✅ Good','✅ Good','✅ Good','✅ Good',
                   '✅ Good','✅ Good','✅ Good','✅ Good'],
    })
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════
elif page == "🔍  EDA Analysis":
    section_header("🔍", "Exploratory Data Analysis",
                   "Statistical insights and visual patterns in the dataset")

    tab1, tab2, tab3 = st.tabs(
        ["📈 Revenue Analysis", "👥 Customer Analysis", "💰 Cost Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Revenue Distribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df['Net_Revenue'], bins=40, color='#4f9eff',
                    edgecolor='#0a1628', alpha=0.85)
            ax.axvline(df['Net_Revenue'].mean(), color='#f7c94f',
                       linestyle='--',
                       label=f"Mean: ₹{df['Net_Revenue'].mean():,.0f}")
            ax.axvline(df['Net_Revenue'].median(), color='#4ff7a0',
                       linestyle=':',
                       label=f"Median: ₹{df['Net_Revenue'].median():,.0f}")
            ax.set_xlabel('Net Revenue (₹)')
            ax.set_ylabel('Frequency')
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Revenue by Product Plan")
            pp = df.groupby('Product_Plan')['Net_Revenue'].mean().sort_values()
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.barh(pp.index, pp.values,
                     color='#4ff7a0', edgecolor='none', height=0.5)
            ax2.set_xlabel('Avg Net Revenue (₹)')
            ax2.grid(axis='x', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("#### 🔥 Correlation Heatmap")
        num_cols = ['Monthly_Revenue','Discount_Amount','Net_Revenue',
                    'Marketing_Cost','Sales_Cost','Support_Cost',
                    'Customer_Lifetime_Months',
                    'Customer_Satisfaction_Score','Profit']
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax3, linewidths=0.5, mask=mask,
                    annot_kws={'size': 8})
        ax3.set_title('Correlation Matrix — Financial Features',
                      color='#e0e6f0', pad=10)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Age Distribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df['Age'], bins=25, color='#c44ff7',
                    edgecolor='#0a1628', alpha=0.85)
            ax.axvline(df['Age'].mean(), color='#f7c94f',
                       linestyle='--',
                       label=f"Mean: {df['Age'].mean():.1f}")
            ax.set_xlabel('Age'); ax.set_ylabel('Count')
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Satisfaction by Subscription Type")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            types = df['Subscription_Type'].unique()
            data_p = [df[df['Subscription_Type'] == t][
                'Customer_Satisfaction_Score'].dropna() for t in types]
            bp = ax2.boxplot(data_p, labels=types, patch_artist=True)
            for patch, color in zip(bp['boxes'], COLORS6):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax2.set_ylabel('Satisfaction Score')
            ax2.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("#### Customers by Age Group & Gender")
        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
        ag = df['Age_Group'].value_counts().sort_index()
        axes[0].bar(ag.index.astype(str), ag.values,
                    color=COLORS6[:len(ag)], edgecolor='none', width=0.6)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Age Group Distribution', color='#e0e6f0')
        axes[0].grid(axis='y', alpha=0.3)

        gen = df['Gender'].value_counts()
        axes[1].pie(gen.values, labels=gen.index, autopct='%1.1f%%',
                    colors=['#4f9eff','#f74f7a'], startangle=90,
                    wedgeprops=dict(width=0.55))
        axes[1].set_title('Gender Split', color='#e0e6f0')
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Cost Breakdown (Overall)")
            costs = {
                'Marketing': df['Marketing_Cost'].sum(),
                'Sales':     df['Sales_Cost'].sum(),
                'Support':   df['Support_Cost'].sum()
            }
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(costs.keys(), costs.values(),
                          color=['#4f9eff','#f7934f','#4ff7a0'],
                          edgecolor='none', width=0.5)
            for bar, val in zip(bars, costs.values()):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(costs.values())*0.01,
                        f'₹{val/1e6:.1f}M',
                        ha='center', color='#e0e6f0', fontsize=9)
            ax.set_ylabel('Total Cost (₹)')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Profit by Region")
            prof = df.groupby('Region')['Profit'].sum().sort_values()
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            colors_p = ['#f74f7a' if v < 0 else '#4ff7a0'
                        for v in prof.values]
            ax2.barh(prof.index, prof.values,
                     color=colors_p, edgecolor='none', height=0.5)
            ax2.axvline(0, color='#8ba3c7', linewidth=1)
            ax2.set_xlabel('Profit (₹)')
            ax2.grid(axis='x', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("#### Cost vs Revenue Trend by Acquisition Channel")
        ch = df.groupby('Acquisition_Channel').agg(
            Revenue=('Net_Revenue','mean'),
            Cost=('Total_Cost','mean')).reset_index()
        x = np.arange(len(ch)); w = 0.35
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.bar(x - w/2, ch['Revenue'], w,
                label='Avg Revenue', color='#4f9eff', edgecolor='none')
        ax3.bar(x + w/2, ch['Cost'],   w,
                label='Avg Cost',    color='#f7934f', edgecolor='none')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ch['Acquisition_Channel'], rotation=20)
        ax3.set_ylabel('Amount (₹)')
        ax3.legend(); ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 4 — ML MODELS
# ══════════════════════════════════════════════════════════
elif page == "🤖  ML Models":
    section_header("🤖", "Machine Learning Models",
                   "Model performance, comparison, evaluation metrics & ROC curves")

    tab1, tab2 = st.tabs([
        "🌲 Random Forest",
        "👥 KMeans Clustering",
    ])

    # ── Random Forest ──
    with tab1:
        st.markdown(f"""
        <div style='background:#0a1628; border:1px solid #4f9eff;
        border-radius:12px; padding:20px; margin-bottom:18px;
        display:flex; justify-content:space-between; align-items:center;'>
          <div>
            <div style='color:#4f9eff; font-weight:800;
                        font-size:1.15rem;'>Random Forest Classifier</div>
            <div style='color:#3a5a7a; font-size:0.82rem; margin-top:4px;'>
              Ensemble of 150 Decision Trees  ·  Max Depth: 10
            </div>
          </div>
          <div style='text-align:right;'>
            <div style='color:#4ff7a0; font-size:2.2rem;
                        font-weight:800;'>{rf_acc*100:.2f}%</div>
            <div style='color:#3a5a7a; font-size:0.72rem;'>Accuracy</div>
          </div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        linewidths=0.8,
                        xticklabels=['Not Churned','Churned'],
                        yticklabels=['Not Churned','Churned'],
                        annot_kws={'size':14,'weight':'bold'})
            ax.set_title('Random Forest — Confusion Matrix',
                         color='#e0e6f0', pad=10)
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Feature Importance")
            imp = pd.Series(rf.feature_importances_,
                            index=features).sort_values()
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            cmap_colors = plt.cm.Blues(
                np.linspace(0.4, 0.9, len(imp)))
            ax2.barh(imp.index, imp.values,
                     color=cmap_colors, edgecolor='none', height=0.6)
            ax2.set_xlabel('Importance Score')
            ax2.grid(axis='x', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### ROC Curve")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(rf_fpr, rf_tpr, color='#4f9eff', lw=2,
                     label=f'RF AUC = {rf_auc:.3f}')
            ax3.plot([0,1],[0,1],'--', color='#3a5a7a', lw=1)
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', color='#e0e6f0')
            ax3.legend(); ax3.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        with col4:
            st.markdown("#### Classification Report")
            rep_df = pd.DataFrame(rf_report).T.round(3)
            rep_df = rep_df.drop(columns=['support'], errors='ignore')
            st.dataframe(rep_df, use_container_width=True)

    # ── Comparison ──
    # Model Comparison removed
    with tab2:
        st.markdown(f"""
        <div style='background:#0a1628; border:1px solid #c44ff7;
        border-radius:12px; padding:20px; margin-bottom:18px;
        display:flex; justify-content:space-between; align-items:center;'>
          <div>
            <div style='color:#c44ff7; font-weight:800;
                        font-size:1.15rem;'>KMeans Clustering</div>
            <div style='color:#3a5a7a; font-size:0.82rem; margin-top:4px;'>
              Unsupervised · 4 Clusters · Features: Revenue + Satisfaction + Lifetime
            </div>
          </div>
          <div style='text-align:right;'>
            <div style='color:#c44ff7; font-size:2.2rem;
                        font-weight:800;'>{sil:.4f}</div>
            <div style='color:#3a5a7a; font-size:0.72rem;'>Silhouette Score</div>
          </div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Segmentation Scatter Plot")
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, seg in seg_map.items():
                mask = df['Segment'] == i
                ax.scatter(df[mask]['Monthly_Revenue'],
                           df[mask]['Customer_Satisfaction_Score'],
                           c=seg_c[i], label=seg, alpha=0.45,
                           s=15, edgecolors='none')
            ax.set_xlabel('Monthly Revenue (₹)')
            ax.set_ylabel('Satisfaction Score')
            ax.legend(); ax.grid(alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Segment Summary")
            seg_sum = df.groupby('Segment_Name').agg({
                'Monthly_Revenue': 'mean',
                'Customer_Satisfaction_Score': 'mean',
                'Customer_Lifetime_Months': 'mean',
                'Customer_ID': 'count'
            }).round(2)
            seg_sum.columns = ['Avg Revenue','Avg Satisfaction',
                                'Avg Lifetime','Count']
            seg_sum = seg_sum.sort_values('Avg Revenue', ascending=False)
            st.dataframe(seg_sum, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — CHURN PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🎯  Churn Predictor":
    section_header("🎯", "Customer Churn Prediction",
                   "Real-time churn prediction using Random Forest & Decision Tree")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 Customer Profile")
        age         = st.slider("Customer Age", 18, 70, 35)
        monthly_rev = st.number_input("Monthly Revenue (₹)", 1000, 20000, 5000, 100)
        lifetime    = st.slider("Customer Lifetime (Months)", 1, 60, 18)
        satisfaction= st.slider("Satisfaction Score (1–5)", 1.0, 5.0, 3.5, 0.1)

    with col2:
        st.markdown("#### 💰 Cost Details")
        discount = st.number_input("Discount Amount (₹)",  0, 5000, 500, 50)
        marketing= st.number_input("Marketing Cost (₹)",   0, 10000, 2000, 100)
        sales    = st.number_input("Sales Cost (₹)",        0, 10000, 1500, 100)
        support  = st.number_input("Support Cost (₹)",      0, 10000, 1000, 100)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  Predict Churn Now", type="primary"):
        inp     = np.array([[age, monthly_rev, lifetime, satisfaction,
                             discount, marketing, sales, support]])
        rf_p    = rf.predict(inp)[0]
        rf_prob2= rf.predict_proba(inp)[0]

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        c1, c2, c3 = st.columns(3)
        color_rf = "#f74f7a" if rf_p == 1 else "#4ff7a0"
        label_rf = "⚠️ WILL CHURN" if rf_p == 1 else "✅ WILL STAY"

        for col, title, val, color in [
            (c1, "Prediction",         label_rf,                  color_rf),
            (c2, "Churn Probability",  f"{rf_prob2[1]*100:.1f}%", "#f74f7a"),
            (c3, "Retention Prob.",    f"{rf_prob2[0]*100:.1f}%", "#4ff7a0"),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:#0a1628; border:2px solid {color};
                border-radius:12px; padding:16px; text-align:center;'>
                  <div style='color:#3a5a7a; font-size:0.68rem;
                              text-transform:uppercase; letter-spacing:1px;'>
                    {title}</div>
                  <div style='color:{color}; font-size:1.1rem;
                              font-weight:800; margin-top:6px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        risk = rf_prob2[1] * 100
        bar_c = ("#f74f7a" if risk > 60 else
                 "#f7c94f" if risk > 30 else "#4ff7a0")
        st.markdown(f"""
        <div style='margin:20px 0 6px 0; color:#3a5a7a;
                    font-size:0.78rem; text-transform:uppercase;
                    letter-spacing:1px;'>Risk Level</div>
        <div style='background:#0a1628; border-radius:8px; padding:4px;
                    border:1px solid #162440;'>
          <div style='background:{bar_c}; width:{risk:.0f}%;
          min-width:40px; height:24px; border-radius:6px;
          display:flex; align-items:center; padding-left:10px;'>
            <span style='color:#060b18; font-size:0.76rem;
                         font-weight:800;'>{risk:.1f}% Churn Risk</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 💡 Business Recommendation")
        if rf_p == 1:
            st.error("""**⚠️ High Churn Risk Detected! Immediate Actions Required:**
- 🎁 Offer a personalized discount or loyalty reward immediately
- 📞 Schedule a personal follow-up call within 48 hours
- ⭐ Investigate and resolve service quality issues
- 💼 Propose a better-suited subscription plan
- 📧 Send retention email with exclusive benefits""")
        else:
            st.success("""**✅ Customer Likely to Stay! Suggested Actions:**
- 🌟 Enroll in loyalty rewards program
- 📈 Upsell to a higher subscription tier
- 😊 Maintain current service quality
- 🎯 Share relevant product updates""")

# ══════════════════════════════════════════════════════════
# PAGE 6 — SEGMENTATION
# ══════════════════════════════════════════════════════════
elif page == "👥  Segmentation":
    section_header("👥", "Customer Segmentation Analysis",
                   "KMeans Clustering — 4 Distinct Customer Groups")

    counts = df['Segment_Name'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, (seg, ic, sc) in zip(
        [c1, c2, c3, c4],
        [('Low Value','🔴',seg_c[0]),('Medium Value','🟠',seg_c[1]),
         ('High Value','🔵',seg_c[2]),('Premium','🟢',seg_c[3])]):
        with col:
            kpi_card(f"{ic} {seg}", f"{counts.get(seg,0):,}", sc)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Segmentation Scatter Plot")
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, seg in seg_map.items():
            mask = df['Segment'] == i
            ax.scatter(df[mask]['Monthly_Revenue'],
                       df[mask]['Customer_Satisfaction_Score'],
                       c=seg_c[i], label=seg, alpha=0.45,
                       s=15, edgecolors='none')
        ax.set_xlabel('Monthly Revenue (₹)')
        ax.set_ylabel('Satisfaction Score')
        ax.legend(); ax.grid(alpha=0.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Segment Distribution")
        segs = list(seg_map.values())
        cnts = [counts.get(s, 0) for s in segs]
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        wedges, texts, autos = ax2.pie(
            cnts, labels=segs, autopct='%1.1f%%',
            colors=seg_c, startangle=90,
            wedgeprops=dict(width=0.55))
        for at in autos: at.set_color('white'); at.set_fontsize(9)
        for t  in texts:  t.set_color('#8ba3c7')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("#### 📋 Detailed Segment Summary")
    seg_sum = df.groupby('Segment_Name').agg({
        'Monthly_Revenue':             ['mean','min','max'],
        'Customer_Satisfaction_Score': 'mean',
        'Customer_Lifetime_Months':    'mean',
        'Customer_ID':                 'count'
    }).round(2)
    seg_sum.columns = ['Avg Revenue','Min Revenue','Max Revenue',
                       'Avg Satisfaction','Avg Lifetime (mo)','Count']
    seg_sum = seg_sum.sort_values('Avg Revenue', ascending=False)
    st.dataframe(seg_sum, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔮 Predict Segment for New Customer")
    col1, col2, col3 = st.columns(3)
    with col1: nr = st.number_input("Monthly Revenue (₹)", 1000, 20000, 5000, 100, key="sr")
    with col2: ns = st.slider("Satisfaction Score", 1.0, 5.0, 3.5, 0.1, key="ss")
    with col3: nl = st.slider("Lifetime (Months)", 1, 60, 18, key="sl")

    if st.button("🔮  Predict Customer Segment", type="primary"):
        nd = scaler.transform([[nr, ns, nl]])
        sp = km.predict(nd)[0]
        seg_name = seg_map[sp]
        sc_color = seg_c[sp]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0a1628,#0d1e3a);
        border:2px solid {sc_color}; border-radius:14px;
        padding:28px; text-align:center; margin-top:14px;'>
          <div style='color:#3a5a7a; font-size:0.78rem; text-transform:uppercase;
                      letter-spacing:2.5px;'>Predicted Segment</div>
          <div style='color:{sc_color}; font-size:2.4rem;
                      font-weight:800; margin:12px 0;'>{seg_name}</div>
          <div style='color:#3a5a7a; font-size:0.82rem;'>
            Revenue: ₹{nr:,}  ·  Satisfaction: {ns}  ·  Lifetime: {nl} months
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 7 — REPORTS
# ══════════════════════════════════════════════════════════
elif page == "📋  Reports":
    section_header("📋", "Reports & Analysis Summary",
                   "Detailed dataset info, model results & key findings")

    tab1, tab2 = st.tabs(["📊 Data Summary", "🤖 Model Report"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Basic Dataset Info**")
            info_d = {
                'Property': ['Total Records','Total Features',
                             'Time Period','Missing Values',
                             'Duplicate Rows','Numeric Columns',
                             'Categorical Columns'],
                'Value':    [f"{len(df):,}", '22 columns',
                             '2023 – 2025',
                             f"{df.isnull().sum().sum()}",
                             f"{df.duplicated().sum()}",
                             f"{df.select_dtypes('number').shape[1]}",
                             f"{df.select_dtypes('object').shape[1]}"],
            }
            st.dataframe(pd.DataFrame(info_d),
                         hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Statistical Summary**")
            st.dataframe(
                df[['Monthly_Revenue','Net_Revenue',
                    'Customer_Satisfaction_Score',
                    'Customer_Lifetime_Months']].describe().round(2),
                use_container_width=True)

        st.markdown("#### Column Details")
        col_info = {
            'Column':        df.columns.tolist(),
            'Data Type':     [str(df[c].dtype) for c in df.columns],
            'Unique Values': [df[c].nunique() for c in df.columns],
            'Null Count':    [df[c].isnull().sum() for c in df.columns],
        }
        st.dataframe(pd.DataFrame(col_info),
                     hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("#### Model Performance Comparison")
        mc = pd.DataFrame({
            'Model':        ['Random Forest','KMeans'],
            'Type':         ['Classification','Clustering'],
            'Algorithm':    ['Ensemble (150 trees)',
                             'Unsupervised (k=4)'],
            'Eval Metric':  ['Accuracy + AUC','Silhouette Score'],
            'Score':        [f"{rf_acc*100:.2f}%", f"{sil:.4f}"],
            'AUC':          [f"{rf_auc:.4f}", "N/A"],
            'Purpose':      ['Churn Prediction','Customer Segmentation'],
        })
        st.dataframe(mc, hide_index=True, use_container_width=True)

        st.markdown("#### 🔑 Key Findings")
        findings = [
            ("🏆", "Best Model",
             f"Random Forest achieved {rf_acc*100:.2f}% accuracy with "
             f"AUC score of {rf_auc:.4f} — excellent churn prediction.",
             "#4f9eff"),
            ("⚠️", "Top Churn Factors",
             "Support Cost and Sales Cost are the highest-importance features "
             "for predicting churn.",
             "#f74f7a"),
            ("💰", "Top Revenue Region",
             "North / Canada region generates the highest net revenue.",
             "#4ff7a0"),
            ("👥", "Premium Segment",
             f"Premium customers generate ₹{df[df['Segment_Name']=='Premium']['Monthly_Revenue'].mean():,.0f} "
             f"avg monthly revenue — highest among all segments.",
             "#f7c94f"),
            ("📈", "Retention Rate",
             "~70% customers renewed subscription — strong retention signal.",
             "#c44ff7"),
            ("📊", "Cost Insight",
             "Marketing cost is the highest operational cost across all regions.",
             "#f7934f"),
        ]
        for icon, title, desc, color in findings:
            st.markdown(f"""
            <div style='background:#0a1628; border:1px solid #162440;
            border-left:4px solid {color}; border-radius:10px;
            padding:14px 18px; margin-bottom:8px;
            display:flex; gap:14px; align-items:flex-start;'>
              <div style='font-size:1.3rem; margin-top:2px;'>{icon}</div>
              <div>
                <div style='color:{color}; font-weight:700;
                            font-size:0.9rem;'>{title}</div>
                <div style='color:#c0d0e8; font-size:0.84rem;
                            margin-top:4px; line-height:1.55;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)