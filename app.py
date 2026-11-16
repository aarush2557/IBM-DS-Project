# =====================================================================
#  DIWALI SALES DASHBOARD  (Streamlit web app)
#  Project : Diwali Sales Analysis - Exploratory Data Analysis (EDA)
#  Author  : Aarush Tyagi | BCA (AI & DS) | K.R. Mangalam University
#
#  HOW TO RUN (in Anaconda Prompt / terminal):
#      pip install -r requirements.txt
#      streamlit run app.py
#  A browser tab opens automatically with the dashboard.
# =====================================================================

import io                                   # io.BytesIO lets us read uploaded/downloaded bytes like a file
import re                                   # re = regular expressions (used to match column names)
from pathlib import Path                    # Path makes it easy to work with files and folders

import numpy as np                          # numbers / maths
import pandas as pd                         # tables (DataFrames)
import plotly.express as px                 # interactive charts (hover, zoom, download)
import streamlit as st                      # the library that turns this script into a web page

# ---------------------------------------------------------------------
# 1. PAGE SETTINGS  (must be the first Streamlit command)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Diwali Sales Dashboard",    # text shown in the browser tab
    page_icon="🪔",                          # small icon in the browser tab
    layout="wide",                          # use the full screen width
)

# ---------------------------------------------------------------------
# 2. SETTINGS YOU CAN CHANGE
# ---------------------------------------------------------------------
# Folder where your dataset may be saved (raw string 'r' keeps Windows backslashes safe)
PROJECT_FOLDER = Path(r"C:\Users\bts\OneDrive\Desktop\IBM DATA SCIENCE")

# Festive colour palette (orange -> gold -> red -> maroon) used by every chart
DIWALI = ["#E85D04", "#FAA307", "#DC2F02", "#9D0208", "#F48C06", "#6A040F", "#FFBA08", "#370617"]

# Logical order of age groups so charts read from youngest to oldest
AGE_ORDER = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]

# Columns the dashboard needs
REQUIRED = ["Gender", "Age Group", "Age", "Marital_Status", "State", "Zone",
            "Occupation", "Product_Category", "Orders", "Amount"]

# ---------------------------------------------------------------------
# 3. SMALL CSS BLOCK to make the page look festive
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Big gradient banner at the top */
    .banner {background: linear-gradient(90deg, #6A040F, #DC2F02, #FAA307);
             padding: 22px 28px; border-radius: 14px; color: white; margin-bottom: 18px;}
    .banner h1 {margin: 0; font-size: 2.1rem;}
    .banner p  {margin: 4px 0 0 0; font-size: 1rem; opacity: 0.92;}
    /* Make the KPI numbers stand out */
    div[data-testid="stMetric"] {background: #FFF4E6; border-left: 6px solid #E85D04;
                                 padding: 12px 16px; border-radius: 10px;}
    div[data-testid="stMetric"] * {color: #370617 !important;}
    </style>
    <div class="banner">
        <h1>🪔 Diwali Sales Dashboard</h1>
        <p>Exploratory Data Analysis of festive-season customer purchases</p>
    </div>
    """,
    unsafe_allow_html=True,                 # allow our own HTML / CSS
)


# ---------------------------------------------------------------------
# 4. FUNCTIONS TO FIND AND READ THE DATA
# ---------------------------------------------------------------------
def find_dataset():
    """Search the usual folders for a Diwali dataset file and return its path (or None)."""
    folders = [PROJECT_FOLDER, Path.cwd(), Path.home() / "Downloads", Path.home() / "Desktop",
               Path.home() / "OneDrive" / "Desktop", Path.home() / "Documents"]
    for folder in folders:
        if not folder.exists():
            continue                                         # skip folders that do not exist
        for pattern in ("*", "*/*"):                         # inside the folder + one level deeper
            for f in folder.glob(pattern):
                if f.is_file() and f.suffix.lower() in (".csv", ".xlsx", ".xls") and "diwali" in f.name.lower():
                    return f
    return None


@st.cache_data(show_spinner="Reading dataset ...")           # cache = do not re-read the file on every click
def read_table(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Turn the raw bytes of a CSV / Excel file into a DataFrame."""
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))         # Excel file

    bad_lines = []                                           # lines that cannot be read go here
    def skip_bad_line(line):
        bad_lines.append(line)
        return None                                          # None tells pandas to skip the line

    for encoding in ("utf-8", "latin-1"):                    # try UTF-8 first, then Latin-1
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding,
                               engine="python", on_bad_lines=skip_bad_line)
        except UnicodeDecodeError:
            bad_lines.clear()                                # wrong encoding -> try the next one
    raise ValueError("Could not read the file. Is it a CSV or Excel file?")


def simplify(name):
    """'Age Group', 'age_group', 'AGE-GROUP' all become 'agegroup' so we can match them."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@st.cache_data(show_spinner="Cleaning data ...")
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as the notebook."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()          # remove spaces around column names

    # Fix small spelling differences in column names (e.g. 'age_group' -> 'Age Group')
    existing = {simplify(c): c for c in df.columns}
    rename_map = {existing[simplify(r)]: r for r in REQUIRED
                  if simplify(r) in existing and existing[simplify(r)] != r}
    df = df.rename(columns=rename_map)

    # Stop early if important columns are still missing
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Columns found: {list(df.columns)}")

    df = df.drop(columns=["Status", "unnamed1"], errors="ignore")   # empty columns
    df = df.dropna(subset=["Amount"])                               # Amount is our key number
    df["Amount"] = df["Amount"].astype(float).astype(int)           # whole rupees
    df = df.drop_duplicates()                                       # remove repeated rows
    df = df.rename(columns={"Cust_name": "Customer_Name"})

    # Marital status 0/1 -> text (only if it is numeric)
    if pd.api.types.is_numeric_dtype(df["Marital_Status"]):
        df["Marital_Status"] = df["Marital_Status"].map({0: "Unmarried", 1: "Married"})
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. SIDEBAR - choose the data source
# ---------------------------------------------------------------------
st.sidebar.header("📂 Data")
uploaded = st.sidebar.file_uploader("Upload Diwali dataset (CSV / Excel)", type=["csv", "xlsx", "xls"])

if uploaded is not None:                                     # 1) a file was uploaded in the browser
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
else:                                                        # 2) otherwise search the computer
    found = find_dataset()
    if found is None:
        st.info("👈 Upload your **Diwali Sales Data** file (CSV or Excel) in the sidebar to start.\n\n"
                f"Or save it in `{PROJECT_FOLDER}` and refresh this page.")
        st.stop()                                            # stop here until data is available
    file_bytes, file_name = found.read_bytes(), found.name
    st.sidebar.success(f"Using: {found.name}")

try:
    df_all = clean_data(read_table(file_bytes, file_name))
except Exception as error:                                   # show a friendly message instead of a crash
    st.error(f"Problem with the dataset: {error}")
    st.stop()

# ---------------------------------------------------------------------
# 6. SIDEBAR - filters
# ---------------------------------------------------------------------
st.sidebar.header("🔎 Filters")


def pick(label, column, order=None):
    """Create a multi-select box for one column and return the chosen values."""
    options = sorted(df_all[column].dropna().unique().tolist())
    if order:                                                # keep a logical order if given
        options = [o for o in order if o in options] + [o for o in options if o not in order]
    return st.sidebar.multiselect(label, options, default=options)


sel_gender = pick("Gender", "Gender")
sel_age = pick("Age Group", "Age Group", AGE_ORDER)
sel_marital = pick("Marital Status", "Marital_Status")
sel_zone = pick("Zone", "Zone")
sel_state = pick("State", "State")
sel_occ = pick("Occupation", "Occupation")
sel_cat = pick("Product Category", "Product_Category")

# Keep only the rows that match ALL the chosen filters
df = df_all[
    df_all["Gender"].isin(sel_gender) & df_all["Age Group"].isin(sel_age)
    & df_all["Marital_Status"].isin(sel_marital) & df_all["Zone"].isin(sel_zone)
    & df_all["State"].isin(sel_state) & df_all["Occupation"].isin(sel_occ)
    & df_all["Product_Category"].isin(sel_cat)
]

if df.empty:                                                 # nothing matches the filters
    st.warning("No data matches these filters. Please select more options in the sidebar.")
    st.stop()

st.sidebar.caption(f"Showing {len(df):,} of {len(df_all):,} records")
age_order = [a for a in AGE_ORDER if a in df["Age Group"].unique()]   # age groups that exist after filtering

# ---------------------------------------------------------------------
# 7. KPI CARDS
# ---------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)                               # 4 equal columns
k1.metric("💰 Total Sales", f"₹ {df['Amount'].sum() / 1e7:.2f} Cr")
k2.metric("🛒 Total Orders", f"{int(df['Orders'].sum()):,}")
k3.metric("👥 Customers", f"{len(df):,}")
k4.metric("🧾 Avg Spend", f"₹ {df['Amount'].mean():,.0f}")
st.write("")                                                 # small blank space

# ---------------------------------------------------------------------
# 8. TABS
# ---------------------------------------------------------------------
tab_overview, tab_geo, tab_prod, tab_heat, tab_stats, tab_insight, tab_data = st.tabs(
    ["📊 Overview", "🗺️ Geography", "🛍️ Products & Occupation", "🔥 Heatmaps",
     "📈 Statistics", "💡 Insights", "📄 Data"])

# ---- TAB 1 : OVERVIEW -------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)

    # Donut chart: sales share by gender
    g = df.groupby("Gender", as_index=False)["Amount"].sum()
    fig = px.pie(g, names="Gender", values="Amount", hole=0.5, color_discrete_sequence=DIWALI,
                 title="Sales Share by Gender")
    c1.plotly_chart(fig, width="stretch")

    # Donut chart: sales share by marital status
    m = df.groupby("Marital_Status", as_index=False)["Amount"].sum()
    fig = px.pie(m, names="Marital_Status", values="Amount", hole=0.5, color_discrete_sequence=DIWALI[2:],
                 title="Sales Share by Marital Status")
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)

    # Bar chart: sales by age group, split by gender
    ag = df.groupby(["Age Group", "Gender"], as_index=False)["Amount"].sum()
    fig = px.bar(ag, x="Age Group", y="Amount", color="Gender", barmode="group",
                 category_orders={"Age Group": age_order}, color_discrete_sequence=DIWALI,
                 title="Sales by Age Group and Gender")
    c3.plotly_chart(fig, width="stretch")

    # Bar chart: sales by zone
    z = df.groupby("Zone", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
    fig = px.bar(z, x="Zone", y="Amount", color="Zone", color_discrete_sequence=DIWALI,
                 title="Sales by Zone")
    fig.update_layout(showlegend=False)
    c4.plotly_chart(fig, width="stretch")

# ---- TAB 2 : GEOGRAPHY ------------------------------------------------
with tab_geo:
    top_n = st.slider("How many top states to show?", 5, 20, 10)   # slider controls the chart size
    st_sales = (df.groupby("State", as_index=False)
                  .agg(Sales=("Amount", "sum"), Orders=("Orders", "sum"), Customers=("Amount", "size"))
                  .sort_values("Sales", ascending=False))
    st_sales["Sales (₹ Lakh)"] = st_sales["Sales"] / 1e5

    c1, c2 = st.columns(2)
    fig = px.bar(st_sales.head(top_n), x="Sales (₹ Lakh)", y="State", orientation="h",
                 color="Sales (₹ Lakh)", color_continuous_scale="YlOrRd", title=f"Top {top_n} States by Sales")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})   # biggest bar on top
    c1.plotly_chart(fig, width="stretch")

    # Pareto: bars + cumulative percentage line (80/20 rule)
    p = st_sales.head(15).copy()
    p["Cumulative %"] = st_sales["Sales"].cumsum().head(15) / st_sales["Sales"].sum() * 100
    fig = px.bar(p, x="State", y="Sales (₹ Lakh)", color_discrete_sequence=[DIWALI[0]],
                 title="Pareto Chart: Do a Few States Drive Most Sales?")
    fig.add_scatter(x=p["State"], y=p["Cumulative %"], mode="lines+markers", name="Cumulative %",
                    yaxis="y2", line=dict(color=DIWALI[5]))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="Cumulative %"),
                      showlegend=False)
    c2.plotly_chart(fig, width="stretch")

    # Bubble chart: orders vs sales, bubble size = customers
    fig = px.scatter(st_sales, x="Orders", y="Sales (₹ Lakh)", size="Customers", color="Sales (₹ Lakh)",
                     hover_name="State", color_continuous_scale="YlOrRd", size_max=55,
                     title="State Bubble Chart (bubble size = number of customers)")
    st.plotly_chart(fig, width="stretch")

# ---- TAB 3 : PRODUCTS & OCCUPATION ------------------------------------
with tab_prod:
    c1, c2 = st.columns(2)

    cat = (df.groupby("Product_Category", as_index=False)["Amount"].sum()
             .sort_values("Amount", ascending=False).head(10))
    fig = px.bar(cat, x="Amount", y="Product_Category", orientation="h", color="Amount",
                 color_continuous_scale="YlOrRd", title="Top Product Categories by Sales")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    c1.plotly_chart(fig, width="stretch")

    occ = df.groupby("Occupation", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
    fig = px.bar(occ, x="Amount", y="Occupation", orientation="h", color="Amount",
                 color_continuous_scale="OrRd", title="Sales by Occupation")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    c2.plotly_chart(fig, width="stretch")

    # 100% stacked bar: gender mix inside each product category
    cg = df.pivot_table(index="Product_Category", columns="Gender", values="Amount", aggfunc="sum", fill_value=0)
    cg = cg.loc[cat["Product_Category"]]                              # keep only the top categories
    cg_pct = cg.div(cg.sum(axis=1), axis=0) * 100                     # every row adds up to 100%
    fig = px.bar(cg_pct, orientation="h", barmode="stack", color_discrete_sequence=DIWALI,
                 labels={"value": "Share of Sales (%)", "Product_Category": "Product Category"},
                 title="Who Buys What? Gender Share of Sales in Each Category")
    st.plotly_chart(fig, width="stretch")

# ---- TAB 4 : HEATMAPS -------------------------------------------------
with tab_heat:
    # Heatmap 1: State x Product Category
    ts = df.groupby("State")["Amount"].sum().nlargest(10).index
    tc = df.groupby("Product_Category")["Amount"].sum().nlargest(8).index
    grid = (df[df["State"].isin(ts) & df["Product_Category"].isin(tc)]
              .pivot_table(index="State", columns="Product_Category", values="Amount",
                           aggfunc="sum", fill_value=0) / 1e5)
    fig = px.imshow(grid.loc[ts], text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(color="₹ Lakh"), title="Heatmap: State × Product Category (Sales in ₹ Lakh)")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)

    # Heatmap 2: Occupation x Age Group
    oa = (df.pivot_table(index="Occupation", columns="Age Group", values="Amount", aggfunc="sum", fill_value=0) / 1e5)
    oa = oa[age_order]
    oa = oa.loc[oa.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(oa, text_auto=".1f", aspect="auto", color_continuous_scale="OrRd",
                    labels=dict(color="₹ Lakh"), title="Occupation × Age Group (₹ Lakh)")
    c1.plotly_chart(fig, width="stretch")

    # Heatmap 3: customer count Gender x Age Group
    cm = pd.crosstab(df["Gender"], df["Age Group"])[age_order]
    fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Oranges",
                    labels=dict(color="Customers"), title="Customer Count: Gender × Age Group")
    c2.plotly_chart(fig, width="stretch")

# ---- TAB 5 : STATISTICS -----------------------------------------------
with tab_stats:
    c1, c2 = st.columns(2)

    # Correlation matrix between numeric columns
    num = df[["Age", "Orders", "Amount"]].copy()
    num["Married (1=yes)"] = (df["Marital_Status"] == "Married").astype(int)
    fig = px.imshow(num.corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title="Correlation Matrix")
    c1.plotly_chart(fig, width="stretch")

    # Histogram of purchase amount
    fig = px.histogram(df, x="Amount", nbins=40, color_discrete_sequence=[DIWALI[0]],
                       title="Distribution of Purchase Amount")
    c2.plotly_chart(fig, width="stretch")

    # Box plot: spread of spending in every age group (dots = unusually big spenders)
    fig = px.box(df, x="Age Group", y="Amount", color="Gender", category_orders={"Age Group": age_order},
                 color_discrete_sequence=DIWALI, title="Spending Spread by Age Group (Box Plot)")
    st.plotly_chart(fig, width="stretch")

    # Scatter: orders vs amount (sample keeps it fast)
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    fig = px.scatter(sample, x="Orders", y="Amount", color="Gender", opacity=0.6,
                     color_discrete_sequence=DIWALI, title="Orders vs Amount (random sample of 2000)")
    st.plotly_chart(fig, width="stretch")

# ---- TAB 6 : INSIGHTS (written automatically from the data) ------------
with tab_insight:
    st.subheader("Key findings (calculated from the filtered data)")

    def top_by_amount(col):
        """Return (name, amount, % share) of the group with the highest total sales."""
        s = df.groupby(col)["Amount"].sum().sort_values(ascending=False)
        return s.index[0], s.iloc[0], s.iloc[0] / s.sum() * 100

    for column, label in [("Gender", "Top gender"), ("Age Group", "Top age group"),
                          ("Marital_Status", "Top marital status"), ("State", "Top state"),
                          ("Zone", "Top zone"), ("Occupation", "Top occupation"),
                          ("Product_Category", "Top product category")]:
        name, amount, share = top_by_amount(column)
        st.markdown(f"- **{label}:** {name} — ₹ {amount:,.0f} ({share:.1f}% of sales)")

    best = df.groupby(["Marital_Status", "Gender"])["Amount"].sum().idxmax()
    st.markdown(f"- **Highest-spending segment:** {best[0]} · {best[1]}")

    st.subheader("Business recommendations")
    st.markdown(
        "- Aim advertising at the highest-spending **gender / age / marital-status** segment shown above.\n"
        "- Stock more of the **top product categories** before the festival.\n"
        "- Focus delivery and offers on the **top states and zones**.\n"
        "- Offer special deals to the **top-spending occupations**.\n"
        "- Run separate campaigns in low-selling states to grow new markets."
    )

# ---- TAB 7 : DATA -------------------------------------------------------
with tab_data:
    st.subheader("Cleaned dataset (after filters)")
    st.dataframe(df, width="stretch", height=420)
    st.download_button("⬇️ Download filtered data as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="diwali_filtered.csv", mime="text/csv")
    st.subheader("Summary statistics")
    st.dataframe(df[["Age", "Orders", "Amount"]].describe(), width="stretch")

st.caption("Built with Python · Pandas · Plotly · Streamlit  |  Aarush Tyagi · BCA (AI & DS) · K.R. Mangalam University")
