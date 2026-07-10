import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px

from sklearn.cluster import KMeans

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Flipkart Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1{
    color:#2874F0;
}

.stButton>button{
    background:#2874F0;
    color:white;
    border-radius:8px;
}

.metric-card{
    padding:15px;
    border-radius:10px;
    background:#FFFFFF;
    box-shadow:0px 2px 8px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# TITLE
# ----------------------------------

st.title("🛒 Flipkart Customer Segmentation Dashboard")
st.caption("K-Means based customer segmentation, with support for your own datasets.")

st.markdown("---")

# ----------------------------------
# LOAD MODEL
# ----------------------------------

model = joblib.load("flipkart_kmeans.pkl")
scaler = joblib.load("flipkart_scaler.pkl")

FEATURES = ["Annual_Spending", "Orders_Count"]

# ----------------------------------
# HELPERS
# ----------------------------------

@st.cache_data
def load_default_data():
    df = pd.read_excel("Dumy_dataset.xlsx")
    return df


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def label_segments(df_with_clusters):
    """Map raw cluster numbers to Budget / Regular / Premium based on mean spend."""
    stats = (
        df_with_clusters.groupby("Cluster")["Annual_Spending"]
        .mean()
        .sort_values()
    )
    ordered_clusters = list(stats.index)
    label_map = {}
    names = ["Budget", "Regular", "Premium"]
    for i, cluster_id in enumerate(ordered_clusters):
        label_map[cluster_id] = names[i] if i < len(names) else f"Segment {i}"
    df_with_clusters["Segment"] = df_with_clusters["Cluster"].map(label_map)
    return df_with_clusters, label_map


STRATEGY = {
    "Premium": (
        "success", "⭐ Premium Customer",
        ["Premium Membership", "Exclusive Offers", "Early Sale Access",
         "Personalized Recommendations", "High-value Cashback Rewards"]
    ),
    "Regular": (
        "warning", "🛍️ Regular Customer",
        ["Seasonal Discounts", "Product Bundles", "Loyalty Reward Points",
         "Personalized Coupons", "Cross-selling Products"]
    ),
    "Budget": (
        "error", "💰 Budget Customer",
        ["Budget Deals", "Flash Sales", "Discount Coupons",
         "Free Delivery Offers", "Price-drop Notifications"]
    ),
}


def render_strategy(segment):
    kind, title, actions = STRATEGY.get(segment, STRATEGY["Regular"])
    getattr(st, kind)(title)
    st.info("### Recommended Business Strategy\n\n" + "\n\n".join(f"✅ {a}" for a in actions))


# ----------------------------------
# SIDEBAR — DATA SOURCE
# ----------------------------------

st.sidebar.title("📁 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset (optional)",
    type=["csv", "xlsx"],
    help="If you don't upload anything, the default Flipkart dataset is used."
)

if uploaded_file is not None:
    raw_df = read_uploaded_file(uploaded_file)
    st.sidebar.success(f"Loaded '{uploaded_file.name}' — {raw_df.shape[0]} rows")
else:
    raw_df = load_default_data()
    st.sidebar.caption("Using default dataset. Upload your own to try it out.")

# Column mapping — lets any dataset work, not just ones with exact column names
st.sidebar.markdown("**Map your columns**")
columns = list(raw_df.columns)


def guess_index(name, options):
    return options.index(name) if name in options else 0


spend_col = st.sidebar.selectbox(
    "Annual Spending column", columns, index=guess_index("Annual_Spending", columns)
)
orders_col = st.sidebar.selectbox(
    "Orders Count column", columns, index=guess_index("Orders_Count", columns)
)

df = raw_df.rename(columns={spend_col: "Annual_Spending", orders_col: "Orders_Count"})
df = clean_data(df)

if df.shape[0] == 0:
    st.error("No usable rows left after cleaning. Check your file and column mapping.")
    st.stop()

X = df[FEATURES]
scaled = scaler.transform(X)
df["Cluster"] = model.predict(scaled)
df, label_map = label_segments(df)

# ----------------------------------
# SIDEBAR — NAVIGATION
# ----------------------------------

st.sidebar.markdown("---")
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Overview", "Dashboard", "Predict New Customer", "Predict from File"]
)

# ==================================
# OVERVIEW
# ==================================

if page == "Overview":

    st.header("Project Workflow")

    st.info(
        "Dataset  →  Cleaning  →  Feature Scaling  →  K-Means Clustering  →  "
        "Dashboard  →  Business Recommendation"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Average Spending", f"₹ {df['Annual_Spending'].mean():,.0f}")
    col3.metric("Average Orders", round(df["Orders_Count"].mean(), 2))
    col4.metric("Segments", df["Segment"].nunique())

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ==================================
# DASHBOARD
# ==================================

elif page == "Dashboard":

    st.header("📊 Customer Segmentation Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(df))
    c2.metric("Average Spending", f"₹ {df['Annual_Spending'].mean():,.0f}")
    c3.metric("Average Orders", round(df["Orders_Count"].mean(), 2))
    c4.metric("Segments", df["Segment"].nunique())

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Segments")
        fig = px.scatter(
            df, x="Annual_Spending", y="Orders_Count",
            color="Segment", title="Customer Segmentation",
            hover_data=[c for c in df.columns if c not in ["Cluster"]]
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Segment Distribution")
        seg_count = df["Segment"].value_counts().reset_index()
        seg_count.columns = ["Segment", "Customers"]
        fig = px.bar(seg_count, x="Segment", y="Customers", text="Customers", color="Segment")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Average Spending by Segment")
        avg = df.groupby("Segment")["Annual_Spending"].mean().reset_index()
        fig = px.bar(avg, x="Segment", y="Annual_Spending", text_auto=".2s", color="Segment")
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.subheader("Elbow Method")
        inertia = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            km.fit(scaled)
            inertia.append(km.inertia_)
        fig = px.line(
            x=list(range(1, 11)), y=inertia, markers=True,
            labels={"x": "Number of Clusters", "y": "WCSS"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Summary")
    summary = df.groupby("Segment")[FEATURES].mean().round(2)
    st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Download Clustered Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", data=csv, file_name="Clustered_Customers.csv", mime="text/csv")

# ==================================
# PREDICT — SINGLE CUSTOMER
# ==================================

elif page == "Predict New Customer":

    st.header("🎯 Predict a Single Customer's Segment")

    col1, col2 = st.columns(2)
    with col1:
        annual_spending = st.number_input("Annual Spending (₹)", min_value=0, value=10000, step=500)
    with col2:
        orders_count = st.number_input("Orders Count", min_value=0, value=5, step=1)

    if st.button("Predict Segment"):
        sample = pd.DataFrame([[annual_spending, orders_count]], columns=FEATURES)
        sample_scaled = scaler.transform(sample)
        cluster = model.predict(sample_scaled)[0]
        segment = label_map.get(cluster, "Regular")

        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric("Predicted Cluster", int(cluster))
        render_strategy(segment)

# ==================================
# PREDICT — BATCH FROM FILE
# ==================================

elif page == "Predict from File":

    st.header("📤 Predict Segments for a New File")
    st.write(
        "Upload a dataset of customers (with spending and order count columns) "
        "to predict their segments in bulk."
    )

    batch_file = st.file_uploader("Upload customer file", type=["csv", "xlsx"], key="batch")

    if batch_file is not None:
        batch_raw = read_uploaded_file(batch_file)
        st.write(f"Loaded {batch_raw.shape[0]} rows.")

        b_cols = list(batch_raw.columns)
        bc1, bc2 = st.columns(2)
        with bc1:
            b_spend_col = st.selectbox(
                "Annual Spending column", b_cols, index=guess_index("Annual_Spending", b_cols), key="b_spend"
            )
        with bc2:
            b_orders_col = st.selectbox(
                "Orders Count column", b_cols, index=guess_index("Orders_Count", b_cols), key="b_orders"
            )

        if st.button("Run Prediction"):
            batch_df = batch_raw.rename(
                columns={b_spend_col: "Annual_Spending", b_orders_col: "Orders_Count"}
            )
            batch_df = clean_data(batch_df)

            b_scaled = scaler.transform(batch_df[FEATURES])
            batch_df["Cluster"] = model.predict(b_scaled)
            batch_df, _ = label_segments(batch_df)

            st.success(f"Predicted segments for {batch_df.shape[0]} customers.")
            st.dataframe(batch_df, use_container_width=True)

            fig = px.pie(batch_df, names="Segment", title="Predicted Segment Split")
            st.plotly_chart(fig, use_container_width=True)

            out_csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Predictions", data=out_csv,
                file_name="Predicted_Segments.csv", mime="text/csv"
            )
    else:
        st.info("Upload a file above to get started.")
