import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
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

st.markdown("---")

# ----------------------------------
# LOAD DATA
# ----------------------------------

@st.cache_data
def load_data():

    df = pd.read_excel("Dumy_dataset.xlsx")

    return df

df = load_data()

# ----------------------------------
# DATA CLEANING
# ----------------------------------

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

model = joblib.load("flipkart_kmeans.pkl")

scaler = joblib.load("flipkart_scaler.pkl")

# ----------------------------------
# FEATURES
# ----------------------------------

features = [
    "Annual_Spending",
    "Orders_Count"
]

X = df[features]

scaled = scaler.transform(X)

df["Cluster"] = model.predict(scaled)

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Section",

    [

        "Home",

        "Dataset",

        "Data Cleaning",

        "Feature Scaling",

        "KMeans Clustering",

        "Dashboard",

        "Prediction"

    ]

)

if page=="Home":

    st.header("Project Workflow")

    st.info("""

Customer Dataset

↓

Data Cleaning

↓

Feature Scaling

↓

K-Means Clustering

↓

Dashboard

↓

Business Recommendation

""")

    col1,col2,col3,col4=st.columns(4)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Average Spending",
        round(df["Annual_Spending"].mean(),2)
    )

    col3.metric(
        "Average Orders",
        round(df["Orders_Count"].mean(),2)
    )

    col4.metric(
        "Clusters",
        len(df["Cluster"].unique())
    )

    elif page=="Dataset":

    st.header("Customer Dataset")

    st.dataframe(df)

    st.write("Rows :",df.shape[0])

    st.write("Columns :",df.shape[1])

    st.write(df.describe())

    elif page=="Data Cleaning":

    st.header("Data Cleaning")

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    st.subheader("Duplicate Rows")

    st.write(df.duplicated().sum())

    st.subheader("Dataset Info")

    st.write(df.dtypes)

    elif page=="Feature Scaling":

    st.header("Standard Scaled Features")

    scaled_df = pd.DataFrame(

        scaled,

        columns=features

    )

    st.dataframe(scaled_df)

    st.success("Features scaled successfully using StandardScaler.")

    elif page=="KMeans Clustering":

    st.header("Clustered Dataset")

    st.dataframe(df.head())

    st.write(df["Cluster"].value_counts())

    fig = px.scatter(

        df,

        x="Annual_Spending",

        y="Orders_Count",

        color=df["Cluster"].astype(str),

        title="Customer Segmentation"

    )

    st.plotly_chart(fig,use_container_width=True)

    elif page == "Dashboard":

    st.header("📊 Customer Segmentation Dashboard")

    # ===============================
    # KPI Cards
    # ===============================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        len(df)
    )

    c2.metric(
        "Average Spending",
        f"₹ {df['Annual_Spending'].mean():,.0f}"
    )

    c3.metric(
        "Average Orders",
        round(df["Orders_Count"].mean(),2)
    )

    c4.metric(
        "Clusters",
        len(df["Cluster"].unique())
    )

    st.markdown("---")

    # ===============================
    # Dataset Preview
    # ===============================

    st.subheader("Customer Dataset")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

        st.subheader("Elbow Method")

    inertia = []

    X = df[["Annual_Spending","Orders_Count"]]

    scaled = scaler.transform(X)

    for i in range(1,11):

        km = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )

        km.fit(scaled)

        inertia.append(km.inertia_)

    fig = px.line(
        x=range(1,11),
        y=inertia,
        markers=True,
        labels={
            "x":"Number of Clusters",
            "y":"WCSS"
        },
        title="Elbow Method"
    )

    st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")

    st.subheader("Customer Segments")

    fig = px.scatter(

        df,

        x="Annual_Spending",

        y="Orders_Count",

        color=df["Cluster"].astype(str),

        title="Customer Segmentation",

        hover_data=df.columns

    )

    st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")

    st.subheader("Cluster Distribution")

    cluster_count = (

        df["Cluster"]

        .value_counts()

        .reset_index()

    )

    cluster_count.columns=["Cluster","Customers"]

    fig = px.bar(

        cluster_count,

        x="Cluster",

        y="Customers",

        text="Customers",

        color="Cluster"

    )

    st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")

    st.subheader("Average Spending by Cluster")

    avg = (

        df.groupby("Cluster")["Annual_Spending"]

        .mean()

        .reset_index()

    )

    fig = px.bar(

        avg,

        x="Cluster",

        y="Annual_Spending",

        text_auto=".2s",

        color="Cluster"

    )

    st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")

    st.subheader("Cluster Summary")

    summary = (

        df.groupby("Cluster")[

            ["Annual_Spending","Orders_Count"]

        ]

        .mean()

        .round(2)

    )

    st.dataframe(summary,use_container_width=True)

    elif page == "Prediction":

    st.header("🎯 Customer Segment Prediction")

    st.write(
        "Enter customer details below to predict which customer segment they belong to."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        annual_spending = st.number_input(
            "Annual Spending (₹)",
            min_value=0,
            value=10000,
            step=500
        )

    with col2:

        orders_count = st.number_input(
            "Orders Count",
            min_value=0,
            value=5,
            step=1
        )

    st.markdown("")

    if st.button("Predict Customer Segment"):

        sample = pd.DataFrame(
            [[annual_spending, orders_count]],
            columns=["Annual_Spending", "Orders_Count"]
        )

        sample_scaled = scaler.transform(sample)

        cluster = model.predict(sample_scaled)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        # -------------------------------
        # Segment Mapping
        # -------------------------------

        cluster_stats = (
            df.groupby("Cluster")["Annual_Spending"]
            .mean()
            .sort_values()
        )

        budget_cluster = cluster_stats.index[0]
        regular_cluster = cluster_stats.index[1]
        premium_cluster = cluster_stats.index[2]

        if cluster == premium_cluster:

            st.success("⭐ Premium Customer")

            st.metric(
                "Predicted Cluster",
                int(cluster)
            )

            st.info(
                """
### Recommended Business Strategy

✅ Premium Membership

✅ Exclusive Offers

✅ Early Sale Access

✅ Personalized Product Recommendations

✅ High-value Cashback Rewards
                """
            )

        elif cluster == regular_cluster:

            st.warning("🛍️ Regular Customer")

            st.metric(
                "Predicted Cluster",
                int(cluster)
            )

            st.info(
                """
### Recommended Business Strategy

✅ Seasonal Discounts

✅ Product Bundles

✅ Loyalty Reward Points

✅ Personalized Coupons

✅ Cross-selling Products
                """
            )

        else:

            st.error("💰 Budget Customer")

            st.metric(
                "Predicted Cluster",
                int(cluster)
            )

            st.info(
                """
### Recommended Business Strategy

✅ Budget Deals

✅ Flash Sales

✅ Discount Coupons

✅ Free Delivery Offers

✅ Price-drop Notifications
                """
            )

    st.markdown("---")

    st.subheader("📌 Business Insights")

    cluster_summary = (
        df.groupby("Cluster")[["Annual_Spending", "Orders_Count"]]
        .mean()
        .round(2)
    )

    st.dataframe(cluster_summary)

    st.markdown("---")

    st.subheader("📈 Overall Recommendations")

    st.success("""
✔ Focus marketing campaigns on Premium Customers.

✔ Retain Regular Customers using loyalty programs.

✔ Convert Budget Customers into Regular Customers through targeted discounts.

✔ Use customer segmentation for personalized recommendations.

✔ Increase customer lifetime value using cluster-specific marketing strategies.
""")
    
    # =====================================
# DOWNLOAD CLUSTERED DATASET
# =====================================

st.markdown("---")

st.subheader("📥 Download Clustered Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download CSV",
    data=csv,
    file_name="Flipkart_Customer_Segmentation.csv",
    mime="text/csv"
)
