# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from kneed import KneeLocator


# Page Setup
st.set_page_config(
    page_icon="👥",
    page_title="Customer Segmentation",
    layout="wide"
)


# Sidebar
with st.sidebar:
    st.title("Customer Segmentation")


# Elbow Method
def elbow(df):
    out = []
    k_values = range(1, 11)

    for i in k_values:
        model = KMeans(
            n_clusters=i,
            random_state=42,
            n_init="auto"
        )

        model.fit(df)
        out.append(model.inertia_)

    KL = KneeLocator(
        k_values,
        out,
        curve="convex",
        direction="decreasing"
    )

    df1 = pd.DataFrame({
        "k_val": k_values,
        "inertia": out
    })

    st.subheader("Elbow Curve")

    st.line_chart(
        data=df1,
        x="k_val",
        y="inertia"
    )

    return KL.elbow


# Upload the file
file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)


# Read CSV
if file:

    df = pd.read_csv(file)

    # Select only numerical features for clustering
    features = [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    df = df[features]

    # Display sample data
    st.subheader("Sample Data")
    st.write(df.sample(min(10, len(df))))


    # Find optimal K
    K = elbow(df)

    st.subheader(f"Optimized K: {K}")


    # Model Training
    model = KMeans(
        n_clusters=K,
        random_state=42,
        n_init="auto"
    )

    model.fit(df)

    # Get cluster labels
    labels = model.labels_

    # Add clusters
    df["clusters"] = labels


    # Cluster Visualization
    st.subheader("Cluster Visualization")

    st.scatter_chart(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="clusters"
    )
