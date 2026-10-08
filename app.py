# Import libraries
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from kneed import KneeLocator


# Page setup
st.set_page_config(
    page_icon="👥",
    page_title="Customer Segmentation",
    layout="wide"
)


# Sidebar
with st.sidebar:
    st.title("👥 Customer Segmentation")


# Elbow method
def elbow(df):

    inertia = []
    k_values = range(1, 11)

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto"
        )

        model.fit(df)
        inertia.append(model.inertia_)

    KL = KneeLocator(
        k_values,
        inertia,
        curve="convex",
        direction="decreasing"
    )

    elbow_df = pd.DataFrame({
        "K": k_values,
        "Inertia": inertia
    })

    st.subheader("Elbow Curve")

    st.line_chart(
        elbow_df,
        x="K",
        y="Inertia"
    )

    return KL.elbow


# Upload file
file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)


if file:

    # Read dataset
    df = pd.read_csv(file)

    # Convert Gender into numbers
    if "Gender" in df.columns:

        df["Gender"] = df["Gender"].map({
            "Male": 0,
            "Female": 1
        })

    # Remove CustomerID
    if "CustomerID" in df.columns:

        df = df.drop("CustomerID", axis=1)


    # Sidebar feature selection
    with st.sidebar:

        features = st.multiselect(
            "Select Features:",
            options=df.columns,
            default=[
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )


    # Check features
    if len(features) < 2:

        st.warning("⚠️ Please select at least 2 features.")

    else:

        # Select features
        df_selected = df[features].copy()

        # Remove missing values
        df_selected = df_selected.dropna()

        st.subheader("Sample Data")

        st.write(
            df_selected.sample(
                min(10, len(df_selected))
            )
        )


        # Find best K
        K = elbow(df_selected)

        if K is None:
            K = 5

        st.subheader(f"Optimized K: {K}")


        # K-Means model
        model = KMeans(
            n_clusters=K,
            random_state=42,
            n_init="auto"
        )

        model.fit(df_selected)

        # Add cluster labels
        df_selected["Cluster"] = model.labels_


        # Show clusters
        st.subheader("Customer Clusters")

        st.write(df_selected)


        # Visualization
        st.subheader("Cluster Visualization")

        if (
            "Annual Income (k$)" in features
            and "Spending Score (1-100)" in features
        ):

            st.scatter_chart(
                df_selected,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                color="Cluster"
            )

        else:

            st.info(
                "Select Annual Income and Spending Score "
                "to see the cluster visualization."
            )
