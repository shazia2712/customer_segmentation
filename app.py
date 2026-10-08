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
    st.title("👥 Customer Segmentation")
    st.write("Select features for customer clustering.")


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

    # Only allow numerical columns for K-Means
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Remove CustomerID because it is only an identifier
    if "CustomerID" in numeric_features:
        numeric_features.remove("CustomerID")

    # Sidebar feature selection
    with st.sidebar:

        features = st.multiselect(
            "Select Features:",
            options=numeric_features,
            default=[
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )


    # Check if enough features are selected
    if len(features) < 2:

        st.warning(
            "⚠️ Please select at least 2 numerical features."
        )

    else:

        # Select chosen features
        df_selected = df[features].copy()

        # Display sample data
        st.subheader("Sample Data")

        st.write(
            df_selected.sample(
                min(10, len(df_selected))
            )
        )


        # Find optimal K
        K = elbow(df_selected)

        # Check if elbow was found
        if K is None:

            st.warning(
                "⚠️ Could not automatically determine the optimal number of clusters."
            )

            K = 5

            st.info(
                "Using K = 5 as the default number of clusters."
            )


        st.subheader(f"Optimized K: {K}")


        # Model Training
        model = KMeans(
            n_clusters=K,
            random_state=42,
            n_init="auto"
        )

        model.fit(df_selected)


        # Get cluster labels
        labels = model.labels_


        # Add clusters
        df_selected["clusters"] = labels


        # Cluster Visualization
        st.subheader("Cluster Visualization")


        # Only create scatter plot when these two features are selected
        if (
            "Annual Income (k$)" in features
            and "Spending Score (1-100)" in features
        ):

            st.scatter_chart(
                data=df_selected,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                color="clusters"
            )

        else:

            st.info(
                "📊 Select 'Annual Income (k$)' and 'Spending Score (1-100)' to see the customer cluster visualization."
            )
