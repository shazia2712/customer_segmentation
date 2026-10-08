# import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


# Page Setup
st.set_page_config(page_icon = "👥", page_title = "Customer Segmentation", layout = "wide")

# Side bar
with st.sidebar:
    st.title("Customer Segmentation")
    #st.image("")

# Pre-processing
def preprocessing(df):
    encoder = LabelEncoder()
    for col in df.columns:              # it iterates throug all cols
        if df[col].dtype==object:       # if col is obj it performs encoding
            df[col] = encoder.fit_transform(df[col])

def elbow(df):
    out = []
    k_values = range(1,11)
    for i in k_values:
        model = KMeans(n_clusters= i)
        model.fit(df)
        out.append(model.inertia_)

    KL = KneeLocator(k_values, out, curve = "convex", direction = "decreasing")
    df1 = pd.DataFrame({"k_val": k_values, "inertia": out})
    st.subheader("Elbow Curve")
    st.line_chart(data = df1, x = "k_val", y = "inertia")
    return KL.elbow

# Upload the file   # before uplaod, page setup must be done
file = st.file_uploader(" ", type=["csv"])

# read csv file
if file:
    df = pd.read_csv(file)
    # st.write(df)


    # Feature selection     # to select what cols user needs to analyse the data
    features = st.sidebar.multiselect("Select Featuress : ", 
                                    options = df.columns,
                                    default = ["Annual Income (k$)", "Spending Score (1-100)"])
    if features:
        df = df.loc[:, features]    # only user selected dataframe will be printed
        preprocessing(df)
        st.subheader("Sample Data")
        st.write(df.sample(10))

    # Model Training
    K = elbow(df)
    st.subheader(f"Optimized K : {K}")     # the optimized k value will be printed
    model = KMeans(n_clusters = K)
    model.fit(df)
    labels = model.labels_      # prediction
    df["clusters"] = labels

    # clusters visualization
    st.subheader("Cluster Visualization")
    st.scatter_chart(data= df, x = "Annual Income (k$)", y= "Spending Score (1-100)", 
                     color = "clusters")

