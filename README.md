# Customer Segmentation
A **Machine Learning clustering project** that groups customers into different segments based on their **age, gender, annual income, and spending score**.

## 📌 Overview
* Uses **Unsupervised Machine Learning** to identify customer groups.
* Uses **K-Means Clustering** to create customer segments.
* Allows users to **upload a customer dataset** through a Streamlit web application.
* Allows users to **select features** from the sidebar.
* Converts **Gender** into numerical values:

  * **Male → 0**
  * **Female → 1**
* Uses the **Elbow Method** to find a suitable number of clusters.
* Provides a **visual representation** of customer segments.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Streamlit**
* **Kneed**

## 🤖 Machine Learning

* **Type:** Unsupervised Learning
* **Algorithm:** K-Means Clustering
* **Method:** Elbow Method

### **Features Used**

* **Age**
* **Gender**
* **Annual Income (k$)**
* **Spending Score (1-100)**

> **Note:** `CustomerID` is not used for clustering because it is only an identifier.

## 🔍 What I Did

* Loaded the customer dataset using **Pandas**.
* Explored the available customer features.
* Removed **CustomerID** from the clustering features.
* Converted **Gender** values into numerical values.
* Created a **feature selection** option using the Streamlit sidebar.
* Applied **K-Means Clustering**.
* Used the **Elbow Method** to determine a suitable number of clusters.
* Assigned each customer to a **cluster**.
* Created a **cluster visualization**.
* Built an interactive **Streamlit web application**.

## 📊 Results

* Customers were successfully divided into different groups using **K-Means Clustering**.
* The clusters represent customers with **similar characteristics and spending behavior**.
* The **Elbow Method** was used to select a suitable number of clusters.
* The application allows users to explore different customer segments by selecting features.

## 📈 Visualization

The application provides a scatter plot showing:

* **Annual Income (k$)**
* **Spending Score (1-100)**
* **Customer Clusters**

This visualization helps understand how customers are grouped based on their **income and spending behavior**.

## 🚀 Streamlit App

👉 **[Try the Customer Segmentation App](https://customersegmentation-z8xbz6swdwuyiaxf8nfdox.streamlit.app/)**

## 💻 How to Run

### **1. Clone the Repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### **2. Install Required Libraries**

```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
customer-segmentation/
│
├── app.py
├── customer_segmentation.ipynb
├── dataset.csv
├── requirements.txt
└── README.md
```

## 📚 Key Learning

* **Unsupervised Machine Learning**
* **K-Means Clustering**
* **Elbow Method**
* **Feature Selection**
* **Data Preprocessing**
* **Customer Segmentation**
* **Data Visualization**
* **Streamlit Application Development**

## ⚠️ Disclaimer

This project is created for **educational and portfolio purposes**. The customer segments are based on the selected dataset and should not be considered definitive business classifications.
