import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==========================================================
# FRAUD AI RESPONSE ENGINE
# ==========================================================

def get_fraud_ai_response(question):

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "You are FraudShield AI, an assistant for a "
                        "Credit Card Fraud Detection educational project. "
                        "Answer questions about credit card fraud, "
                        "transactions, OTP security, risk scores, "
                        "cybersecurity and fraud prevention. "
                        "Give clear, helpful, beginner-friendly answers. "
                        "Do not claim to access live bank transactions. "
                        "Never ask users to share their OTP, PIN, CVV, "
                        "password or full card number."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "AI service is unavailable right now. "
            "Check your API key and internet connection. "
            f"Error: {e}"
        )



# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================================
# UI STYLE
# ==========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b1220,#101a2d 55%,#101827);
    color: #e8eef8;
}
[data-testid="stSidebar"] {
    background: #111b2b;
}
h1,h2,h3 {
    color: #f4f7ff !important;
}
div[data-testid="stMetric"] {
    background: #18253a;
    padding: 16px;
    border: 1px solid #2a3d59;
    border-radius: 14px;
}
div.stButton>button {
    border-radius: 10px;
    font-weight: 700;
}
.small-note {
    color: #aabbd3;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.title("💳 Credit Card Fraud Detection")
st.caption(
    "FraudShield AI • Credit Card Transaction Analysis, "
    "Detection & Prevention"
)
st.markdown("**Developed by Hasti**")

# ==========================================================
# DATA LOADING
# ==========================================================
@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes):
    return pd.read_csv(file_bytes)


def get_dataset():

    uploaded = st.session_state.get("uploaded_data")

    if uploaded is not None:
        return uploaded.copy()

    local = Path("data/creditcard.csv")

    if local.exists():
        return pd.read_csv(local)

    return None


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.markdown("## ⚙️ Data Setup")

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            st.session_state.uploaded_data = load_csv_bytes(
                uploaded_file
            )
            st.success("Dataset loaded successfully!")

        except Exception as e:
            st.error(f"Unable to read CSV: {e}")

    st.markdown("---")
    st.markdown("**Expected label column:** `Class`")
    st.markdown("0 = Genuine • 1 = Fraud")

    st.markdown(
        "You can also place `creditcard.csv` "
        "inside the `data` folder."
    )

# ==========================================================
# NAVIGATION
# ==========================================================
pages = [
    "Overview Dashboard",
    "Dataset Explorer",
    "Data Cleaning",
    "Visualization Lab",
    "Model Training",
    "Fraud Prediction",
    "Fraud Intelligence Center",
    "Live Fraud Center",
    "AI Fraud Simulator Lab",
    "Transaction & OTP Simulator"
    
]

page = st.sidebar.radio(
    "🧭 Navigation",
    pages
)

# ==========================================================
# LOAD DATA
# ==========================================================
df = get_dataset()

# Live Fraud Center can run without transaction CSV.
if df is None and page != "Live Fraud Center":

    st.info(
        "📂 Upload a credit-card transaction CSV in the sidebar "
        "to use dataset analysis, training and prediction modules. "
        "The Live Fraud Center is available without uploading a dataset."
    )

    st.markdown("""
    **Dataset Source:**

    [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

    Download `creditcard.csv` and upload it using the sidebar.
    """)

    st.stop()

# ==========================================================
# LABEL VALIDATION
# ==========================================================
# Allow Live Fraud Center without a dataset.
if df is None:
    st.warning("Please upload your credit card dataset CSV file.")
    st.stop()

if "Class" not in df.columns:
    st.error("Dataset must contain a 'Class' column.")
    st.stop()

# ==========================================================
# DATA CLEANING
# ==========================================================
cleaned = df.copy()

numeric_cols = cleaned.select_dtypes(
    include=np.number
).columns.tolist()

cleaned[numeric_cols] = cleaned[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)

missing_before = int(cleaned.isna().sum().sum())
duplicate_count = int(cleaned.duplicated().sum())

for col in numeric_cols:

    if col == "Class":
        cleaned[col] = cleaned[col].fillna(0)

    else:

        median_value = cleaned[col].median()

        if pd.isna(median_value):
            median_value = 0

        cleaned[col] = cleaned[col].fillna(
            median_value
        )

cleaned = cleaned.drop_duplicates().reset_index(drop=True)

cleaned["Class"] = pd.to_numeric(
    cleaned["Class"],
    errors="coerce"
).fillna(0)

cleaned["Class"] = cleaned["Class"].astype(int)
cleaned["Class"] = cleaned["Class"].clip(0, 1)

# ==========================================================
# PAGE 1 - OVERVIEW DASHBOARD
# ==========================================================
if page == "Overview Dashboard":

    st.subheader("📌 Transaction Overview")

    total = len(cleaned)
    fraud = int((cleaned["Class"] == 1).sum())
    genuine = int((cleaned["Class"] == 0).sum())

    fraud_rate = (
        fraud / total * 100
        if total > 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Transactions", f"{fraud:,}")
    c3.metric("Genuine Transactions", f"{genuine:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.markdown("### Dataset Preview")

    st.dataframe(
        cleaned.head(12),
        use_container_width=True
    )

    st.markdown("### Class Balance")

    counts = cleaned["Class"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        ["Genuine (0)", "Fraud (1)"],
        [counts.get(0, 0), counts.get(1, 0)]
    )

    ax.set_ylabel("Transaction Count")
    ax.set_title("Genuine vs Fraud Transactions")

    st.pyplot(fig)
    plt.close(fig)

# ==========================================================
# PAGE 2 - DATASET EXPLORER
# ==========================================================
elif page == "Dataset Explorer":

    st.subheader("🔎 Dataset Explorer")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{len(cleaned):,}")
    c2.metric("Columns", f"{cleaned.shape[1]:,}")
    c3.metric("Duplicates Removed", f"{duplicate_count:,}")

    st.markdown("### Raw Data Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.markdown("### Column Types")

    type_data = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        type_data,
        use_container_width=True
    )

    st.markdown("### Summary Statistics")

    st.dataframe(
        cleaned.describe().T,
        use_container_width=True
    )

# ==========================================================
# PAGE 3 - DATA CLEANING
# ==========================================================
elif page == "Data Cleaning":

    st.subheader("🧹 Data Cleaning Report")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Missing Values Found",
        f"{missing_before:,}"
    )

    c2.metric(
        "Duplicate Rows Found",
        f"{duplicate_count:,}"
    )

    c3.metric(
        "Rows After Cleaning",
        f"{len(cleaned):,}"
    )

    st.markdown("### Missing Values (Before Cleaning)")

    st.dataframe(
        df.isna().sum().to_frame("Missing Count"),
        use_container_width=True
    )

    st.markdown("### Cleaned Dataset Preview")

    st.dataframe(
        cleaned.head(100),
        use_container_width=True
    )

    csv = cleaned.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Cleaned CSV",
        csv,
        "cleaned_transactions.csv",
        "text/csv"
    )
# ==========================================================
# PAGE 4 - VISUALIZATION LAB
# ==========================================================
elif page == "Visualization Lab":

    st.subheader("📊 Visualization Lab")

    numeric = cleaned.select_dtypes(
        include=np.number
    )

    if len(numeric.columns) > 0:

        chosen = st.selectbox(
            "Choose Numeric Field for Histogram",
            numeric.columns
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(
            cleaned[chosen].dropna(),
            bins=40
        )

        ax.set_title(f"Distribution of {chosen}")
        ax.set_xlabel(chosen)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    # FRAUD RISK ANALYSIS DASHBOARD

    st.markdown("---")
    st.markdown("## 🛡️ Fraud Risk Analysis Dashboard")
    st.caption("Transaction risk classification and monitoring")

    if "Amount" in cleaned.columns:

        risk_data = cleaned.copy()

        risk_data["Risk_Level"] = risk_data["Amount"].apply(
            lambda amount:
                "High Risk" if amount >= 10000
                else "Medium Risk" if amount >= 2000
                else "Low Risk"
        )

        high_risk = risk_data[
            risk_data["Risk_Level"] == "High Risk"
        ]

        medium_risk = risk_data[
            risk_data["Risk_Level"] == "Medium Risk"
        ]

        low_risk = risk_data[
            risk_data["Risk_Level"] == "Low Risk"
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric("🔴 High Risk", len(high_risk))
        c2.metric("🟠 Medium Risk", len(medium_risk))
        c3.metric("🟢 Low Risk", len(low_risk))

        st.markdown("---")
        st.markdown("### 📊 Risk Distribution")

        risk_counts = risk_data["Risk_Level"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            risk_counts.index,
            risk_counts.values
        )

        ax.set_title("Transaction Risk Distribution")
        ax.set_xlabel("Risk Category")
        ax.set_ylabel("Number of Transactions")

        st.pyplot(fig)
        plt.close(fig)

        # RISK PERCENTAGE

        total_transactions = len(risk_data)

        if total_transactions > 0:

            high_percent = (
                len(high_risk) / total_transactions
            ) * 100

            medium_percent = (
                len(medium_risk) / total_transactions
            ) * 100

            low_percent = (
                len(low_risk) / total_transactions
            ) * 100

            st.markdown("### 📈 Risk Percentage")

            c1, c2, c3 = st.columns(3)

            c1.metric("High Risk %", f"{high_percent:.2f}%")
            c2.metric("Medium Risk %", f"{medium_percent:.2f}%")
            c3.metric("Low Risk %", f"{low_percent:.2f}%")

        # HIGH RISK TRANSACTION TABLE

        st.markdown("### 🚨 High Risk Transaction Records")

        if len(high_risk) > 0:

            st.dataframe(
                high_risk.head(100),
                use_container_width=True
            )

        else:
            st.info("No high-risk transactions found.")

        st.success("Risk analysis completed successfully.")

    else:

        st.warning(
            "The dataset does not contain an Amount column. "
            "Risk analysis cannot be performed."
        )

    # TRANSACTION AMOUNT ANALYSIS

    st.markdown("---")
    st.markdown("## 💳 Transaction Amount: Genuine vs Fraud")

    if "Amount" in cleaned.columns:

        genuine_amount = cleaned.loc[
            cleaned["Class"] == 0, "Amount"
        ]

        fraud_amount = cleaned.loc[
            cleaned["Class"] == 1, "Amount"
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.boxplot(
              [genuine_amount, fraud_amount],
              tick_labels=["Genuine", "Fraud"]
              )

        ax.set_ylabel("Transaction Amount")
        ax.set_xlabel("Transaction Type")

        ax.set_title(
            "Genuine vs Fraud Transaction Amount Comparison"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.warning(
            "Amount column is not available for transaction analysis."
        )

    # CORRELATION HEATMAP

    st.markdown("---")
    st.markdown("## 🔥 Correlation Heatmap (Selected Numeric Columns)")

    corr_cols = numeric.columns[
        :min(12, len(numeric.columns))
    ].tolist()

    if len(corr_cols) >= 2:

        corr = cleaned[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(9, 5))

        im = ax.imshow(
            corr,
            aspect="auto"
        )

        ax.set_xticks(range(len(corr_cols)))
        ax.set_xticklabels(
            corr_cols,
            rotation=90
        )

        ax.set_yticks(range(len(corr_cols)))
        ax.set_yticklabels(corr_cols)

        fig.colorbar(im, ax=ax)

        ax.set_title("Correlation Matrix")

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.warning(
            "Not enough numeric columns to create a correlation heatmap."
        )


# ==========================================================
# PAGE 5 - MODEL TRAINING
# ==========================================================
    # ======================================================
    # INTERACTIVE CHART STUDIO
    # ======================================================

    st.markdown("---")
    st.markdown("## 🎨 Interactive Chart Studio")
    st.caption(
        "Customize charts, compare transactions and explore fraud data."
    )

    try:
        import plotly.express as px

        st.markdown("### ⚙️ Chart Configuration")

        chart_data = cleaned.copy()

        # FRAUD FILTER
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            fraud_filter = st.selectbox(
                "Filter Transaction Type",
                [
                    "All Transactions",
                    "Genuine Transactions",
                    "Fraud Transactions"
                ],
                key="interactive_fraud_filter"
            )

        with filter_col2:

            chart_type = st.selectbox(
                "Select Chart Type",
                [
                    "Bar Chart",
                    "Line Chart",
                    "Area Chart",
                    "Scatter Plot",
                    "Histogram",
                    "Box Plot",
                    "Pie Chart"
                ],
                key="interactive_chart_type"
            )

        # APPLY FRAUD FILTER
        if fraud_filter == "Genuine Transactions":

            chart_data = chart_data[
                chart_data["Class"] == 0
            ]

        elif fraud_filter == "Fraud Transactions":

            chart_data = chart_data[
                chart_data["Class"] == 1
            ]

        # COLUMN SELECTION
        available_columns = chart_data.columns.tolist()

        numeric_columns = chart_data.select_dtypes(
            include=np.number
        ).columns.tolist()

        st.markdown("### 📌 Select Chart Columns")

        if chart_type in [
            "Bar Chart",
            "Line Chart",
            "Area Chart"
        ]:

            x_col = st.selectbox(
                "X-Axis",
                available_columns,
                key="interactive_x_axis"
            )

            y_col = st.selectbox(
                "Y-Axis",
                numeric_columns,
                key="interactive_y_axis"
            )

        elif chart_type == "Scatter Plot":

            x_col = st.selectbox(
                "Scatter X-Axis",
                numeric_columns,
                key="interactive_scatter_x"
            )

            y_col = st.selectbox(
                "Scatter Y-Axis",
                numeric_columns,
                key="interactive_scatter_y"
            )

        elif chart_type == "Histogram":

            x_col = st.selectbox(
                "Select Column",
                numeric_columns,
                key="interactive_histogram"
            )

            y_col = None

        elif chart_type == "Box Plot":

            x_col = st.selectbox(
                "Category Column",
                available_columns,
                key="interactive_box_x"
            )

            y_col = st.selectbox(
                "Numeric Column",
                numeric_columns,
                key="interactive_box_y"
            )

        elif chart_type == "Pie Chart":

            x_col = st.selectbox(
                "Category Column",
                available_columns,
                key="interactive_pie"
            )

            y_col = None

        # GRAPH TITLE
        chart_title = st.text_input(
            "✏️ Customize Graph Title",
            value=f"{chart_type} - Fraud Analysis",
            key="interactive_chart_title"
        )

        # RECORD LIMIT
        max_records = st.slider(
            "Number of Records",
            min_value=100,
            max_value=max(100, min(50000, len(chart_data))),
            value=max(100, min(5000, len(chart_data))),
            step=100,
            key="interactive_record_limit"
        )

        # GRAPH HEIGHT
        graph_height = st.slider(
            "Graph Height",
            min_value=300,
            max_value=900,
            value=500,
            step=50,
            key="interactive_graph_height"
        )

        # PREPARE DATA
        plot_data = chart_data.head(
            max_records
        ).copy()

        st.markdown("---")

        st.markdown("### 📊 Generated Interactive Graph")

        if len(plot_data) == 0:

            st.warning(
                "No data available for the selected filter."
            )

        else:

            # BAR CHART
            if chart_type == "Bar Chart":

                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    title=chart_title,
                    template="plotly_dark"
                )

            # LINE CHART
            elif chart_type == "Line Chart":

                fig = px.line(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    title=chart_title,
                    template="plotly_dark"
                )

            # AREA CHART
            elif chart_type == "Area Chart":

                fig = px.area(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    title=chart_title,
                    template="plotly_dark"
                )

            # SCATTER PLOT
            elif chart_type == "Scatter Plot":

                color_col = (
                    "Class"
                    if "Class" in plot_data.columns
                    else None
                )

                fig = px.scatter(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=chart_title,
                    template="plotly_dark"
                )

            # HISTOGRAM
            elif chart_type == "Histogram":

                fig = px.histogram(
                    plot_data,
                    x=x_col,
                    title=chart_title,
                    nbins=40,
                    template="plotly_dark"
                )

            # BOX PLOT
            elif chart_type == "Box Plot":

                fig = px.box(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    color=(
                        "Class"
                        if "Class" in plot_data.columns
                        else None
                    ),
                    title=chart_title,
                    template="plotly_dark"
                )

            # PIE CHART
            elif chart_type == "Pie Chart":

                pie_data = plot_data[
                    x_col
                ].value_counts().reset_index()

                pie_data.columns = [
                    "Category",
                    "Count"
                ]

                fig = px.pie(
                    pie_data,
                    names="Category",
                    values="Count",
                    title=chart_title,
                    hole=0.35,
                    template="plotly_dark"
                )

            # GRAPH DESIGN
            fig.update_layout(
                height=graph_height,
                title_x=0.5,
                font_size=13,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=40
                )
            )

            # SHOW INTERACTIVE GRAPH
            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.success(
                f"{chart_type} generated successfully!"
            )

            # DOWNLOAD GRAPH
            st.markdown("### 📥 Export Chart Data")

            csv_data = plot_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Selected Chart Data",
                data=csv_data,
                file_name="fraud_chart_data.csv",
                mime="text/csv",
                key="interactive_chart_download"
            )

    except ImportError:

        st.warning(
            "Plotly is not installed. "
            "Install it using: pip install plotly"
        )

    except Exception as chart_error:

        st.error(
            f"Interactive Chart Error: {chart_error}"
        )
elif page == "Model Training":

    st.subheader("🤖 Machine Learning Model Training")

    st.caption(
        "Random Forest classifier with a stratified train/test split. "
        "Training uses a sample for speed on large datasets."
    )

    max_rows = st.slider(
        "Maximum Rows Used for Training (Speed Control)",
        min_value=5000,
        max_value=min(100000, max(5000, len(cleaned))),
        value=min(50000, max(5000, len(cleaned))),
        step=5000
    )

    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    if st.button("🚀 Train Random Forest Model", type="primary"):

        features = [
            c for c in cleaned.select_dtypes(include=np.number).columns
            if c != "Class"
        ]

        if len(features) == 0:

            st.error("No numeric features available for training.")

        else:

            train_data = cleaned[
                features + ["Class"]
            ].dropna()

            if train_data["Class"].nunique() < 2:

                st.error(
                    "Training requires both class 0 and class 1 rows."
                )

            else:

                try:

                    if len(train_data) > max_rows:

                        train_data = train_data.sample(
                            max_rows,
                            random_state=42
                        )

                    X = train_data[features]
                    y = train_data["Class"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size / 100,
                        random_state=42,
                        stratify=y
                    )

                    model = RandomForestClassifier(
                        n_estimators=120,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        max_depth=14
                    )

                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    metrics = {
                        "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(
                            y_test, pred, zero_division=0
                        ),
                        "Recall": recall_score(
                            y_test, pred, zero_division=0
                        ),
                        "F1 Score": f1_score(
                            y_test, pred, zero_division=0
                        )
                    }

                    st.session_state.fraud_model = model
                    st.session_state.fraud_features = features
                    st.session_state.model_metrics = metrics

                    st.session_state.confusion = confusion_matrix(
                        y_test, pred
                    )

                    st.session_state.report = classification_report(
                        y_test, pred, zero_division=0
                    )

                    st.success(
                        "Model Training Completed Successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Model training failed: {e}"
                    )

    # SHOW TRAINED MODEL RESULTS

    if "model_metrics" in st.session_state:

        metrics = st.session_state.model_metrics

        cols = st.columns(4)

        for col, (name, value) in zip(
            cols, metrics.items()
        ):

            col.metric(
                name,
                f"{value:.3f}"
            )

        # CONFUSION MATRIX

        st.markdown("### Confusion Matrix")

        cm = st.session_state.confusion

        fig, ax = plt.subplots()

        ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Genuine", "Fraud"])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Genuine", "Fraud"])

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        plt.close(fig)

        # CLASSIFICATION REPORT

        st.markdown("### Classification Report")

        st.code(
            st.session_state.report
        )

        # FEATURE IMPORTANCE

        st.markdown("### Feature Importance")

        model = st.session_state.fraud_model
        feat = st.session_state.fraud_features

        importance = pd.Series(
            model.feature_importances_,
            index=feat
        ).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots()

        importance.sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_title("Top Feature Importance")

        st.pyplot(fig)
        plt.close(fig)
 # ==========================================================
# PAGE 6 - FRAUD PREDICTION
# ==========================================================
elif page == "Fraud Prediction":

    st.subheader("🔍 Predict a Transaction")

    if "fraud_model" not in st.session_state:

        st.warning(
            "First train the model in the Model Training page."
        )

    else:

        features = st.session_state.fraud_features
        model = st.session_state.fraud_model

        st.info(
            "Enter transaction feature values below. "
            "For this anonymized dataset, V1–V28 are transformed features; "
            "use dataset values for meaningful predictions."
        )

        with st.form("predict_form"):

            values = {}
            cols = st.columns(3)

            for i, feature in enumerate(features):

                if feature.lower() == "amount":

                    values[feature] = cols[
                        i % 3
                    ].number_input(
                        feature,
                        min_value=0.0,
                        value=0.0,
                        step=1.0
                    )

                elif feature.lower() == "time":

                    values[feature] = cols[
                        i % 3
                    ].number_input(
                        feature,
                        value=0.0,
                        step=60.0
                    )

                else:

                    values[feature] = cols[
                        i % 3
                    ].number_input(
                        feature,
                        value=0.0,
                        step=0.1,
                        format="%.4f"
                    )

            submitted = st.form_submit_button(
                "🛡️ Analyze Transaction"
            )

        if submitted:

            row = pd.DataFrame(
                [values],
                columns=features
            )

            predicted = int(
                model.predict(row)[0]
            )

            try:

                probability = float(
                    model.predict_proba(row)[0][1]
                )

            except Exception:

                probability = float(predicted)

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            if predicted == 1:

                st.error(
                    "🚨 Model flagged this transaction as POTENTIAL FRAUD."
                )

            else:

                st.success(
                    "✅ Model classified this transaction as LIKELY GENUINE."
                )

            st.caption(
                "This is an educational ML demo, "
                "not a real financial fraud decision system."
            )


# ==========================================================
# PAGE 7 - FRAUD INTELLIGENCE CENTER
# ==========================================================
elif page == "Fraud Intelligence Center":

    st.subheader("🧠 Fraud Intelligence & Prevention Center")

    st.caption(
        "Fraud trends, suspicious transaction patterns "
        "and intelligent prevention recommendations"
    )

    total_transactions = len(cleaned)

    fraud_data = cleaned[
        cleaned["Class"] == 1
    ].copy()

    genuine_data = cleaned[
        cleaned["Class"] == 0
    ].copy()

    total_fraud = len(fraud_data)
    total_genuine = len(genuine_data)

    fraud_percentage = (
        (total_fraud / total_transactions) * 100
        if total_transactions > 0 else 0
    )

    st.markdown("### 📊 Fraud Intelligence Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Detected",
        f"{total_fraud:,}"
    )

    c3.metric(
        "Genuine Transactions",
        f"{total_genuine:,}"
    )

    c4.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.2f}%"
    )

    st.markdown("---")

    # FRAUD TREND ANALYSIS

    st.markdown("## 📈 Fraud Trend Analysis")

    if "Time" in cleaned.columns and total_transactions > 0:

        trend_data = cleaned.sort_values("Time").copy()

        try:

            trend_data["Time_Group"] = pd.qcut(
                trend_data["Time"],
                q=10,
                duplicates="drop"
            )

            trend_summary = trend_data.groupby(
                "Time_Group",
                observed=False
            )["Class"].agg(
                Total_Transactions="count",
                Fraud_Count="sum"
            ).reset_index()

            trend_summary["Fraud_Rate"] = (
                trend_summary["Fraud_Count"]
                / trend_summary["Total_Transactions"]
                * 100
            )

            st.markdown("### Fraud Count Across Time Groups")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                range(len(trend_summary)),
                trend_summary["Fraud_Count"],
                marker="o"
            )

            ax.set_xlabel("Time Group (chronological)")
            ax.set_ylabel("Fraud Transactions")
            ax.set_title("Fraud Trend Across Dataset Time Groups")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Fraud Rate Across Time Groups")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                range(len(trend_summary)),
                trend_summary["Fraud_Rate"],
                marker="o"
            )

            ax.set_xlabel("Time Group (chronological)")
            ax.set_ylabel("Fraud Rate (%)")
            ax.set_title("Fraud Percentage Trend")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                trend_summary,
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Unable to generate time trend analysis: {e}"
            )

    else:

        st.info(
            "Time column is unavailable. "
            "Temporal fraud trend analysis cannot be generated."
        )

    # FRAUD PATTERN ANALYSIS

    st.markdown("---")
    st.markdown("## 🔎 Suspicious Fraud Patterns")

    if "Amount" in cleaned.columns and total_transactions > 0:

        amount_data = cleaned["Amount"].dropna()

        if len(amount_data) > 0:

            amount_threshold = amount_data.quantile(0.95)

            high_amount_data = cleaned[
                cleaned["Amount"] >= amount_threshold
            ]

            high_amount_fraud = high_amount_data[
                high_amount_data["Class"] == 1
            ]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "95th Percentile Amount",
                f"{amount_threshold:.2f}"
            )

            c2.metric(
                "High Amount Transactions",
                f"{len(high_amount_data):,}"
            )

            c3.metric(
                "Fraud in High Amount Group",
                f"{len(high_amount_fraud):,}"
            )

            st.markdown("### 💳 High Amount Transaction Analysis")

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.hist(
                amount_data,
                bins=40
            )

            ax.axvline(
                amount_threshold,
                linestyle="--",
                label="95th percentile threshold"
            )

            ax.set_xlabel("Transaction Amount")
            ax.set_ylabel("Frequency")
            ax.set_title("Transaction Amount Distribution")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### 🚨 High Amount Fraud Records")

            if len(high_amount_fraud) > 0:

                st.dataframe(
                    high_amount_fraud.head(100),
                    use_container_width=True
                )

            else:

                st.info(
                    "No fraud records found in the "
                    "95th percentile amount group."
                )

    else:

        st.info(
            "Amount column unavailable. "
            "Amount-based pattern analysis cannot be performed."
        )

    # FRAUD FEATURE COMPARISON

    st.markdown("---")
    st.markdown("## 🧬 Fraud vs Genuine Feature Analysis")

    common_numeric = [
        c for c in cleaned.select_dtypes(include=np.number).columns
        if c != "Class"
    ]

    if (
        len(common_numeric) > 0
        and total_fraud > 0
        and total_genuine > 0
    ):

        comparison = pd.DataFrame({
            "Feature": common_numeric,

            "Genuine Average": [
                genuine_data[c].mean()
                for c in common_numeric
            ],

            "Fraud Average": [
                fraud_data[c].mean()
                for c in common_numeric
            ]
        })

        comparison["Difference"] = (
            comparison["Fraud Average"]
            - comparison["Genuine Average"]
        ).abs()

        comparison = comparison.sort_values(
            "Difference",
            ascending=False
        )

        top_features = comparison.head(10)

        st.markdown("### Top 10 Features with Average Differences")

        st.dataframe(
            top_features,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            top_features["Feature"].astype(str),
            top_features["Difference"]
        )

        ax.set_xlabel("Absolute Difference in Average")
        ax.set_ylabel("Feature")
        ax.set_title("Fraud vs Genuine Feature Difference")

        ax.invert_yaxis()

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.warning(
            "Feature comparison requires numeric columns "
            "and both fraud and genuine transactions."
        )

            # ------------------------------------------------------
    # PREVENTION ADVISOR
    # ------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🧠 AI Fraud Prevention Advisor")

    st.caption(
        "Automated prevention recommendations based on "
        "dataset statistics and risk indicators."
    )

    solutions = [
        (
            "🚨 High Value Transaction Monitoring",
            "Apply additional verification to unusually large "
            "transactions. Consider OTP, authentication or manual review."
        ),
        (
            "🔐 Multi-Factor Authentication",
            "Use additional authentication for sensitive transactions "
            "and suspicious login activity."
        ),
        (
            "📊 Continuous Fraud Monitoring",
            "Monitor transaction activity continuously and investigate "
            "unusual changes in transaction behavior."
        ),
        (
            "⚠️ Suspicious Activity Alerts",
            "Generate alerts when transactions exceed defined risk "
            "thresholds or show unusual patterns."
        ),
        (
            "🧾 Transaction Verification",
            "Verify transaction details and customer identity "
            "before approving suspicious activity."
        ),
        (
            "🛡️ Risk-Based Security",
            "Use risk indicators to prioritize transactions for "
            "additional checks and review."
        )
    ]

    for title, solution in solutions:

        with st.expander(title):

            st.write(solution)

    # ------------------------------------------------------
    # FRAUD RESPONSE ACTION CENTER
    # ------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📋 Fraud Response Action Center")

    st.markdown("""
    | Risk Situation | Recommended Action |
    |---|---|
    | Unusually high transaction | Additional verification |
    | Suspicious transaction behavior | Manual review |
    | Potential fraud detected | Temporary security review |
    | Repeated suspicious activity | Account security investigation |
    | Fraud pattern identified | Update monitoring rules |
    | Confirmed fraudulent transaction | Follow bank fraud response procedure |
    """)

    # ------------------------------------------------------
    # DOWNLOAD FRAUD REPORT
    # ------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📥 Download Fraud Intelligence Report")

    report_data = pd.DataFrame({
        "Metric": [
            "Total Transactions",
            "Fraud Transactions",
            "Genuine Transactions",
            "Fraud Percentage"
        ],
        "Value": [
            total_transactions,
            total_fraud,
            total_genuine,
            round(fraud_percentage, 2)
        ]
    })

    st.dataframe(
        report_data,
        use_container_width=True
    )

    report_csv = report_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Fraud Intelligence Report",
        report_csv,
        "fraud_intelligence_report.csv",
        "text/csv"
    )

    st.success(
        "Fraud Intelligence Analysis Completed Successfully!"
    )


# ==========================================================
# PAGE 8 - LIVE FRAUD CENTER
# ==========================================================
elif page == "Live Fraud Center":

    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    st.subheader("🌐 Live Fraud Center — Credit Card Scam Watch")

    st.caption(
        "Developed by Hasti • Public scam reports and cyber-fraud "
        "news feed, not live bank transaction monitoring."
    )

    st.markdown("### 🔄 How this center works")

    st.markdown(
        "**1. Fetch public reports** → "
        "**2. Display headlines and dates** → "
        "**3. Categorize/read reports** → "
        "**4. Follow prevention tips** → "
        "**5. Report suspected financial cyber fraud**"
    )

    st.markdown("---")
    st.markdown("## 🚨 Live Fraud News & Alerts")

    st.warning(
        "These are public news reports/advisories, "
        "not verified live financial transactions."
    )

    query = urllib.parse.quote("cyber fraud scam India")

    rss_url = (
        f"https://news.google.com/rss/search?q={query}"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    @st.cache_data(ttl=300, show_spinner=False)
    def get_public_fraud_news(feed_url):

        request = urllib.request.Request(
            feed_url,
            headers={
                "User-Agent": "Mozilla/5.0 FraudShieldAI"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=15
        ) as response:

            xml_data = response.read()

        root = ET.fromstring(xml_data)

        items = []

        for item in root.findall(".//item")[:25]:

            items.append({
                "title": item.findtext(
                    "title",
                    "Untitled report"
                ),
                "link": item.findtext(
                    "link",
                    ""
                ),
                "published": item.findtext(
                    "pubDate",
                    "Date unavailable"
                ),
                "description": item.findtext(
                    "description",
                    ""
                )
            })

        return items

    if st.button("🔄 Refresh Public Fraud News"):

        get_public_fraud_news.clear()
        st.rerun()

    try:

        news_items = get_public_fraud_news(rss_url)

        st.metric(
            "Public Reports Loaded",
            len(news_items)
        )

        if news_items:

            for news in news_items:

                with st.expander(
                    f"📰 {news['title']}"
                ):

                    st.caption(
                        news["published"]
                    )

                    if news["description"]:

                        st.write(
                            news["description"][:450],
                            unsafe_allow_html=True
                        )

                    if news["link"]:

                        st.markdown(
                            f"[Read Original Report]({news['link']})"
                        )

        else:

            st.info(
                "The feed returned no news items at this time."
            )

    except Exception as feed_error:

        st.error(
            f"Public news feed could not be loaded: {feed_error}"
        )

        st.info(
            "Check your internet connection and try Refresh again."
        )

    # ------------------------------------------------------
    # FRAUD PREVENTION CHECKLIST
    # ------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🛡️ Fraud Prevention Checklist")

    checklist = [
        "Never share OTP, PIN, CVV or banking passwords.",
        "Do not approve unknown UPI collect requests.",
        "Verify payments inside your bank app; do not trust screenshots.",
        "Avoid opening unknown SMS or WhatsApp links.",
        "Do not install remote-access apps at the request of strangers.",
        "Verify job and investment offers before paying money.",
        "If financial cyber fraud occurs in India, contact 1930 promptly and report at cybercrime.gov.in."
    ]

    for tip in checklist:

        st.markdown(
            f"- ✅ {tip}"
        )

    # ------------------------------------------------------
    # OFFICIAL REPORTING
    # ------------------------------------------------------
    st.markdown("### Official Reporting Links")

    st.markdown(
        "- [National Cyber Crime Reporting Portal](https://cybercrime.gov.in/)"
    )

    st.markdown(
        "- [CERT-In](https://www.cert-in.org.in/)"
    )

    st.caption(
        "Scam Radar shows public news only; it does not connect "
        "to bank payment systems or confirm individual reports."
    )

# ==========================================================
# TRANSACTION AUTHORIZATION & OTP VERIFICATION SIMULATOR
# ==========================================================

elif page == "Transaction & OTP Simulator":

    import secrets
    import time

    st.markdown("# 💳 Transaction Authorization Center")
    st.caption("Secure payment simulation • OTP verification • Demo mode")

    st.warning(
        "Educational simulation only. No real money is transferred "
        "and no SMS is sent."
    )

    # INITIALIZE SESSION
    if "otp_tx" not in st.session_state:
        st.session_state.otp_tx = None

    if "otp_attempts" not in st.session_state:
        st.session_state.otp_attempts = 0

    if "otp_status" not in st.session_state:
        st.session_state.otp_status = "Not Started"

    # --------------------------------------------------
    # TRANSACTION FORM
    # --------------------------------------------------

    st.markdown("## 📝 Create Transaction")

    with st.form("otp_transaction_form"):

        customer_name = st.text_input(
            "Customer Name",
            placeholder="Enter customer name"
        )

        merchant_name = st.text_input(
            "Merchant Name",
            placeholder="Enter merchant name"
        )

        transaction_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=1,
            max_value=100000,
            value=1000,
            step=100
        )

        payment_type = st.selectbox(
            "Payment Method",
            ["Credit Card", "Online Payment", "POS Payment"]
        )

        initiate = st.form_submit_button(
            "🚀 Initiate Transaction"
        )

    # --------------------------------------------------
    # GENERATE TRANSACTION & OTP
    # --------------------------------------------------

    if initiate:

        if not customer_name.strip() or not merchant_name.strip():

            st.error("Please enter customer and merchant name.")

        else:

            generated_otp = str(
                secrets.randbelow(900000) + 100000
            )

            st.session_state.otp_tx = {
                "customer": customer_name,
                "merchant": merchant_name,
                "amount": transaction_amount,
                "method": payment_type,
                "otp": generated_otp,
                "transaction_id": "TXN-" + secrets.token_hex(4).upper(),
                "created": time.strftime("%d-%m-%Y %H:%M:%S")
            }

            st.session_state.otp_attempts = 0
            st.session_state.otp_status = "Pending"

            st.session_state.otp_verified = False

            st.success("Transaction initiated successfully.")

    # --------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------

    tx = st.session_state.otp_tx

    if tx:

        st.markdown("---")
        st.markdown("## 📋 Transaction Details")

        c1, c2, c3 = st.columns(3)

        c1.metric("Transaction Amount", f"₹{tx['amount']:,.2f}")
        c2.metric("Merchant", tx["merchant"])
        c3.metric("Payment Method", tx["method"])

        st.write("**Customer:**", tx["customer"])
        st.write("**Transaction ID:**", tx["transaction_id"])
        st.write("**Created:**", tx["created"])

        # --------------------------------------------------
        # MOCK OTP
        # --------------------------------------------------

        st.markdown("---")
        st.markdown("## 📲 Customer OTP Verification")

        st.info(
            f"Demo OTP for {tx['customer']}: "
            f"**{tx['otp']}**"
        )

        st.caption(
            "Simulation OTP displayed locally. "
            "It is not sent through SMS or to a real customer."
        )

        # --------------------------------------------------
        # VERIFY OTP
        # --------------------------------------------------

        if st.session_state.otp_status == "Pending":

            with st.form("otp_verify_form"):

                entered_otp = st.text_input(
                    "Enter 6-Digit OTP",
                    max_chars=6
                )

                verify_button = st.form_submit_button(
                    "🔐 Verify & Authorize"
                )

            if verify_button:

                if st.session_state.otp_attempts >= 3:

                    st.session_state.otp_status = "Locked"

                    st.error("Transaction locked.")

                elif entered_otp == tx["otp"]:

                    st.session_state.otp_status = "Approved"
                    st.session_state.otp_verified = True

                    st.success(
                        "✅ OTP verified! Transaction approved successfully."
                    )

                else:

                    st.session_state.otp_attempts += 1

                    remaining = (
                        3 - st.session_state.otp_attempts
                    )

                    if remaining <= 0:

                        st.session_state.otp_status = "Locked"

                        st.error(
                            "🚫 Maximum attempts reached. "
                            "Transaction has been locked."
                        )

                    else:

                        st.error(
                            f"❌ Incorrect OTP. "
                            f"Attempts remaining: {remaining}"
                        )

        # --------------------------------------------------
        # TRANSACTION STATUS
        # --------------------------------------------------

        st.markdown("---")
        st.markdown("## 📊 Transaction Status")

        status = st.session_state.otp_status

        if status == "Pending":

            st.warning("🟠 Pending OTP Verification")

        elif status == "Approved":

            st.success("🟢 Transaction Approved")

        elif status == "Locked":

            st.error("🔴 Transaction Locked")

        # --------------------------------------------------
        # RECEIPT
        # --------------------------------------------------

        if status == "Approved":

            st.markdown("---")
            st.markdown("## 🧾 Transaction Receipt")

            st.success("Payment Simulation Successful!")

            receipt = f"""
FRAUDSHIELD AI - DEMO RECEIPT

Transaction ID: {tx['transaction_id']}
Customer: {tx['customer']}
Merchant: {tx['merchant']}
Amount: Rs. {tx['amount']}
Payment Method: {tx['method']}
Date: {tx['created']}
Status: APPROVED

This is a simulated transaction.
No real money was transferred.
"""

            st.code(receipt)

            st.download_button(
                "📥 Download Receipt",
                data=receipt,
                file_name="Demo_Transaction_Receipt.txt",
                mime="text/plain"
            )

        # --------------------------------------------------
        # RESET TRANSACTION
        # --------------------------------------------------

        if st.button("🔄 New Transaction"):

            st.session_state.otp_tx = None
            st.session_state.otp_attempts = 0
            st.session_state.otp_status = "Not Started"
            st.session_state.otp_verified = False

            st.rerun()

    st.markdown("---")

    st.caption(
        "FraudShield AI • OTP Authorization Simulator • "
        "Demo transactions only"
    )


# ==========================================================
# AI FRAUD SIMULATOR LAB
# Dataset Independent - FraudShield AI
# ==========================================================

elif page == "AI Fraud Simulator Lab":

    import random
    from datetime import datetime

    st.markdown("""
    <style>
    .sim-title {
        font-size: 32px;
        font-weight: 800;
        color: #60A5FA;
    }
    .sim-sub {
        color: #94A3B8;
        font-size: 14px;
    }
    .sim-card {
        background: linear-gradient(135deg, #162338, #0D1728);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #243B55;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sim-title">🧠 AI Fraud Simulator Lab</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sim-sub">Synthetic transaction monitoring • '
        'Risk analysis • Fraud scenario simulation</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This module generates synthetic transactions. "
        "It is not connected to real banks or payment systems."
    )

    # --------------------------------------------------
    # SESSION INITIALIZATION
    # --------------------------------------------------

    if "sim_transactions" not in st.session_state:
        st.session_state.sim_transactions = pd.DataFrame()

    # --------------------------------------------------
    # SIMULATION SETTINGS
    # --------------------------------------------------

    st.markdown("---")
    st.markdown("## ⚙️ Simulation Control Center")

    col1, col2 = st.columns(2)

    with col1:

        scenario = st.selectbox(
            "🎯 Select Fraud Scenario",
            [
                "Normal Transactions",
                "High Amount Suspicious Activity",
                "Rapid Transaction Attack Simulation",
                "Unusual Location Simulation",
                "Night-Time Suspicious Activity",
                "Mixed Fraud Scenarios"
            ]
        )

    with col2:

        transaction_count = st.selectbox(
            "🔢 Number of Transactions",
            [10, 25, 50, 100],
            index=1
        )

    st.markdown("### 🎮 Simulation Controls")

    col1, col2 = st.columns(2)

    with col1:

        run_simulation = st.button(
            "🚀 RUN SIMULATION",
            use_container_width=True,
            type="primary"
        )

    with col2:

        reset_simulation = st.button(
            "🔄 RESET LAB",
            use_container_width=True
        )

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    if reset_simulation:

        st.session_state.sim_transactions = pd.DataFrame()

        st.rerun()

    # --------------------------------------------------
    # GENERATE SYNTHETIC TRANSACTIONS
    # --------------------------------------------------

    if run_simulation:

        generated = []

        customer_list = [
            "CUS-1021", "CUS-1022", "CUS-1023",
            "CUS-1024", "CUS-1025", "CUS-1026",
            "CUS-1027", "CUS-1028"
        ]

        location_list = [
            "Surat", "Ahmedabad", "Mumbai",
            "Delhi", "Unknown Location",
            "Dubai", "London"
        ]

        for i in range(transaction_count):

            customer = random.choice(customer_list)

            amount = random.randint(100, 5000)

            location = random.choice(location_list)

            hour = random.randint(6, 23)

            reasons = []

            # SCENARIO CONTROL

            if scenario == "Normal Transactions":

                amount = random.randint(100, 3000)
                location = random.choice(
                    ["Surat", "Ahmedabad", "Mumbai"]
                )
                hour = random.randint(8, 22)

            elif scenario == "High Amount Suspicious Activity":

                amount = random.randint(20000, 100000)
                reasons.append("High transaction amount")

            elif scenario == "Rapid Transaction Attack Simulation":

                amount = random.randint(1000, 9000)
                reasons.append("Rapid transaction activity")

            elif scenario == "Unusual Location Simulation":

                location = random.choice(
                    ["Dubai", "London", "Unknown Location"]
                )
                reasons.append("Unusual transaction location")

            elif scenario == "Night-Time Suspicious Activity":

                hour = random.randint(0, 4)
                amount = random.randint(5000, 40000)
                reasons.append("Night-time transaction")

            elif scenario == "Mixed Fraud Scenarios":

                choice = random.randint(1, 4)

                if choice == 1:

                    amount = random.randint(25000, 90000)
                    reasons.append("High amount")

                elif choice == 2:

                    location = "Unknown Location"
                    reasons.append("Unknown location")

                elif choice == 3:

                    hour = random.randint(0, 4)
                    reasons.append("Night-time activity")

                else:

                    amount = random.randint(1000, 5000)
                    reasons.append("Unusual activity")

            # RISK ENGINE

            risk_score = 5

            if amount >= 50000:

                risk_score += 55
                reasons.append("Extremely high amount")

            elif amount >= 20000:

                risk_score += 40
                reasons.append("High amount")

            elif amount >= 10000:

                risk_score += 20
                reasons.append("Above normal amount")

            if location in [
                "Unknown Location", "Dubai", "London"
            ]:

                risk_score += 25

                if "Unusual transaction location" not in reasons:
                    reasons.append("Unusual location")

            if hour < 5:

                risk_score += 25

                if "Night-time transaction" not in reasons:
                    reasons.append("Night-time activity")

            if scenario == "Rapid Transaction Attack Simulation":

                risk_score += 45

            risk_score = min(risk_score, 100)

            # DECISION

            if risk_score >= 70:

                decision = "BLOCK"
                risk_level = "HIGH"

            elif risk_score >= 40:

                decision = "REVIEW"
                risk_level = "MEDIUM"

            else:

                decision = "ALLOW"
                risk_level = "LOW"

            if not reasons:

                reasons = ["Normal transaction pattern"]

            generated.append({

                "Transaction ID":
                    f"SIM-{random.randint(100000,999999)}",

                "Customer":
                    customer,

                "Amount":
                    amount,

                "Location":
                    location,

                "Transaction Time":
                    f"{hour:02d}:{random.randint(0,59):02d}",

                "Risk Score":
                    risk_score,

                "Risk Level":
                    risk_level,

                "Decision":
                    decision,

                "Reason":
                    ", ".join(reasons)

            })

        st.session_state.sim_transactions = pd.DataFrame(
            generated
        )

        st.success(
            f"✅ {transaction_count} synthetic transactions generated!"
        )

    # --------------------------------------------------
    # ANALYTICS DASHBOARD
    # --------------------------------------------------

    sim_data = st.session_state.sim_transactions

    if not sim_data.empty:

        st.markdown("---")
        st.markdown("## 📊 Fraud Simulation Dashboard")

        total = len(sim_data)

        blocked = len(
            sim_data[sim_data["Decision"] == "BLOCK"]
        )

        review = len(
            sim_data[sim_data["Decision"] == "REVIEW"]
        )

        allowed = len(
            sim_data[sim_data["Decision"] == "ALLOW"]
        )

        average_risk = sim_data["Risk Score"].mean()

        fraud_percent = (
            blocked / total * 100
            if total > 0 else 0
        )

        # KPI CARDS

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📋 Total Transactions", total)

        c2.metric(
            "🚫 Blocked",
            blocked,
            f"{fraud_percent:.1f}%"
        )

        c3.metric("⚠️ Review", review)

        c4.metric("✅ Allowed", allowed)

        st.markdown("---")

        c1, c2 = st.columns(2)

        c1.metric(
            "📈 Average Risk Score",
            f"{average_risk:.1f}/100"
        )

        c2.metric(
            "💰 Total Transaction Amount",
            f"₹{sim_data['Amount'].sum():,.0f}"
        )

        # --------------------------------------------------
        # RISK SCORE GRAPH
        # --------------------------------------------------

        st.markdown("## 📉 Transaction Risk Score Analysis")

        fig, ax = plt.subplots(figsize=(11, 4))

        ax.bar(
            sim_data["Transaction ID"],
            sim_data["Risk Score"]
        )

        ax.axhline(
            40,
            linestyle="--",
            label="Medium Risk Threshold"
        )

        ax.axhline(
            70,
            linestyle=":",
            label="High Risk Threshold"
        )

        ax.set_ylim(0, 105)

        ax.set_xlabel("Transaction ID")
        ax.set_ylabel("Risk Score")
        ax.set_title("Transaction Risk Score Monitoring")

        ax.tick_params(
            axis="x",
            rotation=90
        )

        ax.legend()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        # --------------------------------------------------
        # DECISION DISTRIBUTION
        # --------------------------------------------------

        st.markdown("## 🛡️ Security Decision Distribution")

        decision_counts = sim_data["Decision"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.pie(
            decision_counts.values,
            labels=decision_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Allow / Review / Block Distribution")

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        # --------------------------------------------------
        # AMOUNT ANALYSIS
        # --------------------------------------------------

        st.markdown("## 💳 Transaction Amount Analysis")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.hist(
            sim_data["Amount"],
            bins=15
        )

        ax.set_xlabel("Transaction Amount (₹)")
        ax.set_ylabel("Frequency")

        ax.set_title("Simulated Transaction Amount Distribution")

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        # --------------------------------------------------
        # FILTER TRANSACTIONS
        # --------------------------------------------------

        st.markdown("---")
        st.markdown("## 🔍 Transaction Investigation Panel")

        selected_status = st.selectbox(
            "Filter Security Decision",
            [
                "All Transactions",
                "BLOCK",
                "REVIEW",
                "ALLOW"
            ]
        )

        if selected_status == "All Transactions":

            filtered_data = sim_data

        else:

            filtered_data = sim_data[
                sim_data["Decision"] == selected_status
            ]

        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # HIGH RISK ALERTS
        # --------------------------------------------------

        st.markdown("## 🚨 Suspicious Transaction Alerts")

        suspicious = sim_data[
            sim_data["Risk Score"] >= 40
        ]

        if not suspicious.empty:

            st.warning(
                f"{len(suspicious)} suspicious transactions identified."
            )

            st.dataframe(
                suspicious,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No suspicious transaction in this simulation."
            )

        # --------------------------------------------------
        # DOWNLOAD REPORT
        # --------------------------------------------------

        st.markdown("---")
        st.markdown("## 📥 Export Simulation Report")

        csv_bytes = sim_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Transaction Report",
            data=csv_bytes,
            file_name="FraudShield_Simulation_Report.csv",
            mime="text/csv"
        )

    else:

        st.markdown("---")

        st.info(
            "👆 Select a scenario and click RUN SIMULATION "
            "to start the fraud simulation."
        )

        st.markdown("""
        ### 🔬 Available Scenarios

        🟢 Normal Transactions

        🔴 High Amount Suspicious Activity

        ⚡ Rapid Transaction Attack Simulation

        🌍 Unusual Location Simulation

        🌙 Night-Time Suspicious Activity

        🎯 Mixed Fraud Scenarios
        """)

    # --------------------------------------------------
    # FOOTNOTE
    # --------------------------------------------------

    st.markdown("---")

    st.caption(
        "FraudShield AI • AI Fraud Simulator Lab • "
        "Synthetic data only • Educational prototype"
    )

    
# ==========================================================
# AI FRAUD ASSISTANT - CHAT SECTION
# ==========================================================

st.markdown("---")
st.markdown("## 🤖 AI Fraud Assistant")
st.caption("Ask questions about credit card fraud, risk, OTP and cyber security.")

if "fraud_ai_chat" not in st.session_state:
    st.session_state.fraud_ai_chat = []

# Suggested questions
st.markdown("### 💡 Quick Questions")

suggestions = [
    "What is credit card fraud?",
    "How can I prevent OTP fraud?",
    "Why are transactions blocked?",
    "What is a risk score?",
]

cols = st.columns(2)

for i, question in enumerate(suggestions):
    with cols[i % 2]:
        if st.button(question, key=f"fraud_suggestion_{i}"):
            st.session_state.fraud_ai_chat.append(
                {"role": "user", "content": question}
            )
            st.session_state.fraud_ai_chat.append(
                {
                    "role": "assistant",
                    "content": get_fraud_ai_response(question)
                }
            )
            st.rerun()

# Display chat history
for message in st.session_state.fraud_ai_chat:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# Chat input
user_question = st.chat_input(
    "Ask AI about fraud, transactions, OTP or security..."
)

if user_question:

    st.session_state.fraud_ai_chat.append(
        {"role": "user", "content": user_question}
    )

    answer = get_fraud_ai_response(user_question)

    st.session_state.fraud_ai_chat.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()

# Clear chat
if st.button("🗑️ Clear AI Chat", key="clear_fraud_ai_chat"):

    st.session_state.fraud_ai_chat = []
    st.rerun()



# ==========================================================
# FOOTER
# ==========================================================
st.sidebar.markdown("---")

st.sidebar.caption(
    "Credit Card Fraud Detection • Developed by Hasti • Educational Prototype"
)

