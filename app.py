import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="FRAUDSENSE UPI", layout="wide")

# =====================================================
# LOAD DATA (LIMIT 50K FOR 8GB LAPTOP)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Upi fraud dataset final.csv")
    df.columns = df.columns.str.strip()
    df = df.sample(n=min(50000, len(df)), random_state=42)
    return df

df = load_data()

# =====================================================
# NAVIGATION
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Fraud Detection"])

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.title("🛡 FRAUDSENSE UPI")
    st.subheader("AI-Powered Fraud Detection System")

    total_txn = len(df)
    fraud_cases = df["fraud"].sum()
    legit_cases = total_txn - fraud_cases
    fraud_rate = (fraud_cases / total_txn) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total_txn:,}")
    c2.metric("Fraud Cases", f"{fraud_cases:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Legitimate Cases", f"{legit_cases:,}")

    st.markdown("---")
    st.write("This system uses ANN with hyperparameter tuning to detect fraudulent transactions.")

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif page == "Dashboard":

    st.title("📊 Fraud Analytics Dashboard")

    st.subheader("Fraud Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="fraud", data=df, ax=ax1)
    st.pyplot(fig1)

    if "Transaction_Type" in df.columns:
        st.subheader("Transaction Type Distribution")
        fig2, ax2 = plt.subplots()
        df["Transaction_Type"].value_counts().head(10).plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

# =====================================================
# FRAUD DETECTION PAGE
# =====================================================
elif page == "Fraud Detection":

    st.title("🔍 Fraud Detection Model")

    # -------------------------------
    # PREPROCESSING
    # -------------------------------
    target = "fraud"

    X = df.drop(columns=[target])
    y = df[target]

    drop_cols = ["Transaction_ID", "Customer_ID", "Merchant_ID", "Device_ID", "IP_Address"]
    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)

    # Handle Date
    if "Date" in X.columns:
        X["Date"] = pd.to_datetime(X["Date"], errors="coerce")
        X["Year"] = X["Date"].dt.year
        X["Month"] = X["Date"].dt.month
        X["Day"] = X["Date"].dt.day
        X.drop(columns="Date", inplace=True)

    # Handle Time
    if "Time" in X.columns:
        X["Time"] = pd.to_datetime(X["Time"], errors="coerce")
        X["Hour"] = X["Time"].dt.hour
        X.drop(columns="Time", inplace=True)

    # Missing values
    X.fillna(X.median(numeric_only=True), inplace=True)
    X.fillna("Unknown", inplace=True)

    # Encoding
    X = pd.get_dummies(X, drop_first=True, dtype="int8")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights (for imbalance)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # -------------------------------
    # HYPERPARAMETER TUNING
    # -------------------------------
    st.sidebar.header("⚙ Hyperparameter Tuning")

    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005])
    hidden_layers = st.sidebar.slider("Hidden Layers", 1, 3, 2)
    neurons = st.sidebar.slider("Neurons per Layer", 32, 128, 64)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3)
    epochs = st.sidebar.slider("Epochs", 10, 30, 20)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64])

    # -------------------------------
    # BUILD MODEL
    # -------------------------------
    def build_model():
        model = keras.Sequential()

        for _ in range(hidden_layers):
            model.add(keras.layers.Dense(neurons, activation="relu"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(1, activation="sigmoid"))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

    # -------------------------------
    # TRAIN BUTTON (ONLY HERE)
    # -------------------------------
    if st.button("🚀 Train Model"):

        with st.spinner("Training model... Please wait"):

            model = build_model()

            history = model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                verbose=0
            )

        # Predictions
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.success(f"Accuracy: {acc:.4f}")
        st.success(f"F1 Score: {f1:.4f}")

        # -------------------------------
        # VISUALIZATIONS
        # -------------------------------
        col1, col2 = st.columns(2)

        with col1:
            fig3, ax3 = plt.subplots()
            ax3.plot(history.history["accuracy"])
            ax3.plot(history.history["val_accuracy"])
            ax3.set_title("Accuracy Curve")
            st.pyplot(fig3)

        with col2:
            fig4, ax4 = plt.subplots()
            ax4.plot(history.history["loss"])
            ax4.plot(history.history["val_loss"])
            ax4.set_title("Loss Curve")
            st.pyplot(fig4)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig5, ax5 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig5)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))