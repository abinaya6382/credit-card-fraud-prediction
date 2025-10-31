import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Prediction", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.write("This app uses **Logistic Regression** to predict fraudulent credit card transactions.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your credit card dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    credit_df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(credit_df.head())

    # Check if target column exists
    if 'Class' not in credit_df.columns:
        st.error("⚠️ The dataset must contain a 'streClass' column (0 = normal, 1 = fraud).")
    else:
        # Split features and target
        X = credit_df.drop(['Class'], axis=1)
        y = credit_df['Class']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        # Train model
        LR = LogisticRegression(max_iter=1000)
        LR.fit(X_train, y_train)

        # Accuracy
        y_pred = LR.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

        # --- PREDICTION SECTION ---
        st.subheader("🔍 Predict on New Data")

        # Option 1: Upload new CSV for prediction
        test_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"], key="test")

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            if 'Class' in test_df.columns:
                X_new = test_df.drop('Class', axis=1)
            else:
                X_new = test_df

            y_new_pred = LR.predict(X_new)

            st.write("### Predictions:")
            pred_df = pd.DataFrame({
                'Prediction': y_new_pred
            })
            st.dataframe(pred_df)

            # Download option
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

        # Option 2: Manual input
        st.subheader("✍️ Enter Feature Values Manually")
        sample = {}
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(X[col].mean()))
            sample[col] = val

        if st.button("Predict Transaction Type"):
            new_df = pd.DataFrame([sample])
            prediction = LR.predict(new_df)[0]
            if prediction == 1:
                st.error("🚨 This transaction is likely FRAUDULENT.")
            else:
                st.success("✅ This transaction looks NORMAL.")
else:
    st.info("📂 Please upload a dataset to begin.")
