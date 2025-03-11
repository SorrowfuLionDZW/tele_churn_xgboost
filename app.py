import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.metrics import accuracy_score, precision_score

# Load the dataset
@st.cache_data  # Cache data to improve performance
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Preprocess the data
def preprocess_data(data):
    # Assuming 'Churn' is the target variable
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    # Convert categorical variables to dummy/indicator variables
    X = pd.get_dummies(X)
    return X, y

# Train the model using XGBoost
def train_model(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42)  # Use XGBoost
    model.fit(X_train, y_train)
    return model

# Predict churn probability
def predict_churn(model, X):
    predictions = model.predict_proba(X)[:, 1]  # Probability of churn (class 1)
    return predictions

# Streamlit app
def main():
    st.title("Customer Churn Prediction Dashboard")
    st.write("Upload your customer data (CSV file) to analyze and predict churn.")

    # File uploader for the dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)
        st.write("### Dataset Overview")
        st.write(data.head())

        # Display key statistics
        st.write("### Key Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            st.metric("Churn Rate", f"{data['Churn'].mean() * 100:.2f}%")
        with col3:
            st.metric("Average Monthly Charges", f"${data['MonthlyCharges'].mean():.2f}")

        # Visualizations
        st.write("### Visualizations")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Churn Distribution
        sns.countplot(x='Churn', data=data, ax=axes[0])
        axes[0].set_title("Churn Distribution")
        axes[0].set_xlabel("Churn")
        axes[0].set_ylabel("Count")

        # Monthly Charges Distribution
        sns.histplot(data['MonthlyCharges'], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Monthly Charges Distribution")
        axes[1].set_xlabel("Monthly Charges")
        axes[1].set_ylabel("Frequency")

        st.pyplot(fig)

        # Preprocess data
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model using XGBoost
        model = train_model(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")

        # Predict churn probability for all customers
        data['Churn Probability'] = predict_churn(model, X)

        # Display predictions
        st.write("### Churn Predictions")
        st.write(data[['CustomerID', 'Churn Probability']])

        # Download predictions as CSV
        st.write("### Download Predictions")
        csv = data[['CustomerID', 'Churn Probability']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()