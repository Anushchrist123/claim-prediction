import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("dataset_balanced.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")  # Standardizing column names

# Drop unnecessary columns
if "region" in df.columns:
    df.drop(columns=["region"], inplace=True)

# Title
st.title("Customer Risk Profiling & Claim Prediction Analysis")

# Feature explanations
st.subheader("Feature Explanations")
feature_info = {
    "policy_tenure": "Duration (in years) the policy has been active.",
    "age_of_car": "Age of the insured car in years.",
    "age_of_policyholder": "Age of the person holding the insurance.",
    "ncap_rating": "Safety rating of the car (0-5 stars).",
    "fuel_type": "Type of fuel used in the car (e.g., Petrol, Diesel).",
    "type_of_vehicle": "Category of the vehicle (e.g., Sedan, SUV).",
    "age_of_driver": "Age of the person driving the vehicle.",
    "years_of_driving_experience": "Number of years the driver has been driving.",
    "previous_accident_history": "Number of accidents the driver has had in the past.",
}
st.write(pd.DataFrame.from_dict(feature_info, orient='index', columns=['Description']))

# Display dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Check missing values
st.subheader("Missing Values Check")
st.write(df.isnull().sum())

# Correlation heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=['number'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Exploratory Data Analysis
st.subheader("Summary Statistics")
st.write(df.describe())

# Outlier Detection - Box Plots
st.subheader("Outlier Detection - Box Plots")
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    fig, ax = plt.subplots()
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Boxplot for {col}")
    st.pyplot(fig)

# Claim Distribution
st.subheader("Claim Distribution")
fig, ax = plt.subplots()
sns.countplot(x=df['claim'], palette='viridis', ax=ax)
st.pyplot(fig)

# Define features and target
numerical_cols = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'ncap_rating', 'age_of_driver', 'years_of_driving_experience', 'previous_accident_history']
categorical_cols = ['fuel_type', 'type_of_vehicle']
target_column = 'claim'

# Handle missing values
df[numerical_cols] = SimpleImputer(strategy='mean').fit_transform(df[numerical_cols])
df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols].astype(str))

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare dataset
X = df.drop(columns=[target_column, 'policy_id'], errors='ignore')
y = df[target_column]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
defined_models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train models and evaluate
results = {}
feature_importances = {}
for model_name, model in defined_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = np.mean(y_pred == y_test)
    results[model_name] = {"Precision": precision, "Recall": recall, "F1-Score": f1, "Accuracy": accuracy}
    
    # Capture feature importance if applicable
    if hasattr(model, 'feature_importances_'):
        feature_importances[model_name] = model.feature_importances_

st.subheader("Model Performance")
st.write(pd.DataFrame(results).T)

# Display feature importance
st.subheader("Feature Importance Analysis")
for model_name, importances in feature_importances.items():
    fig, ax = plt.subplots()
    sns.barplot(x=X.columns, y=importances, ax=ax)
    ax.set_title(f"Feature Importance - {model_name}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Prediction for a new customer
st.subheader("Predict Claim for a New Customer")
new_customer = {
    'policy_tenure': st.number_input("Policy Tenure", min_value=0.1, max_value=10.0, value=3.0),
    'age_of_car': st.number_input("Age of Car", min_value=0.0, max_value=20.0, value=5.0),
    'age_of_policyholder': st.number_input("Age of Policyholder", min_value=18, max_value=80, value=30),
    'ncap_rating': st.selectbox("NCAP Rating", [0, 1, 2, 3, 4, 5]),
    'fuel_type': st.selectbox("Fuel Type", df['fuel_type'].unique()),
    'type_of_vehicle': st.selectbox("Type of Vehicle", df['type_of_vehicle'].unique()),
    'age_of_driver': st.number_input("Age of Driver", min_value=18, max_value=80, value=30),
    'years_of_driving_experience': st.number_input("Years of Driving Experience", min_value=0, max_value=60, value=5),
    'previous_accident_history': st.selectbox("Previous Accident History", list(range(11)))  # Allowing 0-10+ values
}

# Convert new customer input to DataFrame
new_customer_df = pd.DataFrame([new_customer])

# Encode categorical values
for col in categorical_cols:
    if col in label_encoders:
        new_customer_df[col] = new_customer_df[col].map(lambda x: label_encoders[col].classes_.tolist().index(x) if x in label_encoders[col].classes_ else -1)

# Scale input
new_customer_df = new_customer_df[X.columns]
new_customer_scaled = scaler.transform(new_customer_df)

# Predict claim status using models
st.subheader("Prediction Results")
prediction_probs = {}

for model_name, model in defined_models.items():
    prob = model.predict_proba(new_customer_scaled)[0][1]  # Probability of making a claim
    prediction_probs[model_name] = prob
    result = "WILL make a claim" if prob > 0.5 else "will NOT make a claim"
    st.write(f"{model_name}: {result} (Claim Probability: {prob:.2f})")

# Visualizing probabilities
st.subheader("Claim Probability Visualization")
fig, ax = plt.subplots()
sns.barplot(x=list(prediction_probs.keys()), y=list(prediction_probs.values()), palette="coolwarm", ax=ax)
ax.set_ylabel("Claim Probability")
ax.set_title("Model-wise Claim Probability")
st.pyplot(fig)

# Explanation of Feature Contribution
st.subheader("Feature Impact on Prediction")
for model_name, importances in feature_importances.items():
    fig, ax = plt.subplots()
    sns.barplot(x=X.columns, y=importances, ax=ax)
    ax.set_title(f"Feature Contribution - {model_name}")
    plt.xticks(rotation=45)
    st.pyplot(fig)
