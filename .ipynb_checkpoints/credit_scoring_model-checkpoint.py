# Credit Scoring Model – End-to-End Notebook
# ==================================
# This notebook performs:
# 1. Data loading
# 2. Cleaning & preprocessing
# 3. Exploratory data analysis (EDA)
# 4. Feature engineering
# 5. Model training & evaluation
# 6. Model usage for credit scoring

# -----------------------------
# 1. Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv('archive/train.csv')
print(df.shape)
df.head()

# -----------------------------
# 3. Initial Inspection
# -----------------------------
df.info()
df.describe(include='all')

# -----------------------------
# 4. Data Cleaning
# -----------------------------
# Drop identifiers and leakage columns
cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')


# Replace placeholder strings with NaN
for col in df.columns:
    df[col] = df[col].replace(['_', 'nan', 'NaN'], np.nan)

# Convert numeric columns stored as strings
numeric_cols = df.columns.drop('Credit_Score')
for col in numeric_cols:
    #df[col] = pd.to_numeric(df[col], errors='ignore')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -----------------------------
# 5. Exploratory Data Analysis
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_Score', data=df)
plt.title('Distribution of Credit Score')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# -----------------------------
# 6. Feature Engineering
# -----------------------------
# Encode target
target_encoder = LabelEncoder()
df['Credit_Score'] = target_encoder.fit_transform(df['Credit_Score'])

X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', LabelEncoder())
])

# Apply encoding manually for categorical columns
X_encoded = X.copy()
for col in categorical_features:
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 8. Model Training
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# -----------------------------
# 9. Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n')
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 10. Feature Importance
# ---------------s')

# -----------------------------
# 10.5. Save Model as Pickle File
# -----------------------------
with open('credit_scoring_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved to credit_scoring_model.pkl')

# Save target encoder for later use
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print('Target encoder saved to target_encoder.pkl')

# Save feature encoders for later use
feature_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_features}
with open('feature_encoders.pkl', 'wb') as f:
    pickle.dump(feature_encoders, f)
print('Feature encoders saved to feature_encoders.pkl')
plt.show()

# -----------------------------
# 11. Credit Scoring Prediction Function
# -----------------------------
def predict_credit_score(input_data: pd.DataFrame):
    """
    input_data: DataFrame with same structure as training features
    """
    for col in categorical_features:
        input_data[col] = LabelEncoder().fit_transform(input_data[col])
    pred = model.predict(input_data)
    return target_encoder.inverse_transform(pred)

# Example usage
# sample = X_encoded.sample(1)
# print(predict_credit_score(sample))

# -----------------------------
# 12. Business Interpretation
# -----------------------------
# Poor     -> High Risk (Decline / Secure Loans Only)
# Standard -> Medium Risk (Approve with conditions)
# Good     -> Low Risk (Approve with favorable terms)

print('Notebook completed successfully.')
