## Importing Necessary Libraries and Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load the dataset using Pandas
print("--- Loading Dataset ---")
df = pd.read_csv('framingham.csv')

# Check the structure and first few rows of the dataset
print("\nDataset Information:")
df.info()

print("\nFirst 5 rows of the dataset:")
print(df.head())

# Drop the unnecessary 'education' column
df.drop('education', axis=1, inplace=True)
print("\n'education' column dropped.")

# Rename columns for better readability (e.g., from camelCase to snake_case)
df.rename(columns={
    'male': 'is_male',
    'age': 'age',
    'currentSmoker': 'is_current_smoker',
    'cigsPerDay': 'cigs_per_day',
    'BPMeds': 'on_bp_meds',
    'prevalentStroke': 'has_prevalent_stroke',
    'prevalentHyp': 'has_prevalent_hyp',
    'diabetes': 'has_diabetes',
    'totChol': 'total_cholesterol',
    'sysBP': 'systolic_bp',
    'diaBP': 'diastolic_bp',
    'BMI': 'bmi',
    'heartRate': 'heart_rate',
    'glucose': 'glucose',
    'TenYearCHD': 'ten_year_chd'
}, inplace=True)
print("\nColumns renamed for better readability.")
print("\nFirst 5 rows after cleaning:")
print(df.head())


# 2. Data Preprocessing

# Handle missing values by removing rows with NaN values
print("\n--- Data Preprocessing ---")
print(f"Number of rows before dropping NaN: {len(df)}")
df.dropna(inplace=True)
print(f"Number of rows after dropping NaN: {len(df)}")
df.reset_index(drop=True, inplace=True)

# Define features (X) and target (y)
# X contains all columns except the target variable 'ten_year_chd'
X = df.drop('ten_year_chd', axis=1)
# y contains only the target variable
y = df['ten_year_chd']

# Split the dataset into training (70%) and testing (30%) sets
# random_state ensures that the splits are the same every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nDataset split into training and testing sets:")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Normalize numerical features using StandardScaler
# StandardScaler standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nNumerical features have been scaled.")
# Note: Categorical variables like 'is_male' are already in numerical form (0 or 1) and will be scaled along with other features.


# 3. Exploratory Data Analysis (EDA)

print("\n--- Exploratory Data Analysis ---")

# Analyze class distribution of heart disease cases (CHD = 0 or 1)
plt.figure(figsize=(8, 6))
sns.countplot(x='ten_year_chd', data=df)
plt.title('Class Distribution of 10-Year CHD Risk')
plt.xlabel('10-Year CHD Risk (0 = No, 1 = Yes)')
plt.ylabel('Patient Count')
plt.show()

# Visualize data distributions using histograms
print("\nDisplaying histograms for all features...")
X.hist(figsize=(20, 15), bins=20)
plt.suptitle('Histograms of All Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Visualize correlations using a heatmap
print("\nDisplaying correlation heatmap...")
plt.figure(figsize=(15, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of All Features')
plt.show()

# From the EDA, we can observe:
# - Class Imbalance: There are significantly more patients without CHD (class 0) than with CHD (class 1).
# - Key Risk Factors: The heatmap shows that 'age', 'systolic_bp', 'diastolic_bp', and 'glucose' have a relatively higher positive correlation with 'ten_year_chd'.


# predict_chd.py (continued)



# 4. Model Training using Logistic Regression

print("\n--- Model Training ---")

# Define the Logistic Regression Model
# We increase max_iter to ensure the model converges
log_reg = LogisticRegression(max_iter=1000, random_state=42)


# Train the model on the scaled training dataset
# The model uses Binary Cross-Entropy as its default loss function
print("Training the Logistic Regression model...")
log_reg.fit(X_train_scaled, y_train)
print("Model training complete.")


# 5. Model Evaluation and Prediction
print("\n--- Model Evaluation ---")

# Predict on the scaled test data
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate model performance
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Precision, Recall, F1-Score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No CHD', 'CHD']))

# ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
print(f"ROC AUC Score: {roc_auc:.4f}")

# Predict the likelihood of heart disease in new patient data
print("\n--- Prediction on New Patient Data ---")
# Create a sample of new patient data (ensure it has the same columns as X)
new_patient_data = pd.DataFrame({
    'is_male': [1],
    'age': [55],
    'is_current_smoker': [1],
    'cigs_per_day': [20],
    'on_bp_meds': [0],
    'has_prevalent_stroke': [0],
    'has_prevalent_hyp': [1],
    'has_diabetes': [0],
    'total_cholesterol': [250],
    'systolic_bp': [160],
    'diastolic_bp': [95],
    'bmi': [31.5],
    'heart_rate': [80],
    'glucose': [85]
})

print("New patient data:")
print(new_patient_data)

# Scale the new patient data using the same scaler
new_patient_scaled = scaler.transform(new_patient_data)

# Predict the 10-year CHD risk
prediction = log_reg.predict(new_patient_scaled)
prediction_proba = log_reg.predict_proba(new_patient_scaled)

print(f"\nPrediction (0=No CHD, 1=CHD): {prediction[0]}")
print(f"Prediction Probability [No CHD, CHD]: {prediction_proba[0]}")
print(f"Likelihood of Heart Disease: {prediction_proba[0][1]*100:.2f}%")



###b  --- Conclusion ---

# This model provides a foundational tool for early heart disease detection using key health indicators.
# By inputting a patient's health metrics, it can predict the 10-year risk of developing Coronary Heart Disease.

# Future improvements can include:
# 1. Handling Class Imbalance: Using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create a more balanced dataset, which can improve the model's ability to predict the minority class (CHD cases).
# 2. Advanced Models: Implementing more complex models like Random Forest, Gradient Boosting (XGBoost), or Neural Networks could yield higher accuracy and better predictive performance.
# 3. Feature Engineering: Creating new features from the existing ones might help the model find more complex patterns in the data.