
# ### 1. Importing Libraries and Datase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings

# Suppress all warnings for cleaner output in a notebook environment
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Load the dataset using Pandas and explore its structure.

# Load the dataset
try:
    df = pd.read_csv('parkinson_disease.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'parkinson_disease.csv' not found. Make sure the file is in the same directory.")
    exit() # Exit if the file isn't found

# Display the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display basic information about the dataset (columns, non-null counts, dtypes)
print("\nDataset Information:")
df.info()

# Display descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(df.describe())

# Display the shape of the dataset (rows, columns)
print("\nDataset Shape:")
print(df.shape)


### Step 2: Data Preprocessing**


# (Keep your scaling code as is)
# Separate features (X) and target (y)
# The 'class' column is our target variable, and 'id' is just an identifier.
X = df.drop(columns=['class', 'id'])
y = df['class']   

# Identify numerical features for scaling. All features except 'gender' (which is already 0/1) and 'class' are numerical.
# From the df.info() and df.describe(), 'gender' is numerical, but could be considered categorical if not 0/1.
# Given its nature, it's often best treated as a numerical feature or one-hot encoded if more categories existed.
# For simplicity and given its 0/1 nature, we'll include it in scaling if it were float, or leave it as is if it's already binary.
# Let's inspect 'gender' unique values to confirm.
print("Unique values in 'gender' column:", df['gender'].unique())

# Assuming 'gender' is binary (0 or 1) and doesn't need scaling with other numerical features.
# We will scale all other columns.
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
if 'gender' in numerical_features:
    numerical_features.remove('gender') # Exclude gender from scaling as it's binary

print(f"\nNumerical features to be scaled: {numerical_features}")

scaler = StandardScaler()


X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\nFeatures after scaling (first 5 rows):")
print(X.head())


# Check class distribution

print("\n--- Pre-SMOTE Diagnostics for 'y' ---")
print(f"y dtype: {y.dtype}")
print(f"y unique values: {y.unique()}")
print(f"Number of NaN values in y before SMOTE: {y.isnull().sum()}")

if y.isnull().any():
    print("NaN values found in 'y'. Dropping rows with NaN in 'class' column.")
    original_df_shape = df.shape
    df.dropna(subset=['class'], inplace=True)

    # RE-DEFINE X AND Y AFTER DROPPING NaNs IN 'class'
    X = df.drop(columns=['class', 'id'])
    y = df['class']

    print(f"Original DataFrame shape: {original_df_shape}")
    print(f"DataFrame shape after dropping NaNs in 'class': {df.shape}")
    print(f"Number of NaN values in y after handling: {y.isnull().sum()}")

    # RE-SCALE NUMERICAL FEATURES IN X AFTER RE-DEFINITION
    # (Assuming 'scaler' is already defined from an earlier cell)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    if 'gender' in numerical_features:
        numerical_features.remove('gender')
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    print("\nNumerical features re-scaled after NaN removal in 'y'.")
# --- END OF NEW CODE BLOCK ---

# Check class distribution (THIS WAS YOUR ORIGINAL LINE 67, NOW SHIFTED DOWN)
print("Class distribution before SMOTE:")
print(y.value_counts())


print("Class distribution before SMOTE:")
print(y.value_counts())

# Get percentages for individual classes and then format them
# Access the count for class 1 (or the majority class, typically the first index if sorted by default)
# and for class 0 (or the minority class).
# Based on your previous output:
# class
# 1.0    42  (majority)
# 0.0     6  (minority)


# For class 1 (Parkinson's):
percent_class_1_before = (y.value_counts().get(1.0, 0) / len(y)) * 100
# For class 0 (Healthy):
percent_class_0_before = (y.value_counts().get(0.0, 0) / len(y)) * 100
print(f"Percentage of class 1: {percent_class_1_before:.2f}%")
print(f"Percentage of class 0: {percent_class_0_before:.2f}%")


# Apply SMOTE if there's significant imbalance
# A common threshold for imbalance is when the minority class is less than 20-30%
# Assuming 0.0 is the minority class based on your output:
minority_class_percentage = percent_class_0_before / 100 # Convert back to a fraction for comparison

if minority_class_percentage < 0.30: # Example threshold
    print("\nClass imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\nClass distribution after SMOTE:")
    print(y_resampled.value_counts())

    # Get percentages for individual classes after SMOTE
    percent_class_1_after = (y_resampled.value_counts().get(1.0, 0) / len(y_resampled)) * 100
    percent_class_0_after = (y_resampled.value_counts().get(0.0, 0) / len(y_resampled)) * 100

    print(f"Percentage of class 1: {percent_class_1_after:.2f}%")
    print(f"Percentage of class 0: {percent_class_0_after:.2f}%")
    X = X_resampled
    y = y_resampled
else:
    print("\nClass is reasonably balanced. SMOTE not applied.")

# #### Split dataset into training and testing sets (80%-20%)


# Explanation for Class Imbalance and SMOTE:**
# y.value_counts()` shows the number of samples for each class in the target variable.
# The `if` condition checks if the minority class (presumably `1` for Parkinson's based on typical disease datasets) is below a certain percentage.
# If imbalanced, `SMOTE` (Synthetic Minority Over-sampling Technique) is applied. SMOTE works by creating synthetic samples from the minority class, rather than just duplicating existing ones, which helps the model learn more effectively from the minority class.
# The features (`X`) and target (`y`) are updated to `X_resampled` and `y_resampled` after SMOTE, ensuring the models are trained on a balanced dataset.

# #### Split dataset into training and testing sets (80%-20%)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")


# **Explanation for Data Splitting:**
# *   `train_test_split` divides the dataset into training (80%) and testing (20%) sets.
# *   `random_state=42` ensures reproducibility, meaning you'll get the same split every time you run the code.
# *   `stratify=y` is crucial for classification tasks, especially with imbalanced datasets. It ensures that the proportion of target classes in the training and testing sets is similar to the original dataset (or the resampled dataset if SMOTE was applied).

##Visualize feature distributions using histograms and boxplots.


# Add 'class' back to X temporarily for plotting, then remove it
df_eda = X.copy()
df_eda['class'] = y

# Set style for plots
sns.set_style("whitegrid")

# Create histograms for numerical features
print("Visualizing Feature Distributions (Histograms):")
# Select a subset of numerical features for clarity, or loop through all
features_to_plot = numerical_features[:5] # Plot first 5 numerical features for example
if 'PPE' in numerical_features: features_to_plot.append('PPE') # Ensure PPE is included if it's a key feature

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_eda[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Create boxplots to visualize distributions across classes
print("\nVisualizing Feature Distributions (Boxplots by Class):")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='class', y=feature, data=df_eda)
    plt.title(f'{feature} by Class')
plt.tight_layout()
plt.show()


# Explanation for Histograms and Boxplots:**
# Histograms show the distribution of individual features. This helps identify if features are normally distributed, skewed, or have outliers.
# boxplots visualize the distribution of numerical features across different classes. This is useful for seeing if there are noticeable differences in feature values between healthy individuals (class 0) and Parkinson's patients (class 1). For example, if a voice feature shows a significantly different median or spread between the two classes, it could be a strong predictor.

#### Use correlation heatmaps to identify the most important features for prediction.


# Calculate the correlation matrix
correlation_matrix = df_eda.corr()

# Plot the correlation heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f", linewidths=.5) # annot=True can make it very crowded
plt.title('Correlation Heatmap of All Features')
plt.show()

# Display correlations with the target variable 'class'
print("\nCorrelation with 'class' (target variable):")
print(correlation_matrix['class'].sort_values(ascending=False))

# Explanation for Correlation Heatmap:**
# A correlation heatmap visually represents the correlation coefficients between all pairs of features.
# Values close to 1 (red) indicate a strong positive correlation, values close to -1 (blue) indicate a strong negative correlation, and values close to 0 (white/light colors) indicate a weak or no linear correlation.
# By sorting correlations with the 'class' variable, you can identify features that have the strongest linear relationship with Parkinson's disease presence (positive or negative). These features are often important for prediction.

# #### Identify trends in voice features that contribute to Parkinson's disease.


# Voice features typically include jitter, shimmer, harmonicity, MFCCs, etc.
# Let's select some key voice features based on common Parkinson's research (e.g., Jitter, Shimmer, PPE, DFA, RPDE, etc.)
# You can refer to the dataset description or common knowledge about Parkinson's voice analysis for more specific features.
voice_features = ['PPE', 'DFA', 'RPDE', 'locPctJitter', 'locAbsJitter', 'rapJitter',
                  'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer',
                  'meanAutoCorrHarmonicity', 'meanNoiseToHarmHarmonicity', 'meanHarmToNoiseHarmonicity']

# Visualize trends for selected voice features by class
plt.figure(figsize=(18, 12))
for i, feature in enumerate(voice_features):
    if feature in df_eda.columns: # Ensure the feature exists in the DataFrame
        plt.subplot(3, 5, i + 1) # Adjust subplot grid based on number of features
        sns.boxplot(x='class', y=feature, data=df_eda)
        plt.title(f'Trend of {feature} by Class')
plt.tight_layout()
plt.show()

print("\nObservations on voice feature trends contributing to Parkinson's disease:")
print("- Higher jitter (locPctJitter, locAbsJitter, rapJitter) and shimmer (locShimmer, locDbShimmer, apq*Shimmer) values are often associated with Parkinson's disease, indicating voice instability. [INDEX 0]")
print("- Lower harmonicity (meanHarmToNoiseHarmonicity) and higher noise-to-harmonic ratio (meanNoiseToHarmHarmonicity) can also be indicators, suggesting a more 'noisy' voice. [INDEX 0]")
print("- Features like PPE, DFA, and RPDE often show altered values in individuals with Parkinson's, reflecting changes in vocal fold vibration and nonlinear dynamics of speech. [INDEX 0]")
print("These trends suggest that various perturbations in voice production can serve as markers for the disease.")

# **Explanation for Voice Feature Trends:**
# *   This section specifically targets voice-related features, which are a strong focus of your project.
# *   Boxplots are again used to compare the distribution of these features between the two classes (Parkinson's vs. Healthy).
# *   The observations highlight common vocal characteristics associated with Parkinson's disease, such as increased jitter and shimmer (measures of voice instability) and altered harmonicity. This qualitative analysis helps in understanding the predictive power of these features.

### **Step 4: Model Training and Selection**


# 1. Logistic Regression
log_reg = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
log_reg.fit(X_train, y_train)
print("Logistic Regression model trained.")

# 2. Random Forest
# n_estimators: number of trees in the forest
# class_weight: 'balanced' can help with remaining slight imbalances or for robust models
# max_depth: limits the depth of the trees to prevent overfitting
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_clf.fit(X_train, y_train)
print("Random Forest model trained.")

# 3. Support Vector Machine (SVM)
# C: regularization parameter (strength of the regularization is inversely proportional to C)
# gamma: kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var())
svm_clf = SVC(random_state=42, probability=True) # probability=True to get ROC_AUC scores
svm_clf.fit(X_train, y_train)
print("Support Vector Machine (SVM) model trained.")

# 4. XGBoost (Extreme Gradient Boosting)
# n_estimators: number of boosting rounds/trees
# use_label_encoder=False and eval_metric for suppressing warnings
# scale_pos_weight: useful for imbalanced datasets (sum(negative instances) / sum(positive instances))
# if not using SMOTE, you can calculate this value as: scale_pos_weight = y_train.value_counts() / y_train.value_counts()
xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
print("XGBoost model trained.")

# Store models in a dictionary for easy iteration
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
    "Support Vector Machine": svm_clf,
    "XGBoost": xgb_clf
}
print("\nAll models trained and ready for evaluation.")


# Explanation for Model Training:**
# Each model is initialized with `random_state=42` for reproducibility.
# Logistic Regression:** A simple yet effective linear model for binary classification. `solver='liblinear'` is a good default for smaller datasets.
# Random Forest:** An ensemble method that builds multiple decision trees and merges their predictions. `n_estimators` controls the number of trees, and `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequencies, which can be helpful if some imbalance persists or to improve robustness.
# support Vector Machine (SVM):** A powerful model that finds the optimal hyperplane to separate classes. `probability=True` is enabled to allow calculation of ROC-AUC scores, which require probability estimates.
# XGBoost:** A highly efficient and flexible gradient boosting framework. `n_estimators` sets the number of boosting rounds. `use_label_encoder=False` and `eval_metric='logloss'` are used to avoid deprecation warnings.

# Evaluate and compare models
performance_metrics = {}

print("Model Performance Comparison:\n")
for name, model in models.items():
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    performance_metrics[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * (len(name) + 12))

# Identify the best model based on F1-Score (often a good metric for imbalanced classification)
best_model_name = ""
best_f1_score = -1
for name, metrics in performance_metrics.items():
    if metrics["F1-Score"] > best_f1_score:
        best_f1_score = metrics["F1-Score"]
        best_model_name = name

print(f"\nBest model based on F1-Score: {best_model_name} (F1-Score: {best_f1_score:.4f})")
best_model = models[best_model_name]

# **Explanation for Model Comparison:**
# *   Each trained model is used to make predictions on the unseen `X_test` data.
# *   **Accuracy:** Overall correctness of the model.
# *   **Precision:** Proportion of true positive predictions among all positive predictions (minimizes false positives). Important when the cost of a false positive is high.
# *   **Recall (Sensitivity):** Proportion of true positive predictions among all actual positive instances (minimizes false negatives). Important when the cost of a false negative is high (e.g., missing a disease diagnosis).
# *   **F1-Score:** The harmonic mean of precision and recall. It's particularly useful when dealing with imbalanced classes, as it balances both precision and recall.
# *   The model with the highest F1-score is chosen as the "best model" in this comparison, as it indicates a good balance between identifying true positives and avoiding false positives/negatives.

### **Step 5: Model Evaluation and Prediction**
# Use ROC-AUC curves to assess model performance.


# Plot ROC-AUC curves for all models
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, "predict_proba"): # Check if model has predict_proba (needed for ROC curve)
        y_prob = model.predict_proba(X_test)[:, 1] # Probability of the positive class (class 1)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    else:
        print(f"Model '{name}' does not support predict_proba, skipping ROC curve.")

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)') # Random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# **Explanation for ROC-AUC Curves:**
# *   The ROC curve (Receiver Operating Characteristic) plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.
# *   The AUC (Area Under the Curve) measures the entire 2D area underneath the entire ROC curve. It provides an aggregate measure of performance across all possible classification thresholds.
# *   An AUC of 1.0 represents a perfect classifier, while an AUC of 0.5 indicates a classifier no better than random guessing. Higher AUC values generally mean a better model.


# #### Test the model with new patient data to predict Parkinson's disease.

print("\n--- Testing the Best Model with New (Simulated) Patient Data ---")

# Best model identified earlier
print(f"Using the best model: {best_model_name}")

# Simulate new patient data (replace with actual new data if available)
# This example uses a random sample from the original scaled features and modifies the 'gender'
# for demonstration. In a real scenario, you would have new, unlabelled patient data.
# The new data should have the same features and be preprocessed (scaled) in the same way.

# Take a random sample from X (scaled features) and ensure it's a DataFrame
# Taking one row from X_test just to demonstrate the structure, then modifying it.
simulated_new_patient_data_scaled = X_test.sample(1, random_state=np.random.randint(0, 1000)).copy()



# Example modification: Make it seem like a higher risk profile
# (e.g., higher jitter/shimmer - which might increase the probability of Parkinson's)
if 'PPE' in simulated_new_patient_data_scaled.columns:
    simulated_new_patient_data_scaled['PPE'] *= 1.2 # Increase PPE slightly

# ... (rest of the prediction code)
print("\nSimulated New Patient Data (scaled features):")
print(simulated_new_patient_data_scaled)

# Make a prediction using the best model
new_patient_prediction = best_model.predict(simulated_new_patient_data_scaled)
new_patient_prediction_proba = best_model.predict_proba(simulated_new_patient_data_scaled)[:, 1]

print(f"\nPrediction for the simulated new patient:")
if new_patient_prediction[0] == 1:
    print(f"The model predicts Parkinson's Disease (Probability: {new_patient_prediction_proba[0]:.4f}).")
else:
    print(f"The model predicts No Parkinson's Disease (Probability: {new_patient_prediction_proba[0]:.4f}).")


# **Explanation for New Patient Prediction:**
# *   This section simulates how to use the trained best model to predict on unseen, new patient data.
# *   It's critical that any new data is preprocessed using the *same* scaling (and other transformations like one-hot encoding if applicable) that was applied to the training data. This means using the *fitted* `scaler` object from the preprocessing step (`scaler.transform()`, not `fit_transform()`).
# *   `best_model.predict()` gives the class prediction (0 or 1).
# *   `best_model.predict_proba()` gives the probability of belonging to each class. The probability of class 1 (Parkinson's) is usually more informative than just the binary prediction.


# ### Conclusion:


print("\n--- Project Summary ---")
print("This project successfully developed and evaluated multiple machine learning models to predict Parkinson's disease based on health metrics and voice features.")
print(f"The best performing model in this analysis, based on F1-Score, was the {best_model_name} model.")
