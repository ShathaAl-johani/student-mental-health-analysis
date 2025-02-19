# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # معالجة عدم التوازن في البيانات
import xgboost as xgb  # مكتبة XGBoost

# Load the cleaned dataset
df = pd.read_csv("cleaned_student_mental_health.csv")

# Remove the timestamp column (not needed for prediction)
df = df.drop(columns=["Timestamp"])

# Convert categorical variables into numeric values using Label Encoding
label_encoders = {}
categorical_columns = ["Choose your gender", "What is your course?", "Your current year of Study", "Marital status"]

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Define features (X) and target variable (y)
X = df.drop(columns=["Do you have Depression?"])  # Features
y = df["Do you have Depression?"]  # Target variable (Depression)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize numerical features (CGPA, Age)
scaler = StandardScaler()
X_train[["Age", "What is your CGPA?"]] = scaler.fit_transform(X_train[["Age", "What is your CGPA?"]])
X_test[["Age", "What is your CGPA?"]] = scaler.transform(X_test[["Age", "What is your CGPA?"]])

# 🔹 Step 2: Train an XGBoost Model

# Initialize and train the model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# 🔹 Step 3: Make Predictions and Evaluate the Model
y_pred = xgb_model.predict(X_test)

# Print Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ XGBoost Model Accuracy: {accuracy * 100:.2f}%")

# Print Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Depression Prediction (XGBoost)")
plt.show()
