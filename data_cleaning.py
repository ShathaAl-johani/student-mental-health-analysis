import pandas as pd
import numpy as np


file_path = r"C:\Users\sh1th\OneDrive - Taibah University\DV\Student Mental health.csv"
df = pd.read_csv(file_path)

print("\n Missing values before handling:")
print(df.isnull().sum())

df = df.assign(Age=df['Age'].fillna(df['Age'].mean()))

print("\n Missing values after handling:")
print(df.isnull().sum())

categorical_columns = ["Choose your gender", "What is your course?", "Your current year of Study", 
                        "Marital status", "Do you have Depression?", "Do you have Anxiety?", 
                        "Do you have Panic attack?", "Did you seek any specialist for a treatment?"]

for col in categorical_columns:
    print(f"\n Unique values in '{col}':")
    print(df[col].unique())

binary_columns = ["Do you have Depression?", "Do you have Anxiety?", 
                  "Do you have Panic attack?", "Did you seek any specialist for a treatment?"]

for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

print("\n Checking transformation of binary columns:")
print(df[binary_columns].head(5))

def convert_cgpa_to_numeric(cgpa_range):
    """Convert CGPA range (e.g., '3.00 - 3.49') to a single numeric value (mean of range)."""
    try:
        if isinstance(cgpa_range, str) and " - " in cgpa_range:
            min_val, max_val = cgpa_range.split(" - ")
            return (float(min_val) + float(max_val)) / 2  # Calculate the mean
        return np.nan  
    except:
        return np.nan  

df["What is your CGPA?"] = df["What is your CGPA?"].apply(convert_cgpa_to_numeric)

print("\n Summary statistics of 'CGPA' after conversion:")
print(df["What is your CGPA?"].describe())