# Import necessary libraries
import pandas as pd


file_path = r"C:\Users\sh1th\OneDrive - Taibah University\DV\Student Mental health.csv"  

df = pd.read_csv(file_path)

print("First 10 rows of the dataset:")
print(df.head(10))

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\n Dataset information:")
print(df.info())

print("\n Statistical summary of numerical columns:")
print(df.describe())

