import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the cleaned dataset
df = pd.read_csv("cleaned_student_mental_health.csv")

# Display the first 5 rows to verify the data
print(df.head())

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="What is your course?", hue="Do you have Depression?", palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Depression Cases by Field of Study")
plt.xlabel("Field of Study")
plt.ylabel("Number of Students")
plt.legend(title="Depression")
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(df[df["Do you have Depression?"] == 1]["Age"], bins=10, color="red", alpha=0.6, label="Depressed")
sns.histplot(df[df["Do you have Depression?"] == 0]["Age"], bins=10, color="blue", alpha=0.6, label="Not Depressed")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.title("Age Distribution: Depressed vs. Not Depressed Students")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df["Do you have Depression?"], y=df["What is your CGPA?"], hue=df["Do you have Depression?"], palette="coolwarm", legend=False)
plt.xticks([0, 1], ["Not Depressed", "Depressed"])
plt.xlabel("Depression Status")
plt.ylabel("CGPA")
plt.title("Effect of Depression on CGPA")
plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="Marital status", hue="Do you have Depression?", palette="coolwarm")
plt.title("Effect of Marital Status on Depression")
plt.xlabel("Marital Status")
plt.ylabel("Number of Students")
plt.legend(title="Depression")
plt.show()

df.groupby("What is your course?")["Do you have Depression?"].mean().to_csv("depression_by_study_field.csv")
