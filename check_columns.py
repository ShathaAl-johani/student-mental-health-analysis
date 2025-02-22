import pandas as pd

# قراءة ملف CSV
df = pd.read_csv('models_comparison_results.csv')  # استبدل باسم ملفك الفعلي

# عرض أسماء الأعمدة
print("models_comparison_results.csv names", df.columns.tolist())