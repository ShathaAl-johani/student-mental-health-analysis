import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# أداء النماذج المختلفة
models_performance = {
    "Model": ["Random Forest", "Optimized Random Forest", "XGBoost", "Optimized XGBoost"],
    "Accuracy": [0.74, 0.81, 0.70, 0.71],
    "Precision": [0.74, 0.81, 0.70, 0.71],
    "Recall": [0.74, 0.75, 0.70, 0.71],
    "F1-Score": [0.74, 0.77, 0.70, 0.70]
}

# تحويلها إلى DataFrame
df_performance = pd.DataFrame(models_performance)

# حفظ النتائج إلى ملف CSV
df_performance.to_csv("models_comparison_results.csv", index=False)

# رسم مقارنة الأداء بين النماذج
plt.figure(figsize=(10, 6))
bars = np.arange(len(df_performance["Model"]))
width = 0.2

plt.bar(bars, df_performance["Accuracy"], width=width, label="Accuracy", color='b')
plt.bar(bars + width, df_performance["Precision"], width=width, label="Precision", color='g')
plt.bar(bars + 2 * width, df_performance["Recall"], width=width, label="Recall", color='r')
plt.bar(bars + 3 * width, df_performance["F1-Score"], width=width, label="F1-Score", color='y')

plt.xticks(bars + width, df_performance["Model"], rotation=15)
plt.ylabel("Score")
plt.title("Comparison of Model Performance")
plt.legend()
plt.show()
