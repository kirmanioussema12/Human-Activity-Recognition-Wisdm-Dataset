import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw WISDM dataset (assuming you have cleaned it into a structured format)
df = pd.read_csv(r"C:\Users\MSI\Desktop\Mitacs Project\Human Activity Recognition\HAR-WISDM\Data_WISDM\WISDM.csv")


# Visualize the distribution of activity classes
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=df, palette='Set2')
plt.title("Distribution of Activities")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
