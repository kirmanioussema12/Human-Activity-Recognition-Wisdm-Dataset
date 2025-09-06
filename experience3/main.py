from data_loader import load_dataset
from preprocessing import preprocess_features
from augmentation import add_jitter,augment_specific_classes
from train import train_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess
data = load_dataset("C:/Users/MSI/Desktop/Mitacs Project/Human Activity Recognition/HAR-WISDM/Data_WISDM/WISDM_cleaned.csv")
X, y, label_encoder = preprocess_features(data)
# Visualize class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=label_encoder.inverse_transform(y))
plt.title('Final Augmented Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Train the model
train_model(X, y, label_encoder)
