from data_loader import load_dataset
from preprocessing import preprocess_features
from augmentation import add_jitter, augment_minority_classes, augment_specific_classes
from train import train_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess
data = load_dataset("C:/Users/MSI/Desktop/Mitacs Project/Human Activity Recognition/HAR-WISDM/Data_WISDM/WISDM_cleaned.csv")
X, y, label_encoder = preprocess_features(data)

# Augment selected classes
minority_classes = [label_encoder.transform([cls])[0] for cls in ['Downstairs', 'Upstairs', 'Standing']]
X_temp, y_temp = augment_minority_classes(X, y, minority_classes, add_jitter, augment_factor=3)

class_sample_map = {
    label_encoder.transform([cls])[0]: n
    for cls, n in [('Jogging', 1200), ('Walking',1300), ('Sitting', 2300), ('Standing',1900), ('Downstairs', 500), ('Upstairs',400)]
}
X_aug, y_aug = augment_specific_classes(X_temp, y_temp, class_sample_map, add_jitter)

# Visualize class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=label_encoder.inverse_transform(y_aug))
plt.title('Final Augmented Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Train the model
train_model(X_aug, y_aug, label_encoder)
