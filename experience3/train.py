import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from model import build_cnn_lstm_model
from augmentation import augment_specific_classes, add_jitter  # Ensure add_jitter is imported


def train_model(X, y, label_encoder):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    f1_scores = []
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))

    # Define how many new samples to generate for specific classes (based on label names)
    class_sample_map = {
        label_encoder.transform([cls])[0]: n
        for cls, n in [
            ('Jogging',10),
            ('Walking',10),
            ('Sitting', 10),
            ('Standing', 10),
            ('Downstairs',20),
            ('Upstairs', 20)
        ]
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n📚 Training Fold {fold}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Augment only the training set
        X_aug, y_aug = augment_specific_classes(X_train, y_train, class_sample_map, add_jitter)

        # Combine original and augmented training data
        X_train_combined = np.concatenate([X_train, X_aug], axis=0)
        y_train_combined = np.concatenate([y_train, y_aug], axis=0)

        model = build_cnn_lstm_model(input_shape=(10, 3), num_classes=len(np.unique(y)))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]

        model.fit(
            X_train_combined, y_train_combined,
            epochs=100,
            batch_size=64,
            class_weight=class_weight_dict,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        print(f"\n🧾 Fold {fold} Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))

        report = classification_report(y_val, y_pred_classes, output_dict=True)
        f1_scores.append(report['macro avg']['f1-score'])

    print(f"\n✅ Average Macro F1-Score across {skf.n_splits} folds: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
