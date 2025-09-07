# Human Activity Recognition with CNN-LSTM and Residual Connections

## 📌 Overview
This project implements a **Human Activity Recognition (HAR)** model using the **WISDM dataset**.  
The dataset consists of accelerometer and gyroscope sensor data collected from smartphones.  
The proposed model combines **Convolutional Neural Networks (CNNs)** with **Long Short-Term Memory networks (LSTMs)** and integrates **residual connections** to enhance feature extraction and improve gradient flow.

The model achieved **~94% accuracy** with augmentation applied, making it suitable for academic and engineering exploration.

---

## 🎯 Motivation
- Human activities generate sequential time-series signals from sensors.  
- Classical ML methods often fail to capture **temporal dependencies** and **complex motion patterns**.  
- This project leverages:
  - **CNNs** for local temporal feature extraction.  
  - **Residual connections** for stable training of deeper CNNs.  
  - **LSTMs** for long-term sequential modeling.  

---

## 🛠️ Data Preprocessing
- **Dataset:** [WISDM (Wireless Sensor Data Mining)](https://www.cis.fordham.edu/wisdm/dataset.php).  
- **Challenge:** Class imbalance in activity distribution.  
- **Solution:**  
  - Applied **data augmentation** using the **jittering technique** (adding Gaussian noise).  
  - Augmentation performed on both training and test sets to increase variability and improve robustness.  

---

## 🏗️ Model Architecture
The CNN-LSTM model with residual connections is structured as follows:

1. **Input Layer**  
   - Time-series windows from accelerometer & gyroscope data.  

2. **CNN Block 1**  
   - Conv1D (64 filters, kernel size 5) + BatchNorm + Dropout.  
   - Residual Conv1D (64 filters) added back to the block output.  

3. **CNN Block 2**  
   - Conv1D (128 filters) + BatchNorm + Dropout.  

4. **LSTM Layer**  
   - LSTM (128 units) to capture long-term dependencies.  

5. **Dense Layers**  
   - Dense (128 units, ReLU) + Dropout.  
   - Final Dense layer with **Softmax activation** for multi-class classification.  

---

## ⚙️ Training Strategy
- **Cross-validation:** 5-fold **Stratified K-Fold**.  
- **Optimizer:** Adam.  
- **Loss Function:** Sparse categorical cross-entropy.  
- **Regularization:**  
  - L2 weight decay.  
  - Dropout.  
  - Early stopping & ReduceLROnPlateau.  
- **Evaluation Metrics:**  
  - Accuracy.  
  - Macro F1-score.  
  - Detailed classification reports per fold.  

---
 

## 📊 Results
- Achieved an average **accuracy of ~94%** on the WISDM dataset.  
- Residual connections and the CNN-LSTM hybrid improved performance over standalone CNN or LSTM baselines.  
- Macro F1-scores across folds showed **balanced classification across both majority and minority activity classes**.  

---
## ⚠️ Notes
- This work was conducted for **academic and experimental purposes**.  
- Results are not intended as an official benchmark submission.  
- Data augmentation (jittering) was applied to improve robustness but may not reflect standardized evaluation protocols.
  
## 📖 References
- WISDM Dataset: [https://www.cis.fordham.edu/wisdm/dataset.php](https://www.cis.fordham.edu/wisdm/dataset.php)  
- Residual Networks: K. He, et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.  
- CNN-LSTM applications in HAR: Ordóñez & Roggen. *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. Sensors 2016.  
## Requirements

This project requires Python 3.8+ and the following Python libraries:

- [TensorFlow](https://www.tensorflow.org/) (for building and training the CNN-BiLSTM model)
- [Keras](https://keras.io/) (high-level API for neural networks, integrated with TensorFlow)
- [NumPy](https://numpy.org/) (numerical computations)
- [Pandas](https://pandas.pydata.org/) (data handling and preprocessing)
- [Matplotlib](https://matplotlib.org/) (visualization, e.g., loss curves, PCA plots)
- [Scikit-learn](https://scikit-learn.org/) (PCA, preprocessing, metrics)
- [Seaborn](https://seaborn.pydata.org/) (optional, for enhanced plots)
- [tqdm](https://tqdm.github.io/) (optional, progress bars for training loops)

## Installation (via pip)

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn seaborn tqdm




---



---
