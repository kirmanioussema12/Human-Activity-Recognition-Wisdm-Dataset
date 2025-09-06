import numpy as np
from sklearn.preprocessing  import StandardScaler,LabelEncoder
def preprocess_features(data,label_column='class'):
    X=data.drop(columns=[label_column],axis=1)
    y=data[label_column]
    label_encoder=LabelEncoder()
    y_encoded=label_encoder.fit_transform(y)
    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    X_reshaped=X_scaled.reshape(-1,10,3)    #the reshape is based on the assumption that each sample has 10 time steps and 3 features and the -1 indicates the number of samples automatically calculated
    y_reshaped=y_encoded.reshape(-1)
    return X_reshaped, y_reshaped, label_encoder