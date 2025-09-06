from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Add # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

def build_cnn_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    conv1 = Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    conv1 = BatchNormalization()(conv1)
    x = Add()([x, conv1])
    x = Dropout(0.2)(x)

    x = Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
