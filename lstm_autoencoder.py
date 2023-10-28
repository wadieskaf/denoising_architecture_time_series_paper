import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from typing import List, Tuple, Dict, Any
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

tf.random.set_seed(42)


def build_lstm_autoencoder(input_shape: Tuple, layers: List, activation: str = 'tanh',
                           dropout: float = 0.2, loss_function: str = 'mae',
                           optimizer: str = 'adam') -> Model:
    """
    Builds and compiles a LSTM autoencoder.
    """
    # Build the model.
    input_layer = Input(shape=(input_shape[1], input_shape[2]))
    # encoder
    if len(layers) == 1:
        x = LSTM(layers[0], activation=activation, return_sequences=False)(input_layer)
        x = Dropout(dropout)(x)
    elif len(layers) > 1:
        x = LSTM(layers[0], activation=activation, return_sequences=True)(input_layer)
        for layer in layers[1:-1]:
            x = LSTM(layer, activation=activation, return_sequences=True)(x)
            x = Dropout(dropout)(x)
        x = LSTM(layers[-1], activation=activation, return_sequences=False)(x)
        x = Dropout(dropout)(x)

    x = RepeatVector(input_shape[1])(x)

    # decoder
    for layer in layers[::-1]:
        x = LSTM(layer, activation=activation, return_sequences=True)(x)
        x = Dropout(dropout)(x)

    x = TimeDistributed(Dense(input_shape[2], activation='linear'))(x)

    # return the model
    model = Model(input_layer, x)
    model.compile(loss=loss_function, optimizer=optimizer)

    return model


def train_lstm_autoencoder(model: Model, train_gen, valid_gen=None, callbacks: List = None, epochs: int = 50,
                           verbose: int = 1) -> Tuple[Model, Any]:
    """
    Trains a LSTM autoencoder.
    """

    history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, verbose=verbose, callbacks=callbacks)

    return model, history


def evaluate_lstm_autoencoder(model: Model, test_gen, quantile: int = .99) -> Tuple[Dict, np.ndarray]:
    """
    Evaluates a LSTM autoencoder.
    """
    y_real, reconstruction_errors_list = list(), list()
    for samples, labels in test_gen:
        y_real.extend(labels.ravel().tolist())
        preds = model.predict(samples)
        reconstruction_errors = np.abs(preds - samples)
        reconstruction_errors_list.extend(reconstruction_errors.ravel().tolist())

    threshold = np.quantile(reconstruction_errors_list, quantile)
    y_pred = [1 if e > threshold else 0 for e in reconstruction_errors_list]
    # calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_real, y_pred),
        'recall': recall_score(y_real, y_pred),
        'precision': precision_score(y_real, y_pred),
        'f1': f1_score(y_real, y_pred)
    }
    # calculate confusion matrix
    cm = confusion_matrix(y_real, y_pred)

    return metrics, cm
