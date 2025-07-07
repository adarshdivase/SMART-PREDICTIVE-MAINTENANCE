import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dataclasses import dataclass

# We need the data generation logic here as well
@dataclass
class SupervisedConfig:
    sequence_length: int = 100
    feature_dim: int = 10

def create_synthetic_data_for_training(config: SupervisedConfig, num_samples: int = 5000):
    """Creates features (X) and a target (y) for training."""
    time = np.linspace(0, 100, num_samples)
    features = []
    # Using a sine wave for the primary feature we want to predict
    vibration = np.sin(time * 0.5)
    features.append(vibration)

    for i in range(1, config.feature_dim):
        features.append(np.random.uniform(0, 1, num_samples) + np.sin(time * i * 0.1))

    features = np.column_stack(features).astype(np.float32)

    X, y = [], []
    # We'll train the model to predict the next 'vibration' value
    for i in range(len(features) - config.sequence_length):
        X.append(features[i:i + config.sequence_length])
        y.append(features[i + config.sequence_length, 0]) # Target is the next vibration reading

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model():
    """Defines, trains, and saves the LSTM model."""
    print("--- Starting Model Training ---")
    config = SupervisedConfig()

    print("1. Generating synthetic data...")
    X_train, y_train = create_synthetic_data_for_training(config)
    print(f"Data generated. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    print("2. Building LSTM model...")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(config.sequence_length, config.feature_dim)),
        Dense(25, activation='relu'),
        Dense(1) # Output is a single value (the predicted health score)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    print("3. Training model...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    print("4. Saving model to 'health_model.h5'...")
    model.save('health_model.h5')
    print("--- Model Training Complete! ---")

if __name__ == '__main__':
    train_model()
