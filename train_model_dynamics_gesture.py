import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, Dense, Dropout

# Parameters
max_frames = 100  # Set a maximum frame length for padding
num_zones = 64    # Number of zones per frame
input_channels = 2  # Distance and signal channels

# Load Data Function
def load_data(data_dir, gestures, max_frames=None):
    max_frames_found = 0  # Track max sequence length for dynamic gestures
    X, y = [], []  # Features and labels

    # Initialize scalers for distance and signal channels
    distance_scaler = MinMaxScaler(feature_range=(0, 1))
    signal_scaler = MinMaxScaler(feature_range=(0, 1))

    for label, gesture in enumerate(gestures):
        # Load CSV file for each gesture
        file_path = os.path.join(data_dir, f"{gesture}_combined.csv")
        data = pd.read_csv(file_path, header=None)

        # Group data by batch number (first column)
        for batch_num in data[0].unique():
            batch_data = data[data[0] == batch_num]

            group = batch_data.sort_values(by=batch_data.columns[1], ascending=True)

            # Extract distances and signals, reshape each frame to 8x8x2 (8x8 grid with 2 channels: distance, signal)
            distances = group.iloc[:, 2:2 + num_zones].values  # Distance columns
            signals = group.iloc[:, 2 + num_zones:2 + 2 * num_zones].values  # Signal columns

            # Normalize distances and signals separately
            distances = distance_scaler.fit_transform(distances)  # Normalized to [0, 1]
            signals = signal_scaler.fit_transform(signals)        # Normalized to [0, 1]
        
            distances = distances.reshape(len(distances), 8, 8, 1)
            signals = signals.reshape(len(signals), 8, 8, 1)
            # Reshape distances and signals to (num_frames, 8, 8, 1) and stack to get (num_frames, 8, 8, 2)
            # frames = np.stack([distances, signals], axis=-1)  # Shape: (num_frames, 8, 8, 2)
            frames = np.concatenate([distances, signals], axis=-1)
            # Append frames to X and the label to y
            X.append(frames)
            y.append(label)  # Assuming `test_number` serves as a label; replace if you have actual gesture labels

            max_frames_found = max(max_frames_found, len(batch_data))

    # Pad dynamic sequences to the max length
    if max_frames is None:
        max_frames = max_frames_found
    X = pad_sequences(X, maxlen=max_frames, padding='post', dtype='float32')

    y = np.array(y)
    return X, y, max_frames

# Load the data
with open(".\\3D-CNN(1)\\config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
dynamic_gestures = [gesture.strip() for gesture in config["dynamic_gestures"].split(",")]


X_dynamic, y_dynamic, max_frames = load_data(root_path, dynamic_gestures)

# Split dynamic dataset into train and test sets
X_train_dyn, X_test_dyn, y_train_dyn, y_test_dyn = train_test_split(X_dynamic, y_dynamic, test_size=0.2, random_state=42)

print(f"Dynamic Training data shape: {X_train_dyn.shape}")
print(f"Labels shape: {y_train_dyn.shape}")

# Build the 3D CNN model
model_3d = Sequential([
    # First Conv3D + MaxPooling3D layer
    Conv3D(64, (3, 3, 3), activation='relu', padding='same', input_shape=(max_frames, 8, 8, 2)),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),

    # Second Conv3D + MaxPooling3D layer
    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),

    # Use GlobalMaxPooling3D to handle variable-length sequences
    GlobalMaxPooling3D(),

    # Fully connected layers
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(len(dynamic_gestures), activation='softmax')  # Output layer for dynamic gestures
])

# Compile the model
model_3d.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model_3d.summary()

# Train the model
model_3d.fit(X_train_dyn, y_train_dyn, epochs=20, batch_size=8, validation_data=(X_test_dyn, y_test_dyn))

# Evaluate the model on the test set
test_loss, test_accuracy = model_3d.evaluate(X_test_dyn, y_test_dyn)
print(f"Overall Test Accuracy: {test_accuracy * 100:.2f}%")

# Calculate accuracy for each gesture
gesture_accuracies = {}
for gesture_idx, gesture in enumerate(dynamic_gestures):
    # Select test samples for this gesture
    gesture_X_test = X_test_dyn[y_test_dyn == gesture_idx]
    gesture_y_test = y_test_dyn[y_test_dyn == gesture_idx]
    
    if len(gesture_X_test) > 0:
        # Evaluate the model on the gesture-specific test set
        gesture_loss, gesture_accuracy = model_3d.evaluate(gesture_X_test, gesture_y_test, verbose=0)
        gesture_accuracies[gesture] = gesture_accuracy * 100
        print(f"Accuracy for {gesture}: {gesture_accuracy * 100:.2f}%")

# Save the model for future use
model_3d.save('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs_with_signal_2.h5')
print("Model saved as 'gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs_with_signal_2.h5'")

# Load the model and make predictions
loaded_model_3d = load_model('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs_with_signal_2.h5')

# Example prediction for dynamic gesture
sample_dynamic = X_test_dyn[0].reshape(1, *X_test_dyn[0].shape)
prediction_dyn = loaded_model_3d.predict(sample_dynamic)
predicted_dyn_label = np.argmax(prediction_dyn)
print(f"Predicted Dynamic Gesture: {dynamic_gestures[predicted_dyn_label]}")
