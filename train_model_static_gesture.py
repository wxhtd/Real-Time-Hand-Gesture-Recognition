import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load Data Function
def load_data(data_dir, gestures):
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

            # Extract the 64 distance and 64 signal values
            zone_distances = batch_data.iloc[:, 2:66].values  # Columns for distance
            zone_signals = batch_data.iloc[:, 66:130].values  # Columns for signal

            # Find the best frame for static gestures
            best_frame_distance, best_frame_signal = find_best_frame_for_static_gesture(zone_distances, zone_signals)

            # Reshape to 2D array for MinMaxScaler and normalize distances and signals separately
            normalized_distances = distance_scaler.fit_transform(best_frame_distance.reshape(-1, 1)).reshape(8, 8, 1)
            normalized_signals = signal_scaler.fit_transform(best_frame_signal.reshape(-1, 1)).reshape(8, 8, 1)

            # Combine the best frames of distance and signal into two channels (8, 8, 2)
            best_frame = np.concatenate([normalized_distances, normalized_signals], axis=-1)
            
            X.append(best_frame)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y

def find_best_frame_for_static_gesture(zone_distances, zone_signals):
    """
    Selects the frame that best represents the static gesture.
    Criteria: The frame with the highest cumulative activation across all zones.
    """
    # Calculate the sum of activations for each frame in distance and signal channels
    activation_sums_distance = zone_distances.sum(axis=1)
    activation_sums_signal = zone_signals.sum(axis=1)

    # Find the index of the frame with the highest combined activation
    best_frame_index = np.argmax(activation_sums_distance + activation_sums_signal)
    
    # Return the best frame for both distance and signal
    best_frame_distance = zone_distances[best_frame_index]
    best_frame_signal = zone_signals[best_frame_index]
    return best_frame_distance, best_frame_signal

# Load the data
with open(".\\3D-CNN(1)\\config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
static_gestures = [gesture.strip() for gesture in config["static_gestures"].split(",")]

X_static, y_static = load_data(root_path, static_gestures)

# Split static dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_static, y_static, test_size=0.2, random_state=42)

# Static Gesture 2D CNN Model
model_2d = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 2)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(static_gestures), activation='softmax')
])

# Compile and train the 2D CNN model for static gestures
model_2d.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Print the model summary to verify the architecture
model_2d.summary()
model_2d.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Save the static gesture model
model_2d.save('gesture_2dcnn_with_signal_flatten_20epochs_with_signal_2.h5')
print("2D CNN model for static gestures saved as 'gesture_2dcnn_with_signal_flatten_20epochs_with_signal_2.h5'")

# Evaluation for static gestures
test_loss_stat, test_accuracy_stat = model_2d.evaluate(X_test, y_test)
print(f"Static Gesture Model Test Accuracy: {test_accuracy_stat * 100:.2f}%")

# Calculate accuracy for each gesture
gesture_accuracies = {}
for gesture_idx, gesture in enumerate(static_gestures):
    # Select test samples for this gesture
    gesture_X_test = X_test[y_test == gesture_idx]
    gesture_y_test = y_test[y_test == gesture_idx]
    
    if len(gesture_X_test) > 0:
        # Evaluate the model on the gesture-specific test set
        gesture_loss, gesture_accuracy = model_2d.evaluate(gesture_X_test, gesture_y_test, verbose=0)
        gesture_accuracies[gesture] = gesture_accuracy * 100
        print(f"Accuracy for {gesture}: {gesture_accuracy * 100:.2f}%")

# Load the model and make predictions
loaded_model_2d = load_model('gesture_2dcnn_with_signal_flatten_20epochs_with_signal_2.h5')

# Example prediction for a static gesture
sample_static = X_test[0].reshape(1, *X_test[0].shape)
prediction = loaded_model_2d.predict(sample_static)
predicted_label = np.argmax(prediction)
print(f"Predicted Static Gesture: {static_gestures[predicted_label]}")
