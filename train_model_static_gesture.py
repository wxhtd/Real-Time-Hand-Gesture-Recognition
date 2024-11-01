import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, Dense, Dropout, Masking, Conv2D, MaxPooling2D, Flatten

# Load Data Function
def load_data(data_dir, gestures, max_frames=None):
    X, y = [], []  # Features and labels
    max_frames_found = 0  # Track max sequence length for dynamic gestures

    for label, gesture in enumerate(gestures):
        # Load CSV file for each gesture
        file_path = os.path.join(data_dir, f"{gesture}_combined.csv")
        data = pd.read_csv(file_path, header=None)

        # Group data by batch number (first column)
        for batch_num in data[0].unique():
            batch_data = data[data[0] == batch_num]
            
            # Extract the 64 zone distances and reshape
            zone_distances = batch_data.iloc[:, 2:].values  # Ignore batch & time columns
            
            # Method to select the frame that best represents the static gesture
            best_frame = find_best_frame_for_static_gesture(zone_distances)
                
            # Reshape for 2D CNN (8x8 grid)
            X.append(best_frame.reshape(8, 8, 1))
            y.append(label)

    # Normalize the data to the 0-1 range
    X = MinMaxScaler().fit_transform(np.array(X).reshape(-1, 64)).reshape(len(X), 8, 8, 1)
    y = np.array(y)
    return X, y, max_frames

def find_best_frame_for_static_gesture(zone_distances):
    """
    Selects the frame that best represents the static gesture.
    Criteria: The frame with the highest cumulative activation across all zones.
    """

    # Calculate the sum of activations (or other criteria) for each frame
    activation_sums = zone_distances.sum(axis=1)  # Sum across all zones per frame

    # Find the index of the frame with the highest activation
    best_frame_index = np.argmax(activation_sums)
    best_frame = zone_distances[best_frame_index]

    return best_frame

# Load the data
with open("config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
static_gestures = [gesture.strip() for gesture in config["static_gestures"].split(",")]

X_static, y_static, max_frames = load_data(root_path, static_gestures)

# Split dynamic and static datasets into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_static, y_static, test_size=0.2, random_state=42)

print(f"Static Training data shape: {X_train.shape}")

# Static Gesture 2D CNN Model
model_2d = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)),
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
model_2d.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Save the static gesture model
model_2d.save('gesture_2dcnn_actibysum_normal_globalMaxPool_flatten_20epochs.h5')
print("2D CNN model for static gestures saved as 'gesture_2dcnn_actibysum_normal_globalMaxPool_flatten_20epochs.h5'")

# Evaluation for static gestures
test_loss_stat, test_accuracy_stat = model_2d.evaluate(X_test, y_test)
print(f"Static Gesture Model Test Accuracy: {test_accuracy_stat * 100:.2f}%")

# Load the model and make predictions
loaded_model_2d = load_model('gesture_2dcnn_actibysum_normal_globalMaxPool_flatten_20epochs.h5')

# Example prediction for dynamic gesture
sample_static = X_test[0].reshape(1, *X_test[0].shape)
prediction = loaded_model_2d.predict(sample_static)
predicted_label = np.argmax(prediction)
print(f"Predicted Static Gesture: {static_gestures[predicted_label]}")