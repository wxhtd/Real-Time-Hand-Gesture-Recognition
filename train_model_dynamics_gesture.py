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
            
            # Reshape for 3D CNN (frames, 8, 8, 1)
            num_frames = len(zone_distances)
            reshaped = zone_distances.reshape(num_frames, 8, 8, 1)
            X.append(reshaped)
            y.append(label)
            max_frames_found = max(max_frames_found, num_frames)

    # Pad dynamic sequences to the max length
    if max_frames is None:
        max_frames = max_frames_found
    X = pad_sequences(X, maxlen=max_frames, padding='post', dtype='float32')

    # Normalize the data to the 0-1 range
    X = MinMaxScaler().fit_transform(X.reshape(-1, 64)).reshape(X.shape)

    y = np.array(y)
    return X, y, max_frames


# Load the data
with open("config.json", "r") as config_file:
    config = json.load(config_file)

root_path = config["test_data_root_directory"]
dynamic_gestures = [gesture.strip() for gesture in config["dynamic_gestures"].split(",")]

X_dynamic, y_dynamic, max_frames = load_data(root_path, dynamic_gestures)

# Split dynamic and static datasets into train and test sets
X_train_dyn, X_test_dyn, y_train_dyn, y_test_dyn = train_test_split(X_dynamic, y_dynamic, test_size=0.2, random_state=42)

print(f"Dynamic Training data shape: {X_train_dyn.shape}")

# Build the 3D CNN model
model_3d = Sequential([
    # # Masking layer to ignore padded frames
    # Masking(mask_value=0.0, input_shape=(max_frames, 8, 8, 1)),

    # First Conv3D + MaxPooling3D layer
    Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(max_frames, 8, 8, 1)),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),

    # Second Conv3D + MaxPooling3D layer
    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),

    # Use GlobalMaxPooling3D to handle variable-length sequences
    GlobalMaxPooling3D(),

    # Fully connected layers
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(len(dynamic_gestures), activation='softmax')  # Output layer for 8 gestures
])

# Compile the model
model_3d.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model_3d.summary()

# Train the model
model_3d.fit(X_train_dyn, y_train_dyn, epochs=20, batch_size=8, validation_data=(X_test_dyn, y_test_dyn))

# Evaluate the model on the test set
test_loss, test_accuracy = model_3d.evaluate(X_test_dyn, y_test_dyn)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model for future use
model_3d.save('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs.h5')
print("Model saved as 'gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs.h5'")

# Load the model and make predictions
loaded_model_3d = load_model('gesture_3dcnn_normal_globalMaxPool_maskPadding_20epochs.h5')

# Example prediction for dynamic gesture
sample_dynamic = X_test_dyn[0].reshape(1, *X_test_dyn[0].shape)
prediction_dyn = loaded_model_3d.predict(sample_dynamic)
predicted_dyn_label = np.argmax(prediction_dyn)
print(f"Predicted Dynamic Gesture: {dynamic_gestures[predicted_dyn_label]}")