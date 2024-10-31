import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, Masking
from tensorflow.keras.layers import Dense, Dropout

# Define gestures and their corresponding labels
GESTURES = ['thumbs_up', 'thumbs_down', 'palm_left', 'palm_right', 
            'palm_up', 'palm_down', 'palm_forward', 'palm_backward']

def load_data(data_dir):
    """Load and preprocess data from all gesture CSV files."""
    X, y = [], []  # Features and labels
    max_frames = 0  # Track the max sequence length

    for label, gesture in enumerate(GESTURES):
        # Load the CSV file for each gesture
        file_path = os.path.join(data_dir, f"{gesture}_combined.csv")
        data = pd.read_csv(file_path, header=None)

        # Group data by batch number (first column)
        for batch_num in data[0].unique():
            batch_data = data[data[0] == batch_num]

            # Extract 64 zone distances and reshape to (frames, 8, 8, 1)
            zone_distances = batch_data.iloc[:, 2:].values
            num_frames = len(zone_distances)
            reshaped = zone_distances.reshape(num_frames, 8, 8, 1)

            X.append(reshaped)
            y.append(label)

            # Update max_frames if the current sequence is longer
            max_frames = max(max_frames, num_frames)

    # Pad all sequences to the max length
    X = pad_sequences(X, maxlen=max_frames, padding='post', dtype='float32')

    # Normalize the data to the 0-1 range
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, 64)).reshape(X.shape)

    y = np.array(y)  # Convert labels to numpy array
    return X, y, max_frames

# Load the data
data_dir = 'D:\\Repository\\CNN\\Processed Data\\'  # Directory containing gesture CSV files
X, y, max_frames = load_data(data_dir)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Build the 3D CNN model
model = Sequential([
    # Masking layer to ignore padded frames
    Masking(mask_value=0.0, input_shape=(max_frames, 8, 8, 1)),

    # First Conv3D + MaxPooling3D layer
    Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),

    # Second Conv3D + MaxPooling3D layer
    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),

    # Use GlobalMaxPooling3D to handle variable-length sequences
    GlobalMaxPooling3D(),

    # Fully connected layers
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(len(GESTURES), activation='softmax')  # Output layer for 8 gestures
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model for future use
model.save('gesture_3dcnn_model_normal_mask_globalMaxPool_maskPadding_20epochs.h5')
print("Model saved as 'gesture_3dcnn_model_normal_mask_globalMaxPool_maskPadding_20epochs.h5'")

# Load the model and make predictions
loaded_model = load_model('gesture_3dcnn_model_normal_mask_globalMaxPool_maskPadding_20epochs.h5')

# Predict gesture for a new sample from the test set
new_sample = X_test[0].reshape(1, *X_test[0].shape)  # Add batch dimension
prediction = loaded_model.predict(new_sample)
predicted_label = np.argmax(prediction)

print(f"Predicted Gesture: {GESTURES[predicted_label]}")
