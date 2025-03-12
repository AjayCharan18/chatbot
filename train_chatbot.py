import os
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
def load_pickle(filename):
    """Loads a pickle file and handles errors."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: {filename} not found!")
    with open(filename, 'rb') as f:
        return pickle.load(f)

try:
    X_train = np.array(load_pickle('X_train.pkl'))
    y_train = np.array(load_pickle('y_train.pkl'))
except Exception as e:
    raise ValueError(f"Error loading training data: {e}")

# Ensure X_train has correct shape
if X_train.ndim != 2:
    raise ValueError(f"Error: X_train must be 2D, but got shape {X_train.shape}")

# Define the model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model in the new format
model.save('model.keras')  # Recommended format
print("✅ Model trained and saved as 'model.keras'")
