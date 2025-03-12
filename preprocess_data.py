import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import json
import os

# Load intents from data.json
data_file = "data.json"
if not os.path.exists(data_file):
    print(f"❌ Error: {data_file} not found!")
    exit()

with open(data_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare data
texts = []
labels = []

for intent in data.get("intents", []):
    for pattern in intent.get("patterns", []):
        texts.append(pattern)
        labels.append(intent.get("tag", "unknown"))

# Check if there is any data
if not texts or not labels:
    print("❌ Error: No data found in data.json!")
    exit()

# Tokenize texts
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
X_train = tokenizer.texts_to_sequences(texts)

# Convert sequences to a numpy array
try:
    X_train = np.array(X_train, dtype=object)  # Avoids shape errors
except ValueError as e:
    print(f"❌ Error converting X_train to numpy array: {e}")
    exit()

# Convert labels to one-hot encoding
label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}
y_train = np.array([label_to_index[label] for label in labels])
y_train = to_categorical(y_train, num_classes=len(label_to_index))

# Save preprocessed data
pickle.dump(X_train, open("X_train.pkl", "wb"))
pickle.dump(y_train, open("y_train.pkl", "wb"))
pickle.dump(texts, open("texts.pkl", "wb"))  # Save raw texts instead of word_index
pickle.dump(label_to_index, open("labels.pkl", "wb"))

# Explicitly check if texts.pkl is created
if os.path.exists("texts.pkl"):
    print("✅ texts.pkl saved successfully! (Contains raw text data)")
else:
    print("❌ ERROR: texts.pkl NOT saved!")

print("✅ Preprocessed data saved successfully!")
