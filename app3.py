import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle

# -------------------------
# Sidebar: choose model
# -------------------------
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["CNN Model (Task 1)", "Autoencoder CNN (Task 2)"]
)

# -------------------------
# Load the selected model
# -------------------------
if model_choice == "CNN Model (Task 1)":
    model = tf.keras.models.load_model("cnn_model.keras")
else:
    model = tf.keras.models.load_model("cnn_ae.keras")

# -------------------------
# Load tokenizer
# -------------------------
with open("tokenizer2.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# -------------------------
# Load label classes (for Task 1 CNN)
# -------------------------
label_classes = np.load("label_classes.npy", allow_pickle=True)

# -------------------------
# Streamlit UI
# -------------------------
st.title("📊 Text Classification App")
st.write("Enter text below and get predictions from your selected model!")

user_input = st.text_area("✏️ Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        if model_choice == "CNN Model (Task 1)":
            # Tokenize input
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=50)  # match training max_len
            prediction = model.predict(padded)
            pred_label = np.argmax(prediction, axis=1)
            result = label_classes[pred_label][0]  # decode
            st.success(f"✅ Prediction: {result}")
        else:
            # Autoencoder output (Task 2)
            # Scale input features using scaler
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            
            # Example: convert input text to numeric features (you need to define feature extraction)
            # For now, just a placeholder
            st.warning("⚠️ Autoencoder CNN currently requires numeric user features, not text input.")
    else:
        st.warning("⚠️ Please enter some text to classify.")
