import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# -------------------------
# Load artifacts
# -------------------------

st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["CNN Model (Task 1)", "Autoencoder CNN (Task 2)"]
)

# Load the selected model
if model_choice == "CNN Model (Task 1)":
    model = tf.keras.models.load_model("cnn_model.keras")
else:
    model = tf.keras.models.load_model("cnn_ae.keras")

# Load tokenizer (new version)
with open("tokenizer2.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# Load label encoder
with open("labelencoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------
# Streamlit app UI
# -------------------------
st.title("📊 Text Classification App")
st.write("Enter text below and get predictions from your selected model!")

# Input text box
user_input = st.text_area("✏️ Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        # Convert text to padded sequence
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)  # adjust to your training maxlen

        # Predict
        prediction = model.predict(padded)
        pred_label = np.argmax(prediction, axis=1)
        result = label_encoder.inverse_transform(pred_label)

        st.success(f"✅ Prediction: {result[0]}")
    else:
        st.warning("⚠️ Please enter some text to classify.")
