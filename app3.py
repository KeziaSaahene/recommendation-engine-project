import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# -------------------------
# Load artifacts
# -------------------------

# Dropdown to choose model
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["CNN Model (Task 1)", "Autoencoder CNN (Task 2)"]
)

# Load selected model
if model_choice == "CNN Model (Task 1)":
    model = tf.keras.models.load_model("cnn_model.keras")
else:
    model = tf.keras.models.load_model("cnn_ae.keras")

# Load tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# Load label encoder classes (new format)
label_classes = np.load("label_classes.npy", allow_pickle=True)

# -------------------------
# Streamlit app
# -------------------------
st.title("📊 Text Classification App")
st.write("Enter text below and get predictions from your selected model!")

# Input text
user_input = st.text_area("✏️ Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        # Tokenize input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)  # match your training length

        # Make prediction
        prediction = model.predict(padded)
        pred_label = np.argmax(prediction, axis=1)
        result = label_classes[pred_label][0]

        st.success(f"✅ Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some text to classify.")
