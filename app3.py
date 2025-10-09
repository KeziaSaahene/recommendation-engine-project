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

# Load CNN model
model = tf.keras.models.load_model("cnn_model.keras")

# Load new tokenizer
with open("tokenizer2.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# Load label classes (NumPy format)
import numpy as np
label_classes = np.load("label_classes.npy", allow_pickle=True)

# -------------------------
# Streamlit app
# -------------------------
st.title("📊 Text Classification App")
st.write("Enter text and get a prediction from the CNN model!")

# Input box for user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip():
        # Convert text to sequences
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)  # same maxlen as used during training

        # Make prediction
        prediction = model.predict(padded)
        pred_label = np.argmax(prediction, axis=1)
        result = label_classes[pred_label][0]

        st.success(f"✅ Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some text!")
