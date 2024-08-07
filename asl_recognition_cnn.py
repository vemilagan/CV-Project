import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import os

# Function to download the model from OneDrive
def download_model(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(filename, 'wb') as f:
            f.write(response.content)
        st.info("Model downloaded successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the model: {e}")
        return None

# URL of the model file on OneDrive
model_url = 'https://api.onedrive.com/v1.0/shares/u!An8rvMR27KPcioRoeRuLwtzX8ZJKHQ/root/content'
model_filename = 'asl_model_cnn.h5'

# Download the model if it doesn't exist
if not os.path.exists(model_filename):
    st.info("Downloading model...")
    download_model(model_url, model_filename)

# Load the model
try:
    model = load_model(model_filename)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up Streamlit
st.title("ASL Alphabet Recognition")
st.subheader("AIG 210 - Computer Vision - Group 4 Final Project")

# Define the class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Function to draw hand landmarks with red and green points
def draw_hand_landmarks(image, hand_landmarks):
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    mp_drawing.draw_landmarks(
        image, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    )

# Function to predict the ASL letter from an image
def predict_asl(image):
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0
    input_data = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Main function
def main():
    st.subheader("")

    # Allow image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    draw_hand_landmarks(image, hand_landmarks)

        if model is not None:
            predicted_class_name = predict_asl(image)
            st.markdown(f"<h2 style='color: red; font-size: 24px;'><strong>Predicted: {predicted_class_name}</strong></h2>", unsafe_allow_html=True)
        else:
            st.error("Model is not loaded. Unable to make predictions.")

        st.image(image, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
