import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained Keras model
try:
    model = tf.keras.models.load_model('asl_model_mobilenetv2.h5')  # Updated file name
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Ensure model is set to None if loading fails

# Define the labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space'
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up Streamlit
st.title("ASL Alphabet Recognition with Hand Landmarks")

# Function to draw hand landmarks with red and green points
def draw_hand_landmarks(image, hand_landmarks):
    # Draw landmarks with red points
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red points

    # Draw connections with green lines
    mp_drawing.draw_landmarks(
        image, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green connections
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green points
    )

# Main function
def main():
    st.header("Upload an Image")

    # Allow image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    draw_hand_landmarks(image, hand_landmarks)

        # Preprocess image for prediction
        resized_image = cv2.resize(image, (128, 128))
        normalized_image = resized_image / 255.0
        input_data = np.expand_dims(normalized_image, axis=0)

        # Check if model is loaded before making predictions
        if model is not None:
            prediction = model.predict(input_data)
            predicted_character = labels_dict[np.argmax(prediction)]
            st.write(f"Predicted: {predicted_character}")
        else:
            st.error("Model is not loaded. Unable to make predictions.")

        st.image(image, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
