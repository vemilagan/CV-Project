import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Suppress version warnings
warnings.simplefilter('ignore', InconsistentVersionWarning)

# Load the trained RandomForest model and label encoder
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

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

# Function to draw hand landmarks
def draw_hand_landmarks(image, hand_landmarks):
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red points

    mp_drawing.draw_landmarks(
        image, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green connections
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green points
    )

# Function to preprocess hand landmarks for model prediction
def preprocess_landmarks(hand_landmarks):
    # Extract x, y, and z coordinates
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    # Normalize landmarks
    landmarks -= np.min(landmarks)
    landmarks /= np.ptp(landmarks)
    
    # Ensure landmarks have 84 features
    target_length = model.n_features_in_
    if len(landmarks) < target_length:
        landmarks = np.pad(landmarks, (0, target_length - len(landmarks)), 'constant')
    return landmarks

def main():
    if model is None:
        st.error("Model is not loaded. Please check your model file.")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Flip and process frame
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            # Draw landmarks and predict ASL letter
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

                    # Preprocess the landmarks
                    landmarks = preprocess_landmarks(hand_landmarks)
                    landmarks = landmarks.reshape(1, -1)  # Reshape for model input

                    # Check if the input size matches
                    if landmarks.shape[1] == model.n_features_in_:
                        # Predict the ASL letter
                        prediction = model.predict(landmarks)
                        predicted_index = int(prediction[0])
                        predicted_character = labels_dict.get(predicted_index, "Unknown")

                        # Display prediction
                        cv2.putText(frame, f"Predicted: {predicted_character}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        st.warning(f"Model input shape mismatch: expected {model.n_features_in_} features, got {landmarks.shape[1]}.")

            # Display result
            st.image(frame, channels="BGR", use_column_width=True)

            if st.button("Stop", key="stop_button"):
                break

    cap.release()

if __name__ == "__main__":
    main()
