import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained Keras model
try:
    model = tf.keras.models.load_model('asl_model_mobilenetv2.keras')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

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

            # Draw landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

            # Preprocess frame for prediction
            resized_frame = cv2.resize(frame, (128, 128))  # Resize to model's input size
            normalized_frame = resized_frame / 255.0  # Normalize pixel values
            input_data = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

            # Predict the ASL letter
            prediction = model.predict(input_data)
            predicted_character = labels_dict[np.argmax(prediction)]

            # Display prediction
            cv2.putText(frame, f"Predicted: {predicted_character}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display result
            st.image(frame, channels="BGR", use_column_width=True)

            if st.button("Stop"):
                break

    cap.release()

if __name__ == "__main__":
    main()