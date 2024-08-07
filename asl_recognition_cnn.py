import streamlit as st
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the direct download link for Google Drive
direct_link = 'https://drive.google.com/uc?export=download&id=1Zk4ssy2IFYVUU4uFV072EmvQaKq2aOw3'

# Download the model from Google Drive
@st.cache_data
def download_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(response.content)
        temp.close()
        return temp.name
    else:
        st.error("Failed to download the model")
        return None

# Use the function to download and load the model
model_path = download_model(direct_link)
if model_path:
    model = load_model(model_path)
else:
    st.error("Model not loaded due to download failure.")

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
