import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space'
}

# Data Augmentation Setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# Create the model using transfer learning
def create_model():
    # Load the MobileNetV2 model
    base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(labels_dict), activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and load the model (or train if needed)
# Uncomment the following lines if you want to train a new model
# model = create_model()
# model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Load the trained Keras model
model = tf.keras.models.load_model('asl_model.keras')

# Function to preprocess images
def preprocess_image(image):
    image = image.resize((64, 64))  # Ensure this matches the model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape((1, 64, 64, 3))  # Reshape for batch prediction
    return image

# Streamlit app setup
st.title("ASL Alphabet Recognition")

# Function to capture frames from webcam
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to capture frame from webcam.")
        return None
    return frame

# Capture image from webcam
if st.button("Capture Image from Webcam"):
    frame = capture_frame()
    if frame is not None:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption='Captured Image', use_column_width=True)

        # Preprocess the image for model input
        image = cv2.resize(frame_rgb, (64, 64))  # Resize to the expected input size
        image = np.array(image) / 255.0  # Normalize the image
        image = image.reshape((1, 64, 64, 3))  # Reshape to match model input

        # Predict the ASL letter
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_character = labels_dict[predicted_class_index]

        st.write(f"Predicted Class: {predicted_character}")

        # Debugging: Show prediction probabilities
        st.write("Prediction Probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{labels_dict[i]}: {prob:.4f}")

# Option to upload an image for prediction
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to RGB and preprocess
    preprocessed_image = preprocess_image(image)

    # Predict the ASL letter
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_character = labels_dict[predicted_class_index]

    st.write(f"Predicted Class: {predicted_character}")

    # Debugging: Show prediction probabilities
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{labels_dict[i]}: {prob:.4f}")
