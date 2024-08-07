# ASL Alphabet Recognition

## Project Overview

This project focuses on recognizing American Sign Language (ASL) alphabet characters using Convolutional Neural Network (CNN). The project is designed to facilitate communication for individuals who use ASL by translating sign language gestures into text or speech.

## Dataset

The dataset used for this project is the ASL Alphabet Dataset, which is a collection of images representing the ASL alphabet. It contains:

- **Training Data**: 87,000 images of size 200x200 pixels, spread across 29 classes. These classes include the 26 letters (A-Z) and three additional classes for SPACE, DELETE, and NOTHING, aiding in real-time applications and classification.
- **Test Data**: 28 images designed to encourage testing with real-world images.

For more details, visit the ASL Alphabet Dataset page on Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Getting Started

### Prerequisites

To run this project, you will need:
- Python 3.x
- TensorFlow
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Requests

You can install these dependencies using:

```bash
pip install tensorflow opencv-python mediapipe streamlit numpy matplotlib scikit-learn requests
```

### Installation

1. Clone this repository: 
```bash
git clone https://github.com/vemilagan/CV-Project.git
cd CV-Project
```

2. Run the Jupyter Notebook to train and evaluate the model:
Open the asl_alphabet_cnn.ipynb file in Jupyter Notebook and follow the instructions to load the data, train the model, and evaluate its performance.

3. Use Streamlit to run the user interface (asl_recognition_cnn.py):
```bash
streamlit run your_script.py
```
(Replace your_script.py with the name of your Streamlit application file.)

## Model Architecture
This project utilizes a custom Convolutional Neural Network (CNN) architecture designed to effectively recognize American Sign Language (ASL) alphabet characters. The model is built from scratch, tailored to process input images of size 150x150 pixels, and includes multiple convolutional layers followed by pooling layers. This architecture is optimized for both accuracy and computational efficiency, making it suitable for deployment on various platforms.

### Key Features of the Model:
- **Convolutional Layers**: Capture spatial hierarchies in images to recognize features specific to ASL gestures.
- **Pooling Layers**: Reduce dimensionality and computational complexity, helping to improve generalization.
- **Dense Layers**: Facilitate decision-making based on extracted features to predict ASL characters.
- **Dropout Layer**: Reduces overfitting by preventing the network from becoming too reliant on any specific feature set.
- **Activation Functions**: Employ ReLU activations in hidden layers for non-linear transformations, and softmax activation in the output layer for multi-class classification across 29 classes.

The model is trained and validated on the ASL Alphabet dataset to achieve high accuracy in recognizing sign language gestures, with ground truth annotations used for rigorous evaluation.

## Results
- The model achieved high accuracy on both the training and test datasets.
- Confusion matrices and classification reports indicate strong performance across all ASL alphabet classes.
- The project can be further improved by increasing the dataset size, incorporating more diverse real-world test images, and enhancing model robustness.

## Model Evaluation
- The training and validation accuracy and loss curves indicate that the model converges well and performs effectively on unseen data.
- The confusion matrix illustrates the model's ability to correctly classify each ASL alphabet gesture

## Ground Truth
The ground truth for each image in the test dataset corresponds to the true class labels, which were manually verified. The model's predictions are compared against these ground truth labels to compute evaluation metrics such as accuracy, precision, recall, and F1-score. This ensures a reliable assessment of the model's performance.

## Acknowledgements
- Dataset: ASL Alphabet Dataset by Akash Nagaraj available at https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- GitHub Repository: Sign Language to Speech: Unvoiced available at https://github.com/grassknoted/Unvoiced

### Citation:
“ASL Alphabet.” Kaggle, 22 Apr. 2018, www.kaggle.com/datasets/grassknoted/asl-alphabet.

Computer Vision Engineer. “Sign language detection with Python and Scikit Learn | Landmark detection | Computer vision tutorial.” YouTube, 26 Jan. 2023, www.youtube.com/watch?v=MJCSjXepaAM.

Garimella, Mihir. “Sign Language Recognition with Advanced Computer Vision.” Medium, 4 Dec. 2022, towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442.

“Hand landmarks detection guide.” Google AI for Developers, ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker.

## Contributing
If you wish to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.
