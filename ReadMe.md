# ASL Alphabet Recognition with MobileNetV2

## Project Overview

This project focuses on recognizing American Sign Language (ASL) alphabet characters using a MobileNetV2 model. The project is designed to facilitate communication for individuals who use ASL by translating sign language gestures into text or speech.

## Dataset

The dataset used for this project is the ASL Alphabet Dataset, which is a collection of images representing the ASL alphabet. It contains:

- **Training Data**: 87,000 images of size 200x200 pixels, spread across 29 classes. These classes include the 26 letters (A-Z) and three additional classes for SPACE, DELETE, and NOTHING, aiding in real-time applications and classification.
- **Test Data**: 29 images designed to encourage testing with real-world images.

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

You can install these dependencies using:

```bash
pip install tensorflow opencv-python mediapipe streamlit numpy matplotlib scikit-learn
```

### Installation

1. Clone this repository: 
```bash
git clone https://github.com/vemilagan/CV-Project.git
cd cd CV-Project
```

2. Run the Jupyter Notebook to train and evaluate the model:
Open the asl_tensorflow_mobilenetv2.ipynb file in Jupyter Notebook and follow the instructions to load the data, train the model, and evaluate its performance.

3. Use Streamlit to run the user interface:
```bash
streamlit run your_script.py
```
(Replace your_script.py with the name of your Streamlit application file.)

# Model Architecture
This project uses the MobileNetV2 architecture, a lightweight and efficient convolutional neural network suitable for mobile and edge devices. The model is fine-tuned on the ASL Alphabet dataset to achieve high accuracy in recognizing sign language gestures.

## Results
- The model achieved high accuracy (approximately 96%) on both the training and test datasets.
- Confusion matrices and classification reports indicate strong performance across all ASL alphabet classes.
- The project can be further improved by increasing the dataset size, incorporating more diverse real-world test images, and enhancing model robustness.

## Model Evaluation
- The training and validation accuracy and loss curves indicate that the model converges well and performs effectively on unseen data.
- The confusion matrix illustrates the model's ability to correctly classify each ASL alphabet gesture

## Acknowledgements
- Dataset: ASL Alphabet Dataset by Akash Nagaraj
- GitHub Repository: Sign Language to Speech: Unvoiced

### Citation:
{https://www.kaggle.com/grassknoted/aslalphabet_akash_nagaraj_2018,
title={ASL Alphabet},
url={https://www.kaggle.com/dsv/29550},
DOI={10.34740/KAGGLE/DSV/29550},
}

## Contributing
If you wish to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.