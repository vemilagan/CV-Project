{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# **ASL Alphabet Recognition with MobileNetV2**"
      ],
      "metadata": {
        "id": "B0H-H2KHOUvM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Project Overview\n"
      ],
      "metadata": {
        "id": "Em9HRa5lObIW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This project focuses on recognizing American Sign Language (ASL) alphabet characters using a MobileNetV2 model. The project is designed to facilitate communication for individuals who use ASL, by translating sign language gestures into text or speech."
      ],
      "metadata": {
        "id": "cuJu-oRmOdwz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Dataset"
      ],
      "metadata": {
        "id": "1ew6ooY2Og-L"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "The dataset used for this project is the ASL Alphabet Dataset, which is a collection of images representing the ASL alphabet. It contains:"
      ],
      "metadata": {
        "id": "MekSktZwWiIQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "*   **Training Data:** 87,000 images of size 200x200 pixels, spread across 29 classes. These classes include the 26 letters (A-Z) and three additional classes for SPACE, DELETE, and NOTHING, aiding in real-time applications and classification.\n",
        "*   **Test Data:** 29 images designed to encourage testing with real-world images.\n",
        "\n"
      ],
      "metadata": {
        "id": "aTfHyieRWmpV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "For more details, visit the ASL Alphabet Dataset page on Kaggle. https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
      ],
      "metadata": {
        "id": "Kxb9Syl-WxFF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Getting Started"
      ],
      "metadata": {
        "id": "hj3kU0H6W5iR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Prerequisites"
      ],
      "metadata": {
        "id": "6pGUycg4W8kh"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "To run this project, you will need:\n",
        "\n",
        "* Python 3.x\n",
        "* TensorFlow\n",
        "* OpenCV\n",
        "* MediaPipe\n",
        "* Streamlit\n",
        "* NumPy\n",
        "* Matplotlib\n",
        "* Scikit-learn"
      ],
      "metadata": {
        "id": "09nJjhmAW_CN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "You can install these dependencies using:\n",
        "\n"
      ],
      "metadata": {
        "id": "ieytYXS4XJhE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install tensorflow opencv-python mediapipe streamlit numpy matplotlib scikit-learn"
      ],
      "metadata": {
        "id": "IZnMdS18XMgB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Installation"
      ],
      "metadata": {
        "id": "sIHZf7PzXPjm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "1. Clone this repository:"
      ],
      "metadata": {
        "id": "YzyE6hKRXSwa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "git clone https://github.com/vemilagan/CV-Project.git\n",
        "cd asl-alphabet-recognition"
      ],
      "metadata": {
        "id": "Vb_pmN1NXWgq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "2. Download the dataset from Kaggle and place it in the appropriate directory.\n",
        "\n"
      ],
      "metadata": {
        "id": "17MnXMFZXYRl"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "3. Run the Jupyter Notebook to train the model and evaluate its performance."
      ],
      "metadata": {
        "id": "Y2jsC9jZX6gS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "4. Use Streamlit to deploy the model for real-time ASL recognition:"
      ],
      "metadata": {
        "id": "Sh5GDr3zX9ox"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "streamlit run asl_recognition_mobilenetv2.py"
      ],
      "metadata": {
        "id": "DhRs5RX-YA8t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Model Architecture"
      ],
      "metadata": {
        "id": "NRY8powwYQXz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This project uses the MobileNetV2 architecture, a lightweight and efficient convolutional neural network suitable for mobile and edge devices. The model is fine-tuned on the ASL Alphabet dataset to achieve high accuracy in recognizing sign language gestures."
      ],
      "metadata": {
        "id": "fcCVtJsDYSSE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Results and Analysis"
      ],
      "metadata": {
        "id": "9RK2oTV6YVXm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "* The model achieved high accuracy on both the training and test datasets.\n",
        "* Confusion matrices and classification reports indicate strong performance across all ASL alphabet classes.\n",
        "* The project can be further improved by increasing the dataset size, incorporating more diverse real-world test images, and enhancing model robustness."
      ],
      "metadata": {
        "id": "ekk9wIYMYYvo"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Acknowledgements"
      ],
      "metadata": {
        "id": "TjwT9tZ0Yekv"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "* Dataset: ASL Alphabet Dataset by Akash Nagaraj https://www.kaggle.com/datasets/grassknoted/asl-alphabet\n",
        "* GitHub Repository: Sign Language to Speech: Unvoiced https://github.com/grassknoted/Unvoiced"
      ],
      "metadata": {
        "id": "DVkn1DTQYhIy"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Citation"
      ],
      "metadata": {
        "id": "QYRMyBfaY0L-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "{https://www.kaggle.com/grassknoted/aslalphabet_akash_nagaraj_2018,\n",
        "title={ASL Alphabet},\n",
        "url={https://www.kaggle.com/dsv/29550},\n",
        "DOI={10.34740/KAGGLE/DSV/29550},\n",
        "}\n"
      ],
      "metadata": {
        "id": "Urdtcz8rY_ft"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Contributing"
      ],
      "metadata": {
        "id": "Vtbbmz23ZHJG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "If you wish to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome."
      ],
      "metadata": {
        "id": "p2sMWTLxZR32"
      }
    }
  ]
}