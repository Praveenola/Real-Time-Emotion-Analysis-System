
# Real-Time Emotion Analysis System

The "Real-Time Emotion Analysis System" project focuses on identifying human emotions from facial expressions using a Convolutional Neural Network (CNN). Trained on the FER-2013 dataset, the CNN classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The system provides accurate, real-time emotion recognition, with applications in human-computer interaction, customer service, and mental health monitoring.


## Tech Stack

Python, 
Pandas, 
NumPy, 
Scikit-learn (sklearn),
Seaborn,
Matplotlib,
TensorFlow,
Keras,
ResNet50v2,
VGG16,
OpenCV,
Gradio.

## Dataset

- The project uses the FER-2013 dataset, which is a publicly available dataset containing labeled images of facial expressions.
## Features

- **Emotion Classification:** Classifies images and videos into seven emotion categories.
- **Real-time Detection:** Capable of detecting emotions in real-time using webcam input.
- **Pre-trained Model:** Includes a pre-trained model for quick setup and use.
- **Interactive Interface:** Provides a user-friendly interface for testing and visualization.
## Installation

### Prerequisites
- Python 3.7+
- pip
- OpenCV
- Other dependencies listed in 'requirements.txt'

## Setup
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Praveenola/Real-Time-Emotion-Analysis-System.git
    cd Real-Time-Emotion-Analysis-System
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command Line Interface
- **To test the model on an image:**
    ```bash
    python test.py --image path/to/image.jpg
    ```

- **To use the webcam for real-time emotion detection:**
    ```bash
    python test.py --webcam
    ```

## Interactive Interface
- **Run the Gradio app:**
    ```bash
    gradio app.py
    ```



## Models 


The emotion detection model is built using a Convolutional Neural Network (CNN) and includes advanced architectures such as **ResNet50v2** and **VGG16**. It is trained on the FER-2013 dataset to classify facial expressions into seven emotion categories.

## Training

To train the model, use the following command:
```bash
python train.py --dataset path/to/FER-2013 --epochs 50 --batch_size 64
```

### Pre-trained Model
A pre-trained model is available in the directory for quick use and testing.


## Deployment

## Overview

The provided deployment code sets up a real-time emotion detection system using a webcam. It utilizes a pre-trained deep learning model to classify emotions from facial expressions detected in the live video feed. The setup involves using OpenCV for video capture and face detection, and Keras for emotion classification.

## Requirements

- **Python 3.x**: Ensure Python is installed on your system.
- **Libraries**: Install the necessary Python libraries. Use the following command to install them:
    ```bash
    pip install keras opencv-python numpy
    ```
- **Pre-trained Models**: You need a pre-trained emotion detection model saved in the `.keras` or `.h5` format. Replace the model path in the code as needed.

## Model Files

- **Face Classifier**: `haarcascade_frontalface_default.xml`
  - Download the Haar Cascade model from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).
- **Emotion Classification Model**:
  - Ensure you have either `Custom_CNN_model.keras` or `Final_Resnet50_Best_model.keras` available in your working directory.

## Deployment Steps

1. **Setup Model Paths**: Update the paths to the model files in the code if necessary.
    ```python
    classifier = load_model(r'path/to/your/model.keras')
    ```

2. **Run the Code**:
    - Save the provided code in a Python script file, e.g., `deploy_emotion_detection.py`.
    - Execute the script using:
        ```bash
        python deploy_emotion_detection.py
        ```

3. **Interactive Video Feed**:
    - The code starts capturing video from your webcam.
    - It detects faces in each frame, classifies emotions, and displays the results in real-time.

4. **Exit the Application**:
    - Press the 'q' key while the video feed window is active to exit the application.

## Troubleshooting

- **No Webcam Feed**: Ensure your webcam is connected and properly configured. Check webcam permissions in your system settings.
- **Model Not Loading**: Verify the model file path and ensure it matches the format expected by `load_model()`.
- **Dependencies**: Ensure all required Python packages are installed correctly.

## Notes

- The model's accuracy and performance depend on the quality of the training and the dataset used.
- For enhanced performance or additional features, consider fine-tuning the model or integrating additional pre-processing steps.
## Contributing

Contributions are always welcome!

Please see `contributing.md` for ways to get started and adhere to this project's code of conduct.

## How to Contribute

I welcome contributions to the Real-Time Emotion Analysis project! Follow these steps to contribute:

1. **Fork the repository:**
   - Click the "Fork" button at the top right of this repository page to create a copy of the repository under your own GitHub account.

2. **Create your feature branch:**
   - Open a terminal and clone the forked repository to your local machine:
     ```bash
     git clone https://github.com/Praveenola/Real-Time-Emotion-Analysis-System.git
     cd Real-Time-Emotion-Analysis-System
     ```
   - Create a new branch for your feature or bugfix:
     ```bash
     git checkout -b feature/new-feature
     ```

3. **Commit your changes:**
   - Make the necessary changes in your local repository.
   - Stage the changes:
     ```bash
     git add .
     ```
   - Commit the changes with a descriptive message:
     ```bash
     git commit -m 'Add new feature'
     ```

4. **Push to the branch:**
   - Push the changes to your forked repository:
     ```bash
     git push origin feature/new-feature
     ```

5. **Open a Pull Request:**
   - Go to the original repository on GitHub and you will see a prompt to open a Pull Request from your new branch.
   - Provide a descriptive title and detailed description of your changes, and submit the Pull Request.

Thank you for contributing!

## Acknowledgements

We would like to extend our heartfelt thanks to the following individuals and organizations for their contributions and support:

- **FER-2013 Dataset**: We gratefully acknowledge the creators of the FER-2013 dataset, which provided the essential data for training and evaluating the emotion detection model.
- **ResNet and VGG Teams**: Special thanks to the teams behind the ResNet50v2 and VGG16 architectures for their groundbreaking work in deep learning and computer vision, which greatly enhanced the performance of our model.
- **Open Source Community**: Our project benefits immensely from the tools and libraries available in the open-source community. We appreciate the developers and maintainers of the various Python libraries and frameworks we utilized.
- **Contributors**: Thank you to all the contributors who help improve this project. Your efforts in reviewing code, reporting issues, and suggesting features are invaluable.
- **Supporters and Users**: A big thank you to everyone who has supported and used this project. Your feedback and enthusiasm drive us to continuously improve and innovate.

If you have contributed in any way or provided feedback, we appreciate your support and involvement in making this project better.




## 🔗 Links
[![https://www.linkedin.com/in/praveen-ola-991a66256?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/praveen-ola-991a66256?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) 

[![https://github.com/Praveenola](https://img.shields.io/badge/github-0A66C2?style=for-the-badge&logo=github&logoColor=black)](https://github.com/Praveenola)

