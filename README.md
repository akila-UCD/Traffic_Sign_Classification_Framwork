# Traffic Sign Classification Project

## Overview

The project is structured as follows:

1. **Data Retrieval**: The dataset used for training the model is downloaded from Kaggle using the Kaggle API.
2. **Model Training**: The deep learning model is trained on the preprocessed traffic sign images.
3. **API Development**: An API is built using Flask to serve predictions based on input images.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python 3.x
- See `requirements.txt` for a list of Python packages required for both the deep learning and API parts of the project.

## Setup

1. Clone this repository:

   ```bash
   git clone <repository-url>

2. Install dependencies:
   pip install -r requirements.txt

3. Download the dataset:
   kaggle datasets download -d valentynsichkar/traffic-signs-preprocessed
   unzip traffic-signs-preprocessed.zip -d data

## Usage

1. Train the deep learning model:

2. Start the API server:
   python3 api.py

The API server will be running at http://localhost:8083 by default.

# API Endpoints
   /predict: POST endpoint for classifying traffic signs. Send an image file as input.

# Credits

   Dataset: Traffic Signs Preprocessed by Valentyn Sichkar
   TensorFlow: https://www.tensorflow.org/
   Flask: https://flask.palletsprojects.com/
   PIL (Python Imaging Library): https://python-pillow.org/

# License

 This project is licensed under the Apache license 2.0 License.
