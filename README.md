# Anemia & Thrombocytopenia Prediction

Anemia & Thrombocytopenia Prediction is a Python project that aims to predict whether a patient has anemia and thrombocytopenia based on specific health parameters. This project includes a backend model comparison and a user-friendly frontend using Streamlit.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Comparison](#model-comparison)
- [License](#license)

## Overview

Anemia and thrombocytopenia are critical health conditions that require early detection and intervention. This project utilizes machine learning models to predict the likelihood of these conditions based on specific patient data.

## Features

- Predicts the presence of anemia and thrombocytopenia.
- User-friendly web interface for inputting patient data.
- Model comparison to identify the best-fitting prediction model.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine: git clone https://github.com/Sia-11/AIML-Internship-Dlithe.git

2. Set up a virtual environment for the project and install the required packages.

3. Navigate to the project directory

4. Install the required dependencies using pip

5. Run the Streamlit app:
streamlit run app.py

6. Use the web interface to input patient data and make predictions.

## Project Structure

The project structure is organized as follows:

- `app.py`: Streamlit app for the frontend.
- `data.csv`: Sample dataset.
- `requirements.txt`: List of required packages.
- `models/`: Contains the machine learning models.
- `utils/`: Utility functions for data preprocessing.

## Usage

To use this project, follow these steps:

1. Run the Streamlit app as described in the "Getting Started" section.

2. Input patient data through the web interface.

3. Click the "Predict Anemia" or "Predict Thrombocytopenia" buttons to get predictions.

## Model Comparison

-The project includes a model comparison to determine the best-fitting prediction model.
-The accuracy scores of different models are compared and visualized using Matplotlib.

## License

This project is licensed under the [MIT License](LICENSE).
