# Skin-Cancer Detection Using machine learning
This repository contains a machine learning model for detecting skin cancer using image analysis techniques.

# Overview
Skin cancer is one of the most common types of cancer globally, and early detection is crucial for effective treatment. This project aims to develop a model that can accurately classify skin lesions as benign or malignant based on images.

# Dataset
The model is trained on the Skin Cancer MNIST: HAM10000 dataset from Kaggle, which includes a large collection of skin lesion images labeled with their respective diagnoses.

# Model Architecture
The skin cancer detection model is built using deep learning techniques, specifically a convolutional neural network (CNN). The CNN architecture includes multiple convolutional layers followed by pooling layers and fully connected layers for classification.

# Training
The model is trained using the TensorFlow framework with data augmentation techniques to improve generalization and prevent overfitting. Training involves optimizing the model's parameters to minimize the classification error on the training data.

# Evaluation
The model's performance is evaluated using various metrics such as accuracy, precision, recall, and F1 score on a separate validation dataset. Additionally, visualizations such as confusion matrices and ROC curves are used to assess the model's performance.

# Usage
To use the skin cancer detection model:

Clone this repository to your local machine.
Install the required dependencies listed in requirements.txt.
Download the dataset from Kaggle or provide your own dataset.
Preprocess the images and split them into training and validation sets.
Train the model using the provided scripts or Jupyter notebooks.
Evaluate the model's performance and make predictions on new data.
Results
The model achieves a high accuracy rate in classifying skin lesions as benign or malignant, demonstrating its potential for assisting dermatologists in early skin cancer detection.
I have added the code for live detection of skin cancer using device camera or uploading of the images and it is named as CancerPrediction2.py
Desined using spyder and with the help of this model it predicts whether the person has skin cancer or not.

# Contributing
Contributions to improve the model's performance, add new features, or enhance the documentation are welcome. Please fork the repository and submit a pull request with your changes.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Contact
For inquiries or feedback, please contact Deepak Avadhani.
