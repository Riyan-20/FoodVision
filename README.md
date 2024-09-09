# FoodVision 🍕🍔🍣

FoodVision is a deep learning project focused on classifying food images into 30 different categories using Convolutional Neural Networks (CNN). The model is trained on a subset of the **Food-101** dataset, specifically using 30 food classes, to predict the type of food in an image.

## Dataset 📊

The project utilizes a subset of the **Food-101** dataset, focusing on 30 specific food categories. The dataset contains thousands of food images, which makes it suitable for training a deep learning model for image classification tasks.

- **Dataset Link**: [Food-101 on Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)

## Model 🧠

The CNN model architecture is designed to learn from image data and accurately classify it into one of the 30 food categories. The model leverages multiple convolutional and pooling layers to extract features, followed by fully connected layers to make predictions.

### Features:
- **Convolutional Layers** for detecting spatial patterns
- **Pooling Layers** to reduce feature dimensions
- **Batch Normalization** and **Dropout** for regularization