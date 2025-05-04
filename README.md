# Lung Cancer Detection Using Chest CT Scan Images

## Overview

This project uses Convolutional Neural Networks (CNNs) to detect **lung cancer** from **Chest CT Scan images**. The dataset used for this project was sourced from [Kaggle's Chest CT Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images). 

The goal is to classify the scans into two categories:
- **With Cancer**: Images that show signs of lung cancer.
- **Without Cancer**: Normal lung images.

## Importance of Lung Cancer Detection

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection significantly improves survival rates, as it allows for timely intervention and treatment. **Automated detection** using AI models like CNNs can assist healthcare professionals in diagnosing lung cancer more accurately and efficiently. This model can help identify cancerous cells at an early stage, reducing the risk of late-stage diagnoses and improving outcomes for patients.

## Project Workflow

### 1. **Dataset Preparation**
   - The dataset from Kaggle contains CT scan images of lungs, both with and without cancer.
   - The images are first **organized into a binary classification format** (`with_cancer` vs. `without_cancer`) for easier model training.
   - The dataset is split into three subsets: **train**, **validation**, and **test**.

### 2. **Model Building**
   - A **Convolutional Neural Network (CNN)** is designed using TensorFlow and Keras to classify the images.
   - The model includes several convolutional layers for feature extraction, followed by fully connected layers to make predictions.

### 3. **Training**
   - The model is trained on the prepared dataset for several epochs, using **binary cross-entropy loss** and the **Adam optimizer**.
   - The accuracy of the model is monitored during training to ensure proper learning.

### 4. **Evaluation**
   - After training, the model is evaluated on the **test dataset** to check its **accuracy** and **loss**. A high accuracy and low loss indicate that the model is performing well.

### 5. **Visualization**
   - Sample images from the training dataset are displayed to verify that the data is loaded and organized correctly.
   
### 6. **Model Saving**
   - The trained model is saved for future use, so it can be loaded again for inference on new, unseen images.

## Results

After training, the model achieved an impressive **accuracy of 98.79%** on the test dataset, with a **loss of 0.0174**. This high accuracy and low loss indicate that the model performs well in distinguishing between CT scans with and without cancer, demonstrating the potential of CNNs in medical image classification.

## Requirements

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **Matplotlib**
- **NumPy**
- **Pandas**
- **os**, **shutil** (for data organization)

## Running the Project

To run this project, clone this repository and ensure you have all the required libraries installed. Follow these steps:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) and organize the images as per the script.
2. Run each cell in the Jupyter notebook to prepare the dataset, build the model, train it, and evaluate its performance.

## Conclusion

This project showcases the potential of **machine learning** and **AI** in assisting with critical medical diagnoses like lung cancer. By training a model on CT scan images, the system can help doctors identify cancerous lesions quickly and accurately, improving diagnosis and treatment outcomes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

