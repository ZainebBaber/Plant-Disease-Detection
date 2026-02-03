
**Project Overview**
This project implements a Convolutional Neural Network (CNN) to classify Pepper/Bell plant images into healthy or diseased categories. The model is trained on a curated dataset, and data preprocessing includes **duplicate removal, stratified splitting, and data augmentation** to improve generalization.

The pipeline supports:

* Cleaning the dataset from duplicates
* Stratified train/validation/test split
* Data augmentation for robust training
* Training a CNN model with PyTorch
* Visualizing metrics and confusion matrices
* Testing on a dedicated test set


  **Dataset**

The dataset contains images of Pepper/Bell plants categorized into:

* `Pepper__bell___Bacterial_spot`
* `Pepper__bell___healthy`

The dataset is first cleaned from duplicate images and then split into `train`, `val`, and `test` folders in a **stratified manner** to maintain class balance.

## Features

* Duplicate image removal
* Stratified train/val/test split
* CNN-based image classification using PyTorch
* Data augmentation: rotation, horizontal flip, resized crop, color jitter
* Metrics computed on the test set: Accuracy, Precision, Recall, F1-score
* Confusion matrix visualization
* Sample predictions visualization

## Usage

### 1. Training the Model

Run the training script:

```bash
python train.py
```

This will:

* Clean duplicates in the dataset
* Perform stratified splitting into `dataset/train`, `dataset/val`, and `dataset/test`
* Train the CNN model for the number of epochs specified in `config.json`
* Save the **best model** in `saved/models/best_model.pth`



### 2. Testing the Model

Run the testing script:

```bash
python test.py
```

This will:

* Load the saved model
* Load the `dataset/test` images
* Compute metrics: Accuracy, Precision, Recall, F1-score
* Display a confusion matrix and sample predictions

## Project Structure

```
pepper-bell-classification/
│
├── data/
│   └── raw/               # Original dataset
├── dataset/               # Cleaned and stratified folders created by train.py
│   ├── train/
│   ├── val/
│   └── test/
├── model/
│   └── PlantCnn_model.py  # CNN model definition
├── trainer/
│   └── trainer.py          # Training loop
├── utils/
│   ├── util.py             # Helper functions
│   └── split_dataset.py    # Stratified split and save function
├── logger/
│   └── visualization.py    # Metrics and plots
├── saved/
│   └── models/             # Best model saved here
├── train.py
├── test.py
├── config.json             # Hyperparameters and paths
├── requirements.txt
└── README.md

**Results**

After training, the model achieves:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.92  |
| Precision | 0.91  |
| Recall    | 0.92  |
| F1-score  | 0.91  |

Example **confusion matrix** and **sample predictions** are generated after testing.

---

**Future Improvements**

* Add support for more disease classes
* Experiment with pre-trained models like **ResNet** or **EfficientNet**
* Hyperparameter tuning for better performance
* Deploy as a web app or mobile app for farmers

---


