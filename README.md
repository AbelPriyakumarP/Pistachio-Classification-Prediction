# Pistachio Classification Prediction

This project implements a Convolutional Neural Network (CNN) to classify pistachio images into two categories: **Siirt Pistachio** and **Kirmizi Pistachio**. The model is trained on the [Pistachio Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset) from Kaggle and deployed using a Python script.

## Table of Contents
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)
- [Training the Model](#training-the-model)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The [Pistachio Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset) contains 2,148 images of pistachios, split into two classes:
- **Kirmizi Pistachio**: 1,232 images
- **Siirt Pistachio**: 916 images

The dataset is used to train a CNN model to distinguish between these two pistachio types based on visual features.

## Project Overview
This project includes:
- A custom CNN model (`PistachioCNN`) implemented in PyTorch.
- A prediction script (`app.py`) that loads a pre-trained model and classifies a given pistachio image.
- Helper functions in `model_helper.py` to preprocess images and make predictions.

The goal is to accurately classify pistachio images using deep learning techniques.

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- Pillow (PIL)
- NumPy

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd pistachio-classification

## Example Prediction
Here’s an example of a pistachio image classified by the model:

![Pistachio Example](Screenshot 2025-03-09 223848.jpg)
Predicted Class: Siirt_Pistachio

# File Structure
   pistachio-classification/
│
├── app.py                # Main script to run predictions
├── model_helper.py       # CNN model definition and prediction logic
├── model/
│   └── saved_model.pth   # Pre-trained model weights
├── Pistachio_Image_Dataset/  # Dataset folder (after extraction)
│   ├── Kirmizi_Pistachio/    # Kirmizi pistachio images
│   └── Siirt_Pistachio/      # Siirt pistachio images
└── README.md             # This file


---

### Explanation
- **Dataset**: Describes the Kaggle dataset with a link and basic stats.
- **Project Overview**: Summarizes the purpose and components.
- **Requirements/Installation**: Guides users to set up the environment and download the dataset.
- **Usage**: Shows how to run predictions.
- **Model Architecture**: Details the CNN structure for transparency.
- **Training**: Provides a basic training script since the pre-trained model isn’t provided.
- **File Structure**: Reflects your earlier code context.

Save this as `README.md` in your project directory. Let me know if you need help with any specific part, like training or deployment!
