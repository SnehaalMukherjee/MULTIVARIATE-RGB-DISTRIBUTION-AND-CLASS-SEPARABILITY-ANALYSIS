# Multivariate RGB Distribution and Class Separability Analysis

A statistical image-analysis project that studies RGB colour distributions across six scene categories in the **Intel Image Classification** dataset. The project extracts mean Red, Green, and Blue (RGB) intensity values from images and compares the resulting feature distributions across classes.

## Project Overview

The objective is to explore how the colour characteristics of different scene categories vary and how useful mean RGB features are for distinguishing between those categories.

This is a statistical analysis project, not a trained image-classification model.

## Dataset

- **Dataset:** Intel Image Classification
- **Source:** [Kaggle – Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Split analysed:** Training set (`seg_train`)
- **Images analysed:** 14,034
- **Classes:** 6

### Image Classes

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

> **Note:** The dataset is not included in this repository. Download it from Kaggle and place the training images in the folder/path expected by the project code.

## Methodology

1. Load the images and their corresponding class labels.
2. Extract the mean Red, Green, and Blue intensity values from each image.
3. Organise the extracted RGB features by image class.
4. Analyse and compare the class-wise RGB distributions.
5. Use statistical summaries and visualisations to study similarities, differences, and class separability.

## Features Extracted

Each image is represented using three numerical features:

| Feature | Description |
|---|---|
| Mean Red | Average intensity of the red channel |
| Mean Green | Average intensity of the green channel |
| Mean Blue | Average intensity of the blue channel |

## Tools and Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Image-processing and statistical analysis libraries used in the project

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SnehaalMukherjee/MULTIVARIATE-RGB-DISTRIBUTION-AND-CLASS-SEPARABILITY-ANALYSIS.git
cd MULTIVARIATE-RGB-DISTRIBUTION-AND-CLASS-SEPARABILITY-ANALYSIS
```

### 2. Download the dataset

Download the Intel Image Classification dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), extract it, and make sure the training images are available at the path configured in the project code.

### 3. Install dependencies

Install the Python packages required by the project. If a `requirements.txt` file is provided, run:

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

Open and run the project's Python script or Jupyter Notebook. Make sure the dataset path in the code matches the location where you extracted the dataset.

## Expected Outcome

The analysis provides a comparison of mean RGB colour distributions across the six scene categories. The results help illustrate where classes show similar colour characteristics and where their RGB feature distributions differ.

## Academic Project

- **Subject / Domain:** Pattern Matching
- **Project Title:** Multivariate RGB Distribution and Class Separability Analysis
- **Dataset:** Intel Image Classification
- **Analysis Type:** Multivariate statistical image analysis

---

*This repository contains the project code, analysis outputs, and documentation. The original image dataset is hosted separately on Kaggle.*
