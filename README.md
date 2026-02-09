# EfficientNetV2L Transfer Learning CNN for dog breed classification 

## Description
This notebook trains a dog-breed classifier (120 classes) using EfficientNetV2-L transfer learning on the Kaggle Dog Breed Identification dataset.

The training pipeline includes:
- Loading labels.csv with a stratified train/validation split
- A transfer learning model based on EfficientNetV2-L
- Training in two phases -> frozen backbone, then fine-tuning
- Quantitative evaluation via confusion matrix and classification report
- LIME explanations for visual interpretability

For benchmarking purposes, notebook also implements a baseline 5-block CNN.

Dataset used for training and validation: [Kaggle Link](https://www.kaggle.com/competitions/dog-breed-identification/data)

## Model performance

Validation accuracy: ~95%

## How to run 

### 1. Download tha dataset and unzip it

The notebook expects dataset directory to be named exactly 'dataset'

Project structure should look like this:

```
project_root/
  dog_breed_classification_EfficientNetV2L_transfer_learning.ipynb
  dataset/
    labels.csv
    train/
      <image_id>.jpg
```

### 2. Create conda environment

```
conda create --name dogbreed python=3.11
conda activate dogbreed
```

### 3. Install dependencies

```
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scikit-image lime
```

### 4. Launch notebook in your code editor and select created venv

As long as the folder is named dataset, the notebook should execute without path modifications.

### Alternatively you can run notebook in [Google Colab](https://colab.research.google.com/).

## Training time information

Model was trained on RTX 5070 Ti GPU (Blackwell).

Estimated training time:

- Stage 1, 10 Epochs -> ~13 minutes
- Stage 2, 30 Epochs -> ~44 minutes
- Baseline CNN, 50 Epochs -> ~15 minutes

After training, best version of each model is saved to .keras file.

