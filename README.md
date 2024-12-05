# Stellar Classification with Machine Learning

This project applies machine learning techniques to classify celestial objects based on their observed properties. Using the **Stellar Classification Dataset (SDSS17)**, we aim to demonstrate how machine learning models can effectively distinguish between stars, galaxies, and quasars.

## Dataset

The dataset used is the [Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) from Kaggle. It is derived from the Sloan Digital Sky Survey (SDSS) and includes information on celestial objects. Key features of the dataset:

- **Object Classes:** Stars, galaxies, and quasars.
- **Attributes:** Magnitudes, redshift, and other observed characteristics of celestial bodies.
- **Balanced Distribution:** Approximately 100,000 samples, evenly distributed across the three classes.

The dataset is suitable for classification tasks, providing a rich variety of features for model training.

## Objective

The main goal of this study is to build and evaluate machine learning models that can classify celestial objects into their respective categories. This involves:

1. **Exploratory Data Analysis (EDA):** Understanding the dataset structure and visualizing feature distributions.
2. **Feature Engineering:** Cleaning the dataset and selecting the most relevant attributes for classification.
3. **Model Training:** Experimenting with machine learning algorithms to classify objects accurately.
4. **Evaluation and Insights:** Interpreting the results and discussing the model's performance.

## Methodology

The complete workflow is documented in detail in the [case study blog post](https://juanmmaidana.github.io/posts/stellar/). The methodology includes:

1. Preprocessing steps like scaling and handling outliers.
2. Exploratory analysis to visualize relationships and feature importance.
3. Training models such as:
   - Logistic Regression
   - Random Forests
   - Support Vector Machines (SVM)
4. Tuning hyperparameters and optimizing model performance.
5. Evaluating the models with metrics like accuracy, precision, recall, and F1-score.

## Requirements

To reproduce the analysis, the following Python libraries are required:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook (optional, for interactive exploration)

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
