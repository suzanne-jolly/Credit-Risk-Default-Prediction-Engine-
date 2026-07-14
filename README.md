# Credit Risk & Default Prediction Engine

## Overview
This repository contains an end-to-end machine learning pipeline designed to predict credit card defaults. The project focuses on handling heavily imbalanced financial data and prioritizing model interpretability.

## Key Achievements
* **Data Pipeline:** Built a binary classification pipeline for 50K+ records, engineering 12 distinct risk ratios (including credit utilization and payment-to-bill ratios).
* **Imbalance Handling:** Successfully addressed a 9:1 class imbalance utilizing **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Model Optimization:** Trained and optimized an **XGBoost** classifier using Scikit-learn's `GridSearchCV`, improving the cross-validated ROC-AUC score from a baseline of 0.65 to **0.82**.
* **Interpretability:** Integrated **SHAP** (SHapley Additive exPlanations) for comprehensive feature attribution, ensuring both global and local model interpretability for automated risk assessment.

## Repository Structure
* `Credit_Risk_Assessment.ipynb`: The main Google Colab notebook containing EDA, feature engineering, model training, and SHAP visualizations.
* `requirements.txt`: List of dependencies required to run the environment.

## Quick Start
1. Clone the repository: `git clone https://github.com/suzanne-jolly/credit-risk-prediction-engine.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset [https://drive.google.com/file/d/1sbkBJ-sRGsnm_N8ac8-aXjVG2ZA6obPt/view?usp=sharing] and place it in the root directory.
4. Run the Jupyter Notebook.
