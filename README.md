# Credit Card Fraud Detection Using Machine Learning

This repository contains the code and artifacts produced as part of the practical assignment for the Machine Learning course in the Graduate Program in Computer Science at UFRGS. The study applies supervised learning techniques to detect credit card transaction fraud using a synthetic and highly imbalanced dataset.

## Objective

Evaluate the performance of different machine learning algorithms in a realistic fraud detection scenario, prioritizing metrics that favor sensitivity (recall) and analyzing model interpretability.

## Methods Used

The developed pipeline includes the following steps:

1. **Exploratory Data Analysis (EDA)**  
   Descriptive statistics, time-based analysis, feature correlation, and identification of discriminative patterns between legitimate and fraudulent transactions.

2. **Feature Engineering**  
   Creation of derived variables based on time, location, job, and customer-merchant distance, in addition to transformation of categorical variables.

3. **Data Preprocessing**  
   Categorical encoding (One-Hot and Ordinal Encoding), robust scaling, and normalization to the [0, 1] range.

4. **Models Evaluated**  
   - Naive Bayes  
   - Decision Tree  
   - Random Forest  
   - LightGBM  
   - XGBoost

5. **Nested Cross-Validation**  
   Evaluation strategy using `GridSearchCV` for inner hyperparameter tuning and outer model validation.

6. **Evaluation Metrics**  
   - **F2-score** (recall prioritized)
   - **Average Precision (AP)**
   - Confusion Matrix
   - Precision-Recall Curve

7. **Interpretability**  
   SHAP (SHapley Additive exPlanations) used to interpret XGBoost model predictions.

## Key Results

The **XGBoost** model showed the best results on the test set:

- **F2-score**: 0.7154  
- **Recall**: 0.7781  
- **Precision**: 0.5410  
- **Average Precision (AP)**: 0.771  

SHAP analysis revealed the most influential features:

- `amt` (transaction amount)
- `is_late_night` (nighttime transaction indicator)
- `category` and `city_size`

The model captured complex non-linear patterns, though some "invisible" frauds (low-value or daytime transactions) still resulted in false negatives.

## How to Run

### 1. Prerequisites

- Python ≥ 3.9
- Recommended: Google Colab (avoids memory and dependency issues)
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `shap`
  - `matplotlib`
  - `seaborn`

### 2. Run in Google Colab

1. Open the notebook:  
   [Notebook on Google Colab](https://colab.research.google.com/drive/1mvETRCFkSkSqARub5cSYmlmXFrl661Ho)

2. Execute the cells sequentially.

3. The notebook includes:
   - Data loading (train/test)
   - Feature engineering
   - Model training and evaluation
   - SHAP visualizations

> Note: The dataset is available on Kaggle as "Synthetic Financial Dataset For Fraud Detection". Upload the train and test files as prompted in the notebook.

### 3. Run Locally

1. Clone the repository:

```bash
git clone https://github.com/daniel-menna/credit-card-fraud-ml.git
cd credit-card-fraud-ml
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Run the notebook `credit-card-fraud-ml-notebook.ipynb` using Jupyter or VS Code.

## Repository Structure

```
├── credit-card-fraud-ml-notebook.ipynb    # Main notebook
├── README.md
├── requirements.txt                        # Dependency list
```

## References

All references used are listed at the end of the research article, including works by Faceli et al. (2021), Chen & Guestrin (2016), and Lundberg & Lee (2017).

## License

This project is made available for academic and educational purposes only.
