# Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to leave. The model is evaluated with **F1 score** because the churned-customer class is substantially smaller than the retained-customer class, making accuracy alone a poor measure of performance.

The complete analysis is in [`main.ipynb`](main.ipynb).

## Project Objective

Beta Bank wants to identify customers at risk of churn so that retention teams can prioritize timely interventions. The project target was an F1 score of at least **0.59** on a held-out test set.

## Dataset

The included [`datasets/Churn.csv`](datasets/Churn.csv) contains 10,000 customer records and 14 columns.

- **Target:** `Exited` (`1` = customer left, `0` = customer stayed)
- **Identifier columns removed:** `RowNumber`, `CustomerId`, `Surname`
- **Categorical features:** `Geography`, `Gender`
- **Numerical and binary features:** `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Data quality:** `Tenure` contains 909 missing values; no duplicate rows were found
- **Class distribution:** 79.6% stayed and 20.4% exited

## Approach

1. Split the data into train, validation, and test sets using a stratified 60/20/20 split.
2. Fit the `Tenure` median imputation value on the training set only.
3. Encode `Gender` as binary and one-hot encode `Geography` using training-set categories.
4. Standardize numeric features using statistics from the training set only.
5. Train a baseline `RandomForestClassifier` without imbalance correction.
6. Compare two imbalance strategies:
   - Balanced class weights
   - Random upsampling of the minority class
7. Select the best validation model and evaluate it once on the held-out test set.

This preprocessing order keeps validation and test information from influencing fitted transformations or model selection.

## Results

| Approach | Validation F1 | Validation AUC-ROC |
| --- | ---: | ---: |
| Baseline random forest | 0.5919 | 0.8664 |
| Balanced class weights | 0.6363 | 0.8728 |
| Random upsampling | **0.6374** | 0.8696 |

The upsampling model was selected because it achieved the highest validation F1 score. Its final held-out test performance was:

| Metric | Test result |
| --- | ---: |
| F1 score | **0.6110** |
| AUC-ROC | 0.8557 |
| Churn precision | 0.55 |
| Churn recall | 0.68 |

The final model exceeded the project target of 0.59. Its recall indicates that it identified approximately two-thirds of actual churners, while its precision shows that some customers flagged for retention would not have churned.

## Tools

- Python
- pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

## How to Run

From the project folder, install the required packages in a Python environment:

```bash
python -m pip install pandas scikit-learn matplotlib jupyter
```

Launch the notebook:

```bash
jupyter notebook main.ipynb
```

Run the cells from top to bottom. The notebook expects the dataset at `datasets/Churn.csv` relative to the project folder.

## Project Structure

```text
.
├── main.ipynb
├── datasets/
│   └── Churn.csv
└── README.md
```

## Limitations and Next Steps

This project focuses on a small, interpretable comparison of random-forest approaches. Potential extensions include cross-validated hyperparameter search, decision-threshold tuning, feature-importance analysis, calibration, and comparison with gradient-boosting models. Any production use would also require monitoring for data drift, periodic retraining, and a careful review of the business cost of false positives and false negatives.
