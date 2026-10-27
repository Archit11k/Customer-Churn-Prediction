# Customer Churn Prediction and Retention Analysis Using Machine Learning

Mini-project for **CSE3141 Predictive Analytics** (B.Tech CSE, Manipal University Jaipur, Jul–Nov 2026).

Analyses the IBM Telco Customer Churn dataset, builds and compares four classifiers (Logistic Regression,
Decision Tree, Random Forest, LDA) with cross-validation and hyperparameter tuning, and serves the selected
model in a Streamlit dashboard.

## Results at a glance

Held-out test set (1,409 customers, 20 % stratified split, seed 42). Model selected by 5-fold CV ROC-AUC
on the training set only.

| Model (tuned) | CV ROC-AUC | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest (selected)** | 0.8476 | 0.7537 | 0.5240 | 0.7888 | 0.6297 | 0.8454 |
| Logistic Regression | 0.8462 | 0.7381 | 0.5043 | 0.7807 | 0.6128 | 0.8404 |
| LDA | 0.8440 | 0.7395 | 0.5060 | 0.7914 | 0.6173 | 0.8356 |
| Decision Tree | 0.8339 | 0.7495 | 0.5183 | 0.7968 | 0.6280 | 0.8358 |
| Majority-class baseline | 0.5000 | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

Full tables: `reports/model_comparison.csv`. Discussion: `reports/project_report.md`.

## Project structure

```
Customer-Churn-Prediction/
├── app.py                       Streamlit dashboard (5 pages)
├── data/
│   ├── raw/                     Telco-Customer-Churn.csv (+ README with source & checksum)
│   └── processed/               telco_churn_clean.csv (generated)
├── notebooks/
│   ├── 01_data_analysis.ipynb   cleaning, descriptive statistics, EDA
│   └── 02_model_training.ipynb  features, models, CV, tuning, evaluation, interpretation
├── src/
│   ├── config.py                paths, seed, feature lists, allowed input values
│   ├── data_preprocessing.py    loading, cleaning, ColumnTransformer
│   ├── eda.py                   statistics, tests and EDA figures
│   ├── train_model.py           split, CV, ablation, tuning, selection, saving
│   ├── evaluate_model.py        metrics, plots, odds ratios, importance, validation curve
│   └── predict.py               load saved pipeline, validate input, predict
├── models/
│   ├── churn_model.joblib       selected pipeline (preprocessing + model)
│   └── model_metadata.json      features, params, metrics, dataset hash
├── reports/
│   ├── figures/                 all generated plots
│   ├── model_comparison.csv     baseline vs tuned, CV + test metrics
│   ├── cv_results.csv, tuning_results.csv, feature_ablation.csv, ...
│   ├── project_report.md
│   └── viva_questions.md
├── tests/                       pytest suite (pipeline + dashboard smoke tests)
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python 3.9 or newer (tested on 3.9.6, macOS).

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

### Dataset

The CSV is already in `data/raw/`. If it is missing, download it from IBM's repository
(https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data) or Kaggle
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as
`data/raw/Telco-Customer-Churn.csv`. See `data/raw/README.md` for licence notes and the checksum.

## Running

Run all commands from the project root.

```bash
python -m src.eda              # EDA statistics + figures (~10 s)
```
```bash
python -m src.train_model      # CV, tuning, evaluation, saves model (~1 min)
```
```bash
python -m src.predict          # example prediction with the saved model
```
```bash
streamlit run app.py           # dashboard at http://localhost:8501
```
```bash
python -m pytest -q            # tests
```
```bash
jupyter notebook notebooks/    # open the notebooks
```

Re-running training with the same data and seed reproduces the same numbers.

## Method summary

1. **Cleaning** - 11 hidden blank `TotalCharges` (tenure 0 → set to 0); consistent Yes/No coding; redundant
   "No internet/phone service" labels collapsed; no duplicates; no IQR outliers.
2. **EDA** - descriptive statistics, churn rates by contract/tenure/charges/services, chi-square and
   Mann-Whitney tests, correlation heatmap.
3. **Features** - 19 features (customerID excluded); ablation via CV showed no benefit from dropping
   `TotalCharges` or adding a derived add-on count.
4. **Models** - pipelines (`ColumnTransformer` + classifier), class imbalance handled with balanced class weights
   (equal priors for LDA), stratified 5-fold CV.
5. **Tuning** - GridSearchCV (LR, DT, LDA) and RandomizedSearchCV (RF) on training data, scoring ROC-AUC.
6. **Evaluation** - single test-set evaluation; confusion matrices, ROC/PR curves, odds ratios,
   permutation importance, validation curve.

## Dashboard pages

1. **Project Overview** - objective, dataset summary, customer count, churn rate.
2. **Exploratory Data Analysis** - interactive churn-rate charts, distributions, scatter, tests.
3. **Model Evaluation** - comparison table, selected model, confusion matrix, ROC/PR, importance, odds ratios.
4. **Churn Prediction** - form → churn / no churn with probability (uses the saved pipeline; never retrains).
5. **Retention Recommendations** - computed findings, clearly separated from general recommendations.

## Known limitations

IBM sample (fictional) data; single snapshot; possible undocumented timing of `tenure`/`TotalCharges` relative to
churn; associations are not causes; default 0.5 threshold; class weighting inflates probabilities (use for
ranking). Details in `reports/project_report.md` §17.

## Academic integrity

All reported numbers are generated by this code. The dashboard reads every score from files written by
`src/train_model.py` and `src/eda.py`; nothing is hard-coded. The tables in `project_report.md` and this README
were copied from those files - if you re-run with different data or settings, update them.
