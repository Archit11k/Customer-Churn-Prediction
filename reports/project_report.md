# Customer Churn Prediction and Retention Analysis Using Machine Learning

**Course:** CSE3141 Predictive Analytics · B.Tech Computer Science · Manipal University Jaipur
**Semester:** July–November 2026 · **Assessment:** Predictive Modeling Mini Project + Viva
**Student:** _[your name, registration number]_ · **Supervisor:** _[faculty name]_

> All numbers in this report were produced by the code in this repository (`src/`, `notebooks/`) on the
> IBM Telco Customer Churn dataset (SHA-256 `16320c9c…5e91`), with `random_state = 42`. Re-running
> `python -m src.eda` and `python -m src.train_model` regenerates them. Tables are copied from
> `reports/eda_summary.json`, `reports/cv_results.csv`, `reports/model_comparison.csv` and
> `reports/feature_ablation.csv`.

---

## 1. Abstract

Customer churn - customers ending their service - directly reduces a telecom company's revenue. This project
analyses the IBM Telco Customer Churn dataset (7,043 customers, 26.5 % churn) to find patterns associated with
churn and to build classifiers that estimate a customer's churn risk. After cleaning (hidden blank values,
inconsistent labels) and exploratory analysis, four classifiers - Logistic Regression, Decision Tree, Random
Forest and Linear Discriminant Analysis - were trained inside scikit-learn pipelines, compared with 5-fold
stratified cross-validation and tuned with GridSearchCV/RandomizedSearchCV using training data only. The tuned
Random Forest had the highest cross-validated ROC-AUC (0.848) and achieved, on a held-out test set of 1,409
customers, ROC-AUC 0.845, recall 0.789, precision 0.524 and F1 0.630. Tuned Logistic Regression performed
almost identically (CV ROC-AUC 0.846) and is easier to interpret. Contract type, tenure, internet-service type,
payment method and absence of security/support add-ons were the strongest churn indicators. Retention
recommendations are derived from these associations and are flagged as requiring experimental validation.

## 2. Introduction

Acquiring a new telecom customer is generally considered more expensive than keeping an existing one, so
companies try to identify at-risk customers early and target them with retention actions. Predictive analytics
supports this by learning, from historical records, the relationship between customer attributes and the
churn outcome. This is a **binary classification** problem: the output is a category (churn / no churn) and a
probability, not a continuous quantity.

## 3. Problem Statement

Given a customer's demographic, account and service attributes, predict whether the customer is likely to
churn, explain which characteristics are associated with churn, and suggest evidence-informed retention actions.

## 4. Objectives

1. Inspect, clean and describe the dataset (data types, missing values, duplicates, inconsistencies, outliers).
2. Explore churn patterns with descriptive statistics, visualisations and statistical tests.
3. Select features and prevent data leakage.
4. Train and compare Logistic Regression, Decision Tree, Random Forest and LDA.
5. Validate with stratified train/test split and cross-validation; handle class imbalance.
6. Tune hyperparameters without touching the test set; analyse over- and underfitting.
7. Evaluate with accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
8. Interpret the models and derive retention recommendations.
9. Deliver a reproducible codebase, notebooks and a Streamlit dashboard using the saved model.

## 5. Dataset Source and Description

| Item | Detail |
|---|---|
| Name | IBM Telco Customer Churn (sample dataset) |
| File used | `data/raw/Telco-Customer-Churn.csv` |
| Obtained from | IBM GitHub repository `IBM/telco-customer-churn-on-icp4d` (`data/Telco-Customer-Churn.csv`) |
| Also available at | Kaggle, `blastchar/telco-customer-churn` (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) |
| Licence | The IBM GitHub repository is published under the Apache License 2.0. The Kaggle listing's licence terms should be checked on Kaggle before redistribution. **[verify before publishing]** |
| Nature | Fictional/sample data created by IBM for demonstrating analytics; not real customer records |
| Size | 7,043 rows × 21 columns, one row per customer |

**Columns**

| Group | Columns |
|---|---|
| Identifier | `customerID` (excluded from modelling) |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` - Yes if the customer left within the last month |

## 6. Methodology

```
raw CSV ─► deterministic cleaning ─► EDA & statistical tests
                  │
                  ▼
   stratified 80/20 split (seed 42) ──────────────► test set (locked away)
                  │
         training set (5,634)
                  │
   Pipeline[ColumnTransformer ► classifier]
                  │
   5-fold stratified CV: baseline models, feature ablation
                  │
   GridSearchCV / RandomizedSearchCV (CV on training set, scoring = ROC-AUC)
                  │
   select model with best mean CV ROC-AUC
                  │
   single evaluation of all models on test set ─► save selected pipeline (joblib)
                                                       │
                                               Streamlit dashboard
```

## 7. Data Preprocessing

| Issue found | Evidence | Action | Reason |
|---|---|---|---|
| `TotalCharges` stored as text | dtype `object` | Converted to float | Monetary value |
| Hidden missing values | 11 `TotalCharges` cells contain `' '`; `isna()` reports 0 | Set to 0 | All 11 have `tenure = 0` (new customers not yet billed); 0 is the logically correct value, mean/median would invent charges |
| Inconsistent coding | `SeniorCitizen` is 0/1, other binary columns are No/Yes | Recoded to No/Yes | Consistent categorical handling |
| Redundant labels | "No internet service" in 6 add-on columns; "No phone service" in `MultipleLines` | Mapped to "No" | Information already in `InternetService`/`PhoneService`; avoids perfectly collinear one-hot columns |
| Target as text | Yes/No | Mapped to 1/0 | Required by classifiers |
| Duplicates | 0 duplicate rows; 0 duplicate IDs; 22 rows identical once ID is removed | Nothing removed | Different customers with identical profiles are legitimate |
| Outliers | IQR rule finds 0 outliers in `tenure`, `MonthlyCharges`, `TotalCharges` | Nothing removed | Skewed but genuine values |

These rules use no statistics learned from the data, so applying them before the split does not leak.
Learned transformations are inside the model pipeline:

- **Numeric** (`tenure`, `MonthlyCharges`, `TotalCharges`): median imputation (safety net for new inputs) →
  `StandardScaler`.
- **Categorical** (16 columns): most-frequent imputation → `OneHotEncoder(drop="first", handle_unknown="ignore")`.
  This yields 23 model inputs (3 numeric + 20 one-hot).

## 8. Exploratory Data Analysis

### 8.1 Descriptive statistics

| Feature | Mean | Median | Mode | Variance | Std | Min | Max | Skewness |
|---|---|---|---|---|---|---|---|---|
| tenure (months) | 32.37 | 29.00 | 1 | 603.17 | 24.56 | 0 | 72 | 0.24 |
| MonthlyCharges | 64.76 | 70.35 | 20.05 | 905.41 | 30.09 | 18.25 | 118.75 | −0.22 |
| TotalCharges | 2279.73 | 1394.55 | 0 | 5,138,357 | 2266.79 | 0 | 8684.80 | 0.96 |

The mode of tenure is 1 month (many new customers). `TotalCharges` is clearly right-skewed (mean ≫ median).

### 8.2 Target distribution

5,174 customers did not churn and 1,869 churned (**26.54 %**). The classes are imbalanced (≈ 2.8 : 1).
A model that always predicts "No churn" would be 73.5 % accurate and detect no churners.

### 8.3 Churn by key attributes (computed churn rate)

| Attribute | Group | Customers | Churn rate |
|---|---|---|---|
| Contract | Month-to-month | 3,875 | 42.7 % |
| | One year | 1,473 | 11.3 % |
| | Two year | 1,695 | 2.8 % |
| Tenure | 0–12 months | 2,186 | 47.4 % |
| | 13–24 months | 1,024 | 28.7 % |
| | 25–48 months | 1,594 | 20.4 % |
| | 49–72 months | 2,239 | 9.5 % |
| Monthly charges (quartiles) | Q1 (18.25–35.50) | 1,762 | 11.2 % |
| | Q2 (35.50–70.35) | 1,766 | 24.6 % |
| | Q3 (70.35–89.85) | 1,757 | 37.5 % |
| | Q4 (89.85–118.75) | 1,758 | 32.9 % |
| Internet service | Fiber optic | 3,096 | 41.9 % |
| | DSL | 2,421 | 19.0 % |
| | No internet | 1,526 | 7.4 % |
| Payment method | Electronic check | 2,365 | 45.3 % |
| | Mailed check | 1,612 | 19.1 % |
| | Bank transfer (automatic) | 1,544 | 16.7 % |
| | Credit card (automatic) | 1,522 | 15.2 % |
| Online security | No / Yes | 5,024 / 2,019 | 31.3 % / 14.6 % |
| Tech support | No / Yes | 4,999 / 2,044 | 31.2 % / 15.2 % |
| Senior citizen | Yes / No | 1,142 / 5,901 | 41.7 % / 23.6 % |
| Paperless billing | Yes / No | 4,171 / 2,872 | 33.6 % / 16.3 % |
| Partner | No / Yes | 3,641 / 3,402 | 33.0 % / 19.7 % |
| Dependents | No / Yes | 4,933 / 2,110 | 31.3 % / 15.5 % |
| Gender | Female / Male | 3,488 / 3,555 | 26.9 % / 26.2 % |

Note: after cleaning, "No" for add-ons includes customers without internet service (who churn rarely).

**Tenure × contract** (churn %): month-to-month customers churn more than contract customers in *every* tenure
band (e.g. 51.4 % vs 10.5 % vs 0.0 % in the first year), so contract type carries information beyond tenure.

### 8.4 Numeric features by churn group

| Feature | Median (no churn) | Median (churn) | Mann-Whitney p |
|---|---|---|---|
| tenure | 38 | 10 | 2.4 × 10⁻²⁰⁸ |
| MonthlyCharges | 64.43 | 79.65 | 3.3 × 10⁻⁵⁴ |
| TotalCharges | 1679.53 | 703.55 | 5.7 × 10⁻⁸³ |

The non-parametric Mann-Whitney U test was used because the distributions are not normal.

### 8.5 Categorical associations (chi-square test of independence)

Strongest associations by Cramér's V: Contract (0.41), InternetService (0.32), PaymentMethod (0.30),
PaperlessBilling (0.19), OnlineSecurity (0.17), TechSupport (0.16), Dependents (0.16). All features have
p < 0.05 except **gender (p = 0.49)** and **PhoneService (p = 0.34)**.

### 8.6 Correlation

Correlation with churn: tenure −0.35, contract length (1/12/24 months) −0.39, TotalCharges −0.20,
MonthlyCharges +0.19. `TotalCharges` correlates 0.83 with tenure and **0.9996 with tenure × MonthlyCharges**,
i.e. it is almost entirely redundant.

Figures: `reports/figures/eda_*.png`.

## 9. Feature Engineering and Selection

- **Excluded `customerID`:** a unique label with no behavioural meaning; it could only be memorised (overfitting)
  and cannot generalise to new customers.
- **Leakage check:** no feature is computed from the churn outcome. Possible caveat: `tenure`/`TotalCharges` are
  snapshot values whose recording date relative to churn is not documented (see Limitations).
- **Ablation experiment** (Logistic Regression, 5-fold CV on the training set only):

| Feature set | # features | CV ROC-AUC | CV F1 |
|---|---|---|---|
| A: all 19 features | 19 | 0.8459 ± 0.0124 | 0.6298 |
| B: without TotalCharges | 18 | 0.8447 ± 0.0121 | 0.6288 |
| C: + NumAddOns (derived count of add-ons) | 20 | 0.8459 ± 0.0124 | 0.6296 |
| D: without gender & PhoneService | 17 | 0.8462 ± 0.0126 | 0.6288 |

All differences (≤ 0.0015) are much smaller than the fold-to-fold standard deviation (≈ 0.012). **Decision:**
keep the 19 original features. The derived feature added nothing and is an exact sum of existing one-hot
columns (perfect multicollinearity). Tenure groups and charge quartiles were used **only for EDA**.

## 10. Algorithms Used

| Model | How it works | Strengths | Limitations | Imbalance handling |
|---|---|---|---|---|
| Logistic Regression | Models log-odds of churn as a linear function of features; sigmoid gives probability | Interpretable odds ratios, probabilistic, fast, low variance | Linear decision boundary; coefficient interpretation suffers from multicollinearity | `class_weight="balanced"` |
| Decision Tree | Recursive binary splits that reduce impurity (Gini/entropy) | Human-readable rules, captures interactions, no scaling needed | High variance - overfits unless constrained | `class_weight="balanced"` |
| Random Forest | Many trees on bootstrap samples with random feature subsets; average votes (bagging) | Reduces variance of trees, robust, captures non-linearity | Less interpretable, heavier | `class_weight="balanced"` |
| LDA | Assumes Gaussian classes with shared covariance → linear boundary via Bayes' rule | Stable, few parameters | Normality assumption violated by binary one-hot features | `priors=[0.5, 0.5]` |
| Majority-class dummy | Always predicts the most common class | Reference point | Useless predictor | - |

**QDA was not used:** it estimates a separate covariance matrix per class, and with 20 binary one-hot columns
(several of which are constant within subgroups) these matrices are near-singular and the Gaussian assumption
is clearly violated.

## 11. Model Training and Validation

- **Split:** `train_test_split(test_size=0.2, stratify=y, random_state=42)` → 5,634 train / 1,409 test; churn
  rate 26.5 % in both.
- **Pipelines:** every model is a `Pipeline(ColumnTransformer → classifier)`, so scaling/encoding is re-fitted
  inside each CV fold - no information from validation folds or the test set leaks into preprocessing.
- **Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training set.

**Baseline 5-fold CV results (training set)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (CV) | ROC-AUC (train) |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| Logistic Regression | 0.7496 | 0.5182 | 0.8033 | 0.6298 | 0.8459 | 0.8494 |
| Decision Tree | 0.7373 | 0.5049 | 0.4990 | 0.5018 | 0.6611 | 1.0000 |
| Random Forest | 0.7865 | 0.6363 | 0.4589 | 0.5330 | 0.8275 | 0.9999 |
| LDA | 0.7614 | 0.5353 | 0.7652 | 0.6297 | 0.8390 | 0.8419 |

The unconstrained Decision Tree (train 1.00 vs CV 0.66) and Random Forest (train ≈ 1.00 vs CV 0.83) overfit.
LR and LDA have almost equal train and CV scores (no overfitting).

## 12. Evaluation Metrics

With churn as the positive class (TP = churner correctly flagged):

- **Accuracy** = (TP + TN) / N - misleading under imbalance (the dummy model scores 0.73).
- **Precision** = TP / (TP + FP) - share of flagged customers who really churn (cost of wasted offers).
- **Recall** = TP / (TP + FN) - share of churners caught (cost of lost customers).
- **F1** = 2·P·R / (P + R) - balance of precision and recall.
- **ROC-AUC** - probability that a random churner receives a higher score than a random non-churner;
  threshold-independent. Used as the **model-selection metric**.
- **Confusion matrix** - counts of TN, FP, FN, TP at threshold 0.5.

## 13. Hyperparameter Optimisation

All searches used 5-fold stratified CV **on the training set only**, `scoring="roc_auc"`, and refitted the best
configuration on the full training set.

| Model | Method | Search space | Candidates | Best parameters |
|---|---|---|---|---|
| Logistic Regression | GridSearchCV | C ∈ {0.001 … 100}, penalty ∈ {l1, l2} | 12 | C = 100, penalty = l2 |
| Decision Tree | GridSearchCV | max_depth ∈ {3,4,5,6,8,10,None}, min_samples_leaf ∈ {1,10,25,50,100}, criterion ∈ {gini, entropy} | 70 | entropy, max_depth = 6, min_samples_leaf = 50 |
| Random Forest | RandomizedSearchCV (n_iter = 30) | n_estimators, max_depth, min_samples_leaf, max_features | 30 | n_estimators = 300, max_depth = 6, min_samples_leaf = 1, max_features = sqrt |
| LDA | GridSearchCV | solver ∈ {svd, lsqr}, shrinkage ∈ {None, auto, 0.01, 0.1, 0.3, 0.5} | 7 | lsqr, no shrinkage |

**Over- and underfitting.** A validation curve for the Decision Tree (`model_validation_curve_tree_depth.png`)
shows: depth 1–3 → train and CV both low (**underfitting**; depth 1: 0.646 / 0.646); best CV at depth 5
(0.824); deeper trees → training score rises towards 1.0 while CV falls (**overfitting**; depth 11: 0.952 / 0.747).
Tuning fixed exactly this: Decision Tree CV ROC-AUC rose from 0.661 to 0.834; Random Forest from 0.828 to 0.848.

## 14. Model Comparison Results (held-out test set, 1,409 customers)

Model selection used only the CV column; test numbers were computed once afterwards.

| Model | Stage | CV ROC-AUC (mean ± std) | Train ROC-AUC | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Majority-class baseline | baseline | 0.5000 ± 0.0000 | 0.5000 | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | baseline | 0.8459 ± 0.0124 | 0.8494 | 0.7402 | 0.5069 | 0.7861 | 0.6164 | 0.8419 |
| Decision Tree | baseline | 0.6611 ± 0.0141 | 1.0000 | 0.7346 | 0.5000 | 0.4813 | 0.4905 | 0.6544 |
| Random Forest | baseline | 0.8275 ± 0.0126 | 0.9999 | 0.7850 | 0.6195 | 0.4920 | 0.5484 | 0.8262 |
| LDA | baseline | 0.8390 ± 0.0115 | 0.8419 | 0.7480 | 0.5176 | 0.7460 | 0.6112 | 0.8297 |
| Logistic Regression | tuned | 0.8462 ± 0.0128 | 0.8498 | 0.7381 | 0.5043 | 0.7807 | 0.6128 | 0.8404 |
| Decision Tree | tuned | 0.8339 ± 0.0116 | 0.8554 | 0.7495 | 0.5183 | 0.7968 | 0.6280 | 0.8358 |
| **Random Forest** | **tuned (selected)** | **0.8476 ± 0.0112** | 0.8748 | **0.7537** | **0.5240** | 0.7888 | **0.6297** | **0.8454** |
| LDA | tuned | 0.8440 ± 0.0111 | 0.8469 | 0.7395 | 0.5060 | 0.7914 | 0.6173 | 0.8356 |

**Confusion matrix - tuned Random Forest (test set)**

| | Predicted: No churn | Predicted: Churn |
|---|---|---|
| **Actual: No churn** | 767 (TN) | 268 (FP) |
| **Actual: Churn** | 79 (FN) | 295 (TP) |

**Logistic Regression odds ratios (tuned model, selected examples)** - OR > 1 raises churn odds vs. the
reference category, holding other features constant; numeric features are per +1 standard deviation.

| Feature | Odds ratio | Reading |
|---|---|---|
| Contract: Two year (vs month-to-month) | 0.23 | ≈ 77 % lower odds |
| Contract: One year (vs month-to-month) | 0.48 | ≈ 52 % lower odds |
| tenure (+1 SD ≈ 24.6 months) | 0.30 | ≈ 70 % lower odds |
| Payment: Electronic check (vs bank transfer) | 1.49 | ≈ 49 % higher odds |
| Paperless billing: Yes | 1.39 | ≈ 39 % higher odds |
| Dependents: Yes | 0.79 | ≈ 21 % lower odds |
| InternetService: Fiber optic (vs DSL) | 15.7 | see caution below |
| MonthlyCharges (+1 SD ≈ $30.1) | 0.09 | see caution below |

*Caution:* `MonthlyCharges` is almost fully determined by the services subscribed (R² = 0.9988 when regressed on
the service columns). The model therefore splits weight between `MonthlyCharges` and the service indicators
(fiber, streaming, phone), giving large opposite-signed coefficients that must be interpreted **jointly**, not
individually. Full table: `reports/logistic_odds_ratios.csv`.

**Permutation importance (selected Random Forest, test ROC-AUC drop):** InternetService 0.051, Contract 0.051,
tenure 0.033, TotalCharges 0.013, PaymentMethod 0.005; gender and DeviceProtection ≈ 0.

## 15. Discussion

- **Accuracy is misleading.** The dummy model has 73.5 % test accuracy, close to the tuned models (73.8–75.4 %),
  yet catches zero churners. The baseline Random Forest has the *highest* accuracy (78.5 %) but the *lowest*
  recall among real models (0.49) - it misses half the churners.
- **Class weighting shifts the trade-off** toward recall (≈ 0.78–0.80 for tuned models) at the cost of precision
  (≈ 0.50–0.52). For retention, a missed churner is usually costlier than an unnecessary offer, but the right
  threshold depends on real costs that are not available in this dataset.
- **Model choice.** The tuned Random Forest, Logistic Regression and LDA are within 0.004 CV ROC-AUC of each
  other, smaller than one CV standard deviation (≈ 0.011); the tuned Decision Tree is ≈ 0.014 lower. Random Forest was selected by the pre-defined rule, but **Logistic Regression is an equally
  defensible choice** that is more interpretable. The data's main signal (contract, tenure, internet type) is
  largely captured by a linear model.
- **Generalisation.** Test scores are close to CV scores (e.g. RF 0.848 CV vs 0.845 test), indicating the CV
  estimate was reliable and the test set was not overfit by model selection.
- **Consistency.** EDA, chi-square tests, odds ratios and permutation importance agree on the main drivers.

## 16. Customer Retention Recommendations

**Data-driven findings (associations in this dataset):** churn is highest for month-to-month contracts
(42.7 %), first-year customers (47.4 %), electronic-check payers (45.3 %), fiber-optic customers (41.9 %), and
customers without online security (31.3 %) or tech support (31.2 %).

**General recommendations (not validated - require controlled experiments):**

1. **Early-tenure onboarding** for customers in their first 12 months.
2. **Contract-upgrade incentives** for month-to-month customers - noting possible self-selection: customers who
   choose long contracts may already be more committed, so the causal effect of switching is unknown.
3. **Review the fiber-optic experience** (price perception, reliability, competition) - high churn despite
   premium service.
4. **Trial security/tech-support bundles**, recognising the association may reflect customer type.
5. **Encourage automatic payment**, as electronic-check payment may be a marker of lower engagement.
6. **Risk-ranked outreach** using model probabilities, with a control group to measure actual impact.

No claim is made that these actions will reduce churn; this dataset contains no information about interventions.

## 17. Limitations

1. **Sample / fictional data** from IBM - results may not transfer to any real telecom operator.
2. **Single snapshot;** no time dimension, so churn timing and changes in behaviour cannot be modelled.
3. **Possible temporal leakage:** it is not documented whether `tenure`/`TotalCharges` were recorded before or
   after the churn event. If after, they partly encode the outcome. **[verify with the data owner]**
4. **Association ≠ causation:** recommendations need A/B testing.
5. **Default threshold 0.5;** no business cost matrix was available to optimise the threshold.
6. **Probabilities are not calibrated:** class weighting inflates predicted churn probabilities relative to the
   true 26.5 % base rate; they are best used for ranking.
7. **Multicollinearity** (MonthlyCharges vs services; TotalCharges vs tenure × MonthlyCharges) limits
   individual coefficient interpretation in Logistic Regression.
8. **Moderate dataset size** (7,043 rows); results vary by about ±0.01 ROC-AUC across folds.
9. Only one random split was used for the final test; other seeds would give slightly different test numbers.

## 18. Conclusion

A reproducible, leakage-free classification workflow was built for telecom churn. After cleaning and EDA,
four classifiers were compared with stratified cross-validation and tuned on training data only. All tuned models
achieved test ROC-AUC between 0.836 and 0.845 and recall around 0.78–0.80, far better than the majority-class
baseline that accuracy alone would have flattered. The selected Random Forest (test ROC-AUC 0.845, F1 0.630)
is deployed in a Streamlit dashboard via a saved scikit-learn pipeline. Contract type, tenure, internet service,
payment method and support add-ons emerged consistently as the strongest churn indicators, providing a basis for
retention strategies that should be validated experimentally.

## 19. References

1. IBM. *Telco Customer Churn* sample dataset. GitHub repository `IBM/telco-customer-churn-on-icp4d`,
   https://github.com/IBM/telco-customer-churn-on-icp4d (accessed 27 Oct 2026).
2. Kaggle listing: *Telco Customer Churn* (blastchar), https://www.kaggle.com/datasets/blastchar/telco-customer-churn.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. scikit-learn developers. *scikit-learn User Guide*, version 1.6, https://scikit-learn.org/1.6/user_guide.html.
5. James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning*. Springer
   (chapters on classification, resampling methods and tree-based methods).
6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
7. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*.
8. Course handout, CSE3141 Predictive Analytics, Manipal University Jaipur. **[add exact handout details]**

_Check edition/year details of reference 5 against the copy used in your course._
