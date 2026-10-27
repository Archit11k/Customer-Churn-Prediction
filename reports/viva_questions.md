# Viva Preparation - Customer Churn Prediction (CSE3141)

Answers are short enough to say aloud. Numbers come from this project's actual run (seed 42).

---

## 1. Predictive Analytics and the problem

**Q1. What is predictive analytics?**
Using historical data and statistical / machine-learning models to estimate unknown or future outcomes - here,
whether a customer will churn.

**Q2. What is customer churn?**
A customer stopping their service. In this dataset `Churn = Yes` means the customer left within the last month.

**Q3. Why does churn matter to a business?**
Lost customers mean lost recurring revenue, and replacing them is generally costlier than retaining them.
Predicting churn lets the company focus retention effort on high-risk customers.

**Q4. Is this classification or regression? Why?**
Classification - the target is a category (churn / no churn). Regression predicts a continuous number
(e.g. next month's bill). Logistic *Regression* is still a classifier: it models the probability of a class.

**Q5. What is the target variable and how is it encoded?**
`Churn`, mapped Yes → 1, No → 0. Churn is the positive class.

## 2. Dataset

**Q6. Describe the dataset.**
IBM Telco Customer Churn sample data: 7,043 customers, 21 columns - demographics, account information
(tenure, contract, billing, charges), subscribed services, and `Churn`. It is IBM sample (fictional) data.

**Q7. Why did you drop `customerID`?**
It's a unique label with no behavioural meaning. A model could memorise it (overfitting) and it cannot help
predict for a new customer whose ID was never seen.

**Q8. What is the churn rate?**
26.5 % (1,869 of 7,043).

## 3. Preprocessing

**Q9. Were there missing values?**
`isna()` showed none, but 11 `TotalCharges` values were blank strings `' '` - hidden missing values.

**Q10. How did you handle them and why not use the mean?**
All 11 had tenure = 0 - new customers not yet billed - so 0 is the correct value. The mean (≈ 2,280) would invent
charges they never paid.

**Q11. What inconsistent values did you fix?**
`TotalCharges` stored as text → number; `SeniorCitizen` 0/1 → No/Yes like other binary columns;
"No internet service"/"No phone service" → "No", because that information is already in
`InternetService`/`PhoneService` and would create duplicate (collinear) one-hot columns.

**Q12. Duplicates?**
No duplicate rows or IDs. 22 rows are identical once the ID is removed, but they are different customers with the
same profile, so they were kept.

**Q13. How did you detect outliers?**
IQR rule: values below Q1 − 1.5·IQR or above Q3 + 1.5·IQR, plus box plots. None of the numeric features had
outliers, so nothing was removed. `TotalCharges` is skewed but the values are genuine.

**Q14. What transformations did you apply?**
Standardisation (z-score) of numeric features and one-hot encoding (drop first level) of categorical features,
inside a `ColumnTransformer`.

**Q15. Why one-hot instead of label encoding?**
Label encoding (0, 1, 2…) implies an order and distance between categories that doesn't exist (e.g. payment
methods). One-hot creates a separate 0/1 column per category.

**Q16. Why drop the first one-hot category?**
To avoid redundant columns (the dummy-variable trap) and give Logistic Regression a reference category, so
each coefficient reads "compared with the reference".

**Q17. Why scale features?**
Logistic Regression (with regularisation) and LDA are sensitive to feature scale. Trees aren't, but using the
same pipeline for all is simpler and harmless.

## 4. EDA and descriptive statistics

**Q18. Give the mean, median and mode of tenure. What do they say?**
Mean 32.4, median 29, mode 1 month. Many customers are brand new; the distribution has peaks at both ends.

**Q19. Why is TotalCharges' mean much larger than its median?**
It is right-skewed (skewness ≈ 0.96): a few long-tenure customers have very large totals.

**Q20. Variance vs standard deviation?**
Variance is the average squared deviation from the mean; SD is its square root, in the original units
(tenure SD ≈ 24.6 months).

**Q21. What are the main EDA findings?**
Month-to-month contracts churn 42.7 % vs 2.8 % for two-year; first-year customers 47.4 % vs 9.5 % after
four years; fiber optic 41.9 % vs DSL 19.0 %; electronic check 45.3 %; no online security / tech support ≈ 31 %
vs ≈ 15 %. Gender shows no difference (26.9 % vs 26.2 %).

**Q22. Which statistical tests did you use?**
Chi-square test of independence for categorical features vs churn (with Cramér's V for strength), and
Mann-Whitney U for numeric features (non-parametric, since they aren't normal). Gender and PhoneService were
not significant (p > 0.05).

**Q23. What did the correlation heatmap show?**
Tenure (−0.35) and contract length (−0.39) are most negatively related to churn. TotalCharges is almost exactly
tenure × MonthlyCharges (r = 0.9996).

## 5. Feature engineering and selection

**Q24. Did you create derived features?**
I tested `NumAddOns` (count of add-on services) with cross-validation. It gave no improvement (CV ROC-AUC 0.8459
both with and without) and is an exact sum of existing columns, so I didn't keep it. Tenure groups and charge
quartiles were used only for EDA.

**Q25. Did you remove TotalCharges since it's redundant?**
I tested it: dropping it changed CV ROC-AUC from 0.8459 to 0.8447 - within noise (SD ≈ 0.012). With no evidence
either way I kept the original 19 features.

**Q26. What is target leakage and how did you avoid it?**
Leakage is when a feature contains information about the target that wouldn't be available at prediction
time, giving unrealistically good results. No feature here is derived from churn. A possible caveat is that
tenure/TotalCharges might have been recorded after churn - undocumented, so listed as a limitation.

## 6. Models

**Q27. Explain Logistic Regression.**
It models log(p / (1 − p)) = b0 + b1x1 + … and converts it to a probability with the sigmoid function. Predicts
churn if p ≥ 0.5. Coefficients are learned by maximising likelihood (with L1/L2 regularisation).

**Q28. What is an odds ratio? Give an example from your model.**
OR = e^coefficient: the multiplicative change in churn odds, holding other features constant. Two-year contract
OR ≈ 0.23 → about 77 % lower odds than month-to-month. Electronic check OR ≈ 1.49 → 49 % higher odds than bank
transfer. Tenure OR ≈ 0.30 per +1 SD (≈ 24.6 months).

**Q29. Why does MonthlyCharges have OR ≈ 0.09 if high charges are linked to churn?**
Multicollinearity: MonthlyCharges is almost fully determined by the services (R² ≈ 0.999). The model puts a large
positive weight on fiber/streaming and a negative weight on MonthlyCharges; they offset each other. These
coefficients must be read together, not individually.

**Q30. Explain a Decision Tree.**
It repeatedly splits data on the feature/threshold that most reduces impurity (Gini or entropy) until a stopping
rule. Easy to interpret, but an unconstrained tree memorises the training data.

**Q31. Gini vs entropy?**
Both measure node impurity. Gini = 1 − Σp²; entropy = −Σp·log₂p. They usually give similar trees; tuning chose
entropy here.

**Q32. Explain Random Forest.**
An ensemble of decision trees, each trained on a bootstrap sample and considering a random subset of features at
each split. Predictions are averaged. This bagging reduces variance compared with a single tree.

**Q33. Explain LDA and its assumptions.**
LDA assumes each class is Gaussian with a shared covariance matrix, then uses Bayes' rule, giving a linear
decision boundary. Our one-hot features aren't Gaussian, so the assumption is violated - but it still performed
well (CV ROC-AUC 0.844).

**Q34. Why didn't you use QDA?**
QDA estimates a separate covariance matrix per class. With 20 binary one-hot columns, some constant within groups,
those matrices are near-singular and the normality assumption is clearly broken.

**Q35. Which model did you select and why?**
Tuned Random Forest - highest mean 5-fold CV ROC-AUC (0.8476). But Logistic Regression (0.8462) and LDA (0.8440)
are within one CV standard deviation, so the difference isn't meaningful. LR would be an equally good choice if
interpretability matters most.

## 7. Validation

**Q36. Why split into train and test sets?**
To estimate performance on unseen data. The test set (20 %, 1,409 customers) was used only once, at the end.

**Q37. What is stratification?**
Keeping the class ratio identical in each split. Both train and test have ≈ 26.5 % churn.

**Q38. What is k-fold cross-validation?**
Split training data into k folds; train on k − 1 and validate on the remaining one, k times; average the scores.
I used stratified 5-fold. It gives a more stable estimate than a single validation split and also a spread (SD).

**Q39. Why use a Pipeline and ColumnTransformer?**
So preprocessing is learned only from training folds. If I scaled the full dataset first, validation/test
statistics would leak into training. The pipeline also guarantees the dashboard applies exactly the same
transformations.

**Q40. What is data leakage in model selection?**
Using the test set to choose features, models or hyperparameters. Then the test score is optimistic. I selected
everything using CV on the training set only.

## 8. Class imbalance

**Q41. How did you handle class imbalance?**
`class_weight="balanced"` for LR, DT and RF (errors on the minority class are weighted more); equal priors for
LDA. No synthetic resampling was needed.

**Q42. Why is accuracy misleading here?**
A model that always predicts "No churn" scores 73.5 % accuracy and catches zero churners. The baseline Random Forest
had the highest accuracy (78.5 %) but missed half the churners (recall 0.49).

**Q43. What would SMOTE do, and why didn't you use it?**
SMOTE creates synthetic minority samples. Class weighting achieves a similar effect without inventing data and is
simpler. SMOTE would need to be applied inside CV folds only, to avoid leakage.

## 9. Metrics and confusion matrix

**Q44. Define precision, recall and F1.**
Precision = TP / (TP + FP): of flagged customers, how many really churn. Recall = TP / (TP + FN): of churners,
how many we caught. F1 = harmonic mean of the two.

**Q45. Which matters more for churn - precision or recall?**
Usually recall: missing a churner loses the customer, while a false alarm costs only a retention offer. The best
balance depends on the actual costs.

**Q46. What is ROC-AUC?**
The area under the curve of true-positive rate vs false-positive rate at all thresholds. It equals the probability
that a random churner gets a higher score than a random non-churner. 0.5 = random, 1.0 = perfect. Our model: 0.845.

**Q47. Read your confusion matrix.**
Tuned Random Forest on test: TN 767, FP 268, FN 79, TP 295. It catches 295 of 374 churners (recall 0.789); of 563
flagged customers, 295 churn (precision 0.524).

**Q48. How could you increase recall further?**
Lower the decision threshold below 0.5 - more churners caught, but more false positives. The threshold should be
chosen on validation data (not the test set) using business costs.

## 10. Hyperparameter tuning

**Q49. What is a hyperparameter?**
A setting chosen before training (e.g. tree depth, C in LR), unlike parameters learned from data (coefficients,
split thresholds).

**Q50. GridSearchCV vs RandomizedSearchCV?**
Grid tries every combination; randomized samples a fixed number. I used grid for LR/DT/LDA (small spaces) and
randomized with 30 candidates for Random Forest (larger space, slower model).

**Q51. What were the tuning results?**
Decision Tree: CV ROC-AUC 0.661 → 0.834 (max_depth 6, min_samples_leaf 50). Random Forest: 0.828 → 0.848
(max_depth 6). LR and LDA barely changed (they weren't overfitting).

**Q52. Explain overfitting and underfitting with your results.**
Overfitting: the default tree had training ROC-AUC 1.00 but CV 0.66 - it memorised noise. Underfitting: a depth-1
tree scores 0.65 on both training and CV - too simple. The validation curve peaks at depth ≈ 5 (CV 0.824).

**Q53. Bias–variance trade-off?**
Simple models have high bias (underfit); complex models have high variance (overfit). Limiting tree depth and
averaging trees (Random Forest) lowers variance.

## 11. Results and limitations

**Q54. Summarise your final results.**
Tuned Random Forest on the test set: accuracy 0.754, precision 0.524, recall 0.789, F1 0.630, ROC-AUC 0.845.
All tuned models had test ROC-AUC 0.836-0.845.

**Q55. Are the test results close to CV results? Why does that matter?**
Yes (RF: 0.848 CV vs 0.845 test). It shows the CV estimate was reliable and selection didn't overfit.

**Q56. What are the project's limitations?**
Sample data; single snapshot with no time dimension; possible temporal leakage in tenure/TotalCharges; association
isn't causation; default 0.5 threshold without cost data; probabilities inflated by class weighting (not calibrated);
multicollinearity hampers coefficient interpretation.

**Q57. What would you do next?**
Threshold tuning with real costs, probability calibration, gradient boosting, SHAP explanations, time-based data,
and A/B testing of retention actions.

## 12. Retention recommendations

**Q58. What retention actions do you suggest?**
Early-tenure onboarding, contract-upgrade incentives, reviewing the fiber-optic experience, security/support
bundles, auto-pay nudges, and risk-ranked outreach using model probabilities.

**Q59. Can you claim these will reduce churn?**
No. The data shows associations only. Example: two-year customers churn less, but they may have chosen that
contract because they were already loyal (self-selection). Actions must be tested with a controlled experiment.

**Q60. How does the dashboard make predictions?**
It loads the saved pipeline (`models/churn_model.joblib`) once, validates the form input against the allowed
categories and ranges, builds a one-row DataFrame with exactly the 19 training features, and calls
`predict_proba`. No retraining happens in the dashboard.
