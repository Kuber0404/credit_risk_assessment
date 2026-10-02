# Credit Risk Assessment

Binary classification on the Statlog (German Credit) dataset: predicting which
loan applicants turn out to be a bad credit risk. Seven algorithms are built
and evaluated against one shared preprocessing pipeline and one fixed
train/test split, so the comparison between them is apples-to-apples.

## Data

`data/german_credit_data.csv` — 1,000 loan applications, nine predictors
describing the applicant and the loan (age, sex, job skill level, housing,
savings and checking account bands, credit amount, duration, purpose), labelled
`good` or `bad`.

## Layout

```
data/        german_credit_data.csv
notebooks/   00 data exploration & preprocessing   05 svm
             01 logistic regression                06 random forest
             02 knn                                07 gradient boosting / xgboost
             03 naive bayes                        08 model comparison
             04 decision tree
```

Notebook `00` establishes the conventions; `01`-`07` each take one algorithm
through baseline, regularization, cross-validated tuning and threshold
analysis; `08` refits all seven and compares them on the same held-out set.

## Conventions shared by every notebook

- **Target:** `y = (Risk == "bad")` — 1 = bad credit risk is the positive
  class, at a ~30% base rate. Recall on that class is the metric a lender
  cares about, so accuracy is never read on its own.
- **Missing values:** blanks in `Saving accounts` / `Checking account` mean the
  applicant holds no such account — real information, not a missing
  measurement — so they become their own `"none"` level rather than being
  imputed. Notebook `00` tests and rejects the "too young to have an account"
  explanation before settling this.
- **Encoding:** `Job` kept ordinal (0-3); `Sex`, `Housing`, `Saving accounts`,
  `Checking account` and `Purpose` one-hot encoded with `drop_first=True`.
  9 raw predictors become 21 features.
- **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)`,
  identical in every notebook.
- **Scaling:** `StandardScaler` fit on the training set only, for the models
  that need it (Logistic Regression, KNN, SVM); skipped for the tree-based
  models, which are scale-invariant.
- **Tuning:** 5-fold cross-validation on the training set, scored on ROC-AUC.
  The test set takes no part in hyperparameter selection.

## Results

Test-set ROC-AUC on the 200 held-out applications:

| Model | Baseline | Tuned |
| --- | --- | --- |
| Logistic Regression | 0.757 | 0.754 |
| KNN | 0.709 | 0.723 |
| Naive Bayes | 0.708 | 0.734 |
| Decision Tree | 0.681 | 0.719 |
| SVM (RBF) | 0.758 | 0.734 |
| Random Forest | 0.754 | **0.781** |
| XGBoost | **0.787** | 0.772 |

Three things this dataset makes clear:

1. **No single feature separates the classes.** The strongest correlation with
   default is about 0.3, so ROC-AUC in the 0.7-0.8 range is the realistic
   ceiling here, not a disappointment.
2. **Tuning is not reliably an improvement at this sample size.** XGBoost and
   SVM both scored better at their defaults than after a cross-validated
   search — 5-fold CV on 800 rows cannot separate settings a few thousandths of
   ROC-AUC apart, so the search partly fits the folds.
3. **The decision threshold matters more than the algorithm.** At the default
   0.5 cutoff most of these models miss the majority of actual defaulters. The
   threshold sections in `01`-`08` examine that, with an explicit caveat: they
   select the cutoff on the test set, which is leakage. In production the
   cutoff belongs on a validation fold and is then settled in deployment by
   shadow-scoring live applications and adjusting against the realised default
   and approval rates.

## A note on `Sex` as a feature

Using a protected attribute as a model input is a legal and ethical problem,
not just a technical one. It is kept here because it is part of the historical
benchmark dataset, but the honest reading of any coefficient attached to it is
"this is what the historical data encodes", not "this is a valid basis for a
lending decision".

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebooks read `../data/german_credit_data.csv` and are run in order, though
`01`-`08` each repeat the preprocessing from `00` so they also run standalone.
