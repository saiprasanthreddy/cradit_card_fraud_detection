# FraudShield v3 — Credit Card Fraud Detection System

> A production-grade, ML-powered fraud detection web application built with a 3-model calibrated ensemble (XGBoost + Random Forest + LightGBM), a recall-optimized decision threshold, and a user-friendly Flask + HTML/CSS/JS interface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Performance Results](#2-performance-results)
3. [Project Structure](#3-project-structure)
4. [Quick Start — Run the Project](#4-quick-start--run-the-project)
5. [Dataset](#5-dataset)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning Pipeline](#7-machine-learning-pipeline)
8. [Algorithms Explained](#8-algorithms-explained)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Imbalance Handling](#10-imbalance-handling)
11. [Probability Calibration](#11-probability-calibration)
12. [Threshold Tuning](#12-threshold-tuning)
13. [Soft Voting Ensemble](#13-soft-voting-ensemble)
14. [Backend API](#14-backend-api)
15. [Frontend UI](#15-frontend-ui)
16. [Saved Artifacts](#16-saved-artifacts)
17. [API Reference](#17-api-reference)
18. [Deployment Guide](#18-deployment-guide)
19. [Design Decisions & Why](#19-design-decisions--why)
20. [Dependencies](#20-dependencies)

---

## 1. Project Overview

FraudShield v3 is a complete end-to-end fraud detection system trained on the Kaggle ULB Credit Card Fraud dataset (284,807 transactions, 492 frauds — 0.17% fraud rate). The system detects whether a given credit card transaction is fraudulent or legitimate using a three-model ensemble with calibrated probabilities and a recall-optimized decision threshold.

The entire project is divided into three layers:

| Layer | Technology | Purpose |
|---|---|---|
| Model Training | Python, XGBoost, RF, LightGBM | Train ensemble, save artifacts |
| Backend API | Python, Flask | Load models, serve predictions |
| Frontend UI | HTML, CSS, JavaScript | Accept user input, display results |

**What makes this system production-ready:**

- No PCA features required from the user — the system looks up the nearest real transaction internally
- 9 engineered features on top of the base dataset
- 3-model weighted ensemble with isotonic probability calibration
- Recall-optimized threshold (not the default 0.5) to minimize missed frauds
- Risk level classification (LOW / MEDIUM / HIGH) with plain-English explanations

---

## 2. Performance Results

These are the actual results obtained after training on the real Kaggle dataset:

| Metric | Value |
|---|---|
| Recall | **0.86** (86% of all frauds caught) |
| Precision | ~0.95 |
| F1-Score | ~0.90 |
| ROC-AUC | ~0.98 |
| PR-AUC | ~0.88 (primary metric) |

> **Why Recall is the most important metric here:**
> In fraud detection, a False Negative (missing a real fraud) is far more costly than a False Positive (flagging a legitimate transaction). A missed fraud means the customer loses money with no chance of recovery. A false alarm is inconvenient but reversible. The entire threshold and tuning strategy is designed to maximize recall first.

> **On the 86% Recall:** The threshold was optimized using a recall-weighted scoring formula `(2 × recall + precision) / 3`, which biases the model toward catching more fraud. Any further increase in recall would reduce precision significantly. 86% recall on a real-world dataset with 0.17% fraud rate is a strong result.

---

## 3. Project Structure

```
fraudshield/
│
├── model/                          ← All training code and saved artifacts
│   ├── model.py                    ← Main ML training pipeline (run this first)
│   ├── creditcard.csv              ← Kaggle dataset (download separately)
│   
│
├── backend/
│   └── app.py                      ← Flask REST API server
│
├── frontend/
│   ├── index.html                  ← Main UI page
│   ├── style.css                   ← All styles (dark industrial theme)
│   └── script.js                   ← API calls, result rendering, UI logic
│
├── requirements.txt                ← All Python dependencies
└── README.md                       ← This file
```

> **Important:** The `creditcard.csv` file must be placed inside the `model/` folder before running `model.py`. The `.pkl` files are generated automatically after training and must not be moved — `app.py` expects them in the `model/` folder.

---

## 4. Quick Start — Run the Project

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset

1. Go to: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it inside the `model/` folder

### Step 3 — Train the model

```bash
cd model
python model.py
```

Expected training time: **15–20 minutes** on a standard laptop (8-core CPU, 8GB RAM).

You will see progress printed for each step. When complete, 9 `.pkl` files and 3 plots are generated inside the `model/` folder.

### Step 4 — Start the Flask backend

```bash
cd backend
python app.py
```

The API starts at: `http://localhost:5000`

Verify it is running by visiting `http://localhost:5000` in your browser — you should see a JSON health-check response.

### Step 5 — Open the frontend

Open `frontend/index.html` directly in your browser. No server needed for the frontend.

> **Note:** If you get a CORS error, open the file through a local server:
> ```bash
> cd frontend
> python -m http.server 5500
> ```
> Then visit `http://localhost:5500`

---

## 5. Dataset

**Source:** [Kaggle ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Legitimate transactions | 284,315 (99.83%) |
| Class imbalance ratio | ~577:1 |

**Columns:**

- `Time` — Seconds elapsed since the first transaction in the dataset
- `Amount` — Transaction amount in EUR
- `V1` through `V28` — 28 PCA-transformed features (original features are confidential; the bank applied PCA before releasing the data)
- `Class` — Target variable: `0` = Legitimate, `1` = Fraud

**Why V1–V28?** The real features (merchant ID, cardholder name, location, etc.) are confidential. The bank applied Principal Component Analysis (PCA) to transform and anonymize the original features. The result is 28 numerical components that capture the underlying patterns while hiding the raw data.

---

## 6. Feature Engineering

The raw dataset only gives `Time`, `Amount`, and `V1–V28`. Before training, 9 additional features are engineered from `Time` and `Amount`. These features provide stronger signals that the raw values do not directly express.

### Time-Based Features

**`hour`** — Hour of day extracted from `Time`:
```python
df["hour"] = (df["Time"] % 86400) / 3600
```
Fraud has a higher rate during late-night hours (midnight to 6 AM). This feature makes that pattern visible to the model.

---

**`hour_sin` and `hour_cos`** — Cyclic encoding of the hour:
```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```
**Why not use raw `hour` alone?**
Hour 23 (11 PM) and Hour 0 (midnight) are actually very close in time, but numerically they are at opposite ends of a 0–23 scale. A linear model or tree split would never know they are adjacent. Sine and cosine encoding maps the hour onto a circle so that 23 and 0 are numerically close. This improves how the model handles time boundaries.

---

**`is_night`** — Binary flag for high-risk hours:
```python
df["is_night"] = (df["hour"] < 6).astype(int)
```
Transactions between midnight and 6 AM have a measurably higher fraud rate in this dataset. This flag gives the model a direct signal for that window without it needing to learn the boundary from scratch.

### Amount-Based Features

**`amount_log`** — Log-transformed transaction amount:
```python
df["amount_log"] = np.log1p(df["Amount"])
```
Transaction amounts follow an extreme right-skewed distribution — most transactions are small (under $100) but a few go into thousands. Raw amounts give a distorted picture. Log transformation compresses the scale so the model can learn patterns across the full range of amounts equally.

---

**`amount_zscore`** — Standardized amount (how unusual is this amount?):
```python
df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
```
A very high z-score means this transaction amount is unusually large compared to the average. This is a direct anomaly signal — large unusual amounts are a known fraud indicator.

### PCA Aggregate Features

**`v_mean`** — Average value across all 28 PCA components:
```python
df["v_mean"] = df[v_cols].mean(axis=1)
```

**`v_std`** — Standard deviation across all 28 PCA components:
```python
df["v_std"] = df[v_cols].std(axis=1)
```

**`high_risk_v_count`** — Number of PCA components with absolute value greater than 2:
```python
df["high_risk_v_count"] = (df[v_cols].abs() > 2.0).sum(axis=1)
```
Legitimate transactions tend to have PCA projections that are moderate and consistent. Fraudulent transactions often have multiple extreme projections simultaneously (several V features with |value| > 2σ at once). `high_risk_v_count` captures this anomaly density directly — it is one of the most informative engineered features.

---

## 7. Machine Learning Pipeline

The complete training flow, step by step:

```
creditcard.csv
      │
      ▼
Feature Engineering (9 new features)
      │
      ▼
Scale Amount + Time (StandardScaler)
      │
      ▼
Train / Test Split  ─────────────────────────────► Test Set (20%)
      │
      ▼
Train / Calibration Split ──────────────────────► Calibration Set (15% of train)
      │
      ▼
Compute scale_pos_weight = neg/pos ≈ 577
      │
      ├──► XGBoost (RandomizedSearchCV, 3-fold CV, 40 iter)
      │         └── Refit with early stopping → find best n_estimators
      │         └── Clean refit (no early_stopping)
      │         └── Isotonic Calibration on calibration set
      │
      ├──► Random Forest (RandomizedSearchCV, 3-fold CV, 20 iter)
      │
      └──► LightGBM (RandomizedSearchCV, 3-fold CV, 20 iter)
                │
                ▼
      Weighted Soft Voting Ensemble
      (weights = CV PR-AUC scores)
                │
                ▼
      Recall-Weighted Threshold Selection
      score = (2 × recall + precision) / 3
                │
                ▼
      Final Evaluation + Artifact Saving
```

---

## 8. Algorithms Explained

### XGBoost (eXtreme Gradient Boosting)

XGBoost is the primary model in the ensemble and is typically the highest-weighted member.

**How it works:** XGBoost builds an ensemble of decision trees sequentially. Each new tree focuses on correcting the mistakes of all previous trees combined. Specifically:

1. Start with a simple prediction (e.g., the mean of the target)
2. Compute residuals (errors) between predictions and actual labels
3. Fit a shallow decision tree to predict those residuals
4. Add this tree to the ensemble with a small learning rate (shrinkage)
5. Repeat for N rounds, each tree correcting the remaining errors

This process is called **gradient boosting** because each new tree is fit to the gradient of the loss function — the direction that reduces error the most.

**Why XGBoost specifically?**
- Built-in `scale_pos_weight` handles extreme class imbalance natively
- `tree_method='hist'` uses histogram-based splitting (much faster than exact splitting on 280k rows)
- Supports early stopping — automatically stops when validation performance stops improving
- L1 (`reg_alpha`) and L2 (`reg_lambda`) regularization prevent overfitting
- Consistently outperforms Random Forest on tabular, imbalanced datasets

**Key hyperparameters used:**

| Parameter | Role |
|---|---|
| `n_estimators` | Number of boosting rounds (trees). More = more capacity, but slower and risk of overfit |
| `max_depth` | Maximum depth of each tree. Deeper = more complex patterns, but higher overfit risk |
| `learning_rate` | Shrinkage applied to each tree. Lower = more trees needed but more stable |
| `subsample` | Fraction of training rows used per tree. Adds randomness, reduces variance |
| `colsample_bytree` | Fraction of features used per tree. Prevents any single feature from dominating |
| `min_child_weight` | Minimum sum of weights in a leaf node. Higher = more conservative splits |
| `gamma` | Minimum gain required to make a split. Regularizes tree structure |
| `reg_alpha` | L1 regularization — drives unimportant feature weights toward zero |
| `reg_lambda` | L2 regularization — keeps weights small and stable |
| `scale_pos_weight` | Weight multiplier for the minority class (fraud). Set to neg/pos ≈ 577 |
| `early_stopping_rounds` | Stop if validation PR-AUC does not improve for 50 consecutive rounds |

---

### Random Forest

Random Forest is the second ensemble member, providing diversity to the blend.

**How it works:** Random Forest builds many independent decision trees in parallel (not sequentially like XGBoost). Each tree is trained on a random bootstrap sample of the data, and at each split, only a random subset of features is considered. The final prediction averages the probabilities from all trees.

The two sources of randomness (row sampling and feature sampling) ensure each tree is different from the others. This diversity is exactly what makes the ensemble more robust than any single tree — individual trees overfit in different directions, and their errors cancel out when averaged.

**Why include it alongside XGBoost?**
Random Forest and XGBoost make their predictions through fundamentally different mechanisms (parallel independent trees vs. sequential corrective trees). This difference means their errors are not correlated — when XGBoost is wrong on a particular transaction, Random Forest is often right, and vice versa. Blending them reduces the overall error rate.

**Key hyperparameters:**

| Parameter | Role |
|---|---|
| `n_estimators` | Number of trees in the forest |
| `max_depth` | Maximum depth per tree |
| `max_features` | `"sqrt"` or `"log2"` — fraction of features per split |
| `min_samples_split` | Minimum samples required to split a node |
| `class_weight` | `"balanced_subsample"` — each bootstrap sample rebalances class weights automatically |

---

### LightGBM (Light Gradient Boosting Machine)

LightGBM is the third ensemble member and is typically the fastest of the three models to train.

**How it works:** LightGBM is also a gradient boosting framework, but uses two key optimizations over traditional gradient boosting:

1. **Leaf-wise (best-first) tree growth** instead of level-wise: grows the leaf with the highest gain at each step, producing deeper, more asymmetric trees that fit complex patterns better
2. **Histogram-based feature binning**: continuous feature values are grouped into discrete bins (e.g., 255 bins), reducing computation while retaining most information

These optimizations make LightGBM 5–10× faster than standard gradient boosting on large datasets while often matching or exceeding accuracy.

**Additional hyperparameters:**

| Parameter | Role |
|---|---|
| `num_leaves` | Controls tree complexity (31 = conservative, 127 = high complexity) |
| `min_child_samples` | Minimum samples per leaf — prevents overfitting on noisy small leaves |

---

## 9. Hyperparameter Tuning

### RandomizedSearchCV

All three models are tuned using `RandomizedSearchCV` rather than `GridSearchCV`.

**GridSearchCV** tests every single combination of hyperparameters. With the search spaces defined above, that would be thousands of combinations × 3-fold CV × training time per model = days of computation.

**RandomizedSearchCV** randomly samples a fixed number of combinations (`n_iter`) from the search space. Research has shown that random search finds near-optimal configurations in far fewer evaluations than grid search because most hyperparameters have limited interaction with each other — a few combinations dominate performance, and random sampling finds them efficiently.

| Model | n_iter | CV Folds | Scoring |
|---|---|---|---|
| XGBoost | 40 | 3 | average_precision (PR-AUC) |
| Random Forest | 20 | 3 | average_precision (PR-AUC) |
| LightGBM | 20 | 3 | average_precision (PR-AUC) |

**Why PR-AUC as the scoring metric and not accuracy?**

Accuracy is completely misleading on a 0.17% fraud dataset. A model that labels every single transaction as "Not Fraud" would achieve 99.83% accuracy while catching zero frauds. 

PR-AUC (Area Under the Precision-Recall Curve) measures how well the model distinguishes fraud from legitimate transactions across all possible thresholds. It is specifically designed for imbalanced classification and directly penalizes models that ignore the minority class.

### StratifiedKFold

Cross-validation uses `StratifiedKFold` with `n_splits=3`:

```python
cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
```

**Why stratified?** In standard K-Fold, folds are split randomly. With a 0.17% fraud rate, some folds could end up with very few or even zero fraud samples, making the CV score unreliable. Stratified K-Fold guarantees that each fold contains the same proportion of fraud as the full dataset. This gives stable, reliable CV estimates.

**Why 3 folds instead of 5?** 5-fold CV on 280k rows with 40 XGBoost iterations would take ~60 minutes. Reducing to 3 folds saves ~40% of training time with minimal impact on CV estimate quality on a dataset this large.

### Early Stopping (XGBoost)

```python
XGBClassifier(early_stopping_rounds=50, ...)
xgb_base.fit(..., eval_set=[(X_cal, y_cal)])
```

After hyperparameter search finds the best parameters, the model is refit with early stopping enabled. During training, after each boosting round, the model checks validation PR-AUC on `X_cal`. If the score does not improve for 50 consecutive rounds, training stops automatically and the best checkpoint is restored.

**Why this matters:** Without early stopping, you must guess `n_estimators` upfront. Too low and the model underfits. Too high and you waste computation or overfit. Early stopping finds the optimal number of trees automatically by monitoring actual performance on held-out data.

---

## 10. Imbalance Handling

### scale_pos_weight (XGBoost and LightGBM)

```python
spw = (y_train_fit == 0).sum() / (y_train_fit == 1).sum()
# spw ≈ 577
XGBClassifier(scale_pos_weight=spw, ...)
```

With 0.17% fraud, a model trained without any correction will always predict "Not Fraud" because that minimizes the loss function on the overwhelming majority class.

`scale_pos_weight` tells XGBoost to treat each fraud sample as if it were `spw` (≈577) legitimate samples. This makes the model pay 577× more attention to correctly classifying fraud. It is mathematically equivalent to SMOTE in terms of what it tells the model to focus on, but without creating synthetic data.

### Why scale_pos_weight instead of SMOTE?

| | SMOTE | scale_pos_weight |
|---|---|---|
| Method | Creates synthetic fraud samples | Adjusts loss function weights |
| Data leakage risk | Yes — synthetic samples near real fraud | None |
| Computation | Extra preprocessing step | Zero — one parameter |
| Artifacts | Can introduce noise | None |
| Performance | Competitive | Equal or better on XGBoost/LightGBM |

SMOTE was removed in v2 after analysis showed it added noise and complexity without improving PR-AUC. `scale_pos_weight` is the correct and cleaner solution for gradient boosting models.

### class_weight='balanced_subsample' (Random Forest)

Random Forest does not support `scale_pos_weight`. Instead, `class_weight='balanced_subsample'` is used, which recalculates class weights on each bootstrap sample separately. This is more effective than `'balanced'` (which uses global weights) because it adapts to the specific fraud/legit ratio in each tree's training data.

---

## 11. Probability Calibration

```python
best_xgb.set_params(early_stopping_rounds=None)
calibrated_xgb = CalibratedClassifierCV(
    estimator=best_xgb,
    method="isotonic",
    cv=3,
    ensemble=False,
)
calibrated_xgb.fit(X_cal, y_cal)
```

**The problem:** XGBoost with `scale_pos_weight` produces skewed probabilities. Because every fraud sample is treated as 577 legitimate samples during training, the model may output `P(fraud) = 0.95` for transactions where the true rate is only 70%. The raw probabilities are not reliable.

**Why this matters:** The decision threshold (e.g., "flag if P > 0.28") is meaningless if the probabilities themselves are wrong. A threshold of 0.28 means different things depending on whether the model is well-calibrated or not.

**Isotonic regression calibration** fits a monotone step function that remaps raw model probabilities to calibrated ones. After calibration, `P(fraud) = 0.8` genuinely means that among all transactions the model scores 0.8, approximately 80% are real fraud.

**Technical note on `ensemble=False`:** This parameter tells sklearn to use the already-trained `best_xgb` as-is and only fit the isotonic calibration layer on top — without refitting the base model. The model weights are unchanged. Only the probability mapping is adjusted. This is important because `best_xgb` was trained with `early_stopping_rounds`, and `ensemble=True` would try to refit the model internally without an `eval_set`, causing a crash.

---

## 12. Threshold Tuning

```python
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, ensemble_proba)

recall_weighted = (2 * recall_vals[:-1] + precision_vals[:-1]) / 3
best_idx       = np.argmax(recall_weighted)
best_threshold = float(thresholds[best_idx])
```

**The default threshold problem:** By default, classifiers predict "Fraud" if `P(fraud) ≥ 0.5`. On an imbalanced dataset, this is far too conservative — the model rarely reaches 0.5 for fraud cases because the training data is mostly legitimate. The previous version of this project had threshold 0.94, which caused it to miss 16% of all frauds.

**What the Precision-Recall curve is:** For every possible threshold value (0 to 1), you can compute precision and recall. The PR curve plots all these trade-off points. The optimal threshold lies somewhere on this curve depending on what you are willing to trade off.

**Why recall-weighted scoring instead of F1?**

F1 treats precision and recall as equally important:
```
F1 = (2 × Precision × Recall) / (Precision + Recall)
```

In fraud detection, missing a fraud (low recall) is more costly than a false alarm (lower precision). The recall-weighted formula reflects this:
```
score = (2 × Recall + Precision) / 3
```
This gives twice as much weight to recall as to precision. The threshold that maximizes this score will always have higher recall than the F1-optimal threshold, at the cost of some precision — which is exactly the desired trade-off.

---

## 13. Soft Voting Ensemble

```python
w_xgb  = xgb_search.best_score_    # XGB's CV PR-AUC
w_rf   = rf_search.best_score_     # RF's CV PR-AUC
w_lgbm = lgbm_search.best_score_   # LGBM's CV PR-AUC
w_total = w_xgb + w_rf + w_lgbm

ensemble_proba = (
    (w_xgb  / w_total) * xgb_cal_proba +
    (w_rf   / w_total) * rf_proba       +
    (w_lgbm / w_total) * lgbm_proba
)
```

**What soft voting is:** Instead of each model voting "Fraud" or "Not Fraud" (hard voting), each model outputs a fraud probability. These probabilities are blended with weights to produce a final probability.

**Why blend three different models?** Each algorithm makes errors in different ways. XGBoost tends to be overconfident near the class boundary; Random Forest is more conservative and stable. LightGBM learns complex patterns quickly. Because their errors are not perfectly correlated, averaging their outputs reduces the overall error rate — individual mistakes cancel out.

**Why weight by CV PR-AUC?** Models that performed better during cross-validation deserve more influence in the final blend. The weights are automatically computed from each model's CV score, so no manual tuning is needed.

**Expected weights** (approximate, varies per run):
```
XGBoost:       ~50% weight
LightGBM:      ~30% weight
Random Forest: ~20% weight
```

---

## 14. Backend API

`app.py` is a Flask REST API that:

1. Loads all 9 `.pkl` artifacts at startup
2. Pre-loads the `creditcard.csv` reference dataset into memory
3. For each prediction request:
   - Validates user inputs
   - Finds the nearest real transaction by amount and time (Euclidean distance)
   - Borrows its V1–V28 values as the PCA feature approximation
   - Engineers all 9 new features from amount and time
   - Scales amount and time with saved scalers
   - Assembles the 39-feature input vector in the exact training column order
   - Runs all 3 models and blends probabilities
   - Applies rule-based probability adjustments for context signals
   - Applies the saved threshold
   - Returns prediction, risk level, confidence, and reasons

**Rule-based adjustments (added on top of ML probability):**

| Signal | Boost |
|---|---|
| International transaction | +0.08 |
| Unusual spending behavior | +0.10 |
| Online / card-not-present | +0.06 |
| Transaction between midnight and 6 AM | +0.05 |
| Amount > $5,000 | +0.04 |

These adjustments encode domain knowledge that the ML model cannot see (the user's transaction context, location, behavioral pattern). They are conservative and additive, capped at 0.98.

---

## 15. Frontend UI

The UI collects five user-friendly inputs (no PCA features required):

| Input | Type | Options |
|---|---|---|
| Transaction Amount | Number | 0 – 50,000 |
| Transaction Type | Select button | POS / Store, Online, ATM |
| Location | Select button | Domestic, International |
| Spending Pattern | Toggle | Normal, Unusual |
| Time | Auto-detected | Override available via slider |

**Live risk preview bar:** As the user selects options, a contextual risk bar updates in real time (before any API call) based on the selected options. This gives immediate visual feedback.

**Result display:**
- Verdict banner (color-coded: green = safe, amber = medium, red = fraud)
- Animated probability bar with threshold marker
- Signal breakdown list (each reason listed as a bullet)
- Model base score vs. final adjusted score shown in footer

---

## 16. Saved Artifacts

After running `model.py`, the following files are saved in the `model/` folder:

| File | Description | Used by |
|---|---|---|
| `xgb.pkl` | Trained XGBoost classifier (no early stopping) | `app.py` indirectly via ensemble |
| `rf.pkl` | Trained Random Forest classifier | `app.py` |
| `lgbm.pkl` | Trained LightGBM classifier | `app.py` |
| `calibrated_model.pkl` | Isotonic-calibrated XGBoost | `app.py` |
| `ensemble_weights.pkl` | Dict of `{w_xgb, w_rf, w_lgbm}` | `app.py` |
| `amount_scaler.pkl` | StandardScaler fitted on Amount column | `app.py` |
| `time_scaler.pkl` | StandardScaler fitted on Time column | `app.py` |
| `threshold.pkl` | Single float — recall-weighted threshold | `app.py` |
| `feature_columns.pkl` | List of 39 column names in training order | `app.py` |

> **Critical:** `feature_columns.pkl` locks the exact column order used during training. `app.py` uses it to assemble the feature vector in the same order before every prediction. If training column order and inference order differ even slightly, the model receives wrong features and produces incorrect predictions silently.

---

## 17. API Reference

### GET /

Health check.

**Response:**
```json
{
  "status": "ok",
  "version": "v3",
  "model_loaded": true,
  "threshold": 0.2841,
  "features": 39,
  "ensemble": {
    "xgb_weight": 0.504,
    "rf_weight": 0.221,
    "lgbm_weight": 0.275
  }
}
```

### POST /predict

**Request body:**
```json
{
  "amount": 149.62,
  "time": 36000,
  "transaction_type": "online",
  "location": "domestic",
  "behavior": "usual"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `amount` | float | Yes | 0 – 50,000 |
| `time` | float | No | Seconds 0 – 172,800. Defaults to current time-of-day |
| `transaction_type` | string | No | `"online"`, `"pos"`, `"atm"` |
| `location` | string | No | `"domestic"`, `"international"` |
| `behavior` | string | No | `"usual"`, `"unusual"` |

**Response:**
```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.8124,
  "confidence": 81.2,
  "risk_level": "HIGH",
  "reasons": [
    "ML model scored this transaction at 73.2% fraud probability",
    "Online transaction — card-not-present risk",
    "Unusual spending behavior detected"
  ],
  "threshold_used": 0.2841,
  "model_base_score": 0.7320,
  "component_scores": {
    "ml_model_score": 0.7320,
    "transaction_type": 0.06,
    "behavior": 0.10
  }
}
```

**Risk levels:**

| Risk Level | Probability Range |
|---|---|
| LOW | < 0.40 |
| MEDIUM | 0.40 – 0.70 |
| HIGH | ≥ 0.70 |

---

## 18. Deployment Guide

### Backend — Render

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r ../requirements.txt`
   - **Start command:** `gunicorn app:app`
5. Add environment variable: `PORT=5000`
6. Deploy → copy the generated URL

### Frontend — Netlify

1. In `frontend/script.js`, update `API_BASE`:
   ```javascript
   const API_BASE = "https://your-render-url.onrender.com";
   ```
2. Go to [netlify.com](https://netlify.com) → New Site
3. Drag and drop the `frontend/` folder
4. Deploy

### CORS update for production

In `backend/app.py`, add your Netlify URL to the allowed origins:
```python
CORS(app, origins=[
    "http://localhost:5500",
    "https://your-site.netlify.app",   # add this
])
```

---

## 19. Design Decisions & Why

| Decision | Reason |
|---|---|
| XGBoost as primary model | Best-in-class on tabular imbalanced data. Native scale_pos_weight. Fast with hist method. |
| Removed SMOTE | Creates synthetic noise. scale_pos_weight achieves the same effect without data artifacts. |
| PR-AUC for tuning | Accuracy is misleading on 0.17% fraud rate. PR-AUC directly measures minority-class detection quality. |
| Recall-weighted threshold | Missing a fraud (FN) costs more than a false alarm (FP). Threshold biases toward catching more frauds. |
| Isotonic calibration | XGBoost probabilities are skewed with scale_pos_weight. Calibration makes probabilities reliable for threshold decisions. |
| 3-model ensemble | Diverse models make different errors. Blending reduces overall variance and improves recall stability. |
| 3-fold CV instead of 5-fold | 280k rows × 3 models × 40+20+20 iterations is already expensive. 3-fold is reliable at this scale and saves ~40% training time. |
| Nearest-neighbor V1–V28 lookup | Users cannot provide PCA features. The nearest real transaction provides a realistic V1–V28 approximation without requiring user input. |
| feature_columns.pkl | Locks column order at training time. Prevents silent feature mismatch at inference — a common production failure mode. |
| Rule-based probability adjustments | ML model has no access to transaction context (online vs ATM, domestic vs international). Rules encode domain knowledge the model cannot learn from the dataset. |

---

## 20. Dependencies

```
pandas          — Data loading and manipulation
numpy           — Numerical operations
scikit-learn    — Preprocessing, CV, calibration, metrics
xgboost         — XGBoost classifier
lightgbm        — LightGBM classifier
joblib          — Saving and loading model artifacts
matplotlib      — Plotting confusion matrix, PR curve, feature importance
seaborn         — Heatmap styling for confusion matrix
flask           — REST API server
flask-cors      — Cross-origin resource sharing for frontend-backend communication
gunicorn        — Production WSGI server for deployment
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Author

**Sai** — Final Year B.E. (AI & ML), Vemana Institute of Technology, Bengaluru

Built as a portfolio project demonstrating production-grade ML engineering: ensemble modeling, probability calibration, threshold optimization, imbalanced classification, and full-stack deployment.

---

*FraudShield v3 — XGBoost · Random Forest · LightGBM · PR-AUC Optimized · Kaggle ULB Dataset*
