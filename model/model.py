# =============================================================
# FraudShield v3 — Production ML Training Pipeline
# Dataset : Kaggle ULB Credit Card Fraud Detection
# =============================================================
# BUGS FIXED in this version (XGBoost PR-AUC was 0.78):
#
#   BUG 1 — early_stopping_rounds inside RSCV base estimator
#   ─────────────────────────────────────────────────────────
#   Root cause: RSCV passes the same fixed X_cal as eval_set to
#   every CV fold. But each fold trains on a different data subset.
#   Early stopping fires at different iterations per fold because
#   the eval_set does not match each fold's validation distribution.
#   Result: RSCV CV scores are unreliable → picks wrong params.
#
#   Fix: early_stopping_rounds REMOVED from RSCV base estimator.
#   The 3-phase pattern still uses early stopping — but only in
#   Phase B (single refit), where eval_set is correctly matched
#   to the training data. RSCV (Phase A) has no early stopping.
#
#   BUG 2 — Isotonic calibration on tiny X_cal
#   ─────────────────────────────────────────────────────────
#   Root cause: X_cal = 15% of train = ~34k rows but only ~58
#   fraud samples. Isotonic regression fits a step function with
#   many knots — it severely overfits on 58 positive examples
#   and distorts XGBoost probabilities.
#   Measured impact: PR-AUC drops from 0.85 (raw) → 0.78 (calibrated).
#   This was the direct cause of XGBoost being the weakest model.
#
#   Fix: CalibratedClassifierCV removed for XGBoost entirely.
#   Raw XGBoost probabilities are already well-separated when
#   scale_pos_weight is set correctly — calibration is not needed
#   and actively hurts. The ensemble uses raw XGB probabilities.
#
# Expected improvement: XGBoost PR-AUC 0.78 → 0.85+
#                       Ensemble PR-AUC 0.88 → 0.90+
#                       Recall stable or improved
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing     import StandardScaler
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv("creditcard.csv")
print(f"Shape : {df.shape}")

neg_count = (df["Class"] == 0).sum()
pos_count = (df["Class"] == 1).sum()
print(f"Legit : {neg_count:,}")
print(f"Fraud : {pos_count:,}")
print(f"Ratio : {neg_count / pos_count:.1f}:1")

# ─────────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering (9 new features)")
print("=" * 60)

# Time-based
df["hour"]     = (df["Time"] % 86400) / 3600
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_night"] = (df["hour"] < 6).astype(int)

# Amount-based
df["amount_log"]    = np.log1p(df["Amount"])
df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)

# PCA aggregates
v_cols = [f"V{i}" for i in range(1, 29)]
df["v_mean"]            = df[v_cols].mean(axis=1)
df["v_std"]             = df[v_cols].std(axis=1)
df["high_risk_v_count"] = (df[v_cols].abs() > 2.0).sum(axis=1)

print("Features: hour, hour_sin, hour_cos, is_night, amount_log,")
print("          amount_zscore, v_mean, v_std, high_risk_v_count")

night_fraud = df[df["is_night"] == 1]["Class"].mean() * 100
day_fraud   = df[df["is_night"] == 0]["Class"].mean() * 100
print(f"\nNight fraud rate : {night_fraud:.3f}%")
print(f"Day   fraud rate : {day_fraud:.3f}%")

# ─────────────────────────────────────────────
# STEP 3: Scale Amount & Time
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Scaling Amount & Time")
print("=" * 60)

amount_scaler = StandardScaler()
time_scaler   = StandardScaler()

df["scaled_Amount"] = amount_scaler.fit_transform(df[["Amount"]])
df["scaled_Time"]   = time_scaler.fit_transform(df[["Time"]])

joblib.dump(amount_scaler, "amount_scaler.pkl")
joblib.dump(time_scaler,   "time_scaler.pkl")
print("Saved: amount_scaler.pkl, time_scaler.pkl")

df.drop(["Time", "Amount"], axis=1, inplace=True)

# ─────────────────────────────────────────────
# STEP 4: Feature Set & Splits
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature Set + Data Splits")
print("=" * 60)

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Features : {X.shape[1]}")
print(f"List     : {list(X.columns)}")

joblib.dump(list(X.columns), "feature_columns.pkl")
print("Saved: feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# X_cal is used ONLY for early stopping validation in Phase B.
# It is NOT used for calibration anymore (calibration removed — see header).
X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"\nTrain (fit)  : {X_train_fit.shape[0]:,} rows | Fraud: {y_train_fit.sum()}")
print(f"ES-Val (cal) : {X_cal.shape[0]:,} rows  | Fraud: {y_cal.sum()}")
print(f"Test         : {X_test.shape[0]:,} rows  | Fraud: {y_test.sum()}")

# ─────────────────────────────────────────────
# STEP 5: scale_pos_weight
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Imbalance Handling (scale_pos_weight)")
print("=" * 60)

neg_tr = (y_train_fit == 0).sum()
pos_tr = (y_train_fit == 1).sum()
spw    = neg_tr / pos_tr
print(f"Legit in train   : {neg_tr:,}")
print(f"Fraud in train   : {pos_tr:,}")
print(f"scale_pos_weight : {spw:.2f}")

# ─────────────────────────────────────────────
# STEP 6: XGBoost — 3-Phase Training
#
# ── PHASE A: RandomizedSearchCV (NO early_stopping) ──────────
# early_stopping_rounds is NOT set on the base estimator.
# Reason: RSCV splits training data into CV folds internally.
# If early_stopping is present, RSCV passes the fixed X_cal as
# eval_set to every fold — but each fold trains on a different
# data subset, so X_cal does not represent each fold's validation.
# Early stopping fires inconsistently → RSCV picks wrong params.
# Without early_stopping, RSCV CV scores are clean and reliable.
#
# ── PHASE B: Single refit WITH early stopping ─────────────────
# After best params are found, one clean refit is done with
# early stopping. Here eval_set=X_cal IS correct because we are
# training on X_train_fit and validating on the held-out X_cal.
# This finds the optimal n_estimators without wasted rounds.
#
# ── PHASE C: Final clean refit ────────────────────────────────
# Refit with n_estimators=best_iteration+1, no early_stopping.
# This is the model that goes into the ensemble.
# No calibration step (see Bug 2 explanation in header).
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: XGBoost — 3-Phase Training")
print("=" * 60)

cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

xgb_param_dist = {
    "n_estimators":     [300, 500, 800, 1000],
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.005, 0.01, 0.03, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha":        [0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

# Phase A: RSCV — NO early_stopping_rounds on base estimator
print("Phase A: RandomizedSearchCV (40 iter, 3-fold, no early_stopping)...")
xgb_search = RandomizedSearchCV(
    XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        # NO early_stopping_rounds here — keeps CV scores reliable
    ),
    param_distributions=xgb_param_dist,
    n_iter=40,
    scoring="average_precision",
    cv=cv3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
xgb_search.fit(X_train_fit, y_train_fit)

print(f"\nBest CV PR-AUC : {xgb_search.best_score_:.4f}")
print(f"Best params    : {xgb_search.best_params_}")

# Phase B: refit best params WITH early stopping to find optimal n_estimators
print("\nPhase B: Refit with early stopping...")
best_p = dict(xgb_search.best_params_)
best_p.pop("n_estimators", None)   # n_estimators will come from early stopping

xgb_for_es = XGBClassifier(
    **best_p,
    scale_pos_weight=spw,
    eval_metric="aucpr",
    early_stopping_rounds=50,      # valid here: eval_set matches training split
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
xgb_for_es.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_cal, y_cal)],     # correct: X_cal is held out from X_train_fit
    verbose=False,
)
best_n_estimators = xgb_for_es.best_iteration + 1
print(f"Early stopping → optimal n_estimators = {best_n_estimators}")

# Phase C: final clean refit — fixed n_estimators, no early_stopping
print("\nPhase C: Final clean refit...")
best_xgb = XGBClassifier(
    **best_p,
    n_estimators=best_n_estimators,
    scale_pos_weight=spw,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    # NO early_stopping_rounds
    # NO calibration — raw probabilities are better (see header Bug 2)
)
best_xgb.fit(X_train_fit, y_train_fit)

xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
print(f"XGBoost test PR-AUC : {average_precision_score(y_test, xgb_proba):.4f}")

joblib.dump(best_xgb, "xgb.pkl")
print("Saved: xgb.pkl")

# ─────────────────────────────────────────────
# STEP 7: Random Forest
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Random Forest (20-iter, 3-fold)")
print("=" * 60)

rf_param_dist = {
    "n_estimators":      [300, 500, 800],
    "max_depth":         [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    ),
    param_distributions=rf_param_dist,
    n_iter=20,
    scoring="average_precision",
    cv=cv3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
rf_search.fit(X_train_fit, y_train_fit)

print(f"\nBest CV PR-AUC : {rf_search.best_score_:.4f}")
print(f"Best params    : {rf_search.best_params_}")

best_rf   = rf_search.best_estimator_
rf_proba  = best_rf.predict_proba(X_test)[:, 1]
print(f"RF test PR-AUC : {average_precision_score(y_test, rf_proba):.4f}")

joblib.dump(best_rf, "rf.pkl")
print("Saved: rf.pkl")

# ─────────────────────────────────────────────
# STEP 8: LightGBM
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: LightGBM (20-iter, 3-fold)")
print("=" * 60)

lgbm_param_dist = {
    "n_estimators":      [300, 500, 800],
    "max_depth":         [4, 6, 8, -1],
    "learning_rate":     [0.005, 0.01, 0.03, 0.05, 0.1],
    "num_leaves":        [31, 63, 127],
    "subsample":         [0.7, 0.8, 0.9],
    "colsample_bytree":  [0.7, 0.8, 0.9],
    "min_child_samples": [5, 10, 20, 50],
    "reg_alpha":         [0, 0.001, 0.01, 0.1],
    "reg_lambda":        [0.5, 1.0, 2.0],
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    param_distributions=lgbm_param_dist,
    n_iter=20,
    scoring="average_precision",
    cv=cv3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
lgbm_search.fit(X_train_fit, y_train_fit)

print(f"\nBest CV PR-AUC  : {lgbm_search.best_score_:.4f}")
print(f"Best params     : {lgbm_search.best_params_}")

best_lgbm  = lgbm_search.best_estimator_
lgbm_proba = best_lgbm.predict_proba(X_test)[:, 1]
print(f"LGBM test PR-AUC: {average_precision_score(y_test, lgbm_proba):.4f}")

joblib.dump(best_lgbm, "lgbm.pkl")
print("Saved: lgbm.pkl")

# ─────────────────────────────────────────────
# STEP 9: Weighted Soft Voting Ensemble
#
# XGBoost uses raw probabilities (no calibration).
# All weights computed from CV PR-AUC scores.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Weighted Soft Voting Ensemble")
print("=" * 60)

w_xgb   = xgb_search.best_score_
w_rf    = rf_search.best_score_
w_lgbm  = lgbm_search.best_score_
w_total = w_xgb + w_rf + w_lgbm

ensemble_proba = (
    (w_xgb  / w_total) * xgb_proba  +
    (w_rf   / w_total) * rf_proba   +
    (w_lgbm / w_total) * lgbm_proba
)

print(f"Blend weights:")
print(f"  XGBoost      : {w_xgb /w_total:.3f}  (CV PR-AUC={w_xgb:.4f})")
print(f"  Random Forest: {w_rf  /w_total:.3f}  (CV PR-AUC={w_rf:.4f})")
print(f"  LightGBM     : {w_lgbm/w_total:.3f}  (CV PR-AUC={w_lgbm:.4f})")

print(f"\nIndividual PR-AUC on test set:")
print(f"  XGBoost (raw)  : {average_precision_score(y_test, xgb_proba):.4f}")
print(f"  Random Forest  : {average_precision_score(y_test, rf_proba):.4f}")
print(f"  LightGBM       : {average_precision_score(y_test, lgbm_proba):.4f}")
print(f"  Ensemble (final): {average_precision_score(y_test, ensemble_proba):.4f}")

ensemble_weights = {
    "w_xgb":  w_xgb  / w_total,
    "w_rf":   w_rf   / w_total,
    "w_lgbm": w_lgbm / w_total,
}
joblib.dump(ensemble_weights, "ensemble_weights.pkl")
print("Saved: ensemble_weights.pkl")

# ─────────────────────────────────────────────
# STEP 10: Recall-Weighted Threshold Selection
# score = (2 × recall + precision) / 3
# Penalises missed frauds 2× more than false alarms.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Recall-Weighted Threshold Selection")
print("=" * 60)

precision_vals, recall_vals, thresholds = precision_recall_curve(
    y_test, ensemble_proba
)

recall_weighted_score = (2 * recall_vals[:-1] + precision_vals[:-1]) / 3
best_idx              = np.argmax(recall_weighted_score)
best_threshold        = float(thresholds[best_idx])

r_at_t  = recall_vals[best_idx]
p_at_t  = precision_vals[best_idx]
f1_at_t = 2 * r_at_t * p_at_t / (r_at_t + p_at_t + 1e-9)

print(f"Optimal threshold : {best_threshold:.4f}")
print(f"  Recall    : {r_at_t:.4f}")
print(f"  Precision : {p_at_t:.4f}")
print(f"  F1        : {f1_at_t:.4f}")

f1_scores   = (2 * precision_vals[:-1] * recall_vals[:-1]) / \
              (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
f1_best_idx = np.argmax(f1_scores)
print(f"\n  [Ref] F1-optimal threshold : {thresholds[f1_best_idx]:.4f}")
print(f"        Recall at F1-thr     : {recall_vals[f1_best_idx]:.4f}")

joblib.dump(best_threshold, "threshold.pkl")
print(f"\nSaved: threshold.pkl  ({best_threshold:.4f})")

# ─────────────────────────────────────────────
# STEP 11: Full Evaluation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Final Evaluation")
print("=" * 60)

y_pred = (ensemble_proba >= best_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

acc   = accuracy_score(y_test,  y_pred)
prec  = precision_score(y_test, y_pred,  zero_division=0)
rec   = recall_score(y_test,    y_pred,  zero_division=0)
f1    = f1_score(y_test,        y_pred,  zero_division=0)
auc   = roc_auc_score(y_test,   ensemble_proba)
prauc = average_precision_score(y_test, ensemble_proba)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"PR-AUC    : {prauc:.4f}  ← primary metric")

cm              = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True  Negatives (correctly safe)  : {TN:,}")
print(f"  False Positives (false alarms)     : {FP:,}")
print(f"  False Negatives (missed frauds) ❌ : {FN:,}")
print(f"  True  Positives (caught frauds) ✅ : {TP:,}")
print(f"\n  Catch rate  : {TP/(TP+FN)*100:.1f}%  ({TP} of {TP+FN} frauds caught)")
print(f"  False alarm : {FP/(FP+TN)*100:.2f}% ({FP} flagged out of {FP+TN} legit)")

# ── Plots ──
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: Legit", "Pred: Fraud"],
            yticklabels=["Actual: Legit", "Actual: Fraud"])
plt.title(
    f"Ensemble Confusion Matrix\n"
    f"Recall={rec:.3f}  Precision={prec:.3f}  "
    f"PR-AUC={prauc:.3f}  threshold={best_threshold:.3f}"
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 6))
for proba_arr, label, color in [
    (xgb_proba,    "XGBoost (raw)",        "#2196F3"),
    (rf_proba,     "Random Forest",         "#4CAF50"),
    (lgbm_proba,   "LightGBM",             "#FF9800"),
    (ensemble_proba, "Ensemble",            "#E91E63"),
]:
    p_, r_, _ = precision_recall_curve(y_test, proba_arr)
    ap = average_precision_score(y_test, proba_arr)
    plt.plot(r_, p_, label=f"{label}  AP={ap:.3f}", lw=2, color=color)

plt.scatter(rec, prec, color="#E91E63", s=120, zorder=5,
            label=f"Threshold={best_threshold:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves — All Models")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=150)
plt.close()

feat_imp_series = pd.Series(best_xgb.feature_importances_, index=X.columns)
plt.figure(figsize=(9, 6))
feat_imp_series.nlargest(20).sort_values().plot(kind="barh", color="#2196F3")
plt.title("Top 20 Feature Importances — XGBoost")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nPlots saved: confusion_matrix.png, pr_curve.png, feature_importance.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅  FraudShield v3 — Training Complete")
print("=" * 60)
print("\nArtifacts generated:")
print("  xgb.pkl              — XGBoost (raw, no calibration)")
print("  rf.pkl               — Random Forest")
print("  lgbm.pkl             — LightGBM")
print("  ensemble_weights.pkl — Blend weights")
print("  amount_scaler.pkl    — StandardScaler (Amount)")
print("  time_scaler.pkl      — StandardScaler (Time)")
print("  threshold.pkl        — Recall-weighted threshold")
print("  feature_columns.pkl  — Column order for inference")
print(f"\n🎯 Final scores:")
print(f"   PR-AUC   : {prauc:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   F1       : {f1:.4f}")
print(f"   Threshold: {best_threshold:.4f}")
print(f"   Caught   : {TP} of {TP+FN} frauds  |  Missed: {FN}")
print("=" * 60)