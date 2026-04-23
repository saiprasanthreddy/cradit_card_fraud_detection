# =============================================================
# FraudShield v3 — Flask API Backend
# =============================================================
# Updated: loads xgb.pkl directly (calibrated_model.pkl removed)
# Calibration removed — raw XGB probabilities are stronger
# =============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import time as _time

app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "null",
])

# ─────────────────────────────────────────────
# 1. Load artifacts
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

def _load(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        obj = joblib.load(path)
        print(f"  ✅ {filename}")
        return obj
    except FileNotFoundError:
        print(f"  ❌ {filename} — not found (run model.py first)")
        return None
    except Exception as e:
        print(f"  ❌ {filename} — {type(e).__name__}: {e}")
        return None

print("\nLoading artifacts...")
xgb_model        = _load("xgb.pkl")          # raw XGB — no calibration
rf_model         = _load("rf.pkl")
lgbm_model       = _load("lgbm.pkl")
amount_scaler    = _load("amount_scaler.pkl")
time_scaler      = _load("time_scaler.pkl")
threshold        = _load("threshold.pkl")
feature_columns  = _load("feature_columns.pkl")
ensemble_weights = _load("ensemble_weights.pkl")

if ensemble_weights is None:
    ensemble_weights = {"w_xgb": 0.50, "w_rf": 0.25, "w_lgbm": 0.25}
    print("  ⚠  Using equal weights (ensemble_weights.pkl missing)")

all_loaded = all([
    xgb_model is not None, rf_model is not None, lgbm_model is not None,
    amount_scaler is not None, time_scaler is not None,
    threshold is not None, feature_columns is not None,
])

thr_val = f"{float(threshold):.4f}" if threshold is not None else "missing"
print(f"\n{'✅ Ready' if all_loaded else '❌ Some artifacts missing'}")
print(f"   Threshold : {thr_val}")
print(f"   Features  : {len(feature_columns) if feature_columns else '?'}")
print(f"   Weights   : XGB={ensemble_weights['w_xgb']:.3f} "
      f"RF={ensemble_weights['w_rf']:.3f} "
      f"LGBM={ensemble_weights['w_lgbm']:.3f}\n")

# ─────────────────────────────────────────────
# 2. Reference dataset — V1–V28 nearest-neighbour lookup
# ─────────────────────────────────────────────
DATASET_PATH  = os.path.join(MODEL_DIR, "creditcard.csv")
_reference_df = None

def get_reference_df():
    global _reference_df
    if _reference_df is None:
        try:
            cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            ref  = pd.read_csv(DATASET_PATH, usecols=cols)
            ref.attrs["a_min"]   = ref["Amount"].min()
            ref.attrs["a_range"] = ref["Amount"].max() - ref["Amount"].min() + 1e-9
            ref.attrs["t_min"]   = ref["Time"].min()
            ref.attrs["t_range"] = ref["Time"].max() - ref["Time"].min() + 1e-9
            _reference_df = ref
            print(f"✅ Reference dataset: {len(ref):,} rows")
        except Exception as e:
            print(f"❌ Reference dataset unavailable: {e}")
            _reference_df = pd.DataFrame()
    return _reference_df


def lookup_v_features(amount: float, time_seconds: float) -> np.ndarray:
    ref = get_reference_df()
    if ref.empty:
        return np.zeros(28)
    a_min = ref.attrs["a_min"];   a_range = ref.attrs["a_range"]
    t_min = ref.attrs["t_min"];   t_range = ref.attrs["t_range"]
    norm_a = (ref["Amount"] - a_min) / a_range
    norm_t = (ref["Time"]   - t_min) / t_range
    q_a    = (amount        - a_min) / a_range
    q_t    = (time_seconds  - t_min) / t_range
    dist    = np.sqrt(2.0 * (norm_a - q_a) ** 2 + 1.0 * (norm_t - q_t) ** 2)
    best_ix = dist.idxmin()
    v_cols = [f"V{i}" for i in range(1, 29)]
    return ref.loc[best_ix, v_cols].values.astype(float)


# ─────────────────────────────────────────────
# 3. Feature vector — must match model.py column order exactly
# ─────────────────────────────────────────────
def build_feature_vector(amount: float,
                          time_seconds: float,
                          v_features: np.ndarray) -> np.ndarray:
    scaled_amount     = float(amount_scaler.transform([[amount]])[0][0])
    scaled_time       = float(time_scaler.transform([[time_seconds]])[0][0])
    hour              = (time_seconds % 86400) / 3600
    hour_sin          = float(np.sin(2 * np.pi * hour / 24))
    hour_cos          = float(np.cos(2 * np.pi * hour / 24))
    is_night          = 1 if hour < 6 else 0
    amount_log        = float(np.log1p(amount))
    amount_zscore     = scaled_amount
    v_mean            = float(np.mean(v_features))
    v_std             = float(np.std(v_features))
    high_risk_v_count = int((np.abs(v_features) > 2.0).sum())

    feature_dict = {f"V{i+1}": v_features[i] for i in range(28)}
    feature_dict.update({
        "scaled_Amount": scaled_amount, "scaled_Time": scaled_time,
        "hour": hour, "hour_sin": hour_sin, "hour_cos": hour_cos,
        "is_night": is_night, "amount_log": amount_log,
        "amount_zscore": amount_zscore, "v_mean": v_mean,
        "v_std": v_std, "high_risk_v_count": high_risk_v_count,
    })
    return np.array([feature_dict[col] for col in feature_columns]).reshape(1, -1)


# ─────────────────────────────────────────────
# 4. Ensemble inference — raw XGB + RF + LGBM
# ─────────────────────────────────────────────
def ensemble_predict_proba(X_input: np.ndarray) -> float:
    xgb_p  = float(xgb_model.predict_proba(X_input)[0][1])
    rf_p   = float(rf_model.predict_proba(X_input)[0][1])
    lgbm_p = float(lgbm_model.predict_proba(X_input)[0][1])
    w = ensemble_weights
    blended = w["w_xgb"]*xgb_p + w["w_rf"]*rf_p + w["w_lgbm"]*lgbm_p
    return float(np.clip(blended, 0.0, 1.0))


# ─────────────────────────────────────────────
# 5. Rule-based risk adjustments
# ─────────────────────────────────────────────
RISK_RULES = [
    ("location",         "international", 0.08,
     "International transaction — elevated fraud risk"),
    ("behavior",         "unusual",       0.10,
     "Spending behavior flagged as unusual"),
    ("transaction_type", "online",        0.06,
     "Card-not-present online transaction — higher fraud rate"),
    ("_is_night",        True,            0.05,
     "Transaction between midnight and 6 AM"),
]

def apply_rule_adjustments(base_proba, user_inputs, is_night):
    adjusted = base_proba
    reasons  = []
    scores   = {"ml_model_score": round(base_proba, 4)}
    ctx = dict(user_inputs, _is_night=is_night)
    for key, value, boost, reason in RISK_RULES:
        if ctx.get(key) == value:
            adjusted += boost
            reasons.append(reason)
            scores[key] = boost
    if user_inputs.get("amount", 0) > 5000:
        adjusted += 0.04
        reasons.append(f"High transaction amount (${user_inputs['amount']:,.2f})")
        scores["high_amount"] = 0.04
    return min(float(adjusted), 0.98), reasons, scores


def get_risk_level(prob):
    if prob >= 0.70:   return "HIGH"
    elif prob >= 0.40: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# 6. Routes
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", "version": "v3",
        "model_loaded": all_loaded,
        "threshold": round(float(threshold), 4) if threshold is not None else None,
        "features": len(feature_columns) if feature_columns else 0,
        "ensemble": {
            "xgb_weight":  round(ensemble_weights["w_xgb"],  3),
            "rf_weight":   round(ensemble_weights["w_rf"],   3),
            "lgbm_weight": round(ensemble_weights["w_lgbm"], 3),
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not all_loaded:
        return jsonify({"error": "Model not ready. Run model.py first."}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    if "amount" not in body:
        return jsonify({"error": "Missing required field: 'amount'"}), 400
    try:
        amount = float(body["amount"])
    except (ValueError, TypeError):
        return jsonify({"error": "'amount' must be a number."}), 400
    if not (0.0 <= amount <= 50000.0):
        return jsonify({"error": "Amount must be 0–50,000."}), 422

    try:
        default_t    = float(_time.time() % 86400)
        time_seconds = float(body.get("time", default_t))
    except (ValueError, TypeError):
        return jsonify({"error": "'time' must be a number (seconds)."}), 400
    if not (0.0 <= time_seconds <= 172800.0):
        return jsonify({"error": "Time must be 0–172,800 seconds."}), 422

    txn_type = str(body.get("transaction_type", "pos")).lower().strip()
    location = str(body.get("location",         "domestic")).lower().strip()
    behavior = str(body.get("behavior",         "usual")).lower().strip()

    if txn_type not in {"online", "pos", "atm"}:
        return jsonify({"error": "transaction_type: online | pos | atm"}), 422
    if location not in {"domestic", "international"}:
        return jsonify({"error": "location: domestic | international"}), 422
    if behavior not in {"usual", "unusual"}:
        return jsonify({"error": "behavior: usual | unusual"}), 422

    try:
        v_feats = lookup_v_features(amount, time_seconds)
        X_in    = build_feature_vector(amount, time_seconds, v_feats)
    except Exception as e:
        return jsonify({"error": f"Feature construction error: {str(e)}"}), 500

    try:
        base_proba = ensemble_predict_proba(X_in)
    except Exception as e:
        return jsonify({"error": f"Model prediction error: {str(e)}"}), 500

    hour     = (time_seconds % 86400) / 3600
    is_night = hour < 6
    user_ctx = {
        "amount": amount, "transaction_type": txn_type,
        "location": location, "behavior": behavior,
    }
    final_proba, reasons, comp_scores = apply_rule_adjustments(
        base_proba, user_ctx, is_night
    )

    thr             = float(threshold)
    is_fraud        = final_proba >= thr
    prediction_text = "Fraud" if is_fraud else "Not Fraud"
    risk_level      = get_risk_level(final_proba)

    if not reasons:
        reasons.append("No elevated contextual risk signals detected")
    if base_proba >= thr:
        reasons.insert(0,
            f"ML ensemble scored this transaction at {base_proba:.1%} fraud probability")

    return jsonify({
        "prediction":        prediction_text,
        "fraud_probability": round(final_proba, 4),
        "confidence":        round(final_proba * 100, 1),
        "risk_level":        risk_level,
        "reasons":           reasons,
        "threshold_used":    round(thr, 4),
        "model_base_score":  round(base_proba, 4),
        "component_scores":  {k: round(float(v), 4) for k, v in comp_scores.items()},
    }), 200


# ─────────────────────────────────────────────
# 7. Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    get_reference_df()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 FraudShield v3 API on port {port}")
    app.run(debug=True, host="0.0.0.0", port=port)