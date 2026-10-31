import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_BUNDLE_PATH = os.path.join("exported_fintech_models", "fintech_ml_pipeline.joblib")
KERAS_MODEL_PATH = os.path.join("exported_fintech_models", "best_fintech_credit_model.keras")

pipeline_bundle = None
dl_model = None

if os.path.exists(MODEL_BUNDLE_PATH) and os.path.exists(KERAS_MODEL_PATH):
    try:
        import tensorflow as tf
        pipeline_bundle = joblib.load(MODEL_BUNDLE_PATH)
        dl_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print("Loaded trained Kaggle ML + DL models successfully!")
    except Exception as exc:
        print(f"Note: Running with built-in calibrated scoring engine ({exc})")


def engineer_financial_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    d["grade_num"] = d["loan_grade"].map(grade_map)
    d["prior_default_binary"] = (d["cb_person_default_on_file"] == "Y").astype(int)
    d["annual_interest_cost"] = d["loan_amnt"] * (d["loan_int_rate"] / 100.0)
    d["total_burden_to_income"] = (d["loan_amnt"] + d["annual_interest_cost"]) / (d["person_income"] + 1.0)
    d["income_per_age"] = d["person_income"] / d["person_age"]
    d["emp_stability_ratio"] = d["person_emp_length"] / np.maximum(d["person_age"] - 18, 1)
    d["log_income"] = np.log1p(d["person_income"])
    d["log_loan_amnt"] = np.log1p(d["loan_amnt"])
    return d


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if pipeline_bundle is not None and dl_model is not None:
        app_df = pd.DataFrame([payload])
        app_df["emp_length_missing"] = 0
        app_df["int_rate_missing"] = 0
        app_fe = engineer_financial_features(app_df)

        cat_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
        feature_cols = pipeline_bundle["feature_columns"]
        base_cols = [c for c in feature_cols if not c.startswith("cluster_")]
        app_enc = pd.get_dummies(app_fe, columns=cat_cols).reindex(columns=base_cols, fill_value=0)

        app_scaled_1 = pipeline_bundle["scaler"].transform(app_enc)
        cluster_id = int(pipeline_bundle["kmeans"].predict(app_scaled_1)[0])
        for i in range(4):
            app_enc[f"cluster_{i}"] = int(cluster_id == i)

        app_final = pipeline_bundle["final_scaler"].transform(app_enc)
        p_gb = float(pipeline_bundle["gb_model"].predict_proba(app_final)[0, 1])
        p_rf = float(pipeline_bundle["rf_model"].predict_proba(app_final)[0, 1])
        p_dl = float(dl_model(app_final, training=False).numpy().ravel()[0])
        p_hybrid = (0.55 * p_gb) + (0.30 * p_dl) + (0.15 * p_rf)

        return jsonify({
            "source": "trained_kaggle_models",
            "cluster_id": cluster_id,
            "p_gb": round(p_gb, 4),
            "p_dl": round(p_dl, 4),
            "p_rf": round(p_rf, 4),
            "p_hybrid": round(p_hybrid, 4),
        })
    return jsonify({"source": "client_calibrated_engine"})


if __name__ == "__main__":
    print("Starting FinTech Credit Risk AI Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
