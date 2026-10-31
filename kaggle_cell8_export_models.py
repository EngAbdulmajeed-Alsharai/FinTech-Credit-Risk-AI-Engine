# ==============================================================================
# الخلية رقم [8] في دفتر كاجل: حفظ وتصدير جميع النماذج المدربة وأدوات التجهيز
# ==============================================================================
import os
import joblib
import zipfile

os.makedirs("exported_fintech_models", exist_ok=True)

# 1. حفظ الشبكة العصبية العميقة (Keras Model)
dl_model.save("exported_fintech_models/best_fintech_credit_model.keras")

# 2. حفظ نماذج تعلم الآلة (Gradient Boosting, Random Forest) والمقيّسات و K-Means
pipeline_bundle = {
    "gb_model": trained_ml_models["3. Gradient Boosting (Boosting)"],
    "rf_model": trained_ml_models["2. Random Forest (Bagging)"],
    "lr_model": trained_ml_models["1. Logistic Regression (Baseline)"],
    "imputer": imputer,
    "scaler": scaler,
    "kmeans": kmeans,
    "final_scaler": final_scaler,
    "feature_columns": list(X_train_enc.columns),
    "optimal_threshold": 0.25,
}
joblib.dump(pipeline_bundle, "exported_fintech_models/fintech_ml_pipeline.joblib")

# 3. ضغط الملفات في حزمة واحدة لتحميلها من قسم Output في Kaggle
with zipfile.ZipFile("fintech_trained_models_bundle.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("exported_fintech_models/best_fintech_credit_model.keras")
    zf.write("exported_fintech_models/fintech_ml_pipeline.joblib")

print("تم حفظ جميع النماذج بنجاح في: fintech_trained_models_bundle.zip")
