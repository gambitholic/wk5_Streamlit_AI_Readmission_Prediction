import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# =======================
# LOAD DATA
# =======================
df = pd.read_csv("diabetic_data.csv")

# Keep <30 and >30 only
df = df[df["readmitted"] != "NO"]
df["readmitted_flag"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

y = df["readmitted_flag"]
X = df.drop(["readmitted", "readmitted_flag"], axis=1)

# Save original training columns
joblib.dump(X.columns.tolist(), "model_columns.pkl")
# Save original column data types (critical for Streamlit!)
joblib.dump(X.dtypes.astype(str).to_dict(), "model_dtypes.pkl")

# Identify numerical and categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# =======================
# PREPROCESSOR
# =======================
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ],
    remainder="passthrough"
)

# =======================
# MODEL
# =======================
model = XGBClassifier(
    eval_metric="logloss",
    max_depth=5,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# =======================
# TRAIN SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))

# =======================
# SAVE TRAINED MODEL
# =======================
joblib.dump(pipeline, "readmission_model.pkl")
print("\nModel saved as readmission_model.pkl")
print("Columns saved as model_columns.pkl")
print("Dtypes saved as model_dtypes.pkl")