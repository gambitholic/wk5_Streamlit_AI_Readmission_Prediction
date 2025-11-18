import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# ---------------------
# Load Data
# ---------------------
df = pd.read_csv("diabetic_data.csv")

# Keep only <30 and >30
df = df[df["readmitted"] != "NO"]
df["readmitted_flag"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

y = df["readmitted_flag"]
X = df.drop(["readmitted", "readmitted_flag"], axis=1)

# Save the column names (VERY IMPORTANT!)
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# ---------------------
# Preprocessing
# ---------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ],
    remainder="passthrough"
)

# ---------------------
# Model
# ---------------------
model = XGBClassifier(
    eval_metric="logloss",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# ---------------------
# Train/Validation Split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))

# ---------------------
# Save model
# ---------------------
joblib.dump(pipeline, "readmission_model.pkl")
print("Model saved successfully!")
print("Columns saved successfully!")