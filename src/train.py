# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")


# ------------------- Load data -------------------
df = pd.read_csv("data/processed/modeling_dataset.csv")

# Features = everything except CustomerId and target
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"High-risk rate in train: {y_train.mean():.1%}")

# ------------------- Start MLflow -------------------
mlflow.set_experiment("credit_risk_model")

def log_metrics(y_true, y_pred, y_prob, model_name):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }
    mlflow.log_metrics(metrics)
    print(f"{model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics

# ------------------- Model 1: Logistic Regression -------------------
# ------------------- Model 2: XGBoost with tuning -------------------
with mlflow.start_run(run_name="XGBoost_Tuned"):
    model_xgb = XGBClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean()  # handle imbalance
    )
    model_xgb.fit(X_train, y_train)
    
    pred_xgb = model_xgb.predict(X_test)
    prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
    
    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_params(model_xgb.get_params())
    
    # FIXED: Use sklearn flavor for XGBClassifier
    mlflow.sklearn.log_model(model_xgb, "model")
    
    xgb_metrics = log_metrics(y_test, pred_xgb, prob_xgb, "XGBoost")

# ------------------- Register best model (XGBoost usually wins) -------------------
# You can change this based on AUC
best_model_name = "xgboost-credit-risk-model"
client = mlflow.tracking.MlflowClient()
try:
    client.create_registered_model(best_model_name)
except:
    pass


# Register the XGBoost run (now logged with sklearn flavor)
latest_run = mlflow.search_runs(filter_string="tags.mlflow.runName = 'XGBoost_Tuned'").iloc[0]
model_uri = f"runs:/{latest_run.run_id}/model"
mlflow.register_model(model_uri, best_model_name)

print("\nTraining complete! Check MLflow UI:")
print("Run in terminal: mlflow ui")
print("Then open http://127.0.0.1:5000 in browser")