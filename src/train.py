from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

import joblib
import json
import os
from datetime import datetime
from data import load_data, split_data

MODEL_VERSION = "1.0.0"
MODEL_DIR = "../model"

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics including accuracy, ROC-AUC, log loss, precision, recall, and F1-score.
    """
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            # Binary case to 2-column
            scores = np.vstack([-scores, scores]).T
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        y_proba = e / e.sum(axis=1, keepdims=True)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # computing ROC-AUC and log loss
    roc_auc = None
    logloss = None
    try:
        if y_proba is not None:
            import numpy as np
            classes = np.unique(y_test)
            if len(classes) > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
            else:
                # Binary: use positive class column (assumes column 1 is positive)
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            logloss = log_loss(y_test, y_proba)
    except Exception:
        pass  

    return {
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        },
        "class_metrics": {k: v for k, v in report.items() if k.isdigit()}
    }


def fit_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier, evaluate it, and save the model and metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # XGBoost configuration suited for small-to-medium tabular datasets
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",  # outputs class probabilities for multiclass
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    model_path = os.path.join(MODEL_DIR, f"iris_model_v{MODEL_VERSION.replace('.', '_')}.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"model_metrics_v{MODEL_VERSION.replace('.', '_')}.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print("Model metrics:", json.dumps(metrics, indent=2))

    return model, metrics

if __name__ == "__main__":
    # Load and prepare data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and evaluate model
    model, metrics = fit_model(X_train, y_train, X_test, y_test)
