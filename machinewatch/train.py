"""Compare baselines and tree models using held-out machines."""
import argparse
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, mean_absolute_error, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import FEATURES, WINDOW, digest, make_examples, read_csv, save_csv, simulate, split_machines, write_json


def classification_report(actual, scores, threshold=0.5):
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {"average_precision": float(average_precision_score(actual, scores)),
            "roc_auc": float(roc_auc_score(actual, scores)), "brier_score": float(brier_score_loss(actual, scores)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0,
            "missed_failure_windows": int(fn), "false_alarm_windows": int(fp),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": threshold, "positive_rate": float(np.mean(actual)), "windows": len(actual)}


def regression_report(actual, predicted):
    predicted = np.maximum(predicted, 0)
    return {"mae_cycles": float(mean_absolute_error(actual, predicted)),
            "rmse_cycles": float(np.sqrt(np.mean((actual - predicted)**2)))}


def train(output, csv_path=None, machines=60, seed=42, horizon=20):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory to preserve existing experiments")
    rows = read_csv(csv_path) if csv_path else simulate(machines, seed)
    x, y, rul, ids, cycles, histories = make_examples(rows, horizon)
    masks, groups = split_machines(ids, seed)
    output.mkdir(parents=True)
    save_csv(output / "dataset.csv", rows)
    provenance = "user-supplied run-to-failure CSV" if csv_path else "engineered synthetic demonstration"
    classifiers = {
        "prevalence": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=seed),
    }
    regressors = {"median": DummyRegressor(strategy="median"),
                  "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
                  "gradient_boosting": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=seed)}
    validation = {"classification": {}, "rul": {}}
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for name, model in classifiers.items():
            model.fit(x[masks["train"]], y[masks["train"]])
            validation["classification"][name] = classification_report(y[masks["validation"]], model.predict_proba(x[masks["validation"]])[:, 1])
        for name, model in regressors.items():
            model.fit(x[masks["train"]], rul[masks["train"]])
            validation["rul"][name] = regression_report(rul[masks["validation"]], model.predict(x[masks["validation"]]))
        chosen_classifier = max(classifiers, key=lambda name: validation["classification"][name]["average_precision"])
        chosen_regressor = min(regressors, key=lambda name: validation["rul"][name]["mae_cycles"])
        classifier, regressor = classifiers[chosen_classifier], regressors[chosen_regressor]
        # Fixed threshold; neither model selection nor tuning uses the test groups.
        scores = classifier.predict_proba(x[masks["test"]])[:, 1]
        estimates = np.maximum(regressor.predict(x[masks["test"]]), 0)
        test = {"classification": classification_report(y[masks["test"]], scores),
                "rul": regression_report(rul[masks["test"]], estimates),
                "prevalence_baseline": classification_report(y[masks["test"]], classifiers["prevalence"].predict_proba(x[masks["test"]])[:, 1]),
                "median_rul_baseline": regression_report(rul[masks["test"]], regressors["median"].predict(x[masks["test"]]))}
    errors = []
    for machine_id, cycle, actual, score, actual_rul, estimate in zip(ids[masks["test"]], cycles[masks["test"]], y[masks["test"]], scores, rul[masks["test"]], estimates):
        if int(score >= 0.5) != actual:
            errors.append({"machine_id": str(machine_id), "cycle": int(cycle), "actual_failure_window": int(actual),
                           "risk_score": float(score), "actual_rul": int(actual_rul), "estimated_rul": float(estimate)})
    # Most confident mistakes first, including false alarms and missed windows.
    errors.sort(key=lambda row: -abs(row["risk_score"] - 0.5))
    joblib.dump({"classifier": classifier, "regressor": regressor}, output / "model.joblib")
    metadata = {"schema_version": 1, "model_version": output.name, "features": FEATURES, "window": WINDOW,
                "horizon_cycles": horizon, "threshold": 0.5, "data_provenance": provenance,
                "classifier": chosen_classifier, "regressor": chosen_regressor,
                "model_sha256": digest(output / "model.joblib"), "dataset_sha256": digest(output / "dataset.csv"),
                "seed": seed, "python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "training_seconds": time.perf_counter() - started}
    write_json(output / "metadata.json", metadata)
    write_json(output / "metrics.json", {"validation": validation, "test": test, "machine_splits": groups,
               "most_confident_errors": errors[:50], "scope": "Window metrics are correlated within machines; no confidence interval or physical safety claim."})
    demo_machine = groups["test"][0]
    write_json(output / "demo.json", {"machine_id": demo_machine,
               "history": [{k: v for k, v in row.items() if k != "machine_id"} for row in histories[demo_machine]], "source": provenance})
    print(f"Selected classifier={chosen_classifier}, RUL={chosen_regressor}; test AP={test['classification']['average_precision']:.3f}, RUL MAE={test['rul']['mae_cycles']:.2f} cycles")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo-v1"))
    parser.add_argument("--csv", type=Path, help="Completed run-to-failure data, not censored fleet snapshots")
    parser.add_argument("--machines", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()
    train(args.output, args.csv, args.machines, args.seed, args.horizon)
