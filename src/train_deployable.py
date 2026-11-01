"""
Trains the DEPLOYABLE CatBoost model for each real dataset (form + statement
features only: no protected attributes incl. age, no credit-bureau / credit-history
data, no currency amounts) and measures the
"cost of deployability" against the full-feature research model on the SAME
held-out test set.

Usage (from the project root):
    python src\\train_deployable.py --dataset german_credit
    python src\\train_deployable.py --dataset gmsc
    python src\\train_deployable.py --dataset homecredit
Each run updates its own row in results/deployability_cost.csv.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_difference
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import MODEL_PATHS, PROJECT_ROOT
from deployable_features import DEPLOYABLE_SPEC, MONOTONE, select_deployable

DEPLOY_DIR = PROJECT_ROOT / "models" / "deployable"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
SHAP_SAMPLE = 5000
LIME_BACKGROUND = 2000


def get_split(name):
    """Exactly the research pipeline's preprocessing + split (same seed, same rows)."""
    if name == "german_credit":
        from data_loader import load_german_credit
        from preprocessing import preprocess, get_train_test_split
        _, enc = preprocess(load_german_credit())
        return get_train_test_split(enc), enc
    if name == "gmsc":
        from data_loader_gmsc import load_gmsc
        from preprocessing_gmsc import preprocess_gmsc, get_train_test_split_gmsc
        enc = preprocess_gmsc(load_gmsc())
        return get_train_test_split_gmsc(enc), enc
    if name == "homecredit":
        from data_loader_homecredit import load_home_credit
        from preprocessing_homecredit import preprocess_home_credit, get_train_test_split_homecredit
        enc = preprocess_home_credit(load_home_credit())
        return get_train_test_split_homecredit(enc), enc
    raise ValueError(name)


def catboost_params(name, deployable=False):
    # Same settings as each dataset's research CatBoost. The deployable model
    # differs only in its feature set and its monotonic risk directions.
    params = {"verbose": 0, "random_state": RANDOM_STATE}
    if name == "homecredit":
        params["auto_class_weights"] = "Balanced"
    if deployable and MONOTONE.get(name):
        params["monotone_constraints"] = dict(MONOTONE[name])
    return params


def decile_table(scores, y):
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
    bins = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = bins == b
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "default_rate": float(y[m].mean()) if m.any() else None})
    return out


def mean_abs_shap(model, X):
    import shap
    sample = X.sample(min(SHAP_SAMPLE, len(X)), random_state=RANDOM_STATE)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return dict(zip(sample.columns, np.abs(sv).mean(axis=0).astype(float)))


def metrics(y, probs, preds, age):
    return {
        "auc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds),
        "recall": recall_score(y, preds, zero_division=0),
        "dp_ratio": demographic_parity_ratio(y, preds, sensitive_features=age),
        "eo_diff": equalized_odds_difference(y, preds, sensitive_features=age),
    }


def cv_auc(name, X, y, n_splits=5, deployable=False):
    """5-fold CV AUC. Used for German Credit, whose 200-row test set is too
    small for a single-split AUC difference to be trusted on its own."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X, y):
        m = CatBoostClassifier(**catboost_params(name, deployable))
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def save_reference(X_train, path):
    Xn = X_train.astype(float)
    binary = [c for c in X_train.columns if set(Xn[c].dropna().unique()).issubset({0.0, 1.0})]
    numeric = [c for c in X_train.columns if c not in binary]
    ref = {"all_columns": X_train.columns.tolist(), "median": Xn.median().to_dict(),
           "mode": X_train.mode().iloc[0].to_dict(), "binary_columns": binary,
           # Training range of each numeric feature. Live inputs outside it are
           # flagged as out-of-distribution and down-weight that model's vote.
           "q01": Xn[numeric].quantile(0.01).to_dict(), "q99": Xn[numeric].quantile(0.99).to_dict()}
    with open(path, "wb") as f:
        pickle.dump(ref, f)


def update_pickle_dict(path, key, value):
    data = {}
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
    data[key] = value
    with open(path, "wb") as f:
        pickle.dump(data, f)


def run(name):
    print(f"\n=== {name} ===")
    (X_train, X_test, y_train, y_test, age_train, age_test), enc = get_split(name)

    # --- Research model (already trained), evaluated on the same test rows ---
    with open(MODEL_PATHS[name], "rb") as f:
        research = pickle.load(f)
    Xr_test = X_test[list(research.feature_names_)] if hasattr(research, "feature_names_") else X_test
    r = metrics(y_test, research.predict_proba(Xr_test)[:, 1], research.predict(Xr_test), age_test)

    # --- Deployable model ---
    Xd_train = select_deployable(name, X_train)
    Xd_test = select_deployable(name, X_test)
    print(f"  research features: {X_train.shape[1]}  ->  deployable features: {Xd_train.shape[1]}")

    model = CatBoostClassifier(**catboost_params(name, deployable=True))
    model.fit(Xd_train, y_train)
    d_probs = model.predict_proba(Xd_test)[:, 1]
    # Threshold-based metrics (recall, DP, EO) are compared at the SAME flag
    # rate as the research model. A fixed 0.5 cut-off is not comparable across
    # models on imbalanced data (on GMSC it flags almost nobody), so we flag the
    # same share of applicants the research model flags and compare who they are.
    flag_rate = float(np.mean(research.predict(Xr_test)))
    threshold = float(np.quantile(d_probs, 1 - flag_rate))
    d = metrics(y_test, d_probs, (d_probs >= threshold).astype(int), age_test)
    print(f"  Comparison at matched flag rate {flag_rate:.3f} (deployable threshold {threshold:.4f})")

    out_dir = DEPLOY_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "CatBoost.pkl", "wb") as f:
        pickle.dump(model, f)
    save_reference(Xd_train, DEPLOY_DIR / f"{name}_reference.pkl")
    # Small background sample of the training data: LIME perturbs around an
    # applicant using these feature distributions (the full data is not shipped).
    Xd_train.sample(min(LIME_BACKGROUND, len(Xd_train)), random_state=RANDOM_STATE) \
        .to_pickle(DEPLOY_DIR / f"{name}_background.pkl")

    y_arr = np.asarray(y_test).astype(int)
    update_pickle_dict(DEPLOY_DIR / "score_reference.pkl", name, {
        "test_scores_sorted": np.sort(d_probs), "n_test": int(len(y_arr)), "n_train": int(len(Xd_train)),
        "features": Xd_train.columns.tolist(),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "base_rate": float(y_arr.mean()), "auc": float(d["auc"]),
        "deciles": decile_table(d_probs, y_arr),
    })
    shap_w = mean_abs_shap(model, Xd_test)
    update_pickle_dict(DEPLOY_DIR / "shap_weights.pkl", name, shap_w)

    row = {
        "dataset": name,
        "n_features_research": X_train.shape[1], "n_features_deployable": Xd_train.shape[1],
        "matched_flag_rate": flag_rate, "deployable_threshold": threshold,
        "auc_research": r["auc"], "auc_deployable": d["auc"], "auc_cost": r["auc"] - d["auc"],
        "recall_research": r["recall"], "recall_deployable": d["recall"],
        "dp_ratio_research": r["dp_ratio"], "dp_ratio_deployable": d["dp_ratio"],
        "eo_diff_research": r["eo_diff"], "eo_diff_deployable": d["eo_diff"],
    }

    if name == "german_credit":
        X_all = enc.drop(columns=["target", "age_group"])
        y_all = enc["target"]
        row["cv_auc_research"], row["cv_auc_research_sd"] = cv_auc(name, X_all, y_all)
        row["cv_auc_deployable"], row["cv_auc_deployable_sd"] = cv_auc(
            name, select_deployable(name, X_all), y_all, deployable=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    cost_path = RESULTS_DIR / "deployability_cost.csv"
    table = pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame()
    if len(table):
        table = table[table["dataset"] != name]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(cost_path, index=False)

    imp = (pd.Series(shap_w).sort_values(ascending=False).rename("mean_abs_shap")
           .reset_index().rename(columns={"index": "feature"}))
    imp.insert(0, "dataset", name)
    imp_path = RESULTS_DIR / "deployable_feature_importance.csv"
    old = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["dataset"] != name]
    pd.concat([old, imp], ignore_index=True).to_csv(imp_path, index=False)

    print(f"  AUC        research {r['auc']:.4f}  deployable {d['auc']:.4f}  cost {r['auc'] - d['auc']:+.4f}")
    if "cv_auc_research" in row:
        print(f"  5-fold CV  research {row['cv_auc_research']:.4f}+/-{row['cv_auc_research_sd']:.4f}  "
              f"deployable {row['cv_auc_deployable']:.4f}+/-{row['cv_auc_deployable_sd']:.4f}")
    print(f"  Recall     research {r['recall']:.4f}  deployable {d['recall']:.4f}")
    print(f"  DP ratio   research {r['dp_ratio']:.4f}  deployable {d['dp_ratio']:.4f}  (1 = equal)")
    print(f"  EO diff    research {r['eo_diff']:.4f}  deployable {d['eo_diff']:.4f}  (0 = equal)")
    print("  Top deployable features by mean|SHAP|:")
    for _, rr in imp.head(8).iterrows():
        print(f"    {rr['feature']:45s} {rr['mean_abs_shap']:.4f}")
    print(f"  Saved models/deployable/{name}/CatBoost.pkl (+ reference, score_reference, shap_weights)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEPLOYABLE_SPEC) + ["all"])
    args = ap.parse_args()
    for ds in (DEPLOYABLE_SPEC if args.dataset == "all" else [args.dataset]):
        run(ds)
    print("\nDone. See results/deployability_cost.csv")
