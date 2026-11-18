import argparse, datetime, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from components import (FEATURE_SUBSETS, TARGETS, build_full_pipeline, engineer_row_safe_features,
                        TARGET_ENCODE_COLS, FREQ_ENCODE_COLS)

# Winning configurations from your notebook's Sec. 16b output.
# ONE deliberate change: the paper's is_disabling model used 'all_features', which includes
# report_age_days / fast_report. Those mean "how old the report is when the dataset was built",
# which is ~0 for a brand-new report, so the MVP uses 'env_plus_physio' (like the other three).
WINNING_CONFIG = {
    "is_fatal": dict(feature_subset="env_plus_physio", scaler="none", feature_selection="variance",
                     algorithm="RandomForest", imbalance_strategy="class_weight",
                     hyperparams={"clf__n_estimators": 200, "clf__min_samples_leaf": 4, "clf__max_depth": 14}),
    "is_hospitalized": dict(feature_subset="env_plus_physio", scaler="none", feature_selection="variance",
                            algorithm="HistGradientBoosting", imbalance_strategy="class_weight",
                            hyperparams={"clf__max_iter": 200, "clf__max_depth": 6, "clf__learning_rate": 0.1}),
    "is_life_threat": dict(feature_subset="env_plus_physio", scaler="none", feature_selection="variance",
                           algorithm="RandomForest", imbalance_strategy="class_weight",
                           hyperparams={"clf__n_estimators": 400, "clf__min_samples_leaf": 4, "clf__max_depth": 14}),
    "is_disabling": dict(feature_subset="env_plus_physio", scaler="none", feature_selection="none",
                         algorithm="LightGBM", imbalance_strategy="smote",
                         hyperparams={"clf__num_leaves": 63, "clf__n_estimators": 300,
                                      "clf__max_depth": 10, "clf__learning_rate": 0.15}),
}

REQUIRED_RAW_COLS = ["patient_age_years", "patient_weight_kg", "patient_sex", "num_reactions", "num_drugs",
                     "report_age_days", "year", "month", "drug_count_category",
                     *TARGET_ENCODE_COLS, *FREQ_ENCODE_COLS, *TARGETS]
CHOICE_COLS = ["suspect_drug", "brand_name", "manufacturer", "pharm_class", "drug_indication",
               "drug_route", "country", "patient_sex"]


def _strata(df):
    # one small integer per row (bit 0..3 = the four outcomes) instead of building strings: far less memory
    return (df[TARGETS].astype(int).values * np.array([1, 2, 4, 8])).sum(axis=1)


def _safe_split(df, test_size, seed):
    try:
        return train_test_split(df, test_size=test_size, stratify=_strata(df), random_state=seed)
    except ValueError:  # a stratum has <2 members (tiny samples)
        return train_test_split(df, test_size=test_size, random_state=seed)


def load_sample(csv_path, n_rows, seed):
    needed = list(dict.fromkeys(REQUIRED_RAW_COLS))                  # unique, keep order
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [c for c in needed if c not in header]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")
    df = pd.read_csv(csv_path, usecols=needed)                        # load only the columns we use
    for t in TARGETS:
        df[t] = df[t].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    if n_rows and n_rows < len(df):
        df, _ = _safe_split(df, test_size=1 - n_rows / len(df), seed=seed)  # keeps rare positives
    return df.reset_index(drop=True)


def find_best_threshold(y_true, proba):
    if y_true.sum() == 0:
        return 0.5
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        s = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if s > best:
            best_t, best = float(t), s
    return best_t


def metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    out = dict(n=int(len(y)), n_positive=int(y.sum()), threshold=round(float(thr), 2),
               precision=float(precision_score(y, pred, zero_division=0)),
               recall=float(recall_score(y, pred, zero_division=0)),
               f1=float(f1_score(y, pred, zero_division=0)),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    return out


def train_models(csv_path, out_dir="artifacts", n_rows=20000, seed=42):
    t0 = time.time()
    out = Path(out_dir); (out / "models").mkdir(parents=True, exist_ok=True)
    df = engineer_row_safe_features(load_sample(csv_path, n_rows, seed))
    print(f"Sample: {len(df):,} rows | positives: " + ", ".join(f"{t}={int(df[t].sum())}" for t in TARGETS))

    train_df, temp = _safe_split(df, 0.40, seed)
    val_df, test_df = _safe_split(temp, 0.50, seed)
    print(f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    thresholds, all_metrics, used_cfg = {}, {}, {}
    for t in TARGETS:
        cfg = dict(WINNING_CONFIG[t])
        y_tr = train_df[t].astype(int)
        if cfg["imbalance_strategy"] == "smote" and y_tr.sum() <= 6:  # SMOTE needs > k_neighbors positives
            cfg["imbalance_strategy"] = "class_weight"
            print(f"  [{t}] too few positives for SMOTE -> class_weight")
        pipe = build_full_pipeline(FEATURE_SUBSETS[cfg["feature_subset"]], cfg["scaler"],
                                   cfg["feature_selection"], cfg["algorithm"], cfg["imbalance_strategy"])
        pipe.set_params(**cfg["hyperparams"])
        s = time.time(); pipe.fit(train_df, y_tr)
        val_p = pipe.predict_proba(val_df)[:, 1]
        thr = find_best_threshold(val_df[t].astype(int).values, val_p)       # tuned on validation only
        test_p = pipe.predict_proba(test_df)[:, 1]
        thresholds[t] = thr
        all_metrics[t] = metrics(test_df[t].astype(int).values, test_p, thr)  # test scored once
        used_cfg[t] = {k: v for k, v in cfg.items()}
        joblib.dump(pipe, out / "models" / f"{t}.joblib", compress=3)
        print(f"  {t:16s} {cfg['algorithm']:22s} fit {time.time()-s:5.1f}s  thr={thr:.2f}  "
              f"test F1={all_metrics[t]['f1']:.3f}")

    # --- metadata the app needs: dropdown choices, numeric defaults, drug-count bins ---
    choices = {c: [v for v in df[c].astype(str).value_counts().index if v != "Unknown"][:300]
               for c in CHOICE_COLS if c in df.columns}
    ranges = df.groupby("drug_count_category")["num_drugs"].agg(["min", "max"]).reset_index()
    metadata = dict(
        model_version=datetime.date.today().isoformat(),
        trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
        sample_rows=int(len(df)),
        sklearn_version=sklearn.__version__, targets=TARGETS, configs=used_cfg,
        test_metrics=all_metrics, choices=choices,
        numeric_defaults={c: float(df[c].median()) for c in ["num_reactions", "num_drugs"]},
        year_range=[int(df["year"].min()), int(df["year"].max())],
        drug_count_ranges=ranges.to_dict("records"),
    )
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved artifacts to {out.resolve()} in {time.time()-t0:.0f}s")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--n-rows", type=int, default=20000, help="rows to sample (0 = use all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train_models(a.csv, a.out, a.n_rows, a.seed)