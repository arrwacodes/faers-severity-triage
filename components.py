import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RNG = 42

LEAKAGE_COLUMNS = ["report_id", "receive_date", "serious", "serious_flags",
                    "reactions", "primary_reaction", "reaction_outcomes",
                    "patient_recovered", "age_group"]

TARGETS = ["is_fatal", "is_hospitalized", "is_life_threat", "is_disabling"]

TARGET_ENCODE_COLS = ["suspect_drug", "manufacturer", "pharm_class", "drug_indication", "brand_name"]
FREQ_ENCODE_COLS   = ["drug_route", "country", "drug_count_category"]
ONEHOT_COLS        = ["patient_sex", "age_bucket_risk"]
NUMERIC_CORE       = ["patient_age_years", "patient_weight_kg", "num_reactions",
                       "num_drugs", "report_age_days", "reactions_per_drug", "age_x_drugs"]
FLAG_COLS          = ["age_missing", "weight_missing", "is_polypharmacy", "fast_report"]

ALL_FEATURE_COLS = (NUMERIC_CORE + ["year", "month_sin", "month_cos"] + FLAG_COLS +
                     TARGET_ENCODE_COLS + FREQ_ENCODE_COLS + ONEHOT_COLS)
ENV_FEATURE_COLS = TARGET_ENCODE_COLS + FREQ_ENCODE_COLS + ["year", "month_sin", "month_cos"]
PHYSIO_FEATURE_COLS = (["patient_age_years", "patient_weight_kg", "num_reactions", "num_drugs",
                         "reactions_per_drug", "age_x_drugs", "age_missing", "weight_missing",
                         "is_polypharmacy"] + ONEHOT_COLS)
FEATURE_SUBSETS = {
    "all_features":        ALL_FEATURE_COLS,
    "environmental_only":  ENV_FEATURE_COLS,     # drug/manufacturer/route/country/indication/date
    "physiological_only":  PHYSIO_FEATURE_COLS,  # age/weight/sex/reactions/drugs/polypharmacy
    "env_plus_physio":     sorted(set(ENV_FEATURE_COLS) | set(PHYSIO_FEATURE_COLS)),
}

def engineer_row_safe_features(df):
    """Only transformations that use a single row's own values (or fixed
    clinical constants) — cannot leak cross-row/test-set information."""
    df = df.copy()
    df["age_missing"] = df["patient_age_years"].isna().astype(int)
    df["weight_missing"] = df["patient_weight_kg"].isna().astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    for c in ["patient_age_years", "patient_weight_kg", "num_reactions", "num_drugs", "report_age_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["age_bucket_risk"] = pd.cut(df["patient_age_years"], bins=[-1, 17, 64, 200],
                                    labels=["pediatric", "adult", "elderly"])
    df["age_bucket_risk"] = df["age_bucket_risk"].astype(str).replace("nan", "Unknown")
    df["reactions_per_drug"] = df["num_reactions"] / (df["num_drugs"] + 1)
    df["is_polypharmacy"] = (df["num_drugs"] >= 5).astype(int)
    for c in TARGET_ENCODE_COLS + FREQ_ENCODE_COLS + ["patient_sex"]:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


class TrainFittedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Learns weight-imputation medians, the age median, and the fast_report
    quantile threshold from whatever it is fit on (train fold only, inside a
    Pipeline/CV) and applies those FIXED, stored values on transform — never
    recomputed from the data being transformed. Fixes Critical Problem #1
    (peer review §5.4/§7)."""
    def __init__(self, fast_report_quantile=0.25):
        self.fast_report_quantile = fast_report_quantile

    def fit(self, X, y=None):
        X = X.copy()
        self.group_medians_ = X.groupby(["patient_sex", "age_bucket_risk"])["patient_weight_kg"].median()
        self.global_weight_median_ = X["patient_weight_kg"].median()
        self.global_age_median_ = X["patient_age_years"].median()
        self.fast_report_threshold_ = X["report_age_days"].quantile(self.fast_report_quantile)
        return self

    def transform(self, X):
        X = X.copy()
        key = list(zip(X["patient_sex"], X["age_bucket_risk"]))
        mapped = pd.Series(key, index=X.index).map(self.group_medians_)
        X["patient_weight_kg"] = X["patient_weight_kg"].fillna(mapped).fillna(self.global_weight_median_)
        age_filled = X["patient_age_years"].fillna(self.global_age_median_)
        X["age_x_drugs"] = age_filled * X["num_drugs"]
        X["fast_report"] = (X["report_age_days"] < self.fast_report_threshold_).astype(int)
        return X

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """category -> relative frequency, LEARNED ON THE FIT FOLD ONLY."""
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cols)
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.cols}
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols).copy()
        for c in self.cols: X[c] = X[c].map(self.maps_[c]).fillna(0.0)
        return X.values.astype(float)

class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorizes to [Q1-1.5*IQR, Q3+1.5*IQR], bounds LEARNED ON THE FIT FOLD ONLY."""
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cols).astype(float)
        q1, q3 = X.quantile(0.25), X.quantile(0.75); iqr = q3 - q1
        self.lower_ = (q1 - 1.5 * iqr).to_dict(); self.upper_ = (q3 + 1.5 * iqr).to_dict()
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols).astype(float).copy()
        for c in self.cols: X[c] = X[c].clip(self.lower_[c], self.upper_[c])
        return X.values

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Genuine K-fold out-of-fold mean-target encoding on training data (a row
    never sees its own label); transform() on held-out data uses the mapping
    learned from the full fit fold. Verified leakage-free by the peer review (§7)
    — kept exactly as designed."""
    def __init__(self, cols, n_splits=5, smoothing=10, random_state=42):
        self.cols = cols; self.n_splits = n_splits
        self.smoothing = smoothing; self.random_state = random_state
    def fit(self, X, y):
        X = pd.DataFrame(X, columns=self.cols).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.global_mean_ = y.mean(); self.maps_ = {}
        for c in self.cols:
            stats = y.groupby(X[c]).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.maps_[c] = smoothed.to_dict()
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols).reset_index(drop=True).copy()
        for c in self.cols: X[c] = X[c].map(self.maps_[c]).fillna(self.global_mean_)
        return X.values.astype(float)
    def fit_transform(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cols).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.fit(X, y); out = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for c in self.cols:
            oof = np.zeros(len(X))
            for tr_idx, val_idx in kf.split(X):
                stats = y.iloc[tr_idx].groupby(X[c].iloc[tr_idx]).agg(["mean", "count"])
                sm = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
                oof[val_idx] = X[c].iloc[val_idx].map(sm).fillna(self.global_mean_).values
            out[c] = oof
        return out.values.astype(float)

def build_preprocessor(feature_cols, scaler="standard"):
    scaler_obj = {"standard": StandardScaler(), "robust": RobustScaler(),
                  "minmax": MinMaxScaler(), "none": "passthrough"}[scaler]
    numeric_present = [c for c in NUMERIC_CORE if c in feature_cols]
    cyc_present  = [c for c in ["year", "month_sin", "month_cos"] if c in feature_cols]
    flag_present = [c for c in FLAG_COLS if c in feature_cols]
    tgt_present  = [c for c in TARGET_ENCODE_COLS if c in feature_cols]
    freq_present = [c for c in FREQ_ENCODE_COLS if c in feature_cols]
    oh_present   = [c for c in ONEHOT_COLS if c in feature_cols]
    transformers = []
    if numeric_present:
        num_steps = [("impute", SimpleImputer(strategy="median")), ("cap", IQRCapper(numeric_present))]
        if scaler_obj != "passthrough": num_steps.append(("scale", scaler_obj))
        transformers.append(("num", Pipeline(num_steps), numeric_present))
    if cyc_present:  transformers.append(("cyc", "passthrough", cyc_present))
    if flag_present: transformers.append(("flag", "passthrough", flag_present))
    if tgt_present:  transformers.append(("tgt", KFoldTargetEncoder(tgt_present), tgt_present))
    if freq_present: transformers.append(("freq", FrequencyEncoder(freq_present), freq_present))
    if oh_present:   transformers.append(("oh", OneHotEncoder(handle_unknown="ignore"), oh_present))
    return ColumnTransformer(transformers=transformers)


def build_feature_selector(method):
    """'none' is itself a tested condition in the budgeted search (Sec. 14) —
    the peer-review-requested comparison includes NOT selecting features at all."""
    if method == "none": return None
    if method == "variance": return VarianceThreshold(threshold=0.001)
    if method == "kbest_mi": return SelectKBest(score_func=mutual_info_classif, k=15)
    if method == "model_based":
        return SelectFromModel(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RNG, n_jobs=-1),
                                threshold="median")
    raise ValueError(f"Unknown feature-selection method: {method}")
   

def make_algorithms(n_estimators=None):
    """Nine algorithms spanning linear -> single-tree -> bagging -> two independent
    gradient-boosting libraries -> a probabilistic baseline. Each is included for a
    stated reason (peer review §6.1 asked for explicit justification):
      - LogisticRegression: interpretable linear baseline, cheap, gives calibrated-ish
        probabilities with class_weight='balanced'.
      - DecisionTree: single-tree baseline to isolate the marginal value of ensembling.
      - RandomForest / ExtraTrees: bagging ensembles; ExtraTrees adds extra split
        randomness, testing whether RF's edge (peer review §9.4/§11) survives a close
        bagging relative rather than being untuned-hyperparameter luck.
      - HistGradientBoosting / XGBoost / LightGBM: three independently-implemented
        gradient-boosting libraries, directly answering peer-review §11.2's question of
        whether the 'boosting' pattern is implementation-specific.
      - AdaBoost: a distinct (non-gradient) boosting family for contrast.
      - GaussianNB: a fast, very-different-bias probabilistic baseline / sanity floor.
    CatBoost was NOT added — see Sec. 26 (Limitations) for why."""
    n = n_estimators or 300
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RNG),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=RNG),
        "RandomForest": RandomForestClassifier(n_estimators=n, max_depth=18, n_jobs=-1, random_state=RNG),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=n, max_depth=18, n_jobs=-1, random_state=RNG),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=8, random_state=RNG),
        "XGBoost": XGBClassifier(n_estimators=max(n, 200), max_depth=6, learning_rate=0.08, eval_metric="logloss",
                                  n_jobs=-1, random_state=RNG, tree_method="hist"),
        "LightGBM": LGBMClassifier(n_estimators=max(n, 200), max_depth=-1, learning_rate=0.08,
                                    random_state=RNG, n_jobs=-1, verbose=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RNG),
        "GaussianNB": GaussianNB(),
    }


def build_full_pipeline(feature_cols, scaler="standard", fs_method="none", algo_name="RandomForest",
                         imbalance="none", n_estimators=None):
    """leak-safe engineer -> ColumnTransformer -> [feature selection] ->
    [SMOTE] -> classifier, as ONE Pipeline object."""
    clf = make_algorithms(n_estimators)[algo_name]
    if imbalance == "class_weight" and "class_weight" in clf.get_params():
        clf.set_params(class_weight="balanced")
    steps = [("leaksafe", TrainFittedFeatureEngineer()), ("prep", build_preprocessor(feature_cols, scaler))]
    fs = build_feature_selector(fs_method)
    if fs is not None: steps.append(("fsel", fs))
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=RNG, k_neighbors=5)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)
    steps.append(("clf", clf))
    return Pipeline(steps)

RECOMMENDED_ALGORITHMS = ["LogisticRegression", "RandomForest", "HistGradientBoosting", "XGBoost", "LightGBM"]
