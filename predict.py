import datetime, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from components import TARGETS, engineer_row_safe_features

DISCLAIMER = ("Research prototype. Scores come from models trained on voluntary FAERS adverse-event "
              "reports, which do not measure true incidence. Not medical advice; not for clinical decisions.")

FIELD_LABELS = {
    "patient_age_years": "Patient age", "patient_weight_kg": "Patient weight", "patient_sex": "Patient sex",
    "num_reactions": "Number of reactions", "num_drugs": "Number of drugs",
    "suspect_drug": "Suspect drug", "brand_name": "Brand name", "manufacturer": "Manufacturer",
    "pharm_class": "Pharmacologic class", "drug_indication": "Indication",
    "drug_route": "Route of administration", "country": "Reporting country",
}
CATEGORICAL = ["suspect_drug", "brand_name", "manufacturer", "pharm_class", "drug_indication",
               "drug_route", "country", "patient_sex"]


class Predictor:
        def __init__(self, artifacts_dir="artifacts"):
        d = Path(artifacts_dir)
        if not d.exists():                       # files were uploaded next to the code, not in artifacts/
            d = Path(__file__).resolve().parent
        models_dir = d / "models" if (d / "models").exists() else d
        self.meta = json.loads((d / "metadata.json").read_text())
        self.thresholds = json.loads((d / "thresholds.json").read_text())
        self.models = {t: joblib.load(models_dir / f"{t}.joblib") for t in TARGETS}
    # ---------- input handling ----------
    def _drug_count_category(self, n):
        ranges = self.meta["drug_count_ranges"]
        for r in ranges:
            if r["min"] <= n <= r["max"]:
                return r["drug_count_category"]
        return min(ranges, key=lambda r: min(abs(n - r["min"]), abs(n - r["max"])))["drug_count_category"]

    def _row(self, report):
        d = self.meta["numeric_defaults"]
        now = datetime.date.today()
        lo, hi = self.meta["year_range"]
        num_drugs = report.get("num_drugs")
        num_drugs = d["num_drugs"] if num_drugs is None else num_drugs
        row = {
            "patient_age_years": report.get("patient_age_years"),
            "patient_weight_kg": report.get("patient_weight_kg"),
            "num_reactions": d["num_reactions"] if report.get("num_reactions") is None else report["num_reactions"],
            "num_drugs": num_drugs,
            "year": min(max(report.get("year") or now.year, lo), hi),   # clip to the range seen in training
            "month": report.get("month") or now.month,
            "report_age_days": 0,   # required by the research feature engineer; unused by the deployed feature sets
            "drug_count_category": self._drug_count_category(num_drugs),
        }
        for c in CATEGORICAL:
            row[c] = report.get(c) or "Unknown"
        return row

    def _score(self, frame):
        X = engineer_row_safe_features(frame)
        return {t: self.models[t].predict_proba(X)[:, 1] for t in TARGETS}

    # ---------- explanation: model-agnostic occlusion ----------
    def _ablations(self, row):
        """One variant of the row per input field, with that field set to 'unknown/typical'."""
        d = self.meta["numeric_defaults"]
        base = {"patient_age_years": np.nan, "patient_weight_kg": np.nan,
                "num_reactions": d["num_reactions"], "num_drugs": d["num_drugs"],
                **{c: "Unknown" for c in CATEGORICAL}}
        variants = []
        for f in FIELD_LABELS:
            v = dict(row); v[f] = base[f]
            if f == "num_drugs":
                v["drug_count_category"] = self._drug_count_category(base[f])
            variants.append(v)
        return variants

    def predict(self, report, explain=True, top_k=4):
        row = self._row(report)
        rows = [row] + (self._ablations(row) if explain else [])
        scores = self._score(pd.DataFrame(rows))
        out = {}
        for t in TARGETS:
            p = float(scores[t][0]); thr = float(self.thresholds[t])
            res = {"probability": round(p, 4), "threshold": round(thr, 2), "flag": bool(p >= thr)}
            if explain:
                eff = [(f, p - float(scores[t][i + 1])) for i, f in enumerate(FIELD_LABELS)]
                eff = [e for e in sorted(eff, key=lambda e: abs(e[1]), reverse=True) if abs(e[1]) > 1e-4][:top_k]
                res["top_factors"] = [{"feature": f, "label": FIELD_LABELS[f],
                                       "effect": round(v, 4)} for f, v in eff]
            out[t] = res
        return {"model_version": self.meta["model_version"], "trained_on_rows": self.meta["sample_rows"],
                "predictions": out, "disclaimer": DISCLAIMER}
