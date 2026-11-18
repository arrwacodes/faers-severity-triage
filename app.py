import os
import pandas as pd
import streamlit as st
from predict import DISCLAIMER, Predictor

ARTIFACTS = os.environ.get("FAERS_ARTIFACTS", "artifacts")
NAMES = {"is_fatal": "Fatal outcome", "is_hospitalized": "Hospitalization",
         "is_life_threat": "Life-threatening", "is_disabling": "Disabling / incapacitating"}

st.set_page_config(page_title="FAERS Severity Triage", layout="wide")


@st.cache_resource
def load():
    return Predictor(ARTIFACTS)


try:
    pred = load()
except FileNotFoundError:
    st.error(f"No model artifacts in '{ARTIFACTS}'. Run `python train.py ...` first.")
    st.stop()

meta = pred.meta
st.title("FAERS Adverse-Event Severity Triage")
st.caption(DISCLAIMER)

tab_score, tab_perf = st.tabs(["Score a report", "Model performance"])
ch = meta["choices"]


def pick(label, key, help_=None):
    opts = ["(not specified)"] + ch.get(key, [])
    v = st.selectbox(label, opts, help=help_)
    return None if v == "(not specified)" else v


with tab_score:
    with st.form("report"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Patient")
            age = st.number_input("Age (years)", 0, 120, 60)
            age_known = st.checkbox("Age known", True)
            wt = st.number_input("Weight (kg), 0 = unknown", 0.0, 500.0, 0.0)
            sex = pick("Sex", "patient_sex")
        with c2:
            st.subheader("Drug")
            drug = pick("Suspect drug", "suspect_drug")
            brand = pick("Brand name", "brand_name")
            manuf = pick("Manufacturer", "manufacturer")
            pclass = pick("Pharmacologic class", "pharm_class")
        with c3:
            st.subheader("Report")
            indic = pick("Indication", "drug_indication")
            route = pick("Route", "drug_route")
            country = pick("Country", "country")
            nd = st.number_input("Number of drugs on report", 1, 200, int(meta["numeric_defaults"]["num_drugs"]))
            nr = st.number_input("Number of reactions", 1, 200, int(meta["numeric_defaults"]["num_reactions"]))
        go = st.form_submit_button("Score report", type="primary")

    if go:
        res = pred.predict({
            "patient_age_years": age if age_known else None, "patient_weight_kg": wt or None,
            "patient_sex": sex, "suspect_drug": drug, "brand_name": brand, "manufacturer": manuf,
            "pharm_class": pclass, "drug_indication": indic, "drug_route": route, "country": country,
            "num_drugs": nd, "num_reactions": nr})
        cols = st.columns(4)
        for col, (t, r) in zip(cols, res["predictions"].items()):
            with col:
                st.metric(NAMES[t], f"{r['probability']:.0%}", "FLAGGED" if r["flag"] else "below threshold",
                          delta_color="inverse" if r["flag"] else "off")
                st.progress(min(r["probability"], 1.0))
                st.caption(f"Flag threshold: {r['threshold']:.2f}")
                if r["top_factors"]:
                    st.dataframe(pd.DataFrame([{"Factor": f["label"], "Effect on score": f"{f['effect']:+.1%}"}
                                               for f in r["top_factors"]]), hide_index=True, width="stretch")
        st.caption("'Effect on score' = how much the probability drops when that input is replaced by "
                   "'unknown / typical'. It is an explanation of the model, not evidence of causation.")

with tab_perf:
    st.write(f"Model version **{meta['model_version']}**, trained on **{meta['sample_rows']:,}** sampled reports "
             f"(held-out test split, thresholds tuned on a separate validation split).")
    rows = [{"Outcome": NAMES[t], "Algorithm": meta["configs"][t]["algorithm"],
             "Test positives": m["n_positive"], "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "ROC-AUC": m.get("roc_auc"), "PR-AUC": m.get("pr_auc")}
            for t, m in meta["test_metrics"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.info("Rare outcomes (life-threatening, disabling) have few positives in small samples, so these "
            "numbers are noisy. The research paper's full-data results are the reference figures.")
