import json
import os

import pandas as pd
import streamlit as st

from predict import DISCLAIMER, Predictor

ARTIFACTS = os.environ.get("FAERS_ARTIFACTS", "artifacts")
NAMES = {"is_fatal": "Fatal outcome", "is_hospitalized": "Hospitalization",
         "is_life_threat": "Life-threatening", "is_disabling": "Disabling / incapacitating"}
DESCR = {"is_fatal": "Death reported as an outcome", "is_hospitalized": "Hospital admission reported",
         "is_life_threat": "Life-threatening event reported", "is_disabling": "Disability or incapacity reported"}

st.set_page_config(page_title="FAERS Severity Triage", page_icon="💊", layout="wide")

CSS = """
<style>
.hero {padding: 1.5rem 1.7rem; border-radius: 16px; color: #fff; margin-bottom: 1.1rem;
       background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);}
.hero .t {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.hero .s {margin-top: .4rem; opacity: .92; font-size: .98rem;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1rem 1.1rem;
       background: rgba(128,128,128,.07);}
.ctitle {font-size: 1.05rem; font-weight: 650;}
.sub {opacity: .65; font-size: .8rem; margin-bottom: .45rem;}
.pct {font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin-bottom: .2rem;}
.chip {display: inline-block; padding: .1rem .65rem; border-radius: 999px; font-size: .75rem;
       font-weight: 600; color: #fff;}
.bar {position: relative; height: 10px; border-radius: 999px; background: rgba(128,128,128,.25);
      margin: .8rem 0 .3rem 0;}
.bar .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}
.bar .thr {position: absolute; top: -4px; bottom: -4px; width: 2px; background: currentColor; opacity: .75;}
.small {font-size: .75rem; opacity: .7;}
.frow {display: flex; align-items: center; gap: .5rem; margin: .32rem 0; font-size: .82rem;}
.frow .lab {width: 42%;}
.frow .track {flex: 1; height: 8px; border-radius: 999px; background: rgba(128,128,128,.18); position: relative;}
.frow .seg {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}
.frow .val {width: 4.2rem; text-align: right; font-variant-numeric: tabular-nums;}
.step {border: 1px dashed rgba(128,128,128,.4); border-radius: 14px; padding: 1rem 1.1rem;}
.step .n {font-size: 1.4rem; font-weight: 700; color: #7C3AED;}
.banner {border-radius: 12px; padding: .75rem 1rem; margin: .2rem 0 1rem 0; font-weight: 600;
         background: rgba(124,58,237,.12); border: 1px solid rgba(124,58,237,.35);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    # st.markdown treats lines indented 4+ spaces as code, so collapse the HTML onto one line
    st.markdown(" ".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


@st.cache_resource
def load():
    return Predictor(ARTIFACTS)


try:
    pred = load()
except FileNotFoundError:
    st.error(f"No model artifacts found in '{ARTIFACTS}'. Run `python train.py ...` first.")
    st.stop()

meta = pred.meta
ch = meta["choices"]

html("""<div class="hero"><div class="t">FAERS Adverse-Event Severity Triage</div>
<div class="s">Enter the details of an adverse-event report and get four severity risk scores,
with the inputs that moved each score the most.</div></div>""")


def pick(label, key):
    v = st.selectbox(label, ["(not specified)"] + ch.get(key, []))
    return None if v == "(not specified)" else v


# ---------------- sidebar: input form ----------------
with st.sidebar:
    st.header("Report details")
    with st.form("report"):
        with st.expander("Patient", expanded=True):
            age_known = st.checkbox("Age is known", True)
            age = st.number_input("Age (years)", 0, 120, 60, disabled=not age_known)
            wt = st.number_input("Weight (kg), 0 = unknown", 0.0, 500.0, 0.0)
            sex = pick("Sex", "patient_sex")
        with st.expander("Drug", expanded=True):
            drug = pick("Suspect drug", "suspect_drug")
            brand = pick("Brand name", "brand_name")
            manuf = pick("Manufacturer", "manufacturer")
            pclass = pick("Pharmacologic class", "pharm_class")
        with st.expander("Report", expanded=False):
            indic = pick("Indication", "drug_indication")
            route = pick("Route", "drug_route")
            country = pick("Country", "country")
            nd = st.number_input("Number of drugs on report", 1, 200, int(meta["numeric_defaults"]["num_drugs"]))
            nr = st.number_input("Number of reactions", 1, 200, int(meta["numeric_defaults"]["num_reactions"]))
        go = st.form_submit_button("Score report", type="primary")
    st.caption(DISCLAIMER)

if go:
    inp = {"patient_age_years": age if age_known else None, "patient_weight_kg": wt or None,
           "patient_sex": sex, "suspect_drug": drug, "brand_name": brand, "manufacturer": manuf,
           "pharm_class": pclass, "drug_indication": indic, "drug_route": route, "country": country,
           "num_drugs": nd, "num_reactions": nr}
    st.session_state["inp"] = inp
    st.session_state["res"] = pred.predict(inp)


def render_card(key, r):
    pct, thr = r["probability"] * 100, r["threshold"] * 100
    color = "#E5484D" if r["flag"] else "#30A46C"
    chip = "FLAGGED" if r["flag"] else "Below threshold"
    facs = r.get("top_factors") or []
    mx = max([abs(f["effect"]) for f in facs], default=0) or 1
    rows = ""
    for f in facs:
        e = f["effect"]
        col = "#E5484D" if e > 0 else "#3E63DD"
        rows += (f'<div class="frow"><div class="lab">{f["label"]}</div>'
                 f'<div class="track"><div class="seg" style="width:{abs(e) / mx * 100:.0f}%;background:{col}"></div></div>'
                 f'<div class="val">{e * 100:+.1f} pts</div></div>')
    if not rows:
        rows = '<div class="small">No single input moved this score noticeably.</div>'
    html(f"""
    <div class="card">
      <div class="ctitle">{NAMES[key]}</div>
      <div class="sub">{DESCR[key]}</div>
      <div class="pct" style="color:{color}">{pct:.0f}%</div>
      <span class="chip" style="background:{color}">{chip}</span>
      <div class="bar"><div class="fill" style="width:{min(pct, 100):.1f}%;background:{color}"></div>
      <div class="thr" style="left:{min(thr, 100):.1f}%"></div></div>
      <div class="small">Vertical marker = flag threshold ({thr:.0f}%)</div>
      <div class="small" style="margin-top:.8rem"><b>What moved this score</b></div>
      {rows}
    </div>""")


tab_res, tab_perf, tab_about = st.tabs(["Results", "Model performance", "About"])

with tab_res:
    if "res" not in st.session_state:
        c = st.columns(3)
        steps = [("1", "Fill in the report", "Use the sidebar. Anything you leave unspecified is treated as unknown."),
                 ("2", "Click Score report", "Four models each return a probability for one severity outcome."),
                 ("3", "Read the cards", "A flag means the score passed that outcome's tuned threshold.")]
        for col, (n, t, d) in zip(c, steps):
            with col:
                html(f'<div class="step"><div class="n">{n}</div><div class="ctitle">{t}</div><div class="small">{d}</div></div>')
    else:
        res = st.session_state["res"]
        flagged = [NAMES[k] for k, v in res["predictions"].items() if v["flag"]]
        msg = (f"{len(flagged)} of 4 outcomes flagged: " + ", ".join(flagged)) if flagged else \
              "No outcome passed its flag threshold for this report"
        html(f'<div class="banner">{msg}</div>')
        cols = st.columns(4)
        for col, (k, r) in zip(cols, res["predictions"].items()):
            with col:
                render_card(k, r)
        st.caption("Bars show how many percentage points the score changes when that input is replaced by "
                   "'unknown / typical' (red raises the score, blue lowers it). This explains the model, "
                   "not cause and effect.")
        st.download_button("Download this result (JSON)",
                           json.dumps({"input": st.session_state["inp"], **res}, indent=2),
                           file_name="faers_triage_result.json", mime="application/json")

with tab_perf:
    a, b, c = st.columns(3)
    a.metric("Model version", meta["model_version"])
    b.metric("Reports used for training", f"{meta['sample_rows']:,}")
    c.metric("Outcomes modelled", len(meta["test_metrics"]))
    rows = [{"Outcome": NAMES[t], "Algorithm": meta["configs"][t]["algorithm"],
             "Test positives": m["n_positive"], "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "ROC-AUC": m.get("roc_auc"), "PR-AUC": m.get("pr_auc")}
            for t, m in meta["test_metrics"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.info("Scores are from a held-out test split of the training sample, with thresholds tuned on a separate "
            "validation split. Rare outcomes have few positives, so their numbers are noisy.")

with tab_about:
    st.markdown(f"""
**What this is.** A research prototype that estimates, for one adverse-event report, the chance of four
severity outcomes: death, hospitalization, life-threatening event and disability.

**Where the data comes from.** FDA Adverse Event Reporting System (FAERS) reports, 2015 to 2026.
FAERS is voluntary reporting, so it shows what was reported, not how often events truly occur.

**How to read a score.** A probability is the model's estimate for that outcome. A flag means it passed a
threshold tuned to balance precision and recall, so flags on rare outcomes will include many false alarms.

**Limits.** Drugs, brands or manufacturers not seen in training are treated as unknown. Performance dropped
when tested on later time periods in the research, so the models need periodic retraining.

> {DISCLAIMER}
""")
