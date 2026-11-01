"""FAIRScore AI -- loan officer dashboard.   Run:  streamlit run app\\dashboard.py"""
import glob
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import live_scoring as ls  # noqa: E402
from demo_personas import PERSONAS, statement_path, write_all  # noqa: E402
from report_pdf import build_report  # noqa: E402
from statement_parser import parse_statement  # noqa: E402

st.set_page_config(page_title="FAIRScore AI", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root { --paper:#F6F4EF; --card:#FBFAF7; --ink:#1B2027; --ink-soft:#5B6270; --navy:#16324A;
        --gold:#B8863B; --gold-soft:#F3E9D3; --risk:#9E3A2E; --risk-soft:#F3E0DC;
        --safe:#2E6B4F; --safe-soft:#DEEBE3; --line:#DCD7CC; }
html, body, [class*="css"], .stApp { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
h1, h2, h3, h4 { font-family:'Source Serif 4',serif !important; color:var(--ink); }
.block-container { padding-top:3.6rem; max-width:1320px; }
.topbar { background:var(--navy); color:#F4F1E8; padding:16px 26px; border-radius:4px; margin-bottom:14px;
          display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
.topbar .name { font-family:'Source Serif 4',serif; font-size:24px; font-weight:600; }
.topbar .tag { font-size:12.5px; color:#B9C3CC; }
.eyebrow { font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-soft);
           font-weight:600; margin:14px 0 6px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:18px 20px; margin-bottom:12px; }
.verdict { border-left:6px solid var(--c); }
.verdict .label { font-family:'Source Serif 4',serif; font-size:28px; font-weight:700; color:var(--c); line-height:1.1; }
.verdict .sub { color:var(--ink-soft); font-size:13.5px; margin-top:6px; }
.verdict ul { margin:10px 0 0 0; padding-left:18px; font-size:13.5px; }
.check { display:flex; gap:12px; padding:9px 0; border-bottom:1px solid #ECE8DE; font-size:13.5px; align-items:flex-start; }
.check:last-child { border-bottom:none; }
.pill { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; padding:3px 8px; border-radius:3px;
        min-width:64px; text-align:center; flex-shrink:0; }
.pass { background:var(--safe-soft); color:var(--safe); }
.caution { background:var(--gold-soft); color:var(--gold); }
.fail { background:var(--risk-soft); color:var(--risk); }
.check .nm { font-weight:600; }
.check .dt { color:var(--ink-soft); font-size:12.5px; }
.chip { display:inline-block; border:1px solid var(--line); background:#fff; padding:8px 12px; margin:0 8px 8px 0;
        font-size:13px; border-radius:3px; }
.meter { position:relative; height:12px; border-radius:6px; margin:8px 0 4px;
         background:linear-gradient(90deg, var(--safe-soft) 0 50%, var(--gold-soft) 50% 80%, var(--risk-soft) 80% 100%); }
.meter .dot { position:absolute; top:-4px; width:4px; height:20px; background:var(--ink); border-radius:2px; }
.meter-scale { display:flex; justify-content:space-between; font-size:11px; color:var(--ink-soft); }
.mrow { display:grid; grid-template-columns: 170px 1fr 54px; gap:12px; align-items:center; font-size:13px; padding:6px 0; }
.mbar { height:8px; background:#ECE8DE; border-radius:4px; position:relative; }
.mbar span { position:absolute; left:0; top:0; bottom:0; border-radius:4px; background:var(--navy); }
.mono { font-family:'IBM Plex Mono',monospace; }
.factor { font-size:13.5px; padding:4px 0; }
.up { color:var(--risk); font-weight:600; }
.down { color:var(--safe); font-weight:600; }
.muted { color:var(--ink-soft); font-size:12.5px; }
.demo-note { background:var(--gold-soft); color:#6B4E1F; padding:8px 12px; border-radius:3px; font-size:12.5px; margin:6px 0 4px; }
div[data-testid="stForm"] { background:var(--card); border:1px solid var(--line); border-radius:4px; }
</style>
<div class="topbar"><span class="name">FAIRScore AI</span>
<span class="tag">Explainable, fairness-aware credit decision support &middot; recommendations for a credit officer, not automatic decisions</span></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Form state + demo applicants
# ---------------------------------------------------------------------------
FIELDS = {   # key: default
    "age": 30, "dependents": 0, "education": "Graduate", "housing": "Rented home", "owns_car": False,
    "employment_type": "Salaried (private sector)", "employment_years": 2.0, "monthly_income": 40000,
    "requested_amount": 200000, "requested_duration_months": 24,
    "purpose": "Electronics (TV, phone, laptop)",
    "active_credit_lines": 0, "property_loans": 0, "total_savings": 20000,
}
for k, v in FIELDS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("demo", None)
st.session_state.setdefault("result", None)


def load_demo(persona):
    if not statement_path(persona).exists():
        write_all()
    for k, v in PERSONAS[persona]["applicant"].items():
        st.session_state[k] = v
    st.session_state["demo"] = persona
    st.session_state["result"] = None


def clear_demo():
    st.session_state["demo"] = None


@st.cache_resource(show_spinner=False)
def warm_up():
    for name in ls.VOTING_MODELS:      # load models + explainers once per server
        ls.load_model(name)
        ls.explainer(name)
    return True


def money(x):
    return f"₹{x:,.0f}"


VERDICT = {
    "approve": ("Recommend approve", "var(--safe)", "All lending-policy checks passed and the risk models "
                "did not flag elevated risk."),
    "refer": ("Refer to credit officer", "var(--gold)", "The application needs a human look before a decision."),
    "decline": ("Recommend decline", "var(--risk)", "One or more lending-policy checks failed."),
}

tab_assess, tab_how, tab_research = st.tabs(["Assess an applicant", "How it decides", "Research results"])

# ===========================================================================
# Assess
# ===========================================================================
with tab_assess:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown('<p class="eyebrow">Try a demo applicant</p>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.button("Strong", on_click=load_demo, args=("strong",), use_container_width=True)
        b2.button("Borderline", on_click=load_demo, args=("borderline",), use_container_width=True)
        b3.button("Weak", on_click=load_demo, args=("weak",), use_container_width=True)
        if st.session_state["demo"]:
            p = PERSONAS[st.session_state["demo"]]
            st.markdown(f'<div class="demo-note"><b>{p["title"]}</b> loaded with its 6-month demo '
                        f'statement. {p["summary"]}</div>', unsafe_allow_html=True)

        with st.form("intake"):
            st.markdown('<p class="eyebrow">Applicant</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.number_input("Age", 18, 80, key="age")
            c2.number_input("Dependents", 0, 10, key="dependents")
            c1.selectbox("Education", list(ls.EDUCATION_LEVELS), key="education")
            c2.selectbox("Housing", list(ls.HOUSING_OPTIONS), key="housing")
            st.checkbox("Owns a car or two-wheeler", key="owns_car")

            st.markdown('<p class="eyebrow">Work and income</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.selectbox("Employment type", list(ls.EMPLOYMENT_TYPES), key="employment_type")
            c2.number_input("Years in current work", 0.0, 50.0, step=0.5, key="employment_years")
            st.number_input("Monthly income (₹, take-home)", 0, 10_000_000, step=1000, key="monthly_income")

            st.markdown('<p class="eyebrow">Loan request</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.number_input("Amount (₹)", 10_000, 50_000_000, step=10_000, key="requested_amount")
            c2.number_input("Term (months)", 6, 84, step=6, key="requested_duration_months")
            st.selectbox("Purpose", list(ls.PURPOSE_OPTIONS), key="purpose")

            st.markdown('<p class="eyebrow">Existing credit and savings</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.number_input("Active loans + credit cards", 0, 30, key="active_credit_lines")
            c2.number_input("Of which home / property loans", 0, 10, key="property_loans")
            st.number_input("Total savings (₹)", 0, 100_000_000, step=5000, key="total_savings")

            st.markdown('<p class="eyebrow">Bank statement (CSV, last 6 months)</p>', unsafe_allow_html=True)
            upload = st.file_uploader("Statement", type=["csv"], label_visibility="collapsed",
                                      help="Columns like Date, Description/Narration, Debit, Credit, Balance.")
            submitted = st.form_submit_button("Assess applicant", type="primary", use_container_width=True)

        if st.session_state["demo"]:
            st.button("Use my own statement instead of the demo one", on_click=clear_demo)

    if submitted:
        applicant = {k: st.session_state[k] for k in FIELDS}
        applicant["employment_years"] = float(applicant["employment_years"])
        stmt_df, stmt, warnings, source = None, None, [], None
        try:
            if upload is not None:
                tmp = Path(tempfile.gettempdir()) / "fairscore_upload.csv"
                tmp.write_bytes(upload.getbuffer())
                stmt_df, stmt, warnings = parse_statement(str(tmp))
                source = upload.name
            elif st.session_state["demo"]:
                stmt_df, stmt, warnings = parse_statement(str(statement_path(st.session_state["demo"])))
                source = statement_path(st.session_state["demo"]).name
        except Exception as exc:   # noqa: BLE001
            warnings = [f"The statement could not be read ({exc}). Assessed without it."]
        with st.spinner("Assessing..."):
            warm_up()
            result = ls.assess(applicant, stmt, stmt_df)
        st.session_state["result"] = {"applicant": applicant, "r": result,
                                      "warnings": warnings, "source": source}

    # -------------------------------------------------------------------
    with right:
        saved = st.session_state["result"]
        if not saved:
            st.markdown('<p class="eyebrow">Result</p>', unsafe_allow_html=True)
            st.markdown('<div class="card"><b>No assessment yet.</b><p class="muted">Load a demo applicant '
                        'or fill in the form, attach a 6-month bank statement, and press '
                        '<i>Assess applicant</i>.</p></div>', unsafe_allow_html=True)
        else:
            a, r = saved["applicant"], saved["r"]
            for w in saved["warnings"]:
                st.warning(w)
            label, color, sub = VERDICT[r["verdict"]]
            reasons = "".join(f"<li>{x}</li>" for x in r["verdict_reasons"])
            st.markdown(f'<div class="card verdict" style="--c:{color}"><div class="label">{label}</div>'
                        f'<div class="sub">{sub} Statement: {saved["source"] or "none provided"}.</div>'
                        f'{"<ul>" + reasons + "</ul>" if reasons else ""}</div>', unsafe_allow_html=True)

            if r["recourse"]:
                st.markdown('<p class="eyebrow">What would help</p>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="chip">{x["change"]}</span>' for x in r["recourse"]),
                            unsafe_allow_html=True)

            # --- policy checks
            st.markdown('<p class="eyebrow">Lending-policy checks (these decide)</p>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="check"><span class="pill {c["status"]}">{c["status"].upper()}</span>'
                f'<div><div class="nm">{c["name"]}</div><div class="dt">{c["detail"]}</div></div></div>'
                for c in r["policy"]["checks"])
            st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

            # --- model evidence
            st.markdown('<p class="eyebrow">Risk-model evidence (supporting; can only escalate to refer)</p>',
                        unsafe_allow_html=True)
            pct = r["consensus_percentile"]
            if pct is None:
                st.markdown('<div class="card">No risk model had enough evidence to vote.</div>',
                            unsafe_allow_html=True)
            else:
                mrows = ""
                for m in r["models"].values():
                    if m["qualifies"]:
                        note = (f'Applicants it scored like this defaulted {m["decile_default_rate"]:.0%} '
                                f'of the time (average {m["base_rate"]:.0%}).')
                        if m["out_of_range"]:
                            note += " Some inputs are outside what it was trained on, so its vote counts less."
                        mrows += (f'<div class="mrow"><span>{m["label"]}</span><div class="mbar">'
                                  f'<span style="width:{max(m["percentile"], 1):.0f}%"></span></div>'
                                  f'<span class="mono">{m["percentile"]:.0f}</span></div>'
                                  f'<div class="muted" style="margin:-2px 0 6px">{note}</div>')
                    else:
                        mrows += (f'<div class="mrow"><span>{m["label"]}</span>'
                                  f'<span class="muted">did not vote: {m["reason"]}</span><span></span></div>')
                conf = r["confidence"]
                st.markdown(
                    f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:baseline">'
                    f'<span>Riskier than <b class="mono" style="font-size:20px">{pct:.0f}%</b> of past applicants</span>'
                    f'<span class="pill {"pass" if conf["label"] == "High" else "caution" if conf["label"] == "Medium" else "fail"}">'
                    f'{conf["label"].upper()} AGREEMENT</span></div>'
                    f'<div class="meter"><div class="dot" style="left:calc({pct:.0f}% - 2px)"></div></div>'
                    f'<div class="meter-scale"><span>safer</span><span>50</span><span>80</span><span>riskier</span></div>'
                    f'<div style="margin-top:12px">{mrows}</div>'
                    f'<div class="muted">{conf["detail"]}</div></div>', unsafe_allow_html=True)

                exp = r["explanation"]
                if exp:
                    ups = "".join(f'<div class="factor"><span class="up">&uarr;</span> {f["label"]}</div>'
                                  for f in exp["risk_up"])
                    downs = "".join(f'<div class="factor"><span class="down">&darr;</span> {f["label"]}</div>'
                                    for f in exp["risk_down"])
                    c1, c2 = st.columns(2)
                    c1.markdown(f'<div class="card"><div class="eyebrow" style="margin-top:0">Raises risk</div>'
                                f'{ups or "<span class=muted>Nothing notable</span>"}</div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="card"><div class="eyebrow" style="margin-top:0">Lowers risk</div>'
                                f'{downs or "<span class=muted>Nothing notable</span>"}</div>', unsafe_allow_html=True)
                    if any(f["key"] == "age" for f in exp["risk_up"]):
                        st.caption("Fairness note: age raised the models' risk estimate for this applicant. "
                                   "Our audit found these models less fair by age group than the full research "
                                   "models, which is one reason they cannot decide on their own.")

            if r["statement_insights"]:
                st.markdown('<p class="eyebrow">From the bank statement</p>', unsafe_allow_html=True)
                st.markdown('<div class="card">' + "".join(f'<div class="factor">&middot; {x}</div>'
                                                          for x in r["statement_insights"]) + "</div>",
                            unsafe_allow_html=True)

            with st.expander("Audit view: per-model detail, evidence tags and SHAP values"):
                for name, m in r["models"].items():
                    st.markdown(f"**{m['label']}** — evidence {m['applicability']:.0%} "
                                f"(vote needs {ls.APPLICABILITY_THRESHOLD:.0%})"
                                + (f", probability {m['probability']:.3f}, percentile {m['percentile']:.0f}, "
                                   f"vote weight {m['weight']:.3f}" if m["qualifies"] else ", did not vote"))
                    if m["qualifies"] and m["out_of_range"]:
                        st.caption("Outside training range: " + ", ".join(
                            f"{o['feature']}={o['value']:.2f} (trained on {o['lo']:.2f}–{o['hi']:.2f})"
                            for o in m["out_of_range"]))
                    if r["explanation"] and name in r["explanation"]["per_model"]:
                        df = pd.DataFrame(r["explanation"]["per_model"][name])[["feature", "value", "shap", "source"]]
                        df = df[df["shap"].abs() > 1e-4].round(4)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                adv = r.get("africa_advisory")
                if adv and adv.get("probability") is not None:
                    st.markdown(f"**Africa Synth behavioural model (advisory, not counted):** "
                                f"risk probability {adv['probability']:.3f}")
                    st.caption(adv["caveat"])

            st.download_button("Download decision report (PDF)", data=build_report(a, r),
                               file_name="fairscore_decision_report.pdf", mime="application/pdf",
                               use_container_width=True)

# ===========================================================================
# How it decides
# ===========================================================================
with tab_how:
    P = ls.POLICY
    st.markdown("### Two layers, one recommendation")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown(f"""
<div class="card"><div class="eyebrow" style="margin-top:0">1 · Lending-policy checks: these decide</div>
<div class="factor"><b>Affordability (FOIR):</b> existing EMIs + new EMI (at {P['annual_interest_rate']:.0%} p.a.)
must stay within {P['foir_pass']:.0%} of verified income; above {P['foir_fail']:.0%} fails.</div>
<div class="factor"><b>Account conduct:</b> balance below zero on at most {P['overdraft_pass']:.0%} of
transactions; above {P['overdraft_fail']:.0%}, or 3+ bounce/penalty charges, fails.</div>
<div class="factor"><b>Income stability:</b> month-to-month variation up to {P['volatility_pass']:.0%};
above {P['volatility_fail']:.0%} fails.</div>
<div class="factor"><b>Gambling / betting:</b> up to {P['betting_pass']:.0%} of spending; above
{P['betting_fail']:.0%} fails.</div>
<div class="factor"><b>Evidence and honesty:</b> 6 months of statement, declared income supported,
declared loans match the EMIs seen.</div>
<p class="muted">Any fail → recommend decline. Any caution → refer. These are configurable lender settings.</p></div>
""", unsafe_allow_html=True)
    c2.markdown(f"""
<div class="card"><div class="eyebrow" style="margin-top:0">2 · Risk models: supporting evidence</div>
<div class="factor">Three CatBoost models trained on public credit datasets, restricted to features a real
applicant can provide: <b>no gender, no nationality, no currency amounts, no credit-bureau data</b>.</div>
<div class="factor">A model votes only if at least {ls.APPLICABILITY_THRESHOLD:.0%} of its decision-relevant
information is evidenced (SHAP-weighted). Inputs outside its training range reduce its vote.</div>
<div class="factor">Each score becomes a <b>percentile</b> against that model's own test set, so models trained on
populations with different default rates can be combined; votes are weighted by evidence and accuracy (Gini).</div>
<div class="factor">If the consensus is in the riskiest {100 - ls.DECLINE_FROM:.0f}%, an otherwise clean case is
<b>referred</b>. The models never approve or decline on their own.</div></div>
""", unsafe_allow_html=True)
    st.info("Why models can't decide alone: trained on 1970s Germany, 2008 USA and an international lender, "
            "they disagree strongly on Indian applicants (different credit-card use, ages and spending patterns). "
            "The framework measures that disagreement instead of hiding it.")

# ===========================================================================
# Research results (read live from results/)
# ===========================================================================
def latest(pattern):
    files = sorted(glob.glob(str(ROOT / "results" / pattern)))
    return pd.read_csv(files[-1]) if files else None


def catboost_auc(pattern):
    df = latest(pattern)
    if df is None:
        return None
    row = df[df["model"] == "CatBoost"]
    return float(row["roc_auc"].iloc[0]) if len(row) else None


with tab_research:
    st.markdown("### Does the framework hold up across datasets?")
    cost = latest("deployability_cost.csv")
    mit = {d: latest(f"mitigation/before_after_v2_{d}.csv") for d in ["german_credit", "gmsc", "homecredit"]}
    meta = {
        "german_credit": ("German Credit", "1,000", "baseline_2*.csv"),
        "gmsc": ("Give Me Some Credit", "150,000", "gmsc_baseline_2*.csv"),
        "homecredit": ("Home Credit", "307,511", "homecredit_baseline_balanced_*.csv"),
    }
    rows = []
    for d, (label, n, pattern) in meta.items():
        crow = cost[cost["dataset"] == d].iloc[0] if cost is not None and (cost["dataset"] == d).any() else None
        m = mit[d]
        before = m.iloc[0] if m is not None else None
        after = m.iloc[-1] if m is not None else None
        rows.append({
            "Dataset": label, "Rows": n,
            "AUC: research model": catboost_auc(pattern),
            "AUC: deployable model": crow["auc_deployable"] if crow is not None else None,
            "AUC: fairness-mitigated": after["roc_auc"] if after is not None else None,
            "DP ratio before → after mitigation":
                f'{before["demographic_parity_ratio"]:.2f} → {after["demographic_parity_ratio"]:.2f}'
                if m is not None else "",
            "EO diff before → after":
                f'{before["equalized_odds_difference"]:.2f} → {after["equalized_odds_difference"]:.2f}'
                if m is not None else "",
        })
    africa = latest("africa_baseline_metrics.csv")
    rows.append({"Dataset": "Africa Synth (behavioural)", "Rows": "101",
                 "AUC: research model": float(africa["roc_auc"].iloc[0]) if africa is not None else None,
                 "AUC: deployable model": None, "AUC: fairness-mitigated": None,
                 "DP ratio before → after mitigation": "n/a (21 test rows)", "EO diff before → after": ""})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in
        ["AUC: research model", "AUC: deployable model", "AUC: fairness-mitigated"]})
    st.caption("All models are CatBoost (within 0.002 AUC of the best of five algorithms on German Credit and "
               "GMSC). DP ratio: 1.0 = equal approval rates across age groups. EO diff: 0 = equal error rates. "
               "Africa Synth's AUC of 1.0 on 21 test rows is evidence of overfitting, not skill.")

    chart = table[table["Dataset"] != "Africa Synth (behavioural)"].melt(
        id_vars="Dataset", value_vars=["AUC: research model", "AUC: deployable model", "AUC: fairness-mitigated"],
        var_name="Model", value_name="AUC").dropna()
    if len(chart):
        import altair as alt
        order = ["AUC: research model", "AUC: deployable model", "AUC: fairness-mitigated"]
        st.altair_chart(alt.Chart(chart).mark_bar(clip=True).encode(
            x=alt.X("Model:N", title=None, sort=order, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.5, 0.9], clamp=True), title="ROC-AUC"),
            color=alt.Color("Model:N", sort=order,
                            scale=alt.Scale(domain=order, range=["#16324A", "#B8863B", "#9E3A2E"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Dataset", "Model", alt.Tooltip("AUC:Q", format=".3f")],
            column=alt.Column("Dataset:N", title=None, sort=list(table["Dataset"])),
        ).properties(width=180, height=220))

    st.markdown("#### Key findings")
    st.markdown("""
- **The cost of deployability.** Restricting models to what an applicant can actually provide (no bureau data,
  no protected attributes, no currency amounts) costs 0.03 AUC on German Credit (5-fold CV) and 0.12–0.17 on
  GMSC and Home Credit, whose strongest features are credit-bureau data.
- **Removing protected attributes did not make models fairer.** At the same flag rate the age-group DP ratio
  stayed at 0.55 on German Credit and fell on GMSC (0.48 → 0.16) and Home Credit (0.54 → 0.42): with bureau data
  gone the models lean harder on age (fairness through unawareness fails).
- **Mitigation trades one kind of fairness for another.** Exponentiated-gradient mitigation raises the
  demographic-parity ratio sharply on every dataset, but the equalized-odds difference worsens on German Credit
  and GMSC, consistent with the known incompatibility of the two criteria.
- **Foreign models don't transfer.** On the Indian demo applicants the three models' risk percentiles differ by
  54–72 points. One feature (count of open credit lines) put every Indian applicant in GMSC's riskiest 2% and was
  removed. Hence the policy-first design.
""")
