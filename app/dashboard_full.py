"""FAIRScore AI -- loan officer dashboard.   Run:  streamlit run app\\dashboard.py"""
import glob
import hashlib
from datetime import datetime
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import audit_store  # noqa: E402
import live_scoring as ls  # noqa: E402
import service  # noqa: E402
from report_pdf import build_report  # noqa: E402

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
.info { background:#E7EEF4; color:var(--navy); }
.fail { background:var(--risk-soft); color:var(--risk); }
.check .nm { font-weight:600; }
.check .dt { color:var(--ink-soft); font-size:12.5px; }
.chip { display:inline-block; border:1px solid var(--line); background:#fff; padding:8px 12px; margin:0 8px 8px 0;
        font-size:13px; border-radius:3px; }
.meter { position:relative; height:12px; border-radius:6px; margin:12px 0 4px;
         background:linear-gradient(90deg, var(--safe-soft) 0 35%, #E7EEF4 35% 65%, var(--gold-soft) 65% 80%, var(--risk-soft) 80% 100%); }
.meter .dot { position:absolute; top:-4px; width:4px; height:20px; background:var(--ink); border-radius:2px; }
.meter-scale { display:grid; grid-template-columns:35% 30% 15% 20%; font-size:11px; color:var(--ink-soft); text-align:center; }
.mrow { display:grid; grid-template-columns: 170px 1fr 54px; gap:12px; align-items:center; font-size:13px; padding:6px 0; }
.mbar { height:8px; background:#ECE8DE; border-radius:4px; position:relative; }
.mbar span { position:absolute; left:0; top:0; bottom:0; border-radius:4px; background:var(--navy); }
.mono { font-family:'IBM Plex Mono',monospace; }
.factor { font-size:13.5px; padding:4px 0; }
.up { color:var(--risk); font-weight:600; }
.down { color:var(--safe); font-weight:600; }
.muted { color:var(--ink-soft); font-size:12.5px; }
.ref { color:var(--ink-soft); font-size:11.5px; margin-top:10px; font-family:'IBM Plex Mono',monospace; }
.tiles { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin:4px 0 6px; }
@media (max-width: 900px) { .tiles { grid-template-columns:repeat(2, 1fr); } }
.tile { background:var(--card); border:1px solid var(--line); border-top:3px solid var(--line); border-radius:4px; padding:12px 14px; }
.tile.good { border-top-color:var(--safe); } .tile.warn { border-top-color:var(--gold); } .tile.bad { border-top-color:var(--risk); }
.t-label { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-soft); font-weight:600; }
.t-val { font-family:'IBM Plex Mono',monospace; font-size:21px; font-weight:600; margin:4px 0 2px; }
.t-sub { font-size:11.5px; color:var(--ink-soft); line-height:1.35; }
.lvl-row { display:flex; gap:14px; align-items:center; font-size:14px; }
.lvl { font-family:'Source Serif 4',serif; font-weight:700; font-size:20px; padding:4px 14px; border-radius:4px; }
.lvl-low { background:var(--safe-soft); color:var(--safe); } .lvl-moderate { background:#E7EEF4; color:var(--navy); }
.lvl-elevated { background:var(--gold-soft); color:var(--gold); } .lvl-high { background:var(--risk-soft); color:var(--risk); }
.fgrid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:8px; }
@media (max-width: 900px) { .fgrid { grid-template-columns:1fr; } }
.frow { display:grid; grid-template-columns: 1fr 58px; gap:2px 10px; font-size:13px; padding:4px 0;
        border-bottom:1px solid #ECE8DE; }
.frow .muted { grid-column: 1 / span 2; font-size:11.5px; }
.req { font-size:10px; letter-spacing:.04em; color:var(--risk); background:var(--risk-soft); padding:1px 6px;
       border-radius:3px; margin-left:6px; vertical-align:middle; }
div[data-testid="stForm"] { background:var(--card); border:1px solid var(--line); border-radius:4px; }
</style>
<div class="topbar"><span class="name">FAIRScore AI</span>
<span class="tag">Explainable, fairness-aware credit decision support &middot; recommendations for a credit officer, not automatic decisions</span></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Form state. Required fields start empty so nothing is assumed about the applicant.
# ---------------------------------------------------------------------------
FIELDS = ls.REQUIRED_FIELDS + ls.OPTIONAL_FIELDS
for k in FIELDS:
    st.session_state.setdefault(k, None)
st.session_state.setdefault("result", None)
st.session_state.setdefault("auto_assess", False)

LABELS = {
    "age": "Age", "employment_type": "Employment type", "employment_years": "Years in current work",
    "housing": "Housing", "requested_amount": "Loan amount", "requested_duration_months": "Term",
    "purpose": "Loan purpose", "dependents": "Dependents", "owns_car": "Owns a vehicle",
    "total_savings": "Total savings",
}
STATUS_WORD = {"pass": "OK", "caution": "REVIEW", "fail": "FAIL", "info": "NOTE"}


@st.cache_resource(show_spinner=False)
def _statement_store():
    return {}


def read_statements(files, passwords, progress=None):
    """Reads the uploaded statement(s) once per (files, passwords); re-assessments are instant.
    Files are never stored: each PDF is read from a private temporary file deleted straight away."""
    key = hashlib.sha256(b"".join(hashlib.sha256(d).digest() for _, d in files) + passwords.encode()).hexdigest()
    store = _statement_store()
    if key not in store:
        pw = [p.strip() for p in passwords.split(",") if p.strip()] if passwords else []
        store[key] = service.read_statements(files, pw, progress)
        while len(store) > 8:                      # keep only the most recent few in memory
            store.pop(next(iter(store)))
    df, stmt, warnings, sha = store[key]
    return df.copy(), dict(stmt), list(warnings), sha


def confirm_loans(n):
    st.session_state["active_credit_lines"] = n
    st.session_state["auto_assess"] = True


@st.cache_resource(show_spinner=False)
def warm_up():
    for name in ls.VOTING_MODELS:      # load models + explainers once per server
        ls.load_model(name)
        ls.explainer(name)
    return True


def money(x):
    return f"−₹{abs(x):,.0f}" if x < 0 else f"₹{x:,.0f}"


def meta_line(r):
    """Decision confidence + the per-application fairness check, in one line."""
    conf, flip = r.get("decision_confidence") or {}, r.get("age_flip") or {}
    parts = []
    if conf:
        tip = "; ".join(conf.get("notes") or []) or "clean statement, clear of every limit"
        parts.append(f'<span title="{tip}"><b>Confidence:</b> {conf["label"]}</span>')
    if flip.get("ages_tested"):
        parts.append("<b>Fairness check:</b> " + (
            f'same recommendation at ages {", ".join(map(str, flip["ages_tested"]))} ✓' if flip["passed"]
            else '<span style="color:var(--risk)">changes with age alone: review</span>'))
    return f'<div class="muted" style="margin-top:10px">{" &nbsp;·&nbsp; ".join(parts)}</div>' if parts else ""


VERDICT = {
    "approve": ("Recommend approve", "var(--safe)", "All lending-policy checks passed and the risk models "
                "did not flag elevated risk."),
    "refer": ("Refer to credit officer", "var(--gold)", "The application needs a human look before a decision."),
    "decline": ("Recommend decline", "var(--risk)", "One or more lending-policy checks failed."),
}

tab_assess, tab_history, tab_how, tab_research = st.tabs(
    ["Assess an application", "History", "How it decides", "Research results"])

# ===========================================================================
# Assess
# ===========================================================================
with tab_assess:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown('<p class="eyebrow">New loan application</p>', unsafe_allow_html=True)
        with st.form("intake"):
            st.markdown('<p class="eyebrow">1 · Bank statement <span class="req">required</span></p>',
                        unsafe_allow_html=True)
            upload = st.file_uploader(
                "Last 6 months: PDF e-statement from net banking, or Account Aggregator data",
                type=["pdf", "json", "xml"], accept_multiple_files=True,
                help="Upload one statement per bank account the applicant uses (up to 4): transfers between "
                     "them are matched and not counted twice. A text PDF from net banking works for any bank; "
                     "Account Aggregator data (RBI AA framework, JSON / XML) is bank-signed and needs no reading.")
            st.text_input("PDF password (only if the statement is protected)", key="pdf_password",
                          type="password", placeholder="often your date of birth or part of name / PAN",
                          help="Banks usually protect e-statements with a password such as DDMMYYYY or the first "
                               "letters of your name plus date of birth. For several files with different "
                               "passwords, separate them with commas. Used only to open the files.")

            st.markdown('<p class="eyebrow">2 · Loan request <span class="req">required</span></p>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.number_input("Amount (₹)", min_value=10_000, max_value=50_000_000, step=10_000,
                            key="requested_amount", placeholder="e.g. 300000")
            c2.number_input("Term (months)", min_value=6, max_value=84, step=6,
                            key="requested_duration_months", placeholder="e.g. 24")
            st.selectbox("Purpose", list(ls.PURPOSE_OPTIONS), key="purpose", index=None,
                         placeholder="Select purpose")

            st.markdown('<p class="eyebrow">3 · About the applicant <span class="req">required</span></p>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.number_input("Age", min_value=18, max_value=80, key="age", placeholder="e.g. 32")
            c2.selectbox("Housing", list(ls.HOUSING_OPTIONS), key="housing", index=None,
                         placeholder="Select")
            c1.selectbox("Employment type", list(ls.EMPLOYMENT_TYPES), key="employment_type",
                         index=None, placeholder="Select")
            c2.number_input("Years in current work", min_value=0.0, max_value=50.0, step=0.5,
                            key="employment_years", placeholder="e.g. 3")
            c1.number_input("Dependents", min_value=0, max_value=10, key="dependents", placeholder="e.g. 1")
            c2.number_input("Total savings (₹)", min_value=0, max_value=100_000_000, step=5000,
                            key="total_savings", placeholder="e.g. 50000")
            st.radio("Owns a car or two-wheeler", [True, False], key="owns_car", index=None, horizontal=True,
                     format_func=lambda v: "Yes" if v else "No")

            with st.expander("4 · Optional details: improve accuracy and enable cross-checks"):
                st.caption("Leave blank if unknown. Blank items are read from the statement where possible "
                           "(income, existing EMIs). Anything you enter is cross-checked against the statement.")
                c1, c2 = st.columns(2)
                c1.number_input("Declared monthly income (₹)", min_value=0, max_value=10_000_000, step=1000,
                                key="monthly_income", placeholder="from statement")
                c2.number_input("Active loans + credit cards", min_value=0, max_value=30,
                                key="active_credit_lines", placeholder="from statement")
                c1.number_input("Home / property loans", min_value=0, max_value=10,
                                key="property_loans", placeholder="unknown")
                c2.selectbox("Education", list(ls.EDUCATION_LEVELS), key="education", index=None,
                             placeholder="unknown")
                c1.number_input("Price of the item being bought (₹)", min_value=0, max_value=1_000_000_000,
                                step=1000, key="asset_price", placeholder="vehicle / appliance loans",
                                help="On-road price or quotation. Checks the applicant's own contribution "
                                     "(loan-to-value): on Indian new-to-credit vehicle loans, financing more "
                                     "than 80% of the price went with markedly more defaults.")
                st.text_input("Name as on the bank account", key="account_holder_name",
                              placeholder="to confirm the statement is the applicant's own",
                              help="Compared with the account holder's name printed on the statement. "
                                   "Only the result (match / no match) is kept; the name is not stored.")

            submitted = st.form_submit_button("Assess application", type="primary", width="stretch")

        st.caption("Sample PDF statements for testing are in data/samples (see TESTING_GUIDE.md).")

    if st.session_state["auto_assess"]:
        st.session_state["auto_assess"] = False
        submitted = True

    if submitted:
        applicant = {k: st.session_state[k] for k in FIELDS}
        problems = [f"Upload the applicant's bank statement"] if not upload else []
        missing = ls.missing_required(applicant)
        if missing:
            problems.append("Fill in: " + ", ".join(LABELS[m] for m in missing))
        if problems:
            with left:
                st.error(" · ".join(problems))
        else:
            stmt_df = stmt = None
            with left:
                bar = st.progress(0.0, text="Reading the statement...")
            try:
                files = [(u.name, bytes(u.getbuffer())) for u in upload]
                parsed = read_statements(files, st.session_state.get("pdf_password") or "",
                                         lambda i, n: bar.progress(i / n, text=f"Reading page {i} of {n}..."))
                stmt_df, stmt = parsed[0], parsed[1]
            except service.ApplicationError as exc:
                with left:
                    st.error(str(exc))
            except Exception as exc:   # noqa: BLE001
                with left:
                    st.error(f"This statement could not be read ({type(exc).__name__}). Please upload the PDF "
                             f"e-statement downloaded from net banking.")
            finally:
                bar.empty()

            if stmt is not None:
                try:
                    with st.spinner("Assessing..."):
                        warm_up()
                        out = service.assess_application(applicant, parsed, channel="dashboard")
                except service.ApplicationError as exc:
                    with left:
                        st.error(str(exc))
                else:
                    st.session_state["result"] = {"applicant": out["applicant"], "r": out["result"],
                                                  "warnings": out["warnings"],
                                                  "source": ", ".join(u.name for u in upload),
                                                  "ref": out["reference"], "stmt": stmt, "stmt_df": stmt_df,
                                                  "cash_flow": ls.cash_flow(stmt_df, {**stmt, "income_basis": out["result"].get("income_basis") or ""}),
                                                  "detected_loans": stmt.get("detected_credit_signal_count", 0)}

    # -------------------------------------------------------------------
    with right:
        saved = st.session_state["result"]
        if not saved:
            st.markdown('<p class="eyebrow">Result</p>', unsafe_allow_html=True)
            st.markdown('<div class="card"><b>No assessment yet.</b><p class="muted">Upload the applicant\'s '
                        'last 6 months of bank statement (PDF), fill in the loan request and the details about '
                        'the applicant, then press <i>Assess application</i>.</p><p class="muted">The '
                        'statement is the main evidence: verified income, existing EMIs, balances, overdrafts '
                        'and spending are all read from it.</p></div>', unsafe_allow_html=True)
        else:
            a, r = saved["applicant"], saved["r"]
            pol = r["policy"]
            for w in saved["warnings"]:
                st.warning(w)

            # --- 1. verdict
            label, color, sub = VERDICT[r["verdict"]]
            reasons = "".join(f"<li>{x}</li>" for x in r["verdict_reasons"])
            st.markdown(f'<div class="card verdict" style="--c:{color}"><div class="label">{label}</div>'
                        f'<div class="sub">{sub}</div>'
                        f'{"<ul>" + reasons + "</ul>" if reasons else ""}'
                        f'{meta_line(r)}'
                        f'<div class="ref">Reference {saved["ref"]} · statement {saved["source"]} · '
                        f'policy {r.get("policy_version", "")}</div></div>',
                        unsafe_allow_html=True)

            if r.get("consistency") == "conflict" and saved.get("detected_loans"):
                n = int(saved["detected_loans"])
                st.button(f"Applicant confirms {n} active loan{'s' if n > 1 else ''}: update the form and re-assess",
                          on_click=confirm_loans, args=(max(n, int(a["active_credit_lines"] or 0)),),
                          type="primary", width="stretch")

            # --- 2. key numbers
            inc, new_emi, ex_emi = pol["income_used"], pol["new_emi"], pol["existing_emi"]
            surplus = inc - (pol["monthly_spend"] or 0) - new_emi if inc else None
            foir_cls = {"pass": "good", "caution": "warn", "fail": "bad"}[
                next(c["status"] for c in pol["checks"] if c["key"] == "affordability")]
            tiles = [
                ("Verified monthly income", money(inc) if inc else "–",
                 f"from {r.get('income_basis') or saved['stmt'].get('income_basis', 'the statement')}", ""),
                ("New EMI", money(new_emi), f"{money(a['requested_amount'])} over {a['requested_duration_months']} months "
                                             f"at {ls.POLICY['annual_interest_rate']:.0%}", ""),
                ("All EMIs vs income", f"{pol['foir']:.0%}" if inc else "–",
                 f"existing {money(ex_emi)} + new · limit {ls.POLICY['foir_pass']:.0%}", foir_cls),
                ("Eligible amount", money((r.get("eligibility") or {}).get("eligible_amount") or 0),
                 f"over {(r.get('eligibility') or {}).get('term') or a['requested_duration_months']} months · limit: "
                 f"{(r.get('eligibility') or {}).get('binding', '–')}",
                 "good" if ((r.get("eligibility") or {}).get("eligible_amount") or 0) >= a["requested_amount"]
                 else "warn"),
            ]
            st.markdown('<div class="tiles">' + "".join(
                f'<div class="tile {cls}"><div class="t-label">{t}</div><div class="t-val">{v}</div>'
                f'<div class="t-sub">{s_}</div></div>' for t, v, s_, cls in tiles) + "</div>",
                unsafe_allow_html=True)

            # --- 2b. FAIRScore + eligibility + offer
            fs, elig, offer = r.get("fairscore"), r.get("eligibility") or {}, r.get("offer")
            if fs:
                pos = (fs["score"] - 300) / 6
                weakest = [f for f in fs["factors"] if f["points"] < f["max"]][:4]
                strongest = sorted(fs["factors"], key=lambda f: f["points"] / max(f["max"], 1), reverse=True)[:3]
                rows_w = "".join(f'<div class="frow"><span>{f["label"]}</span><span class="mono">{f["points"]}/{f["max"]}'
                                 f'</span><span class="muted">{f["reason"]}</span></div>' for f in weakest) \
                    or '<div class="muted">Full points on every factor</div>'
                rows_s = "".join(f'<div class="frow"><span>{f["label"]}</span><span class="mono">{f["points"]}/{f["max"]}'
                                 f'</span><span class="muted">{f["reason"]}</span></div>' for f in strongest)
                band_cls = {"Excellent": "lvl-low", "Good": "lvl-low", "Fair": "lvl-moderate", "Weak": "lvl-elevated",
                            "Poor": "lvl-high"}.get(fs["band"], "lvl-moderate")
                elig_txt = ""
                if elig:
                    amt = elig.get("eligible_amount") or 0
                    elig_txt = (f'<div style="margin-top:10px"><b>Eligible amount:</b> {money(amt)} over '
                                f'{elig.get("term", a["requested_duration_months"])} months'
                                f'<span class="muted"> · binding limit: {elig.get("binding", "")}</span></div>')
                if offer:
                    o = offer[0]
                    elig_txt += (f'<div style="margin-top:6px"><b>Counter-offer:</b> approvable at {money(o["amount"])} '
                                 f'over {o["term"]} months (EMI {money(o["emi"])})</div>')
                st.markdown('<p class="eyebrow">FAIRScore (300-900) · banking behaviour + applicant profile, no credit history needed</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card"><div class="lvl-row"><span class="lvl {band_cls}">{fs["score"]}</span>'
                    f'<span><b>{fs["band"]}</b> · every point comes from the statement or the application; no age, '
                    f'gender or bureau data</span></div>'
                    f'<div class="meter" style="background:linear-gradient(90deg,var(--risk-soft) 0 50%,var(--gold-soft) 50% 58%,'
                    f'#E7EEF4 58% 67%,var(--safe-soft) 67% 100%)"><div class="dot" style="left:calc({pos:.0f}% - 2px)"></div></div>'
                    f'<div class="meter-scale" style="grid-template-columns:50% 8% 9% 33%"><span>Poor</span><span>Weak</span>'
                    f'<span>Fair</span><span>Good / Excellent</span></div>'
                    f'<div class="fgrid"><div><div class="eyebrow">Weakest factors</div>{rows_w}</div>'
                    f'<div><div class="eyebrow">Strongest factors</div>{rows_s}</div></div>{elig_txt}</div>',
                    unsafe_allow_html=True)

            # --- 3. what would help
            if r["recourse"]:
                st.markdown('<p class="eyebrow">What would help</p>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="chip">{x["change"]}</span>' for x in r["recourse"]),
                            unsafe_allow_html=True)

            if r.get("documents"):
                st.markdown('<p class="eyebrow">Documents to collect before sanction</p>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="chip">{d}</span>' for d in r["documents"]), unsafe_allow_html=True)

            # --- 4. checks
            st.markdown('<p class="eyebrow">Checks</p>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="check"><span class="pill {c["status"]}">{STATUS_WORD[c["status"]]}</span>'
                f'<div><div class="nm">{c["name"]}</div><div class="dt">{c["detail"]}</div></div></div>'
                for c in pol["checks"])
            st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

            # --- 5. risk level + reasons
            pct, level = r["consensus_percentile"], r.get("risk_level")
            if pct is not None:
                agree = r["confidence"]["label"]
                agree_txt = ("The independent risk models agree on this." if agree == "High" else
                             "The independent risk models partly agree." if agree == "Medium" else
                             "The independent risk models disagree, so treat this level as indicative.")
                st.markdown('<p class="eyebrow">Credit risk level</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card"><div class="lvl-row"><span class="lvl lvl-{level.lower()}">{level}</span>'
                    f'<span>Riskier than <b>{pct:.0f}%</b> of past borrowers with similar information.</span></div>'
                    f'<div class="meter"><div class="dot" style="left:calc({pct:.0f}% - 2px)"></div></div>'
                    f'<div class="meter-scale"><span>Low</span><span>Moderate</span><span>Elevated</span>'
                    f'<span>High</span></div><div class="muted" style="margin-top:8px">{agree_txt} '
                    f'The risk level can send a case for review but never approves or declines on its own.</div></div>',
                    unsafe_allow_html=True)
                exp = r["explanation"]
                if exp:
                    ups = "".join(f'<div class="factor"><span class="up">&#9650;</span> {f["label"]}</div>'
                                  for f in exp["risk_up"])
                    downs = "".join(f'<div class="factor"><span class="down">&#9660;</span> {f["label"]}</div>'
                                    for f in exp["risk_down"])
                    c1, c2 = st.columns(2)
                    c1.markdown(f'<div class="card"><div class="eyebrow" style="margin-top:0">Working in favour</div>'
                                f'{downs or "<span class=muted>Nothing notable</span>"}</div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="card"><div class="eyebrow" style="margin-top:0">Working against</div>'
                                f'{ups or "<span class=muted>Nothing notable</span>"}</div>', unsafe_allow_html=True)

            # --- 6. cash flow
            cf = saved.get("cash_flow")
            if cf is not None and len(cf):
                import altair as alt
                st.markdown('<p class="eyebrow">Money in and out, month by month</p>', unsafe_allow_html=True)
                long = cf.melt(id_vars="Month", value_vars=["income", "other_in", "money_out"], var_name="Flow",
                               value_name="Amount")
                names = {"income": "Income counted", "other_in": "Not counted as income",
                         "money_out": "Money out"}
                long["Side"] = long["Flow"].map({"income": "in", "other_in": "in", "money_out": "out"})
                long["Flow"] = long["Flow"].map(names)
                long = long[long["Amount"] > 0]
                order = list(cf["Month"])
                bars = alt.Chart(long).mark_bar().encode(
                    x=alt.X("Month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("Side:N"),
                    y=alt.Y("sum(Amount):Q", title="Money in / out (₹)", axis=alt.Axis(format="~s")),
                    color=alt.Color("Flow:N", scale=alt.Scale(domain=list(names.values()),
                                                              range=["#2E6B4F", "#A9C7B5", "#B8863B"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    order=alt.Order("Flow:N", sort="ascending"),
                    tooltip=["Month", "Flow", alt.Tooltip("Amount:Q", format=",.0f")])
                layers = [bars]
                if "closing_balance" in cf.columns:
                    layers.append(alt.Chart(cf).mark_line(point=True, color="#16324A", strokeDash=[4, 3]).encode(
                        x=alt.X("Month:N", sort=order),
                        y=alt.Y("closing_balance:Q", title="Month-end balance (₹)",
                                axis=alt.Axis(format="~s", orient="right")),
                        tooltip=["Month", alt.Tooltip("closing_balance:Q", title="Closing balance", format=",.0f")]))
                chart = alt.layer(*layers).resolve_scale(y="independent").properties(height=240)
                st.altair_chart(chart, width="stretch")
                if r["statement_insights"]:
                    st.caption(" · ".join(r["statement_insights"]) + ". Dashed line: month-end balance (right axis).")

            # --- 6b. month-wise banking summary (as in a credit appraisal)
            monthly = (r.get("banking") or {}).get("monthly")
            if monthly:
                with st.expander("Month-wise banking summary"):
                    mdf = pd.DataFrame(monthly)
                    keep = [c for c in ["month", "income_counted", "credits", "debits", "emis", "investments",
                                        "cash_withdrawals", "abb", "min_balance", "closing_balance", "bounces",
                                        "transactions"] if c in mdf.columns]
                    st.dataframe(mdf[keep], hide_index=True, width="stretch", column_config={
                        "month": "Month", "income_counted": st.column_config.NumberColumn("Income counted (₹)", format="%d"),
                        "credits": st.column_config.NumberColumn("All credits (₹)", format="%d"),
                        "debits": st.column_config.NumberColumn("All debits (₹)", format="%d"),
                        "emis": st.column_config.NumberColumn("EMIs (₹)", format="%d"),
                        "investments": st.column_config.NumberColumn("Investments (₹)", format="%d"),
                        "cash_withdrawals": st.column_config.NumberColumn("Cash withdrawn (₹)", format="%d"),
                        "abb": st.column_config.NumberColumn("Average balance (₹)", format="%d"),
                        "min_balance": st.column_config.NumberColumn("Lowest balance (₹)", format="%d"),
                        "closing_balance": st.column_config.NumberColumn("Month-end balance (₹)", format="%d"),
                        "bounces": "Bounces", "transactions": "Transactions"})
                    st.caption("Average balance = mean of the day-end balances (ABB), as banks compute it. "
                               "Partial first / last months are shown but not used for averages.")

            # --- 7. what-if
            with st.expander("Try a different amount or term"):
                c1, c2 = st.columns(2)
                w_amt = c1.number_input("Amount (₹)", 10_000, 50_000_000, int(a["requested_amount"]), 10_000,
                                        key="wi_amount")
                w_term = c2.number_input("Term (months)", 6, 84, int(a["requested_duration_months"]), 6,
                                         key="wi_term")
                if (w_amt, w_term) != (a["requested_amount"], a["requested_duration_months"]):
                    q = ls.quick_decision({**a, "requested_amount": w_amt, "requested_duration_months": w_term},
                                          saved["stmt"], saved["stmt_df"])
                    wl, wc, _ = VERDICT[q["verdict"]]
                    st.markdown(f'<div class="card verdict" style="--c:{wc};padding:12px 16px">'
                                f'<div class="label" style="font-size:20px">{wl}</div><div class="sub">'
                                f'New EMI {money(q["policy"]["new_emi"])} · all EMIs {q["policy"]["foir"]:.0%} of income'
                                f'{" · risk " + q["risk_level"].lower() if q["risk_level"] else ""}</div>'
                                + ("<ul>" + "".join(f"<li>{x}</li>" for x in q["reasons"]) + "</ul>" if q["reasons"] else "")
                                + "</div>", unsafe_allow_html=True)
                else:
                    st.caption("Change the amount or term to see how the recommendation would change.")

            from applicant_letter import build_letter
            c1, c2 = st.columns(2)
            c2.download_button("Letter to the applicant (PDF)", data=build_letter(a, r, saved["ref"]),
                               file_name=f"fairscore_letter_{saved['ref']}.pdf", mime="application/pdf",
                               width="stretch", help="Plain-language outcome, main reasons and what would help, "
                                                     "as the RBI Fair Practices Code asks lenders to give in writing.")
            c1.download_button("Download credit appraisal memo (PDF)", data=build_report(a, r, saved["ref"]),
                               file_name=f"fairscore_CAM_{saved['ref']}.pdf", mime="application/pdf",
                               width="stretch")

# ===========================================================================
# History (audit trail)
# ===========================================================================
with tab_history:
    st.markdown("### Assessment history")
    try:
        hist = audit_store.history()
    except Exception as exc:   # noqa: BLE001
        hist = None
        st.error(f"The history database could not be opened ({type(exc).__name__}).")
    if hist is not None and len(hist):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Assessments", len(hist))
        for col, key in ((c2, "approve"), (c3, "refer"), (c4, "decline")):
            col.metric({"approve": "Recommended approve", "refer": "Referred", "decline": "Recommended decline"}[key],
                       int((hist["recommendation"] == key).sum()))
        pending = hist[(hist["recommendation"] == "refer") & hist["officer_decision"].isna()]
        c5.metric("Awaiting officer decision", len(pending))

        # --- human in the loop: record the officer's final decision on referred cases
        st.markdown('<p class="eyebrow">Record a credit officer\'s decision</p>', unsafe_allow_html=True)
        if len(pending):
            with st.form("officer"):
                c1, c2 = st.columns([1.2, 1])
                ref = c1.selectbox("Referred application", pending["reference"].tolist(),
                                   format_func=lambda x: f"{x} · ₹{pending.set_index('reference').loc[x, 'amount']:,.0f}")
                decision = c2.radio("Final decision", ["approve", "decline"], horizontal=True,
                                    format_func=str.capitalize)
                reason = st.text_input("Reason (recorded in the audit trail)",
                                       placeholder="e.g. Salary slips and Form 16 verified; employer confirmed")
                officer = st.text_input("Officer name / ID", placeholder="optional")
                if st.form_submit_button("Save decision", type="primary"):
                    try:
                        audit_store.record_officer_decision(ref, decision, reason, officer or None)
                        st.success(f"Saved: {ref} → {decision}")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            st.caption("No referred applications are waiting for a decision.")

        # --- closing the loop: how did approved loans turn out?
        st.markdown('<p class="eyebrow">Record how a loan turned out</p>', unsafe_allow_html=True)
        lent = hist[hist["final_decision"] == "approve"] if "final_decision" in hist else hist.iloc[0:0]
        if len(lent):
            with st.form("outcome"):
                c1, c2 = st.columns([1.2, 1])
                ref_o = c1.selectbox("Approved application", lent["reference"].tolist(), key="outcome_ref")
                outcome = c2.selectbox("Outcome", list(audit_store.OUTCOMES), key="outcome_value")
                if st.form_submit_button("Save outcome"):
                    try:
                        audit_store.record_outcome(ref_o, outcome)
                        st.success(f"Saved: {ref_o} → {outcome}")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        n_out = int(hist["outcome"].notna().sum()) if "outcome" in hist else 0
        st.caption(f"{n_out} outcome(s) recorded. With at least 200 repaid / defaulted loans (30 or more defaults), "
                   f"`python src\\train_from_outcomes.py` re-estimates the FAIRScore points from YOUR customers and "
                   f"proposes them for review (they are never applied automatically).")

        # --- fairness monitor on real decisions
        st.markdown('<p class="eyebrow">Fairness monitor: outcomes by age group</p>', unsafe_allow_html=True)
        table, alerts = audit_store.fairness_summary(hist)
        if table is not None and len(table):
            show = table.rename(columns={
                "age_group": "Age group", "assessments": "Assessments", "recommended_approve": "Recommended approve",
                "referred": "Referred", "recommended_decline": "Recommended decline",
                "final_approve": "Approved (final)", "disparate_impact": "Approval rate vs best group",
                "age_flip_failures": "Age-flip failures", "avg_fairscore": "Average FAIRScore"})
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="percent") for c in
                ["Recommended approve", "Referred", "Recommended decline", "Approved (final)"]} | {
                "Approval rate vs best group": st.column_config.NumberColumn(format="%.2f")})
            for a_ in alerts:
                st.warning(a_)
            small = table[table["assessments"] < audit_store.MIN_GROUP_SIZE]["age_group"].tolist()
            st.caption(
                "Age is never a model input; it is used only for the eligibility rule (loan must end before the "
                "maximum age). This table checks the outcome anyway, because other inputs (e.g. years in work) "
                "can act as proxies. A ratio below 0.80 (four-fifths rule) is a prompt to investigate, not proof "
                "of bias: groups also differ in income and loan size. Every assessment is also re-decided at "
                "other ages (age-flip test)."
                + (f" Too few assessments yet to judge: {', '.join(small)} (needs {audit_store.MIN_GROUP_SIZE})."
                   if small else ""))

        # --- model monitoring: drift of the cash-flow model's inputs
        try:
            import monitoring
            drift = monitoring.drift_report(hist)
        except Exception:   # noqa: BLE001  (monitoring must never break the page)
            drift = None
        if drift is not None and len(drift):
            st.markdown('<p class="eyebrow">Model monitoring: are applicants like the training data? (PSI)</p>',
                        unsafe_allow_html=True)
            st.dataframe(drift.rename(columns={"input": "Model input", "psi": "PSI", "status": "Status",
                                               "training_median": "Training median",
                                               "applicants_median": "Applicants median"}),
                         hide_index=True, width="stretch")
            n = drift.attrs.get("applications", 0)
            st.caption(f"Population Stability Index over {n} assessment(s): below 0.10 stable, 0.10-0.25 moderate shift, "
                       f"above 0.25 major shift. A major shift means the cash-flow model's percentiles no longer "
                       f"describe these applicants (it was trained on a 1990s Czech bank), which is why it can only "
                       f"refer cases: record loan outcomes so the scorecard can be recalibrated on your own customers."
                       + ("" if drift.attrs.get("enough") else f" Fewer than {monitoring.MIN_APPLICATIONS} assessments "
                          f"so far: treat these values as indicative."))

        st.markdown('<p class="eyebrow">All assessments</p>', unsafe_allow_html=True)
        cols = ["reference", "assessed_at", "recommendation", "fairscore", "officer_decision", "officer_reason",
                "outcome", "amount",
                "term_months", "purpose", "verified_income", "foir", "risk_level", "confidence_label", "age_group",
                "channel", "policy_version"]
        st.dataframe(hist[[c for c in cols if c in hist.columns]], hide_index=True, width="stretch", column_config={
            "reference": "Reference", "assessed_at": "Assessed at", "recommendation": "Recommendation",
            "fairscore": "FAIRScore", "outcome": "Loan outcome",
            "officer_decision": "Officer decision", "officer_reason": "Officer's reason", "term_months": "Term",
            "purpose": "Purpose", "risk_level": "Risk level", "confidence_label": "Confidence",
            "age_group": "Age group", "channel": "Channel", "policy_version": "Policy version",
            "amount": st.column_config.NumberColumn("Amount (₹)", format="%d"),
            "verified_income": st.column_config.NumberColumn("Verified income (₹)", format="%d"),
            "foir": st.column_config.NumberColumn("EMIs vs income", format="percent")})
        st.download_button("Download history (CSV)", data=hist.drop(columns=["applicant"]).to_csv(index=False).encode(),
                           file_name="fairscore_history.csv", mime="text/csv")
        st.caption("Each row records the policy and model versions used and a fingerprint of the statement file. "
                   "Uploaded statements themselves are never stored.")
    elif hist is not None:
        st.info("No assessments yet. Every assessment you run is recorded here with its reference number.")

# ===========================================================================
# How it decides
# ===========================================================================
with tab_how:
    P = ls.POLICY
    st.markdown("### Two layers, one recommendation")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown(f"""
<div class="card"><div class="eyebrow" style="margin-top:0">1 · Lending-policy checks: these decide</div>
<div class="factor"><b>No credit history needed:</b> no bureau score, past loans or card data are used. Everything is
read from the bank statement and the application, so first-time borrowers, gig workers and small business owners
can be assessed.</div>
<div class="factor"><b>Eligibility:</b> at least {P['min_age']} years old, and the loan must end before the maximum
age for the employment type (e.g. {P['max_age_at_maturity'].get('Salaried (private sector)')} salaried,
{P['max_age_at_maturity'].get('Self-employed / business owner')} self-employed). This is the <i>only</i> use of age.
Students and people without a job are judged on the income their statement proves: with ₹{P['own_income_min']:,.0f}
a month or more of their own (stipend, part-time, freelance) they are assessed like anyone else; below it a student
is referred for a co-applicant. Farmers' seasonal harvest income is expected to vary month to month.</div>
<div class="factor"><b>Verified income:</b> read from the statement. Regular salary (or any payer crediting a steady
amount every month) is relied on; refunds, reversals, transfers from the applicant's own accounts, loan disbursals
and deposit maturities are never counted. With no regular income (self-employed), all genuine inflows count.</div>
<div class="factor"><b>Income source:</b> salaried / pensioner applicants must show regular salary or pension
credits; self-employed applicants must show inflows from several different payers. Money that goes back and forth
with the same person (round-trip transfers) is not counted as income; when it is a large share of the money
received, the case is referred to ask what the transfers are.</div>
<div class="factor"><b>Affordability (FOIR):</b> existing EMIs + new EMI (at {P['annual_interest_rate']:.0%} p.a.)
must stay within {P['foir_pass']:.0%} of verified income; above {P['foir_fail']:.0%} fails.</div>
<div class="factor"><b>Account conduct:</b> balance below zero on at most {P['overdraft_pass']:.0%} of
transactions; above {P['overdraft_fail']:.0%}, or 3+ bounce/penalty charges, fails.</div>
<div class="factor"><b>Income stability:</b> month-to-month variation up to {P['volatility_pass']:.0%};
above {P['volatility_fail']:.0%} fails.</div>
<div class="factor"><b>Gambling / betting:</b> up to {P['betting_pass']:.0%} of spending; above
{P['betting_fail']:.0%} fails.</div>
<div class="factor"><b>Take-home after EMIs:</b> enough must remain for the household (₹{P['min_take_home']:,}
plus ₹{P['take_home_per_dependent']:,} per dependent). Money left after today's spending is shown as a note: on real
loans it did not separate defaulters.</div>
<div class="factor"><b>Evidence and honesty:</b> 6 months of statement; the running balance must add up on
every row (edited files are caught); declared income and loans must match the statement.</div>
<div class="muted" style="margin-top:8px">Any fail → recommend decline. Any caution → refer. These are lender settings in config/policy.toml (version {ls.POLICY_VERSION}), recorded with every decision.</div>
<div class="muted" style="margin-top:8px"><b>Reading the statement:</b> works with any bank's PDF layout. Columns are identified from their content and verified against the running balance, so every assessment states how many transactions were read and whether every balance reconciles.</div></div>
""", unsafe_allow_html=True)
    c2.markdown(f"""
<div class="card"><div class="eyebrow" style="margin-top:0">2 · Risk models: supporting evidence</div>
<div class="factor"><b>Cash-flow model</b> (the main one): trained on 679 REAL loans whose bank-account history and
repayment outcome are known (Berka, a Czech bank). Inputs are only statement ratios -- EMIs vs income, income
regularity, overdrafts, days at a low balance, lowest balance, average balance vs EMI, penalty charges -- with
monotone constraints so a worse habit can never lower the risk. Cross-validated AUC 0.86.</div>
<div class="factor">Three supporting CatBoost models trained on public credit datasets (German Credit, Give Me Some
Credit, Home Credit), restricted to features a real applicant can provide.</div>
<div class="factor">None of them uses <b>age, gender, nationality, currency amounts or credit-bureau data</b>.</div>
<div class="factor"><b>FAIRScore 300-900</b>: an expert points scorecard of banking behaviour (conduct, affordability,
income regularity, balances, other borrowing, gambling, savings, household bills paid, evidence, tenure, cash use)
plus the applicant profile (type of work, home and assets, dependents, declared savings vs the loan, about 10% of
the points), so two people with the same statement score close but not the same.
Below 600 an otherwise clean case is referred. On the real loans it ranked risk with AUC 0.77 without being fitted to
them.</div>
<div class="factor">A model votes only if at least {ls.APPLICABILITY_THRESHOLD:.0%} of its decision-relevant
information is evidenced (SHAP-weighted). Inputs outside its training range reduce its vote.</div>
<div class="factor">Each score becomes a <b>percentile</b> against that model's own test set, so models trained on
populations with different default rates can be combined; votes are weighted by evidence and accuracy (Gini).</div>
<div class="factor">If the consensus is in the riskiest {100 - ls.DECLINE_FROM:.0f}%, an otherwise clean case is
<b>referred</b>. The models never approve or decline on their own.</div>
<div class="factor"><b>Decision confidence:</b> lowered when the statement did not fully reconcile, income is business inflows rather than a salary,
EMIs sit within 5 points of the limit, or the models place the applicant in the elevated band or disagree. Below
{P['min_confidence_to_approve']:.0%} an otherwise approvable case goes to an officer (manual review).</div>
<div class="factor"><b>Fairness check on every decision:</b> the application is re-decided at other ages; if the
recommendation changed with age alone, it is flagged. The History tab tracks outcomes by age group.</div></div>
""", unsafe_allow_html=True)
    st.info("Why models still can't decide alone: the cash-flow model learned from a 1990s Czech bank and the "
            "others from 1970s Germany, 2008 USA and an international lender. They rank risk, but their percentiles are "
            "relative to those populations, so a lending decision rests on transparent policy checks on verified "
            "statement facts. Recording outcomes in the History tab lets the scorecard be recalibrated on your own "
            "customers. Where the models disagree, decision confidence drops instead of the disagreement being hidden.")

with tab_how:
    st.markdown("### Explanation check: SHAP and LIME")
    st.caption("SHAP attributes each model's own prediction exactly; LIME fits a simple local model around the "
               "applicant. When both name the same main factors in the same direction, the explanation is robust.")
    saved = st.session_state.get("result")
    if not saved:
        st.info("Assess an application first; its explanation can then be checked here.")
    elif saved["r"]["voters"]:
        key = f"lime_{saved['ref']}"
        if st.button(f"Compare SHAP and LIME for {saved['ref']}", type="primary") or key in st.session_state:
            if key not in st.session_state:
                with st.spinner("Running LIME (a few seconds)..."):
                    st.session_state[key] = ls.lime_explanation(saved["r"], saved["applicant"], saved["stmt"])
            cols = st.columns(len(st.session_state[key]))
            for col, (name, m) in zip(cols, st.session_state[key].items()):
                with col:
                    st.markdown(f"**{m['label']}**")
                    st.caption(f"Top-5 factors shared: {m['top_overlap']:.0%} · same direction: "
                               f"{m['direction_agreement']:.0%} · LIME local fit R² {m['local_fit_r2']:.2f}")
                    df_ = pd.DataFrame(m["factors"]).rename(columns={"factor": "Factor", "shap": "SHAP", "lime": "LIME"})
                    st.dataframe(df_, hide_index=True, width="stretch", column_config={
                        "SHAP": st.column_config.NumberColumn(format="%+.3f"),
                        "LIME": st.column_config.NumberColumn(format="%+.3f")})
            st.caption("Positive = raises risk. SHAP values are in log-odds, LIME weights in probability: compare "
                       "the ranking and the sign, not the size. Factors that were not evidenced are left out.")


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
    # ---- validation on REAL loan outcomes (Berka) --------------------------------------
    bsum, cfm, cff = latest("berka_backtest_summary.csv"), latest("cashflow_model_metrics.csv"), latest("cashflow_fairness.csv")
    if bsum is not None and cfm is not None:
        import altair as alt
        st.markdown("### Tested on real loans: 679 bank statements with known repayment outcomes")
        st.caption("Berka / PKDD'99 data set (a Czech bank, 1993-98): for each loan, the 6 months of account "
                   "transactions BEFORE the loan were rebuilt into a statement and run through FAIRScore unchanged; "
                   "the recommendation is then compared with whether the loan was actually repaid. "
                   "No credit-bureau data exists in this data set, just as for a new-to-credit applicant.")
        rec = bsum[bsum["table"] == "Recommendation with the cash-flow model added"].copy()
        rec["group"] = pd.Categorical(rec["group"], ["approve", "refer", "decline"], ordered=True)
        rec = rec.sort_values("group")
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            base = float((bsum[bsum["table"] == "FAIRScore band"]["default_rate"]
                          * bsum[bsum["table"] == "FAIRScore band"]["loans"]).sum()
                         / bsum[bsum["table"] == "FAIRScore band"]["loans"].sum())
            st.altair_chart(alt.Chart(rec.assign(rate=rec["default_rate"] * 100)).mark_bar().encode(
                x=alt.X("group:N", title=None, sort=["approve", "refer", "decline"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("rate:Q", title="Loans that defaulted (%)"),
                color=alt.Color("group:N", scale=alt.Scale(domain=["approve", "refer", "decline"],
                                                           range=["#2E6B4F", "#B8863B", "#9E3A2E"]), legend=None),
                tooltip=["group", "loans", alt.Tooltip("rate:Q", format=".1f", title="default %")]
            ).properties(height=230, title=f"Default rate by recommendation (all loans: {base:.1%})"), width="stretch")
        with c2:
            show = cfm.rename(columns={"model": "Model", "cv_auc": "AUC", "cv_auc_sd": "± sd"})
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "AUC": st.column_config.NumberColumn(format="%.3f"), "± sd": st.column_config.NumberColumn(format="%.3f")})
            st.caption("AUC = how well the score ranks defaulters above repayers (0.5 = chance, 1 = perfect); "
                       "5-fold cross-validation repeated 5 times. The scorecard's points are expert settings, not "
                       "fitted to this data (one factor, household bills paid, was added after it proved predictive "
                       "here); the cash-flow model is trained on it (monotone, statement-only inputs).")
        q = bsum[bsum["table"].str.startswith("Cash-flow model quintile")]
        bands = bsum[bsum["table"] == "FAIRScore band"].copy()
        bands["group"] = pd.Categorical(bands["group"], ["Excellent", "Good", "Fair", "Weak", "Poor"], ordered=True)
        c1, c2 = st.columns(2, gap="large")
        c1.dataframe(bands.sort_values("group")[["group", "loans", "default_rate"]].rename(
            columns={"group": "FAIRScore band", "loans": "Loans", "default_rate": "Defaulted"}), hide_index=True,
            width="stretch", column_config={"Defaulted": st.column_config.NumberColumn(format="percent")})
        c2.dataframe(q[["group", "loans", "default_rate"]].rename(
            columns={"group": "Cash-flow model fifth", "loans": "Loans", "default_rate": "Defaulted"}), hide_index=True,
            width="stretch", column_config={"Defaulted": st.column_config.NumberColumn(format="percent")})
        checks = bsum[bsum["table"].str.startswith("Check:")].copy()
        checks["check"] = checks["table"].str[7:]
        piv = checks.pivot_table(index="check", columns="group", values="default_rate").reindex(
            columns=["pass", "caution", "fail"])
        with st.expander("Which lending-policy checks separate defaulters (default rate when the check passes / "
                         "refers / fails)"):
            st.dataframe(piv, width="stretch", column_config={c: st.column_config.NumberColumn(format="percent")
                                                              for c in ["pass", "caution", "fail"]})
            st.caption("This is how the policy was tuned: 'money left each month after current spending' flagged "
                       "85% of borrowers but barely separated defaulters (11.4% vs 6.9%), so it is now shown to the "
                       "officer as a note instead of referring the case.")
        if cff is not None:
            st.markdown("#### Fairness on real outcomes")
            f = cff.rename(columns={"attribute": "Attribute", "group": "Group", "loans": "Loans",
                                    "default_rate": "Defaulted", "approved_among_repayers": "Repayers approved",
                                    "approved_among_repayers_excl_age_rule": "Repayers approved (excl. age rule)",
                                    "flagged_among_defaulters": "Defaulters caught by model",
                                    "flagged_among_repayers": "Repayers wrongly flagged by model",
                                    "default_rate_if_approved": "Defaulted if approved"})
            cols = ["Attribute", "Group", "Loans", "Defaulted", "Repayers approved", "Repayers approved (excl. age rule)",
                    "Defaulters caught by model", "Repayers wrongly flagged by model", "Defaulted if approved"]
            st.dataframe(f[cols], hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="percent") for c in cols[3:]})
            st.caption("Equal opportunity in lending = people who would repay are approved at similar rates in every "
                       "group. Leaving out the legal / product age rule (minimum age 21, loan ending before the "
                       "maximum age), repayers are approved at similar rates across age groups and genders, and the "
                       "cash-flow model wrongly flags repayers at similar rates. Gender is never collected; it is "
                       "used here only to audit.")

    # ---- Indian evidence: L&T vehicle loans (src/india_ntc_study.py) ----------------------
    iov, imod, ifair, iltv = (latest(f"india_ntc_{n}.csv") for n in ("overview", "models", "fairness", "ltv"))
    if iov is not None and imod is not None:
        import altair as alt
        st.markdown("### Indian evidence: 233,154 real vehicle loans, half with no credit history")
        st.caption("L&T Financial Services two-wheeler / vehicle loans disbursed in India, Aug-Oct 2018 (Kaggle). "
                   "Outcome: default on the first EMI. 116,950 borrowers had 'No Bureau History Available' -- the "
                   "new-to-credit applicants FAIRScore is built for. The data has no bank statements, so it shows "
                   "what can be known WITHOUT one, how fairly, and which application facts carry the risk.")
        ntc = iov.set_index("segment")["default_rate"]
        c1, c2, c3 = st.columns(3)
        c1.metric("New to credit: defaulted", f"{ntc['New to credit (no bureau history)']:.1%}",
                  f"{ntc['New to credit (no bureau history)'] - ntc['Bureau-scored']:+.1%} vs bureau-scored",
                  delta_color="inverse")
        app_only = imod[(imod["population"] == "New to credit") &
                        (imod["model"] == "Application only, age-blind (CatBoost)")].iloc[0]
        with_bureau = imod[imod["model"] == "Application + bureau record"].iloc[0]
        c2.metric("Application only, no bureau (AUC)", f"{app_only['cv_auc']:.3f}",
                  f"{app_only['cv_auc'] - with_bureau['cv_auc']:+.3f} vs with a bureau record", delta_color="off")
        age = imod[imod["model"] == "Application + age (ablation)"].iloc[0]
        c3.metric("AUC lost by leaving age out", f"{age['cv_auc'] - app_only['cv_auc']:.3f}")
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            st.dataframe(imod.rename(columns={"population": "Borrowers", "model": "Model", "cv_auc": "AUC (5-fold)",
                                              "oot_auc": "AUC (October, out-of-time)", "note": "Note"})[
                ["Borrowers", "Model", "AUC (5-fold)", "AUC (October, out-of-time)", "Note"]].fillna({"Note": ""}),
                hide_index=True, width="stretch", column_config={
                    "AUC (5-fold)": st.column_config.NumberColumn(format="%.3f"),
                    "AUC (October, out-of-time)": st.column_config.NumberColumn(format="%.3f")})
        if iltv is not None:
            lt = iltv[iltv["population"] == "New to credit"]
            with c2:
                st.altair_chart(alt.Chart(lt.assign(rate=lt["default_rate"] * 100)).mark_bar(color="#16324A").encode(
                    x=alt.X("ltv_band:N", title="Loan as a share of the vehicle price", sort=None,
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("rate:Q", title="Missed the first EMI (%)"),
                    tooltip=["ltv_band", "loans", alt.Tooltip("rate:Q", format=".1f")]
                ).properties(height=250, title="New-to-credit borrowers: own contribution matters"), width="stretch")
        if ifair is not None:
            fa = ifair[ifair["attribute"] == "age_group"].pivot_table(index="group", columns="model",
                                                                     values="repayers_flagged")
            fa = fa.reindex(["under 25", "25-34", "35-44", "45-54", "55 and over"])
            with st.expander("Fairness on Indian outcomes: repayers wrongly flagged, by age group"):
                st.dataframe(fa.rename(columns={"age-blind": "Age-blind model (FAIRScore's rule)",
                                                "with age": "Model using age"}), width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="percent") for c in fa.columns})
                st.caption("Each model flags its riskiest 23% (the default rate). Using age adds almost no accuracy "
                           "but flags young repayers about twice as often as older ones; leaving age out keeps the "
                           "groups much closer. Real default rates differ little by age (21-24%).")
        st.caption("What it shows: (1) borrowers with no credit history default more and are the hardest to judge; "
                   "(2) with no bureau record and no statement, the application ranks their risk only weakly "
                   "(AUC about 0.60), and even the bureau score reaches about 0.61 on this outcome -- the gap that "
                   "bank-statement evidence is meant to fill; (3) leaving age out costs almost nothing and is fairer; "
                   "(4) the share of the price the borrower pays is the strongest application fact, which is why "
                   "FAIRScore now checks the loan-to-value of purchase loans when the price is entered.")

    # ---- bank-style model validation (src/validate_models.py) ---------------------------
    val = latest("model_validation.csv")
    if val is not None:
        import altair as alt
        st.markdown("### Model validation (as a bank's model-risk team would test it)")
        st.caption("Every figure comes from predictions made without seeing the loan's own outcome (out-of-fold or "
                   "out-of-time), with 95% bootstrap intervals. With 73 defaults the intervals are wide: that is the "
                   "honest size of the evidence.")

        def fmt(r):
            v = f"{r['value']:.3f}"
            return v + (f"  ({r['ci_low']:.3f} – {r['ci_high']:.3f})" if pd.notna(r["ci_low"]) else "")
        show = val[val["section"] != "Data"].copy()
        show = show[~show["metric"].str.startswith(("Gini:", "KS: Logistic", "KS: Foreign"))]
        show["Result (95% interval)"] = show.apply(fmt, axis=1)
        show["note"] = show["note"].fillna("")
        st.dataframe(show.rename(columns={"section": "Test", "metric": "Measure", "loans": "Loans", "note": "Note"})[
            ["Test", "Measure", "Result (95% interval)", "Loans", "Note"]], hide_index=True, width="stretch")
        cal, feats = latest("model_validation_calibration.csv"), latest("model_validation_features.csv")
        c1, c2 = st.columns(2, gap="large")
        if cal is not None:
            long = cal.melt(id_vars="decile", value_vars=["predicted", "actual"], var_name="series",
                            value_name="rate")
            c1.altair_chart(alt.Chart(long.assign(rate=long["rate"] * 100)).mark_line(point=True).encode(
                x=alt.X("decile:O", title="Risk decile (10 = riskiest)"), y=alt.Y("rate:Q", title="Default rate (%)"),
                color=alt.Color("series:N", scale=alt.Scale(domain=["predicted", "actual"],
                                                            range=["#16324A", "#B8863B"]), title=None),
                tooltip=["decile", "series", alt.Tooltip("rate:Q", format=".1f")]
            ).properties(height=240, title="Calibration: predicted vs actual default rate"), width="stretch")
        if feats is not None:
            names = {"foir": "All EMIs vs income (FOIR)", "emi_to_income": "New EMI vs income",
                     "loan_to_annual_income": "Loan vs annual income", "term_months": "Term",
                     "income_volatility": "Income volatility", "regular_income_share": "Share of income that is regular",
                     "overdraft_share": "Transactions in overdraft", "penalty_charges": "Penalty / bounce charges",
                     "low_balance_days": "Days under Rs 1,000", "negative_days": "Days below zero",
                     "abb_to_emi": "Average balance vs EMI", "min_balance_to_income": "Lowest balance vs income"}
            feats = feats.assign(inputs=feats["inputs"].map(lambda x: names.get(x, x)))
            c2.dataframe(feats.rename(columns={"kind": "Kind", "inputs": "Inputs", "auc": "AUC",
                                               "nonzero_share": "Present in"}), hide_index=True, width="stretch",
                         height=260, column_config={"AUC": st.column_config.NumberColumn(format="%.3f"),
                                                    "Present in": st.column_config.NumberColumn(format="percent")})
        st.warning("What this does NOT show: performance on Indian borrowers. The model is mostly a balance-buffer "
                   "model (lowest balance vs income alone reaches AUC 0.82), and today's UPI-era accounts hold far "
                   "less money relative to EMIs than 1990s Czech accounts did, so its percentile is not calibrated for "
                   "Indian applicants. That is why it can only refer, never decide, and why docs/PILOT_PLAN.md starts "
                   "with a shadow-mode pilot that records Indian outcomes before the model is relied on.")

    st.markdown("### Does the framework hold up across datasets? (public benchmark data)")
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
    st.dataframe(table, hide_index=True, width="stretch", column_config={
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

    abl = latest("age_ablation.csv")
    if abl is not None:
        st.markdown("#### Age ablation: the live models are age-blind")
        names = {"german_credit": "German Credit", "gmsc": "Give Me Some Credit", "homecredit": "Home Credit"}
        t = abl.assign(Dataset=abl["dataset"].map(names)).rename(columns={
            "variant": "Model", "auc": "AUC (test)", "cv_auc": "AUC (5-fold CV)", "dp_ratio": "DP ratio",
            "eo_diff": "EO diff", "flag_rate_under_25": "Flagged: under 25", "flag_rate_25_and_over": "Flagged: 25+"})
        st.dataframe(t[["Dataset", "Model", "AUC (test)", "AUC (5-fold CV)", "DP ratio", "EO diff",
                        "Flagged: under 25", "Flagged: 25+"]], hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ["AUC (test)", "AUC (5-fold CV)", "DP ratio", "EO diff"]}
            | {c: st.column_config.NumberColumn(format="percent") for c in ["Flagged: under 25", "Flagged: 25+"]})
        st.caption("Same split, same flag rate as the research model. Reproduce with python src/age_ablation.py.")

    st.markdown("#### Key findings")
    st.markdown("""
- **It works on real loans.** On 679 real bank statements with known outcomes, applications FAIRScore would
  approve defaulted about 3% of the time against 10.8% overall, and declines about 20%; the cash-flow model
  reaches a cross-validated AUC of 0.86 (95% interval 0.81-0.91), the expert scorecard 0.77 without being fitted
  to the data.
- **Honest limit: not yet Indian evidence.** The cash-flow model is mostly a balance-buffer model, and UPI-era
  Indian accounts hold much less money relative to EMIs than the 1990s Czech accounts it learned from. It can
  therefore only refer; a shadow-mode pilot on Indian outcomes comes before relying on it (docs/PILOT_PLAN.md).
- **No credit history needed.** Every live input comes from the application form or the bank statement: no
  bureau score, delinquency history or card utilisation, so applicants without a credit record can be assessed.
- **The cost of deployability.** Restricting models to what an applicant can actually provide costs 0.03 AUC on
  German Credit (5-fold CV) but 0.12–0.22 on Home Credit and GMSC, whose strongest features are bureau data.
  That is why lending-policy checks on verified statement facts decide, and the models only support.
- **Removing age costs almost nothing and helps error-rate fairness.** Age-blind models lose 0.00 AUC on German
  Credit (CV), 0.006 on Home Credit and 0.044 on GMSC, while the equalized-odds gap between age groups shrinks on
  all three (0.18 → 0.08, 0.18 → 0.06, 0.37 → 0.33).
- **Unawareness is not enough.** Home Credit still flags 59% of under-25s vs 26% of older applicants without
  seeing age: other inputs (years in work, income type) act as proxies. GMSC swings the other way (under-25s
  almost never flagged), which also lowers its DP ratio. Hence the outcome monitor and age-flip test on live use.
- **Mitigation trades one kind of fairness for another.** Exponentiated-gradient mitigation raises the
  demographic-parity ratio sharply on every dataset, but the equalized-odds difference worsens on German Credit
  and GMSC, consistent with the known incompatibility of the two criteria.
- **Foreign models don't transfer.** On Indian test applicants the three models' risk percentiles can differ by
  70+ points. The framework measures that disagreement (decision confidence) instead of hiding it.
""")
