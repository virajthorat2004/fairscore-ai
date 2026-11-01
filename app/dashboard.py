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
import ai_engine  # noqa: E402
import live_scoring as ls  # noqa: E402
import security  # noqa: E402
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
.ai { border:1px solid var(--line); border-top:4px solid var(--navy); }
.ai-grid { display:grid; grid-template-columns: 210px 1fr; gap:22px; align-items:center; }
@media (max-width: 900px) { .ai-grid { grid-template-columns:1fr; } }
.pd-big { font-family:'IBM Plex Mono',monospace; font-size:44px; font-weight:600; line-height:1; }
.pd-lbl { font-size:12px; color:var(--ink-soft); margin-top:6px; }
.pdbar { position:relative; height:14px; border-radius:7px; margin:16px 0 6px;
         background:linear-gradient(90deg, var(--safe-soft) 0 var(--a), #E7EEF4 var(--a) var(--d), var(--risk-soft) var(--d) 100%); }
.pdbar .dot { position:absolute; top:-5px; width:4px; height:24px; background:var(--ink); border-radius:2px; }
.pdscale { position:relative; height:16px; font-size:10.5px; color:var(--ink-soft); }
.pdscale span { position:absolute; transform:translateX(-50%); white-space:nowrap; }
.badge { display:inline-block; font-size:11.5px; font-weight:600; padding:3px 9px; border-radius:12px; margin:0 6px 6px 0; }
.b-safe { background:var(--safe-soft); color:var(--safe); } .b-gold { background:var(--gold-soft); color:var(--gold); }
.b-risk { background:var(--risk-soft); color:var(--risk); } .b-navy { background:#E7EEF4; color:var(--navy); }
.cmp { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.cmp div { border:1px solid var(--line); border-radius:4px; padding:10px 12px; background:#fff; font-size:13px; }
.kpi { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
@media (max-width: 900px) { .kpi { grid-template-columns:repeat(2,1fr); } }
.kpi div { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:14px; }
.kpi b { display:block; font-family:'IBM Plex Mono',monospace; font-size:24px; color:var(--navy); }
.kpi span { font-size:12px; color:var(--ink-soft); }
.ladder { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:8px; margin-top:10px; }
.rung { border:1px solid var(--line); border-left:4px solid var(--safe); border-radius:4px; padding:8px 10px; background:#fff;
        display:flex; flex-direction:column; gap:2px; font-size:12.5px; }
</style>
<div class="topbar"><span class="name">FAIRScore AI</span>
<span class="tag">AI credit decisions for people with no credit history &middot; calibrated risk, guaranteed uncertainty, explained and fairness-checked</span></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Form state. Required fields start empty so nothing is assumed about the applicant.
# ---------------------------------------------------------------------------
FIELDS = ls.REQUIRED_FIELDS + ls.OPTIONAL_FIELDS
for k in FIELDS:
    st.session_state.setdefault(k, None)
if st.session_state["bureau_severe_24m"] is None:
    st.session_state["bureau_severe_24m"] = False
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
    ai = r.get("ai")
    if r.get("decided_by") == "AI" and ai:
        parts.append(f'<b>AI certainty:</b> {ai["certainty"]} (conformal, {100 * (1 - ai["alpha"]):.0f}%)')
    elif conf:
        tip = "; ".join(conf.get("notes") or []) or "clean statement, clear of every limit"
        parts.append(f'<span title="{tip}"><b>Confidence:</b> {conf["label"]}</span>')
    if flip.get("ages_tested"):
        parts.append("<b>Fairness check:</b> " + (
            f'same recommendation at ages {", ".join(map(str, flip["ages_tested"]))} ✓' if flip["passed"]
            else '<span style="color:var(--risk)">changes with age alone: review</span>'))
    return f'<div class="muted" style="margin-top:10px">{" &nbsp;·&nbsp; ".join(parts)}</div>' if parts else ""


VERDICT = {
    "approve": ("Recommend approve", "var(--safe)", "Low risk, and the AI is confident."),
    "refer": ("Refer to credit officer", "var(--gold)", "The application needs a human look before a decision."),
    "decline": ("Recommend decline", "var(--risk)", "High risk or a lending rule that cannot be met."),
}
DECIDED_BY = {"AI": ("Decided by the AI engine", "b-navy"),
              "policy (hard stop)": ("Decided by a lending rule (hard stop)", "b-gold"),
              "policy (AI advisory)": ("Decided by the lending policy · AI advisory", "b-gold"),
              "policy (version 1)": ("Decided by the lending policy", "b-gold")}


def ai_card(r):
    """The AI engine's view: probability of default, certainty, typicality, pricing."""
    ai = r.get("ai")
    if not ai:
        return
    pdv, amax, dmin = ai["pd"], ai["approve_max_pd"], ai["decline_min_pd"]
    scale = lambda x: min(100.0, 100 * (x ** 0.5) / (0.6 ** 0.5))        # square-root scale: low risks readable
    cls = "b-safe" if ai["outcome_set"] == "repays" else "b-risk" if ai["outcome_set"] == "defaults" else "b-gold"
    cert = {"repays": "Confident: will repay", "defaults": "Confident: likely to default",
            "uncertain": "Not certain either way"}[ai["outcome_set"]]
    typ = ai["typicality"]
    typ_txt = ("typical of the borrowers it learned from" if typ >= 0.10 else
               "less common, but within its experience" if ai["in_experience"] else "unlike anything it learned from")
    note = r.get("ai_note")
    st.markdown(
        f'<p class="eyebrow">AI risk assessment · trained on 679 real loans with their bank statements</p>'
        f'<div class="card ai"><div class="ai-grid"><div><div class="pd-big">{ai["pd_pct"]:.1f}%</div>'
        f'<div class="pd-lbl">probability of default<br>{ai_engine.odds_text(pdv)} similar borrowers defaulted</div></div>'
        f'<div><span class="badge {cls}">{cert}</span>'
        f'<span class="badge b-navy">{ai_engine.ordinal_risk(pdv)} risk</span>'
        f'<span class="badge b-navy">Profile {typ_txt}</span>'
        f'<div class="pdbar" style="--a:{scale(amax):.1f}%;--d:{scale(dmin):.1f}%">'
        f'<div class="dot" style="left:calc({scale(pdv):.1f}% - 2px)"></div></div>'
        f'<div class="pdscale"><span style="left:0%;transform:none">0%</span>'
        f'<span style="left:{scale(amax):.1f}%">auto-approve up to {100 * amax:.1f}%</span>'
        f'<span style="left:{scale(dmin):.1f}%">decline from {100 * dmin:.0f}%</span>'
        f'<span style="left:100%;transform:translateX(-100%)">60%</span></div>'
        f'<div class="muted" style="margin-top:8px">Risk-based rate {100 * ai["risk_based_rate"]:.1f}% a year · '
        f'expected loss {money(ai["expected_loss"])}'
        + (f'<br><b>Security:</b> covers {100 * ai["secured_coverage"]:.0f}% of the loan, so the loss if the borrower '
           f'defaults falls from {ai["lgd_unsecured"]:.0%} to {ai["lgd"]:.0%}.'
           if ai.get("secured_coverage") else "")
        + (f'<br><b>Note:</b> {note}.' if note else "") + '</div></div></div></div>', unsafe_allow_html=True)


def why_chart(r):
    """SHAP: which habits pushed this applicant's risk up or down (exact contributions of the AI model)."""
    ai = r.get("ai")
    if not ai:
        return
    import altair as alt
    g = pd.DataFrame(ai["explanation"]["groups"])
    g = g[g["impact"].abs() > 0.01]
    if g.empty:
        return
    g = g.reindex(g["impact"].abs().sort_values(ascending=False).index).head(6)
    g["Effect"] = g["impact"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    g["label"] = g["group"]
    st.markdown('<p class="eyebrow">Why · what the AI weighed most (SHAP)</p>', unsafe_allow_html=True)
    ch = alt.Chart(g).mark_bar(cornerRadius=2).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("impact", op="sum", order="descending"), title=None,
                axis=alt.Axis(labelLimit=260)),
        x=alt.X("impact:Q", title="← lowers risk      raises risk →"),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                     range=["#9E3A2E", "#2E6B4F"]), legend=None),
        tooltip=["group", "evidence", alt.Tooltip("impact:Q", format=".2f")]).properties(height=34 * len(g) + 20)
    st.altair_chart(ch, width="stretch")


def recourse_ai(r):
    """What would change the AI's decision: each option re-checked by the model. Returns plain strings."""
    rc = (r.get("ai") or {}).get("recourse")
    if not rc or rc.get("error"):
        return []
    items = []
    if rc.get("loan"):
        L = rc["loan"]
        items.append(f'Borrow {money(L["amount"])} over {L["term"]} months (EMI {money(L["emi"])}): the AI would '
                     f'approve' + (f' (risk {100 * L["pd"]:.1f}%)' if L.get("pd") is not None else ""))
    h = rc.get("habits") or {}
    for x in h.get("single") or []:
        items.append(f'{x["change"]}: risk falls to {100 * x["new_pd"]:.1f}%, enough to approve')
    if h.get("combined"):
        items.append(" + ".join(h["combined"]["changes"]) + f': risk falls to {100 * h["combined"]["new_pd"]:.1f}%')
    for x in h.get("closest") or []:
        items.append(f'{x["change"]}: risk would fall to {100 * x["new_pd"]:.1f}% (not yet enough alone)')
    return items


def starter_card(r, a):
    """FAIRScore 3.0 starter-loan ladder: what the bank can say yes to today."""
    s = r.get("starter")
    if not s:
        return
    cls = {"approved": "b-safe", "starter": "b-safe", "learning": "b-navy", "suggested": "b-gold"}[s["kind"]]
    steps = "".join(
        f'<div class="rung"><b>{"Today" if x["after_emis"] == 0 else "After " + str(x["after_emis"]) + " on-time EMIs"}</b>'
        f'<span class="mono">{money(x["limit"])}</span>'
        f'<span class="muted">{x.get("note") or ("first loan" if x["after_emis"] == 0 else "re-assessed with a fresh statement")}</span></div>'
        for x in s["rungs"])
    st.markdown(
        f'<p class="eyebrow">Starter-loan ladder · FAIRScore 3.0</p>'
        f'<div class="card ai" style="border-top-color:var(--safe)"><span class="badge {cls}">{s["label"]}</span>'
        f'<div style="margin:6px 0 2px;font-size:15px"><b>{money(s["amount"])}</b> over {s["term"]} months · EMI '
        f'{money(s["emi"])} at {100 * s["rate"]:.1f}% · default risk at this size {100 * s["pd"]:.1f}%</div>'
        f'<div class="ladder">{steps}</div>'
        f'<div class="muted" style="margin-top:6px">The requested {money(a["requested_amount"])} stays referred. '
        f'Repaying builds a credit-bureau history.'
        + (f' This is a <b>learning offer</b> (a random {100 * s["propensity"]:.0f}% of similar cases get one), so the '
           f'AI learns about people it would otherwise never see.' if s["kind"] == "learning" else "")
        + '</div></div>', unsafe_allow_html=True)


def compare_card(r):
    v1, v2 = r.get("v1_verdict"), r["verdict"]
    if not v1:
        return
    lab = {"approve": "Approve", "refer": "Refer", "decline": "Decline"}
    st.markdown(f'<div class="cmp"><div><b>Version 1 · rules only:</b> {lab[v1]}</div>'
                f'<div><b>Version 2 · AI engine:</b> {lab[v2]}'
                f'{" (same)" if v1 == v2 else " (changed)"}</div></div>', unsafe_allow_html=True)

tab_assess, tab_evidence, tab_how = st.tabs(["Assess an application", "AI model evidence", "How it works"])

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

            with st.expander("5 · Security and co-applicant (optional)"):
                st.caption("Collateral lowers the bank's LOSS if a borrower defaults; it never replaces the check that "
                           "the borrower can afford the EMI. A co-applicant's income counts towards affordability once "
                           "verified.")
                c1, c2 = st.columns(2)
                c1.selectbox("Security offered", security.SECURITY_TYPES, key="security_type", index=None,
                             placeholder="None",
                             help="Gold: RBI 2025 caps (85% of value up to Rs 2.5 lakh, 80% up to Rs 5 lakh, 75% above). "
                                  "Fixed deposit and property caps are lender settings in config/policy.toml.")
                c2.number_input("Value of the security (₹)", min_value=0, max_value=10_000_000_000, step=10000,
                                key="security_value", placeholder="market / appraised value")
                c1.number_input("Co-applicant's monthly income (₹)", min_value=0, max_value=100_000_000, step=1000,
                                key="co_applicant_income", placeholder="spouse / parent",
                                help="Added to the income used for affordability; to be verified with the co-applicant's "
                                     "statement or salary slips.")

            with st.expander("6 · Credit bureau report (optional: only if the applicant has a record)"):
                st.caption("FAIRScore is built for applicants with no bureau record. If the applicant has one (a card, "
                           "a BNPL line, a small loan), copy it here: it adds hard stops and cautions, and its EMIs are "
                           "used for affordability when they are higher than those seen in the statement.")
                b1, b2 = st.columns(2)
                b1.selectbox("Bureau record", ls.bureau.RECORDS, key="bureau_record", index=None, placeholder="Not checked")
                b2.number_input("Bureau score", min_value=1, max_value=999, step=1, key="bureau_score",
                                placeholder="300-900, or the new-to-credit score")
                b1.selectbox("Worst days past due, last 12 months", [0, 30, 60, 90], key="bureau_max_dpd_12m",
                             index=None, placeholder="0")
                b2.number_input("Hard enquiries, last 6 months", min_value=0, max_value=100, step=1,
                                key="bureau_enquiries_6m", placeholder="0")
                b1.number_input("EMIs on the report (₹ / month)", min_value=0, max_value=100_000_000, step=500,
                                key="bureau_emi", placeholder="all open loans")
                b2.checkbox("90+ DPD, write-off or settlement in 24 months", key="bureau_severe_24m")

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
            _ai = r.get("ai") or {}
            if r["verdict"] == "approve" and r.get("decided_by") == "AI" and _ai.get("outcome_set") != "repays":
                sub = (f"Secured loan: the AI is not certain on its own (risk {_ai['pd_pct']:.1f}%), but the security "
                       f"keeps the expected loss within the bank's appetite.")
            elif r["verdict"] == "approve" and r.get("decided_by") != "AI":
                sub = "Approved by the lending policy; every check passed."
            reasons = "".join(f"<li>{x}</li>" for x in r["verdict_reasons"])
            by_txt, by_cls = DECIDED_BY.get(r.get("decided_by"), ("", "b-navy"))
            conds = r.get("conditions") or []
            cond_html = ("<div class='muted' style='margin-top:8px'><b>Approved subject to:</b> " +
                         "; ".join(c.split(":")[0] for c in conds) + "</div>") if conds and r["verdict"] == "approve" else ""
            st.markdown(f'<div class="card verdict" style="--c:{color}"><div class="label">{label}</div>'
                        f'<div style="margin-top:8px"><span class="badge {by_cls}">{by_txt}</span></div>'
                        f'<div class="sub">{sub}</div>'
                        f'{"<ul>" + reasons + "</ul>" if reasons else ""}{cond_html}'
                        f'{meta_line(r)}'
                        f'<div class="ref">Reference {saved["ref"]} · statement {saved["source"]}</div></div>',
                        unsafe_allow_html=True)

            if r.get("consistency") == "conflict" and saved.get("detected_loans"):
                n = int(saved["detected_loans"])
                st.button(f"Applicant confirms {n} active loan{'s' if n > 1 else ''}: update the form and re-assess",
                          on_click=confirm_loans, args=(max(n, int(a["active_credit_lines"] or 0)),),
                          type="primary", width="stretch")

            ai_card(r)
            why_chart(r)
            starter_card(r, a)

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

            # --- 3. counter-offer, if the bank can lend a smaller amount
            fs, elig, offer = r.get("fairscore"), r.get("eligibility") or {}, r.get("offer")
            if offer and r["verdict"] != "approve":
                o = offer[0]
                st.markdown(f'<div class="card" style="padding:10px 16px"><b>Counter-offer:</b> approvable at '
                            f'{money(o["amount"])} over {o["term"]} months (EMI {money(o["emi"])})</div>',
                            unsafe_allow_html=True)

            # --- 4. what would help (AI re-checked options first, then policy suggestions)
            helps = recourse_ai(r)[:3]
            helps += [x["change"] for x in (r["recourse"] or []) if x["change"] not in helps][:max(0, 4 - len(helps))]
            if helps:
                st.markdown('<p class="eyebrow">What would help</p>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="chip">{x}</span>' for x in helps), unsafe_allow_html=True)

            # --- 5. lending checks: only the ones that need attention are shown here
            def _rows(cs):
                return "".join(
                    f'<div class="check"><span class="pill {c["status"]}">{STATUS_WORD[c["status"]]}</span>'
                    f'<div><div class="nm">{c["name"]}</div><div class="dt">{c["detail"]}</div></div></div>'
                    for c in cs)
            flagged = [c for c in pol["checks"] if c["status"] in ("fail", "caution")]
            n_all = len(pol["checks"])
            head = (f"all {n_all} passed" if not flagged else
                    f"{len(flagged)} need{'s' if len(flagged) == 1 else ''} attention, {n_all - len(flagged)} passed")
            st.markdown(f'<p class="eyebrow">Lending checks · {head}</p>', unsafe_allow_html=True)
            if flagged:
                st.markdown(f'<div class="card">{_rows(flagged)}</div>', unsafe_allow_html=True)

            # --- 6. everything else, for the credit officer, in one place
            with st.expander("Full details: scorecard, all checks, statement summary, documents"):
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
                    st.markdown('<p class="eyebrow">FAIRScore (300-900) · expert scorecard, also an input to the AI</p>',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="card"><div class="lvl-row"><span class="lvl {band_cls}">{fs["score"]}</span>'
                        f'<span><b>{fs["band"]}</b> · from the statement and the application only; no age, '
                        f'gender or bureau data</span></div>'
                        f'<div class="meter" style="background:linear-gradient(90deg,var(--risk-soft) 0 50%,var(--gold-soft) 50% 58%,'
                        f'#E7EEF4 58% 67%,var(--safe-soft) 67% 100%)"><div class="dot" style="left:calc({pos:.0f}% - 2px)"></div></div>'
                        f'<div class="meter-scale" style="grid-template-columns:50% 8% 9% 33%"><span>Poor</span><span>Weak</span>'
                        f'<span>Fair</span><span>Good / Excellent</span></div>'
                        f'<div class="fgrid"><div><div class="eyebrow">Weakest factors</div>{rows_w}</div>'
                        f'<div><div class="eyebrow">Strongest factors</div>{rows_s}</div></div></div>',
                        unsafe_allow_html=True)

                st.markdown(f'<p class="eyebrow">All {n_all} lending checks</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="card">{_rows(pol["checks"])}</div>', unsafe_allow_html=True)

                if r.get("statement_insights"):
                    st.markdown('<p class="eyebrow">What the bank statement shows</p>', unsafe_allow_html=True)
                    st.markdown('<div class="card">' + "".join(f'<div class="factor">• {x}</div>'
                                                               for x in r["statement_insights"]) + '</div>',
                                unsafe_allow_html=True)

                monthly = (r.get("banking") or {}).get("monthly")
                if monthly:
                    st.markdown('<p class="eyebrow">Month-wise banking summary</p>', unsafe_allow_html=True)
                    mdf = pd.DataFrame(monthly)
                    keep = [c for c in ["month", "income_counted", "emis", "abb", "min_balance", "closing_balance",
                                        "bounces"] if c in mdf.columns]
                    st.dataframe(mdf[keep], hide_index=True, width="stretch", column_config={
                        "month": "Month", "income_counted": st.column_config.NumberColumn("Income (₹)", format="%d"),
                        "emis": st.column_config.NumberColumn("EMIs (₹)", format="%d"),
                        "abb": st.column_config.NumberColumn("Average balance (₹)", format="%d"),
                        "min_balance": st.column_config.NumberColumn("Lowest balance (₹)", format="%d"),
                        "closing_balance": st.column_config.NumberColumn("Month-end (₹)", format="%d"),
                        "bounces": "Bounces"})

                if r.get("documents"):
                    st.markdown('<p class="eyebrow">Documents to collect before sanction</p>', unsafe_allow_html=True)
                    st.markdown("".join(f'<span class="chip">{d}</span>' for d in r["documents"]),
                                unsafe_allow_html=True)

                v1 = r.get("v1_verdict")
                if v1:
                    lab = {"approve": "Approve", "refer": "Refer", "decline": "Decline"}
                    st.caption(f"Rules alone (version 1) would say: {lab[v1]} · with the AI (version 2): "
                               f"{lab[r['verdict']]} · policy {r.get('policy_version', '')}")

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
# AI model evidence (every number: nested cross-validation on 679 real loans; src/train_ai_engine.py)
# ===========================================================================
def _csv(name):
    p = ROOT / "results" / name
    return pd.read_csv(p) if p.exists() else None


with tab_evidence:
    import altair as alt
    met, dec, conf = _csv("ai_engine_metrics.csv"), _csv("ai_engine_decisions.csv"), _csv("ai_engine_conformal.csv")
    trade, fair = _csv("ai_engine_tradeoff.csv"), _csv("ai_engine_fairness.csv")
    seqm, ftrade, sl = _csv("seq_model_metrics.csv"), _csv("ai_engine_fairness_tradeoff.csv"), _csv("starter_ladder_evidence.csv")
    zoo, roc_, cal_, cm_, sig_, shg = (_csv("model_zoo.csv"), _csv("eval_roc.csv"), _csv("eval_calibration.csv"),
                                       _csv("eval_confusion.csv"), _csv("eval_v1_v2_bootstrap.csv"),
                                       _csv("eval_shap_global.csv"))
    pct = lambda cols: {c_: st.column_config.NumberColumn(format="percent") for c_ in cols}
    if met is None or dec is None:
        st.info("Run python src/train_ai_engine.py to produce the evidence.")
    else:
        st.markdown("### How good is the AI? Tested on 679 real loans it never saw while learning")
        st.caption("Each loan's 6-month bank statement before the loan, and whether it was repaid (73 defaulted). "
                   "Nested cross-validation: every loan is judged by a model that never saw it.")
        g = lambda v, s_: dec[(dec.version == v) & (dec.verdict == s_)].iloc[0]
        a1, a2, r1, r2 = g("v1", "approve"), g("v2", "approve"), g("v1", "refer"), g("v2", "refer")
        m0 = met.iloc[0]
        offered = None
        if sl is not None:
            offered = sl.set_index("group").loc["offered credit today: approval or starter (version 3)", "share_of_applicants"]
        st.markdown(
            f'<div class="kpi"><div><b>{m0.auc:.2f}</b><span>AUC (95% interval {m0.auc_ci_low:.2f}-{m0.auc_ci_high:.2f})'
            f'</span></div>'
            f'<div><b>{a2.share:.0%}</b><span>approved, {a2.default_rate:.1%} of them defaulted '
            f'(rules alone: {a1.share:.0%} at {a1.default_rate:.1%})</span></div>'
            f'<div><b>{r2.share:.0%}</b><span>sent to an officer (rules alone: {r1.share:.0%})</span></div>'
            + (f'<div><b>{offered:.0%}</b><span>offered some credit today, with starter loans '
               f'(approvals only: {a2.share:.0%})</span></div>' if offered is not None else "")
            + '</div>', unsafe_allow_html=True)

        # 1. the decision result: more approvals at lower risk, and why rules + AI together
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### More approvals at lower risk")
            t = trade[trade.ranking.isin(["AI engine", "FAIRScore scorecard"])]
            pts = trade[~trade.ranking.isin(["AI engine", "FAIRScore scorecard"])]
            line = alt.Chart(t).mark_line(point=True).encode(
                x=alt.X("approval_rate:Q", title="Share of applicants approved", axis=alt.Axis(format="%")),
                y=alt.Y("default_rate:Q", title="Default rate among approved", axis=alt.Axis(format="%")),
                color=alt.Color("ranking:N", title=None, scale=alt.Scale(range=["#16324A", "#B8863B"])))
            dots = alt.Chart(pts).mark_point(size=160, filled=True, shape="diamond").encode(
                x="approval_rate:Q", y="default_rate:Q",
                color=alt.Color("ranking:N", title=None, scale=alt.Scale(range=["#9E3A2E", "#2E6B4F"])),
                tooltip=["ranking", alt.Tooltip("approval_rate:Q", format=".0%"), alt.Tooltip("default_rate:Q", format=".1%")])
            st.altair_chart((line + dots).resolve_scale(color="independent").properties(height=300), width="stretch")
            st.caption("Lower and further right is better. Diamonds: the actual decisions of rules only (version 1) "
                       "and rules + AI (version 2).")
        with c2:
            st.markdown("#### Rules + AI vs AI alone vs rules alone")
            d3 = dec[dec.version.isin(["v1", "ai_only", "v2"])].pivot_table(index="version", columns="verdict",
                                                                            values="share").reindex(["v1", "ai_only", "v2"])
            dr = dec[(dec.version.isin(["v1", "ai_only", "v2"])) & (dec.verdict == "approve")].set_index("version")["default_rate"]
            d3["Default rate among approved"] = dr
            d3.index = ["Rules only", "AI only", "Rules + AI (FAIRScore)"]
            st.dataframe(d3[["approve", "refer", "decline", "Default rate among approved"]].rename(
                columns={"approve": "Approved", "refer": "Referred", "decline": "Declined"}), width="stretch",
                column_config=pct(["Approved", "Referred", "Declined", "Default rate among approved"]))
            st.caption("On risk the AI alone does about as well, but on the forged sample statements it would approve "
                       "5 of 7: an edited statement looks like a good borrower. Rules catch fraud and protect the "
                       "borrower; the AI measures risk.")

        # 2. standard evaluation
        if zoo is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ROC curves")
                st.altair_chart((alt.Chart(roc_).mark_line().encode(
                    x=alt.X("fpr:Q", title="Repayers wrongly flagged (false positive rate)"),
                    y=alt.Y("tpr:Q", title="Defaulters caught (true positive rate)"),
                    color=alt.Color("model:N", title=None, legend=alt.Legend(orient="bottom")))
                    + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#999")
                    .encode(x="x:Q", y="y:Q")).properties(height=280), width="stretch")
            with c2:
                st.markdown("#### Calibration: predicted vs actual")
                st.altair_chart((alt.Chart(cal_).mark_line(point=True, color="#16324A").encode(
                    x=alt.X("predicted:Q", title="Predicted probability of default", axis=alt.Axis(format="%")),
                    y=alt.Y("observed:Q", title="Actual default rate", axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("predicted:Q", format=".1%"), alt.Tooltip("observed:Q", format=".1%"), "loans"])
                    + alt.Chart(pd.DataFrame({"x": [0, 0.6], "y": [0, 0.6]})).mark_line(strokeDash=[4, 4], color="#999")
                    .encode(x="x:Q", y="y:Q")).properties(height=280), width="stretch")
                st.caption("Close to the dashed line: when the AI says 4%, about 4 in 100 such borrowers defaulted.")

            st.markdown("#### Nine algorithms on the same data")
            z = zoo[["model", "auc_mean", "auc_ci_low", "auc_ci_high", "gini", "ks", "overfit_gap", "delong_p"]].copy()
            z["95% interval"] = z.apply(lambda r_: f"{r_.auc_ci_low:.2f}-{r_.auc_ci_high:.2f}", axis=1)
            st.dataframe(z[["model", "auc_mean", "95% interval", "gini", "ks", "overfit_gap", "delong_p"]].rename(
                columns={"model": "Algorithm", "auc_mean": "AUC", "gini": "Gini", "ks": "KS",
                         "overfit_gap": "Overfit gap", "delong_p": "p vs deployed (DeLong)"}),
                hide_index=True, width="stretch",
                column_config={c_: st.column_config.NumberColumn(format="%.3f") for c_ in
                               ["AUC", "Gini", "KS", "Overfit gap", "p vs deployed (DeLong)"]})
            st.caption("The deployed monotone CatBoost ties the best algorithms (p > 0.1) with the smallest overfitting "
                       "among the boosted models, and beats a single tree, the expert scorecard and a neural network.")

        # 3. uncertainty and fairness
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Guaranteed uncertainty (conformal prediction)")
            cf = conf.copy()
            cf["Significance"] = cf.alpha.map(lambda a_: f"{a_:.0%}" + (" (used)" if abs(a_ - 0.10) < 1e-9 else ""))
            st.dataframe(cf[["Significance", "defaulters_auto_approved", "auto_approve_share", "uncertain_share"]].rename(
                columns={"defaulters_auto_approved": "Defaulters auto-approved", "auto_approve_share": "Auto-approved",
                         "uncertain_share": "Sent to officer"}), hide_index=True, width="stretch",
                column_config=pct(["Defaulters auto-approved", "Auto-approved", "Sent to officer"]))
            st.caption("The promise: at 10%, at most 10% of borrowers who default can be auto-approved. "
                       "Measured: within the limit at every level.")
        with c2:
            st.markdown("#### Fairness on real outcomes")
            if fair is not None:
                f2 = fair.pivot_table(index=["attribute", "group"], columns="version",
                                      values="approved_among_repayers").reset_index()
                f2.columns = ["Attribute", "Group", "Version 1", "Version 2"]
                f2["Attribute"] = f2["Attribute"].str.replace("_", " ").str.capitalize()
                st.dataframe(f2, hide_index=True, width="stretch", column_config={
                    "Version 1": st.column_config.NumberColumn("Repayers approved · rules", format="percent"),
                    "Version 2": st.column_config.NumberColumn("Repayers approved · rules + AI", format="percent")})
                st.caption("Of the people who DID repay, how many each group gets approved. "
                           "Gender and age are never inputs to the AI.")

        # 4. what drives the model, and errors at the decision
        if shg is not None and cm_ is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### What drives the model")
                st.altair_chart(alt.Chart(shg.head(8)).mark_bar(color="#16324A").encode(
                    y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                    x=alt.X("mean_abs_shap:Q", title="Average impact on the risk (mean |SHAP|)"),
                    tooltip=["feature", "group", alt.Tooltip("mean_abs_shap:Q", format=".3f")]).properties(height=260),
                    width="stretch")
            with c2:
                st.markdown("#### Confusion matrix at the decision")
                cmv = cm_[cm_.rule.str.startswith("Version 2")].iloc[0]
                st.dataframe(pd.DataFrame({"Flagged (refer / decline)": [cmv.defaulters_flagged_TP, cmv.repayers_flagged_FP],
                                           "Approved": [cmv.defaulters_approved_FN, cmv.repayers_approved_TN]},
                                          index=["Actually defaulted", "Actually repaid"]).round(1), width="stretch")
                st.caption(f"Loans per run. {cmv.recall_defaulters_caught:.0%} of defaulters are caught; "
                           f"{cmv.default_rate_among_approved:.1%} of approved loans default. Accuracy is not used: "
                           "with 10.8% defaults, approving everyone would be 89% 'accurate' and useless.")

        # ------------------------------------------------------------------ advanced studies, folded away
        st.markdown("")
        with st.expander("More research studies (for detailed questions)"):
            if seqm is not None:
                st.markdown("#### Deep learning on raw transactions")
                st.dataframe(seqm.rename(columns={"model": "Model", "auc": "AUC", "auc_sd": "± sd", "folds": "Folds"}),
                             hide_index=True, width="stretch",
                             column_config={"AUC": st.column_config.NumberColumn(format="%.3f"),
                                            "± sd": st.column_config.NumberColumn(format="%.3f")})
                st.caption("A GRU encoder pre-trained without labels on all 4,500 Berka accounts' transactions.")
            if sig_ is not None:
                st.markdown("#### Is rules + AI really better than rules alone? (bootstrap, 95% interval)")
                sg = sig_.copy()
                sg["measure"] = sg["measure"].map({"approval_gain": "More applicants approved",
                                                   "default_rate_change": "Change in default rate among approved",
                                                   "referral_change": "Change in cases sent to an officer"})
                st.dataframe(sg[["measure", "estimate", "ci_low", "ci_high"]].rename(columns={
                    "measure": "Version 2 vs version 1", "estimate": "Estimate", "ci_low": "95% low",
                    "ci_high": "95% high"}), hide_index=True, width="stretch", column_config={
                    c_: st.column_config.NumberColumn(format="%.3f") for c_ in ["Estimate", "95% low", "95% high"]})
                st.caption("More approvals and fewer referrals are statistically significant. The lower default rate "
                           "points the right way in 94% of resamples, but 679 loans are too few to prove it at 95%.")
            if ftrade is not None:
                st.markdown("#### Fairness-constrained training (equalized odds) vs the deployed model")
                ft = ftrade.rename(columns={"attribute": "Attribute", "variant": "Model", "auc": "AUC",
                                            "eo_gap": "Equalized-odds gap", "approval_gap_repayers": "Repayer approval gap",
                                            "dp_ratio": "Approval-rate ratio", "default_rate_approved": "Default rate approved"})
                st.dataframe(ft, hide_index=True, width="stretch", column_config={
                    c_: st.column_config.NumberColumn(format="%.3f") for c_ in
                    ["AUC", "Equalized-odds gap", "Repayer approval gap", "Approval-rate ratio", "Default rate approved"]})
                st.caption("Forcing equal error rates cost most of the accuracy and tripled defaults among approved loans, "
                           "while the unconstrained model already treats groups similarly.")
            ltv = _csv("india_ntc_ltv.csv")
            if ltv is not None:
                st.markdown("#### Why collateral and own contribution matter (116,950 Indian new-to-credit loans, L&T)")
                l_ = ltv[(ltv.population == "New to credit") & (ltv.loans >= 1000)]
                st.altair_chart(alt.Chart(l_).mark_bar(color="#16324A").encode(
                    x=alt.X("ltv_band:N", title="Loan as a share of the vehicle's price", sort=None),
                    y=alt.Y("default_rate:Q", title="Defaulted", axis=alt.Axis(format="%")),
                    tooltip=["ltv_band", alt.Tooltip("default_rate:Q", format=".1%"), "loans"]).properties(height=220),
                    width="stretch")
                st.caption("The more the borrower pays from their own pocket, the fewer defaults. FAIRScore uses collateral "
                           "to lower the expected loss (not to replace the repayment check) and checks the own contribution "
                           "on purchase loans.")
            alfa = _csv("alfabattle_label_efficiency.csv")
            if alfa is not None:
                st.markdown("#### Modern transactions at scale: AlfaBattle 2.0 (132,242 clients, 3,108 defaults)")
                a_ = alfa.copy()
                a_["Labelled training clients"] = a_["train_fraction"].map(lambda f: f"{f:.0%}")
                st.altair_chart(alt.Chart(a_).mark_line(point=True).encode(
                    x=alt.X("train_fraction:Q", title="Share of labelled training clients used", scale=alt.Scale(type="log"),
                            axis=alt.Axis(format="%", values=[0.01, 0.05, 0.2, 1.0])),
                    y=alt.Y("auc:Q", title="AUC on 39,673 held-out clients", scale=alt.Scale(domain=[0.55, 0.76])),
                    color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom", columns=2)),
                    tooltip=["method", "Labelled training clients", alt.Tooltip("auc:Q", format=".3f"), "train_defaults"]
                ).properties(height=300), width="stretch")
                st.caption("Card transactions before a loan, labelled with default. Self-supervised pre-training makes the "
                           "neural network 4-5x more label-efficient (pre-trained with 5% of labels ~ scratch with 20%); "
                           "gradient boosting on designed features stays strongest (AUC 0.748, 0.731-0.763), and adding the "
                           "pre-trained embedding helps most when labelled loans are scarce. Research only: the data's "
                           "categories are anonymised codes that cannot be matched to Indian statements.")
            anom = _csv("statement_anomaly_metrics.csv")
            if anom is not None:
                st.markdown("#### Research module: detecting manipulated statements (unsupervised)")
                a_ = anom[anom.statements == "all"][["manipulation", "auc", "detected_at_5pct_false_alarms"]]
                st.dataframe(a_.rename(columns={"manipulation": "Manipulation injected", "auc": "AUC",
                                                "detected_at_5pct_false_alarms": "Caught at 5% false alarms"}),
                             hide_index=True, width="stretch", column_config={
                                 "AUC": st.column_config.NumberColumn(format="%.3f"),
                                 "Caught at 5% false alarms": st.column_config.NumberColumn(format="percent")})
                st.caption("Isolation Forest trained only on genuine statements (1,388 real Berka windows + 440 "
                           "Indian-style), tested on held-out statements with injected manipulations whose balances still "
                           "add up. Not used in live decisions yet: real Indian statements differ from both training "
                           "sources and would raise false alarms.")

            sl = _csv("starter_ladder_evidence.csv")
            if sl is not None:
                st.markdown("#### Starter-loan ladder: a small yes instead of a dead-end referral")
                g_ = sl.set_index("group")
                v3 = g_.loc["offered credit today: approval or starter (version 3)"]
                v2_ = g_.loc["offered credit today: approval only (version 2)"]
                s_ = g_.loc["  ... of which starter loan (PD <= max_pd)"]
                full = g_.loc["  ... the same starter cases if lent the FULL amount"]
                st.markdown(
                    f'<div class="kpi"><div><b>{v3.share_of_applicants:.0%}</b><span>offered credit today (version 2: '
                    f'{v2_.share_of_applicants:.0%})</span></div>'
                    f'<div><b>{s_.share_of_applicants:.0%}</b><span>of applicants move from "refer" to a starter loan; '
                    f'{s_.default_rate:.1%} of them defaulted</span></div>'
                    f'<div><b>−{1 - s_.loss_per_applicant / full.loss_per_applicant:.0%}</b><span>loss on those cases vs '
                    f'lending them the full amount</span></div>'
                    f'<div><b>{v3.default_rate:.1%}</b><span>default rate across everyone offered credit '
                    f'(version 2: {v2_.default_rate:.1%})</span></div></div>', unsafe_allow_html=True)
                st.caption("Nested out-of-fold decisions on the 679 loans. Conservative: a starter loan is assumed to default "
                           "whenever the full loan did. The starter is at most one month of verified income (median "
                           f"{sl.starter_size_vs_requested.iloc[0]:.0%} of the request), so the bank's loss stays small while "
                           "the applicant builds a repayment record with the credit bureaus.")
            d0, sc, scal, sdec = (_csv("shift_day0.csv"), _csv("shift_conformal.csv"), _csv("shift_calibration.csv"),
                                  _csv("shift_decisions.csv"))
            if sc is not None and scal is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Uncertainty that stays honest when the population changes")
                    t_ = sc[sc.variant != "weighted Mondrian, always"].copy()
                    t_["Scenario"] = t_.scenario.str.replace(r"^\d+ ", "", regex=True)
                    t_["Method"] = t_.variant.map({"standard (source calibration)": "Standard conformal (2.0)",
                                                   "FAIRScore 3.0 (weighted if shift detected)": "Shift-aware (3.0)"})
                    ch = alt.Chart(t_).mark_bar().encode(
                        y=alt.Y("Scenario:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
                        yOffset="Method:N",
                        x=alt.X("defaulters_not_flagged:Q", title="Would-be defaulters the AI could auto-approve",
                                axis=alt.Axis(format="%")),
                        color=alt.Color("Method:N", title=None, scale=alt.Scale(range=["#B8863B", "#16324A"]),
                                        legend=alt.Legend(orient="bottom")),
                        tooltip=["Scenario", "Method", alt.Tooltip("defaulters_not_flagged:Q", format=".1%"),
                                 alt.Tooltip("auto_approvable_share:Q", format=".0%", title="auto-approvable share")])
                    rule = alt.Chart(pd.DataFrame({"x": [0.10]})).mark_rule(color="#9E3A2E", strokeDash=[4, 3]).encode(x="x:Q")
                    lab = alt.Chart(t_).mark_text(align="left", dx=3, fontSize=10, color="#555").encode(
                        y=alt.Y("Scenario:N", sort=None), yOffset="Method:N", x="defaulters_not_flagged:Q",
                        text=alt.Text("defaulters_not_flagged:Q", format=".1%"))
                    st.altair_chart((ch + lab + rule).properties(height=360), width="stretch")
                    st.caption("The guarantee is at most 10% (red line). When applicants look safer than the training ones but "
                               "some still default, standard conformal breaks it (up to 17%); weighting the training cases by "
                               "how much they resemble the new applicants restores it. Weighting is switched on only when a "
                               "shift is detected: always weighting breaks it under a pure default-rate change. 30 repeats.")
                with c2:
                    st.markdown("#### How many local repayments are needed?")
                    opts = list(scal.scenario.unique())
                    pick = st.selectbox("Scenario", opts, index=opts.index(next(o for o in opts if o.startswith("2"))),
                                        key="shift_pick")
                    c_ = scal[(scal.scenario == pick) & scal.method.isin(
                        ["Platt on local outcomes", "Bayesian recalibration (FAIRScore 3.0)", "retrain on local loans only",
                         "retrain on source + local (local x3)"])]
                    base = scal[(scal.scenario == pick) & (scal.method == "none (source engine)")]["brier"].iloc[0]
                    ln = alt.Chart(c_).mark_line(point=True).encode(
                        x=alt.X("n_labels:Q", title="Repayment outcomes recorded by the lender", scale=alt.Scale(type="log")),
                        y=alt.Y("brier:Q", title="Brier score (lower = better)", scale=alt.Scale(zero=False)),
                        color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom", columns=2)),
                        tooltip=["method", "n_labels", alt.Tooltip("brier:Q", format=".4f"), alt.Tooltip("ece:Q", format=".3f")])
                    rl = alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(strokeDash=[4, 3], color="#5B6270").encode(y="y:Q")
                    st.altair_chart((ln + rl).properties(height=290), width="stretch")
                    st.caption("Dashed: the engine as trained. Bayesian recalibration (prior: 'no change') is never worse than "
                               "doing nothing and beats plain Platt scaling at every sample size; retraining needs a few hundred "
                               "outcomes, so FAIRScore proposes it only as a challenger once enough defaults are recorded.")
                if d0 is not None:
                    st.markdown("#### Day 0: what can be known before any loan is repaid")
                    t_ = d0[["scenario", "true_default_rate", "estimate_mean_pd", "estimate_em_prior_shift", "shift_detected",
                             "max_psi"]].copy()
                    st.dataframe(t_.rename(columns={"scenario": "Scenario", "true_default_rate": "True default rate",
                                                    "estimate_mean_pd": "Estimate: mean PD",
                                                    "estimate_em_prior_shift": "Estimate: EM (prior shift)",
                                                    "shift_detected": "Shift detected", "max_psi": "Largest PSI"}),
                                 hide_index=True, width="stretch", column_config={
                                     c_: st.column_config.NumberColumn(format="percent") for c_ in
                                     ("True default rate", "Estimate: mean PD", "Estimate: EM (prior shift)", "Shift detected")})
                    st.caption("From 400 unlabelled applications. The mean PD is right when only the applicants change; the EM "
                               "estimate is right when only the default rate changes and badly wrong otherwise. Neither can be "
                               "trusted alone, so FAIRScore shows both and treats a gap as an alarm, not as a correction.")
            ex = _csv("exploration_study.csv")
            if ex is not None:
                st.markdown("#### Learning about the people a bank used to turn away")
                acc = st.radio("The old policy accepted", [0.4, 0.6], format_func=lambda v: f"{v:.0%} of applicants",
                               horizontal=True, key="explore_accept")
                e_ = ex[(ex.accept_share == acc) & ~ex.strategy.str.startswith("cost")].copy()
                st.dataframe(e_[["strategy", "auc_all", "auc_rejected", "brier_all", "calibration_gap_rejected"]].rename(columns={
                    "strategy": "How the new model was trained", "auc_all": "AUC, all applicants",
                    "auc_rejected": "AUC among those the old policy rejected", "brier_all": "Brier",
                    "calibration_gap_rejected": "PD minus actual default rate (rejected)"}),
                    hide_index=True, width="stretch", column_config={
                        "AUC, all applicants": st.column_config.NumberColumn(format="%.3f"),
                        "AUC among those the old policy rejected": st.column_config.NumberColumn(format="%.3f"),
                        "Brier": st.column_config.NumberColumn(format="%.4f"),
                        "PD minus actual default rate (rejected)": st.column_config.NumberColumn(format="%+.3f")})
                cost = ex[(ex.accept_share == acc) & ex.strategy.str.startswith("cost")]
                st.caption(f"The old policy (the expert scorecard) accepted {acc:.0%} of applicants; outcomes are known only for "
                           "them, plus a random share of the rejected who received a starter loan. Weighting those by 1 / "
                           "probability of being explored (inverse-propensity weighting) gives the model back a view of "
                           "everyone. Fuzzy augmentation (the classic fix) recovers under 10% of the lost accuracy; a 10% "
                           "exploration budget about half; 20% up to 80%. Full 1 / e weights leave PDs about 2 points too high "
                           "for the rejected group; the learning loop's recalibration corrects the level. Cost: " + "; ".join(f"{r_.strategy.replace('cost of exploration ', '')} exploration "
                                                         f"lends to {r_.explored_share_of_applicants:.1%} of applicants, "
                                                         f"{r_.explored_default_rate:.0%} of whom default"
                                                         for r_ in cost.itertuples()) + ".")
            pl, pf = _csv("proxy_leakage.csv"), _csv("proxy_fairness.csv")
            if pl is not None and pf is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Proxy test: do the AI's inputs reveal who or where someone is?")
                    pl = pl.copy()
                    if "boosted_auc" in pl:
                        pl["p_holm"] = pl[["linear_p_holm", "boosted_p_holm"]].min(axis=1)
                    cols = [c for c in ("attribute", "linear_auc", "boosted_auc", "boosted_null_auc", "p_holm", "reading")
                            if c in pl] if "boosted_auc" in pl else ["attribute", "leakage_auc", "reading"]
                    num = {"linear_auc": "Linear probe AUC", "boosted_auc": "Boosted probe AUC", "boosted_null_auc": "Chance",
                           "p_holm": "p (Holm-corrected)", "leakage_auc": "AUC"}
                    st.dataframe(pl[cols].rename(columns={"attribute": "Attribute", "reading": "Reading", **num}),
                        hide_index=True, width="stretch", column_config={
                            v: st.column_config.NumberColumn(format="%.2f" if k == "p_holm" else "%.3f") for k, v in num.items()})
                    st.caption("A linear and a boosted probe each try to recover the attribute from the 24 inputs, compared with "
                               "100 label-shuffled runs and corrected for testing 8 times. In India the worry is the pin code "
                               "standing in for community or caste; Berka's district and district income are the closest "
                               "testable analogue.")
                with c2:
                    st.markdown("#### Repayers approved, by district income")
                    f_ = pf[pf.attribute == "district_income"].copy()
                    f_["Version"] = f_.version.map({"v1": "1 · rules", "v2": "2 · AI"})
                    st.altair_chart(alt.Chart(f_).mark_bar().encode(
                        x=alt.X("group:N", title=None, sort=["poorer districts", "middle districts", "richer districts"],
                                axis=alt.Axis(labelAngle=0)),
                        xOffset="Version:N",
                        y=alt.Y("approved_among_repayers:Q", title="Repayers approved", axis=alt.Axis(format="%")),
                        color=alt.Color("Version:N", title=None, scale=alt.Scale(range=["#B8863B", "#16324A"]),
                                        legend=alt.Legend(orient="bottom")),
                        tooltip=["group", "Version", alt.Tooltip("approved_among_repayers:Q", format=".1%"),
                                 alt.Tooltip("ci_low:Q", format=".1%"), alt.Tooltip("ci_high:Q", format=".1%")]
                    ).properties(height=250), width="stretch")
                    reg = pf[(pf.attribute == "region") & (pf.version == "v2")]
                    low = reg[~reg.four_fifths_ok]
                    inc_ok = bool(f_[f_.version == "v2"].four_fifths_ok.all())
                    st.caption("Equal opportunity: borrowers who repaid should be approved at similar rates wherever they live. "
                               + ("Every district-income group passes the four-fifths rule. " if inc_ok else
                                  "A district-income group falls below the four-fifths rule. ")
                               + ("By region (8 regions, 56-127 loans each): " + ", ".join(
                                   f"{r_.group} at {r_.ratio_to_best:.2f} of the best-treated region "
                                   f"(interval {r_.ci_low:.0%}-{r_.ci_high:.0%})" for r_ in low.itertuples())
                                  + " is below 0.80, with overlapping intervals; a pilot keeps monitoring it."
                                  if len(low) else "Every region passes it too."))

with tab_how:
    st.markdown("""
### How FAIRScore decides (version 3.0)
1. **Read and verify** the bank statement: every transaction read, every running balance re-added, edits detected.
2. **Lending-rule hard stops** (bank policy): age limits, EMIs above 55% of income, 3+ bounced payments, 5+ lenders,
   heavy gambling → decline. Possible forgery, contradictions, missing months or limits on loan size → an officer.
3. **AI engine** (trained on 679 real loans with their statements):
   - **Probability of default**, calibrated: "4%" means about 4 in 100 such borrowers defaulted.
   - **Certainty with a guarantee** (conformal prediction): the AI decides alone only when it can rule out the other
     outcome; otherwise it refers the case and says why.
   - **Knows its limits**: for applicants unlike its training data, or kinds of applicant it has never seen
     (students, farmers, gig workers), it advises and the lending policy decides.
   - **Explains** every decision (SHAP) and checks **what would change it** by re-running the model.
4. **Fairness**: no gender, caste, religion or age in the AI; every decision re-tested at five ages; outcomes
   monitored by group.
5. **Price and loss**: risk-based interest rate and expected loss for the bank.
6. **Starter-loan ladder** (3.0): when the AI is unsure about the full loan but not confident of default, a small first
   loan (at most one month of income) is offered, re-checked by every rule at that size; on-time repayment unlocks
   larger limits and builds a credit-bureau history. A small random, logged share of slightly riskier cases gets a
   *learning offer*, so the bank learns about people its policy would otherwise never see.
7. **Learning loop** (3.0): live applicants are compared with the training ones (drift, shift detection); uncertainty
   is re-weighted when they differ; recorded repayments recalibrate the AI with a Bayesian prior, and a retrained
   challenger is compared with the current engine. Nothing changes without the model-risk team's approval.
8. **Data and bureau** (3.0): Account Aggregator data (bank-signed) is preferred over PDFs; if the applicant has a
   bureau record, it adds hard stops, cautions and its EMIs to affordability.
The credit officer always takes the final decision.
""")
