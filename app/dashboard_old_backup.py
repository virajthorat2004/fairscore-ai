import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from statement_parser import parse_statement
from ensemble_predictor import run_ensemble
from explain_recourse import compute_shap_breakdown, compute_confidence, get_calibration_notes, generate_recourse

st.set_page_config(page_title="FAIRScore AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #F6F4EF; --ink: #1B2027; --ink-soft: #5B6270; --navy: #16324A;
    --gold: #B8863B; --risk: #9E3A2E; --risk-soft: #F3E0DC; --safe: #2E6B4F;
    --safe-soft: #DEEBE3; --line: #DCD7CC;
}
html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: var(--paper); color: var(--ink); }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; }
.topbar { background: var(--navy); color: #F4F1E8; padding: 18px 28px; margin: -1rem -1rem 1.5rem -1rem;
          display: flex; align-items: baseline; gap: 10px; }
.topbar h1 { font-size: 22px; margin: 0; font-weight: 600; color: #F4F1E8 !important; }
.topbar span { font-size: 12px; color: #B9C3CC; }
.mono { font-family: 'IBM Plex Mono', monospace; }
.verdict-card { border: 1px solid var(--line); background: #FBFAF7; padding: 24px; margin-bottom: 14px; }
.verdict-risk { color: var(--risk); background: var(--risk-soft); padding: 6px 14px; border-radius: 2px;
                font-family: 'IBM Plex Mono'; font-weight: 600; display: inline-block; }
.verdict-safe { color: var(--safe); background: var(--safe-soft); padding: 6px 14px; border-radius: 2px;
                font-family: 'IBM Plex Mono'; font-weight: 600; display: inline-block; }
.model-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 13.5px; }
.flag-box { background: var(--risk-soft); color: var(--risk); padding: 12px 14px; border-radius: 2px; font-size: 13px; margin: 10px 0; }
.panel-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-soft); font-weight: 500; }
.shap-row { display: flex; justify-content: space-between; font-size: 12.5px; padding: 5px 0; border-bottom: 1px solid #EFECE3; }
.confidence-badge { padding: 5px 12px; border-radius: 2px; font-family: 'IBM Plex Mono'; font-weight: 600; font-size: 13px; display: inline-block; }
.conf-high { background: var(--safe-soft); color: var(--safe); }
.conf-medium { background: #F3E9D3; color: var(--gold); }
.conf-low { background: var(--risk-soft); color: var(--risk); }
.recourse-chip { border: 1px solid var(--line); background: #fff; padding: 9px 12px; margin: 0 8px 8px 0;
                  font-size: 12.5px; border-radius: 2px; display: inline-block; }
</style>
<div class="topbar"><h1>FAIRScore AI</h1><span>Responsible credit decision support</span></div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Live Assessment", "Research Results"])

with tab1:
    col_form, col_result = st.columns([1, 1.3], gap="large")

    with col_form:
        st.markdown('<p class="panel-title">Applicant Intake</p>', unsafe_allow_html=True)
        st.subheader("New assessment")
        st.caption("Core details are required. A bank statement is strongly recommended — "
                   "it lets the system route automatically instead of asking you to choose.")

        with st.form("intake_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=40000, step=1000)
                requested_amount = st.number_input("Requested amount (₹)", min_value=1000, value=200000, step=5000)
                total_savings = st.number_input("Total savings (₹)", min_value=0, value=20000, step=1000)
            with c2:
                employment_days = st.number_input("Employment duration (days)", min_value=0, value=500, step=30)
                dependents = st.number_input("Dependents", min_value=0, value=0, step=1)
                requested_duration = st.number_input("Requested duration (months)", min_value=1, value=24, step=1)

            gate_answer = st.selectbox(
                "Do you currently have any active loans, credit cards, "
                "or a bank account older than 2 years?", ["Yes", "No"]
            )

            st.markdown("---")
            st.markdown('<p class="panel-title">Bank statement — 6 months recommended</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV statement", type=["csv"])

            submitted = st.form_submit_button("Assess applicant", use_container_width=True)

    if submitted:
        core = {
            "age": age, "monthly_income": monthly_income, "requested_amount": requested_amount,
            "requested_duration_months": requested_duration, "total_savings": total_savings,
            "dependents": dependents, "employment_days": employment_days,
        }

        stmt_df, stmt_features, parse_warnings = None, {}, []
        if uploaded_file is not None:
            try:
                tmp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(tmp_dir, "uploaded_statement.csv")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                stmt_df, stmt_features, parse_warnings = parse_statement(tmp_path)
            except ValueError as e:
                st.error(f"Could not parse this statement: {e}")

        with st.spinner("Running ensemble assessment..."):
            result = run_ensemble(core, stmt_features, stmt_df, gate_answer)
            confidence = compute_confidence(result)
            calibration_notes = get_calibration_notes(result["models_used"])
            shap_breakdown = compute_shap_breakdown(core, stmt_features, stmt_df, gate_answer, result)
            recourse = generate_recourse(core, stmt_features, stmt_df, gate_answer, result)

        overall_manual_review = result["manual_review_required"] or confidence["manual_review"]

        with col_result:
            st.markdown('<p class="panel-title">Assessment Result</p>', unsafe_allow_html=True)

            for w in parse_warnings:
                st.warning(w)

            if result["consistency_flagged"]:
                st.markdown(f'<div class="flag-box">⚑ {result["consistency_message"]}</div>', unsafe_allow_html=True)

            if result["consensus_risk_probability"] is None:
                st.markdown("""
                <div class="verdict-card">
                  <span class="verdict-risk">Insufficient data</span>
                  <p style="margin-top:14px; color:var(--ink-soft); font-size:13.5px;">
                  No model had enough genuinely derivable information to produce a reliable
                  assessment. Upload a bank statement, or route this application for full
                  manual underwriting.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                risk = result["consensus_risk_probability"]
                verdict_class = "verdict-risk" if risk >= 0.5 else "verdict-safe"
                verdict_label = "Higher risk" if risk >= 0.5 else "Lower risk"
                st.markdown(f"""
                <div class="verdict-card">
                  <span class="{verdict_class}">{verdict_label}</span>
                  <span class="mono" style="float:right; font-size:26px; font-weight:600;">{risk:.4f}</span>
                  <p style="color:var(--ink-soft); font-size:12.5px; margin-top:4px;">Consensus risk probability</p>
                </div>
                """, unsafe_allow_html=True)

            # --- Confidence ---
            conf_class = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}[confidence["label"]]
            st.markdown(f"""
            <p class="panel-title" style="margin-top:18px;">Confidence</p>
            <span class="confidence-badge {conf_class}">{confidence["label"]}</span>
            <span style="font-size:12.5px; color:var(--ink-soft); margin-left:10px;">{confidence["explanation"]}</span>
            """, unsafe_allow_html=True)

            if overall_manual_review:
                st.info("Flagged for manual review.")

            # --- Model participation ---
            st.markdown('<p class="panel-title" style="margin-top:20px;">Model participation</p>', unsafe_allow_html=True)
            for name, detail in result["details"].items():
                status = "✓ voted" if detail["qualifies"] else "excluded"
                color = "var(--safe)" if detail["qualifies"] else "var(--ink-soft)"
                prob_str = f"risk={detail['risk_probability']:.4f}" if detail["qualifies"] else ""
                st.markdown(f"""
                <div class="model-row">
                  <span>{name.replace('_', ' ').title()}</span>
                  <span class="mono" style="color:{color};">applicability={detail['applicability']:.2f} · {status} {prob_str}</span>
                </div>
                """, unsafe_allow_html=True)
                if not detail["qualifies"]:
                    st.caption(detail["exclusion_reason"])

            # --- SHAP breakdown ---
            if shap_breakdown:
                st.markdown('<p class="panel-title" style="margin-top:20px;">Why this verdict</p>', unsafe_allow_html=True)
                for name, rows in shap_breakdown.items():
                    with st.expander(f"{name.replace('_', ' ').title()} — top factors"):
                        for r in rows:
                            direction = "↑ risk" if r["shap_value"] > 0 else "↓ risk"
                            color = "var(--risk)" if r["shap_value"] > 0 else "var(--safe)"
                            tag = "" if r["source"] == "real" else f" ({r['source']})"
                            st.markdown(f"""
                            <div class="shap-row">
                              <span>{r['feature']}{tag}</span>
                              <span class="mono" style="color:{color};">{r['shap_value']:+.4f} {direction}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        if name in calibration_notes:
                            st.caption(f"Calibration note: {calibration_notes[name]}")

            # --- Counterfactual recourse ---
            if recourse:
                st.markdown('<p class="panel-title" style="margin-top:20px;">If rejected — what could change the outcome</p>',
                           unsafe_allow_html=True)
                for s in recourse:
                    flip_note = " → flips to lower risk" if s["flips_to_lower_risk"] else ""
                    st.markdown(f"""
                    <span class="recourse-chip">{s['change']} → risk drops to
                    <b class="mono">{s['new_risk']}</b>{flip_note}</span>
                    """, unsafe_allow_html=True)
            elif result["consensus_risk_probability"] is not None and result["consensus_risk_probability"] >= 0.5:
                st.caption("No realistic adjustment to loan amount or duration meaningfully reduced this applicant's risk.")

with tab2:
    st.subheader("Does the framework hold up across datasets?")
    st.caption("Independent training and audit results across four datasets of different scale, "
               "structure, and applicant population.")
    research_df = pd.DataFrame([
        {"Dataset": "German Credit", "n": 1000, "Role": "Primary benchmark", "Best model": "Logistic Regression",
         "ROC-AUC": 0.811, "DP ratio (before)": 0.545, "DP ratio (after)": 0.661, "Status": "Validated"},
        {"Dataset": "Give Me Some Credit", "n": 150000, "Role": "Scale benchmark", "Best model": "LightGBM",
         "ROC-AUC": 0.867, "DP ratio (before)": 0.481, "DP ratio (after)": 0.654, "Status": "Validated"},
        {"Dataset": "Home Credit Default Risk", "n": 307511, "Role": "Thin-file benchmark", "Best model": "CatBoost",
         "ROC-AUC": 0.751, "DP ratio (before)": 0.543, "DP ratio (after)": 0.936, "Status": "Trade-off noted"},
        {"Dataset": "Africa Synth", "n": 101, "Role": "Alternative-data demo", "Best model": "CatBoost",
         "ROC-AUC": None, "DP ratio (before)": None, "DP ratio (after)": None, "Status": "Qualitative only"},
    ])
    st.dataframe(research_df, use_container_width=True, hide_index=True)
    st.caption("Sample sizes and mitigation outcomes vary by dataset — Home Credit's demographic-parity "
              "gain came with a recall trade-off; Africa Synth (n=101) is used for explainability "
              "demonstration only, not statistical validation.")