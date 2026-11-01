# FAIRScore AI

Explainable, fairness-aware credit decision support for people with no credit history. FAIRScore reads a bank
statement (a PDF from any bank, or Account Aggregator data), checks that it is genuine, measures 24 money habits and
gives a calibrated probability of default. An AI model and the bank's lending rules decide together: approve only
when confident, refer to a credit officer when unsure (with a small starter-loan offer), and decline only through
hard rules. Every decision is explained and checked for fairness.

**Results on 679 real loans (nested cross-validation):** AUC 0.86 (0.81-0.91); 38% approved at 1.8% default,
against 31% at 3.3% for rules alone; referrals down from 51% to 40%.

## Try it

Open the app, then upload a sample statement from `data/samples/realistic/` (for example
`R04_kirana_shop_owner.pdf`) and fill in the form with the inputs listed in `data/samples/TESTING_GUIDE.md`.
All sample statements are synthetic (fictional DEMO BANK, marked SAMPLE).

Please use sample statements only: this is a demonstration, and an uploaded file is processed on the hosting server.

## Run locally

```
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## What is in this repository

| Folder | Contents |
|---|---|
| `app/dashboard.py` | The Streamlit dashboard |
| `src/` | Statement reader, scoring, AI engine, rules, explanations, reports |
| `config/policy.toml` | All lending rules and thresholds |
| `models/` | The trained models used by the dashboard |
| `results/` | Evaluation results shown in the "AI model evidence" tab |
| `data/samples/` | Synthetic sample statements for the demo |
| `docs/` | Method, results and verification report |

This repository holds the deployable app only. Training data, research notebooks and experiments are kept in the
full project.
