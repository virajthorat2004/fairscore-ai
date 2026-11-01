# FAIRScore 3.0: a credit decision system that keeps working in a new market

Version 2.0 made the AI the decision engine: a calibrated, monotone model of default risk with conformal
uncertainty, lending rules as hard stops, explanations and model-checked recourse. It was tested on real loans.
Version 3.0 deals with what happens **after** a lender switches it on: different applicants, a different default
rate, outcomes that arrive a year later, and a bank that only ever sees the outcomes of the loans it granted.
Those are the problems that sink real credit models, and the reason Indian lenders stay cautious with
new-to-credit (NTC) applicants.

Every number below comes from code in this repository and can be re-run (section 8).

---

## 1. How Indian lenders decide today (2026), and where it fails NTC applicants

**The usual retail pipeline.**
1. KYC: CKYC / e-KYC / video KYC, PAN, and duplicate and negative-list checks.
2. A **bureau pull** with a score cut-off. This is the main gate.
3. Income proof (salary slips, ITR, GST, bank statements run through a statement analyser).
4. Policy rules: FOIR (EMIs as a share of income), income multiples, age, employer category and pin-code lists.
5. An application scorecard, then a credit manager who approves "deviations".
6. After disbursal: behaviour scorecards and early-warning signals.

**What changed recently.**
- **Account Aggregator (AA) data** is now used at scale: ₹3.82 lakh crore lent across 3.68 crore loans in FY26.
  - It brings in more NTC borrowers: 9% of personal loans with AA vs 3.8% without, and 28.5% vs 8.6% of
    mid-ticket business loans.
  - Two-wheeler loans made with AA data had 0.5% 90+ days past due (DPD), against 5.3% without.
  - So cash-flow data is both more inclusive and safer. That is FAIRScore's premise, now visible at national scale.
- **RBIH's Unified Lending Interface (ULI)** has 64 lenders and 136+ data services, including land records,
  satellite data and dairy data.
- The **public-sector MSME model** combines GST, AA, ITR, bureau and fraud checks with straight-through processing.

**The NTC squeeze.** The NTC share of loan originations fell from 20% (2023) to 18% to **16%** (Q1 FY26) as lenders
turned cautious. In practice an NTC applicant is rejected, pushed to a gold loan or an FD-backed card, or given a
small, expensive loan decided by an opaque fintech model.

| Gap in current practice | Effect |
|---|---|
| G1 bureau-first gate | no score means no loan, whatever the cash flow |
| G2 statements read by fixed rules | average balance and bounce counts against thresholds; no learned risk |
| G3 edited PDFs | document fraud; rules and tools chase forgers |
| G4 hard cut-offs, no uncertainty | everything ambiguous goes to a person: slow and inconsistent |
| G5 opaque fintech models | weak reasons; RBI's 2026 draft model-risk guidance and its FREE-AI framework ask for validation and explainability |
| G6 collateral as a yes/no | not combined with risk into expected loss and price |
| G7 "refer" is a dead end | an officer who sees no history usually declines |
| G8 models learn only from approved loans | the people turned away are never learned about (selection bias) |
| G9 a model built elsewhere drifts | no routine recalibration, no shift detection |
| G10 fairness unmeasured | proxies such as pin code go untested |

## 2. What 2.0 already answers (G2, G3, G4, G5, G6)

Nested cross-validation on 679 real loans with their pre-loan statements (Berka):

| Measure | Result |
|---|---|
| Discrimination | AUC 0.862 (0.812-0.910) |
| Calibration | ECE 0.019 |
| Approvals and their default rate | 38.1% approved at 1.8% default, vs 30.8% at 3.3% for rules alone |
| Referrals | 11.5 points fewer (significant) |

Conformal prediction bounds how many would-be defaulters can be auto-approved. Rules kept as hard stops catch
forged statements that the AI alone would approve. Details are in [AI_ENGINE.md](AI_ENGINE.md).

## 3. What 3.0 adds

| Part | Gap | Module |
|---|---|---|
| Starter-loan ladder | G1, G7 | `src/starter_ladder.py`, `[starter]` in the policy |
| Learning loop: drift, day-0 estimates, Bayesian recalibration, champion/challenger | G9 | `src/learning_loop.py`, `[learning]` |
| Shift-aware conformal prediction | G4 under G9 | `learning_loop.DensityRatio`, `ai_engine.conformal_shift_aware` |
| Exploration with logged propensities (learning offers) | G8 | `starter_ladder`, `audit_store.propensity` |
| Credit-bureau layer | G1 for thin files | `src/bureau.py`, `[bureau]` |
| Data provenance: AA first | G3 | `live_scoring` "Data source" check |
| Proxy-fairness audit | G10 | `src/proxy_fairness.py` |

### 3.1 Starter-loan ladder: a small yes instead of a dead end

**Rule.** When the AI is unsure about the requested loan (its conformal set is not {repays}) but not confident of
default, FAIRScore offers a first loan of at most ₹50,000 and at most one month of verified income, over 12 (or 6)
months. The offer is re-checked by the **whole** decision at that size: every policy rule, the AI, and its
conformal test.
- It is approved if the AI's default risk at that size is at most 12%.
- It is priced by the risk-based rate.
- Further rungs unlock after 6 and 12 on-time EMIs, each re-assessed with a fresh statement and the repayment
  record.
- Each repaid loan is reported to the credit bureaus, so **the ladder creates the credit history that the
  bureau-first system demands.**
- For applicant types the AI is advisory for (students, farmers, freelancers), the starter loan is a suggestion
  that an officer confirms.

**Evidence** (nested out-of-fold decisions; conservative: a starter loan is assumed to default whenever the full
loan did; run `src/starter_ladder.py`):

| | Version 2 | Version 3 |
|---|---|---|
| Applicants offered credit today | 38.1% | **60.4%** |
| Default rate among them | 1.8% | 3.3% |
| Indicative profit per applicant | 2,749 | 2,714 (−1%) |

The 22.0% moved from "refer" to a starter loan default at 5.8%. Lent at starter size, their loss is **81% lower**
than lending them the full amount: 341 vs 1,826 per applicant (statement currency).

**Honest reading.**
- Starter loans roughly break even on their own: they are small and short, so interest income is small.
- On Berka in hindsight, lending the full amount to this group would have earned more, *if* the AI's risk
  estimate holds.
- The ladder is a **risk-control and data-collection product**: these are exactly the applicants the model is
  unsure about. In a new market its PDs can be wrong. Under stress (the group's true default rate multiplied by
  k), the starter's loss stays about 4.5 times smaller at every k:

  | Stress k | Starter | Full amount |
  |---|---|---|
  | 1 | +11 | +1,491 |
  | 2 | −322 | −106 |
  | 3 | −656 | −1,702 |

Live examples (synthetic statements):
- A kirana-shop owner referred for ₹3 lakh is approved today for ₹50,000 over 12 months, which passes every rule.
- A first-job applicant with a thin file who asked for ₹50,000 gets ₹24,000 (one month's income) at a 6.3% risk.
- A freelancer with lumpy income who asked for ₹4 lakh gets a suggested ₹50,000 that an officer confirms.
- A delivery partner the AI is confident will default gets nothing.

### 3.2 Learning loop: keep the engine honest in a new population

`src/shift_study.py` recreates "trained in one market, used in another" inside Berka, where every outcome is
known. Eight scenarios, 30 repeats each:
- no shift;
- the default rate doubled (prior shift);
- applicants tilted towards thin or thick financial cushions (covariate shift, moderate and strong);
- covariate + prior shift together;
- two natural shifts: trained up to 1996 and used on 1997-98 loans, and trained in Bohemia and used in Moravia.

Applications arrive as a stream, and outcomes become known for the first n.

**Day 0 (no outcomes yet).** Two label-free estimates of the new default rate, from 400 applications:

| Scenario | True | Mean PD | EM (prior-shift) |
|---|---|---|---|
| no shift | 11.0% | 10.9% | 11.9% |
| default rate doubled | 20.0% | 13.9% | **21.3%** |
| thin cushion | 17.6% | **16.8%** | 31.3% |
| thick cushion (strong) | 5.5% | **5.5%** | 0.6% |
| covariate + prior | 24.3% | 20.1% | 40.9% |

The mean PD is right when only the *applicants* change; the EM estimate (Saerens et al., 2002) is right when only
the *default rate* changes and badly wrong otherwise. Without outcomes the two cannot be told apart, so FAIRScore
applies neither to PDs: it reports both and raises an alarm when they disagree. (A version that applied the EM
correction made calibration much worse under covariate shift, so it was dropped.)

**Shift-aware uncertainty.** A classifier that tells training applicants from live ones gives each training case
a weight w(x) = p_live(x) / p_train(x). Weighted Mondrian conformal prediction (Tibshirani et al., 2019) then
computes the p-values on the re-weighted cases.
- The weights are cross-fitted, clipped to 0.2-5 and normalised.
- They are used only when the shift is detectable: the cross-validated AUC must be more than two standard errors
  above 0.5.

Share of would-be defaulters the AI could auto-approve (the guarantee is at most 10%):

| Scenario | Standard (2.0) | Always weighted | **3.0 (weighted if shift detected)** |
|---|---|---|---|
| no shift | 9.1% | 9.0% | 9.1% |
| default rate doubled | 8.7% | 13.0% | 9.1% |
| thick cushion | 12.1% | 7.6% | 10.3% |
| thick cushion, strong | **16.7%** | 2.6% | **2.6%** |
| covariate + prior | 5.5% | 10.4% | 9.6% |

- When new applicants look safer than the training ones but some still default, standard conformal breaks its
  own guarantee (16.7%). Weighting restores it.
- Weighting *always* breaks the guarantee under a pure default-rate change, which is why it is gated on
  detection.
- A Mondrian (class-conditional) test is already valid under a pure prior shift, because p(x | y) does not change.
- A recalibration with an increasing map leaves conformal p-values unchanged, so the two parts compose cleanly.
- The cost: under a strong shift towards safer-looking applicants, the share the AI dares to auto-approve falls
  from 60% to 16%. That is the price of keeping the guarantee with few comparable training defaulters.

**Day N (outcomes recorded).** Brier score on the target population (lower is better; "none" = the engine as
trained):

| Scenario | none | n | Platt | **Bayesian (3.0)** | retrain (source + local) |
|---|---|---|---|---|---|
| default rate doubled | 0.1080 | 25 | 0.1175 | **0.1064** | n/a |
| | | 100 | 0.1053 | **0.1035** | 0.1050 |
| thin cushion | 0.0822 | 25 | 0.0955 | **0.0824** | n/a |
| covariate + prior | 0.1033 | 100 | 0.1057 | **0.1025** | 0.1071 |
| out of time | 0.0542 | 50 | 0.0572 | 0.0526 | **0.0488** |
| | | 200 | 0.0520 | 0.0518 | **0.0450** |
| other region | 0.0650 | 100 | 0.0664 | **0.0645** | 0.0701 |

- Bayesian recalibration, PD' = sigmoid(a + b · logit PD) with a Gaussian prior on "no change" (a = 0, b = 1), is
  never meaningfully worse than doing nothing and beats plain Platt scaling at every sample size.
- Platt scaling is *worse than doing nothing* below about 100-200 outcomes.
- Where the meaning of default itself changed (out of time: recent loans are still running), only retraining
  helps. FAIRScore therefore proposes a retrained challenger (Berka + local loans) once 200 outcomes and 30 defaults
  are recorded, and compares it with the current engine with bootstrap intervals.
- Nothing is promoted automatically: the model-risk team switches `use_adaptation` on.

### 3.3 Exploration: learning about the people a bank turns away

`src/exploration_study.py` uses the fact that every Berka applicant got the loan:
- the expert scorecard plays the lender's old policy, accepting the best 60% (or a stricter 40%);
- outcomes are hidden for the rest;
- a random share of the rejected receives a starter loan, so their outcomes are known.

Five-fold cross-validation, 10 repeats; judged on everyone who applied.

AUC on all applicants (AUC among those the old policy rejected in brackets). The oracle knows every outcome:
AUC 0.873.

| Training data | Old policy accepts 60% | Old policy accepts 40% |
|---|---|---|
| Accepted loans only (selection bias) | 0.835 (0.821) | 0.799 (0.764) |
| Fuzzy augmentation (classic reject inference) | 0.836 (0.831) | 0.805 (0.776) |
| + 5% exploration, clipped IPW | 0.847 (0.846) | 0.827 (0.818) |
| + 10% exploration, clipped IPW | **0.854 (0.849)** | **0.839 (0.838)** |
| + 20% exploration, clipped IPW | 0.857 (0.854) | 0.858 (0.862) |
| Share of applicants explored / their default rate (10%) | 4.1% / 18% | 5.9% / 15% |

- Training only on approved loans costs AUC: 0.04 when the old policy is generous and 0.07 when it is strict. The
  loss is largest among the applicants the old policy rejected: the new-to-credit growth market.
- The classic reject-inference fix (fuzzy augmentation) recovers under 10% of the gap.
- A 10% exploration budget recovers about half of the gap (52-55%); 20% recovers 60-80%. It does so by lending
  small amounts to about 4-6% of applicants, of whom 15-18% default.
- Weighting the explored outcomes (IPW) helps ranking a little but leaves the PDs slightly too high for the
  rejected group (by about 2 points). The capped weights (FAIRScore 3.0) sit between unweighted and full 1 / e
  weights, and the learning loop's recalibration step corrects the level.
- Fold-to-fold spread is about ±0.05 AUC; the comparisons are paired on the same folds and repeats, so the
  ordering is consistent, but differences of 0.01 between weighting schemes are not meaningful.

Live, learning offers are deterministic (a hash of the application, so audit can reproduce them). Their
propensity is stored with the assessment, and `learning_loop` weights recorded outcomes by min(1 / propensity, 5).

### 3.4 Credit-bureau layer (thin files)

FAIRScore is built for applicants with no bureau record, but many have something. When the officer enters the
report:
- a 90+ DPD, write-off or settlement within 24 months → decline;
- an established file below 650 → decline;
- a thin file below 650, a score below 700, any 30+ DPD in 12 months, or more than 6 enquiries in 6 months →
  officer review;
- **bureau EMIs above those seen in the statement** are used for FOIR, and the gap is flagged: loans are being
  paid from another account.

This is a rule layer, not a learned blend: no data links Indian bureau records, statements and outcomes, so a
blend could not be fitted honestly.

### 3.5 Data provenance: Account Aggregator first

Every assessment states its evidence tier:
- **Account Aggregator data** is signed by the bank and fetched with consent; it cannot be edited, and the
  document-forgery checks are not needed.
- **A PDF** is checked for balance-chain breaks, font outliers, printed-total mismatches and editor metadata.

In 2.0 an AI with no rules would have approved 5 of 7 forged samples. AA input removes that attack surface
entirely.

### 3.6 Proxy-fairness audit

In India the worrying proxy is the pin code, which can stand in for community or caste. Berka's district (77
districts, 8 regions, district average salary) is the closest testable analogue.
- **Leakage.** Can the 24 inputs predict the region, a poorer district, gender or being under 30?
  A linear and a boosted probe, each compared with label-shuffled runs and Holm-corrected over the 8 tests: no
  attribute is recoverable. The one raw signal, a weak nonlinear trace of gender (AUC 0.556 vs 0.506 by chance;
  raw p 0.05, corrected 0.40), is not significant, and women repayers are approved slightly more often.
- **Outcomes** (repayers approved, version 2):

  | District income | Repayers approved | Ratio to best |
  |---|---|---|
  | poorer | 44.0% | 1.00 |
  | middle | 43.8% | 1.00 |
  | richer | 37.9% | 0.86 |

  Every group passes the four-fifths rule.
- **By region,** Prague is at 0.79 of the best-treated region in both versions (overlapping intervals). It is
  reported, and a pilot would keep monitoring it.

## 4. Today's practice vs FAIRScore 3.0

| | Bank rulebook | Fintech ML | FAIRScore 3.0 |
|---|---|---|---|
| NTC applicant | rejected or secured-only | small, costly, opaque | AI decision on cash flow, or a priced starter loan |
| Statement evidence | thresholds | learned, unexplained | calibrated PD + SHAP reasons + verified recourse |
| Uncertainty | none (cut-off) | none | conformal sets with a bound, kept under shift |
| Forged documents | manual / tools | varies | hard stops + AA first |
| New market | static | retrain when broken | drift and shift detection, Bayesian recalibration, champion/challenger |
| Rejected applicants | never learned | never learned | logged exploration, capped IPW |
| Collateral | yes/no | n/a | expected loss and risk-based price |
| Fairness | not measured | rarely | age-flip per decision, group and proxy audits |
| Governance | policy file | varies | versioned policy, audit trail, nothing auto-promoted |

## 4b. Added after the final verification (update 28)

The independent audit (docs/VERIFICATION_REPORT.md) added two rules.

**No AI-only declines before local validation.** On seven real Indian statements, read perfectly by the system,
the AI's inputs sat at the edge of its Czech training data:
- low-balance days beyond 94-100% of training borrowers;
- balance relative to the EMI in the bottom 0-5%.

It would have called most of them likely defaulters. Until an adaptation fitted on local repayment outcomes is
approved (`use_adaptation`), the AI may recommend approval but its "confident default" becomes a referral with a
warning (`[learning] ai_decline_needs_local_validation`). Hard-stop rules still decline. This is **asymmetric
autonomy**: the costly error in a new market is wrongly excluding people, which is what this project exists to
prevent, so the unvalidated model is not allowed to make it alone.

**Window dressing assessed with and without the money.** When a large unexplained credit arrived in the last
30 days and is still in the account, the whole assessment also runs as if it had never arrived, and the worse of
the two stands.
- On the 679 real loans, the 26 borrowers with this pattern defaulted at 26.9%, against 10.1% for the rest (odds
  ratio 3.3, Fisher exact p = 0.015).
- Version 2 had auto-approved 5 such cases (pooled over repeats), 3 of which defaulted.

## 5. What is still not solved, and what would solve it

1. **No Indian outcomes.** Everything above is measured on Berka. A lender pilot (docs/PILOT_PLAN.md) supplies
   them; the learning loop is the machinery that uses them.
2. **Shift detection power.** With 400 applications, moderate covariate shifts are detected about 45% of the time;
   the guarantee is then only approximately kept. More applications, or a fixed weighting once the lender's
   population is stable, close this.
3. **Concept change** (the meaning of default changes) cannot be fixed by recalibration; only retraining on local
   outcomes can. That is why the challenger exists.
4. **Starter-loan economics** are near break-even on their own. Their return is information and graduation, which
   needs a pilot to measure.
5. **Bureau fusion** is rules, not learned.
6. **Proxy audit** uses Czech districts as a stand-in for pin codes. In a pilot the same code runs on pin-code
   groups.
7. **Exploration** lends to riskier applicants by design (18-20% defaulted at small size). The budget and its
   ceiling (`explore_share`, `explore_max_pd`) are a lender's risk-appetite decision.

## 6. Configuration

`config/policy.toml` version 2026.10-11 holds the 3.0 settings in `[starter]`, `[learning]` and `[bureau]`
(defaults as above; `explore_share = 0` switches learning offers off, and `use_adaptation = false` until the
model-risk team approves).

## 7. Tests

- `tests/test_backend.py` covers:
  - weighted conformal equals the engine when unweighted;
  - Bayesian recalibration moves only with evidence;
  - EM recovers a pure prior shift;
  - shift detection;
  - the learning loop on recorded outcomes (incl. the challenger);
  - the starter ladder (and that confident defaults and hard stops never get one);
  - the bureau layer;
  - learning-loop fields in the audit log.
- `tests/test_invariants.py` adds starter-ladder and bureau invariants on random applications.

## 8. Reproduce

    python src/train_ai_engine.py        # engine (stores calibration inputs for the learning loop)
    python src/starter_ladder.py         # 3.1
    python src/shift_study.py            # 3.2   (~20 min)
    python src/exploration_study.py      # 3.3   (~10 min)
    python src/proxy_fairness.py         # 3.6
    python src/learning_loop.py          # lender-side run on recorded outcomes

## Sources

- [CRIF-Sahamati Account Aggregator lending report FY26 (KhabarPatri)](https://english.khabarpatri.com/2026/10/16/crif-sahamati-account-aggregator-lending-fy26/)
- [Unified Lending Interface: 64 lenders, 136 data services (Medianama)](https://www.medianama.com/2026/01/223-unified-lending-interface-64-lenders-136-data-services/)
- [PSB digital MSME credit assessment model (PIB)](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2216047&reg=6&lang=1)
- [New-to-credit loan share dips to 16% in Q1 (Business Standard)](https://www.business-standard.com/amp/finance/news/new-to-credit-loan-share-dips-to-16-in-q1-as-lenders-stay-cautious-report-125092401219_1.html)
- [RBI draft Model Risk Management guidelines 2026 (Vinod Kothari)](https://vinodkothari.com/2026/06/rbi-draft-model-risk-management-guidelines-2026/)
- [TransUnion CIBIL CreditVision New-to-Credit Score](https://www.transunioncibil.com/lp/creditvision-new-to-credit-score)
- Saerens, Latinne & Decaestecker (2002), Adjusting the outputs of a classifier to new a priori probabilities.
- Tibshirani, Barber, Candès & Ramdas (2019), Conformal prediction under covariate shift.
- Vovk et al. (2005), Algorithmic Learning in a Random World (Mondrian conformal prediction).
