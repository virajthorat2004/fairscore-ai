# FAIRScore AI decision engine (2.0, extended in 3.0)

Version 1 let written lending rules decide and allowed the machine-learning models only to refer a case. Tested on
real loans, the learned model ranks risk far better than the rules. Version 2 therefore lets the AI **decide**,
inside the bank's hard stops, and adds what a bank needs before trusting it: calibrated probabilities, a
statistical guarantee on its uncertainty, knowledge of its own limits, exact explanations, model-checked recourse,
fairness measurement and risk-based pricing.

All numbers below are **nested cross-validation** on 679 real loans with their rebuilt pre-loan bank statements
(Berka, 73 defaults): the model, its calibration, the conformal scores, the approval threshold and the
unusual-applicant detector are all re-learned without the loans being tested (5 folds x 3 repeats). Reproduce with
`python src\train_ai_engine.py` (about 3 minutes); results in `results/ai_engine_*.csv`.

## 1. Method

| Part | What it does | Why |
|---|---|---|
| **Knowledge-guided model** | Monotone CatBoost on 24 inputs: 12 cash-flow ratios read from the statement + the 12 behaviour scores of the expert FAIRScore scorecard | Expert knowledge as features; monotone constraints guarantee a better habit can never raise the risk |
| **Calibration** | Platt scaling fitted on out-of-fold predictions | "4%" means about 4 in 100 such borrowers defaulted |
| **Conformal prediction** | Mondrian (class-conditional) split-conformal p-values for "repays" and "defaults" at alpha = 10% | The AI decides alone only when it can rule out the other outcome; then at most alpha of would-be defaulters can be auto-approved (distribution-free guarantee) |
| **Knows its limits** | Isolation Forest typicality score; and the AI decides only for applicant types present in its training data (salaried, pension, self-employed). For students, farmers and gig workers it is advisory and the expert policy decides | No guessing outside experience |
| **Decision rule** | Hard stops (age, EMIs > 55% of income, 3+ bounces, 5+ lenders, heavy gambling) decline; possible forgery, contradictions, missing months, borderline affordability and product limits go to an officer; otherwise **approve** if confident "repays" and PD <= risk appetite (3% target default rate, learned threshold 4.2%), **decline** if confident "defaults" and PD >= 30%, else **refer** | Rules stay where they are regulation / product policy; the AI decides risk |
| **Explanations** | Exact TreeSHAP contributions, grouped into 7 habits (balance kept, affordability, income regularity, conduct, other borrowing / gambling, saving and bills, completeness) | Every decision explained; contributions add up exactly to the model output (tested) |
| **Recourse** | (1) smaller loan / longer term re-assessed through the whole decision; (2) habit changes re-scored by the model | "What would change the decision", never an unchecked suggestion |
| **Pricing** | Expected loss = PD x LGD (65%) x amount; rate = (cost of funds + operating cost + margin + yearly expected loss) / (1 - yearly PD), between 10.5% and 26% | The bank's question "what should we charge?" |

Settings are in `config/policy.toml`, section `[ai]` (mode, alpha, risk appetite, decline threshold, which
cautions always go to an officer, pricing). `mode = "policy_first"` restores version 1.

## 2. Results

**Ranking and calibration**

| Model | AUC (95% interval) | Gini | Brier | Calibration error |
|---|---|---|---|---|
| **FAIRScore 2.0 AI engine** | **0.862 (0.812-0.910)** | 0.724 | 0.064 | 0.019 |
| Cash-flow model v1 (12 ratios) | 0.850 (0.795-0.895) | 0.701 | 0.066 | 0.022 |
| Logistic regression (12 ratios) | 0.822 (0.761-0.876) | 0.644 | 0.070 | 0.020 |
| Expert scorecard (no training) | 0.774 (0.710-0.841) | 0.548 | | |
| Predict the average | 0.500 | | 0.096 | |

**Decisions: version 1 (rules) vs version 2 (AI)**

| | Approved | Defaulted among approved | Referred to an officer | Declined |
|---|---|---|---|---|
| Version 1 | 30.8% | 3.3% | 51.1% | 18.1% |
| **Version 2** | **38.1%** | **1.8%** | **39.7%** | 22.2% (26.0% of them defaulted) |

**24% more applicants are approved and referrals drop by a fifth** -- both statistically significant (bootstrap
over loans: +7.4 points, 95% interval 4.4 to 10.4; referrals -11.5 points, -14.7 to -8.2). Defaults among approved
loans fall from 3.3% to 1.8%; that drop points the right way in 94% of resamples but 679 loans are too few to
prove it at the 95% level (-1.5 points, -3.7 to +0.4). Indicative lender profit per rupee lent: +51% (1.35 to 2.04 per 100 applicants). The AI decided 69% of
cases; the rest were hard stops. 13.7 points of the declines are the legal age rule in this data.

**The conformal guarantee holds on real loans**

| alpha | Defaulters flagged as possible | Defaulters auto-approved (must be <= alpha) | Auto-approved | Default rate auto-approved | Uncertain |
|---|---|---|---|---|---|
| 5% | 95.9% | 4.1% | 28.1% | 1.6% | 62.0% |
| **10%** | 92.2% | **7.8%** | 46.1% | 1.8% | 38.5% |
| 15% | 86.8% | 13.2% | 61.4% | 2.3% | 18.0% |
| 20% | 81.7% | 18.3% | 69.2% | 2.8% | 4.0% |

A bank chooses alpha as its trade-off between automation and safety.

**Fairness (gender and age are never inputs)**

| Group | Repayers approved, v1 | Repayers approved, v2 | Default rate if approved, v2 |
|---|---|---|---|
| Female | 33.6% | 43.3% | 2.0% |
| Male | 33.1% | 40.5% | 1.6% |
| Under 30 | 25.9% | 31.4% | 1.1% |
| 30-44 | 40.9% | 50.1% | 2.7% |
| 45 and over | 32.0% | 42.9% | 1.1% |

Every group gains. Excluding the legal age rule, repayers under 30 are approved at 47.9%. **Fairness-constrained
training** (fairlearn exponentiated gradient, equalized odds; `src/ai_fairness.py`) lowered AUC from 0.87 to
about 0.69 and raised defaults among approved loans from 1.8% to about 6.9%, while the unconstrained model's
equalized-odds gap was already small (gender 0.07, age 0.06). The deployed model is unconstrained and outcomes are
monitored by group; the trade-off is reported, not hidden.

**Standard evaluation** (`python src\model_comparison.py`; results/model_zoo.csv, results/eval_*.csv)

| Algorithm (same 24 inputs, 5 folds x 3 repeats) | AUC (95% interval) | KS | PR-AUC | Train AUC | Overfit gap | DeLong p vs deployed |
|---|---|---|---|---|---|---|
| CatBoost, no constraints | 0.870 (0.825-0.913) | 0.585 | 0.612 | 0.978 | 0.107 | 0.33 |
| **CatBoost, monotone (deployed)** | **0.870 (0.815-0.911)** | 0.576 | 0.603 | 0.939 | **0.069** | - |
| Random forest | 0.868 (0.824-0.910) | 0.570 | 0.620 | 0.987 | 0.119 | 0.62 |
| XGBoost | 0.857 (0.798-0.898) | 0.582 | 0.571 | 0.982 | 0.124 | 0.24 |
| LightGBM | 0.851 (0.793-0.896) | 0.569 | 0.557 | 0.992 | 0.141 | 0.13 |
| Logistic regression | 0.834 (0.772-0.888) | 0.552 | 0.581 | 0.886 | 0.053 | 0.06 |
| Decision tree (depth 4) | 0.798 (0.750-0.860) | 0.494 | 0.447 | 0.902 | 0.104 | 0.002 |
| Expert scorecard (no training) | 0.774 (0.711-0.834) | 0.456 | 0.372 | - | - | <0.001 |
| Neural network (MLP 32-16) | 0.691 (0.614-0.767) | 0.370 | 0.380 | 0.716 | 0.025 | <0.001 |

The deployed model ties the best algorithms (no significant difference) with the smallest overfitting gap of the
boosted models, and its monotone constraints guarantee sensible behaviour; it is significantly better than a
single tree, the expert scorecard and a neural network (679 loans are too few for one). **Tuning sensitivity:** AUC
0.848-0.864 across 12 settings of depth (2-6), trees (300/600) and learning rate (0.03/0.1).

**Why three AUC figures (0.862, 0.870, 0.864) and which to quote.**
- **0.862** (the headline) is one AUC over all 679 out-of-fold predictions of the *complete* recipe: model,
  calibration, conformal scores and threshold, all re-fitted in every outer fold (nested).
- **0.870** is the *mean of the 15 per-fold AUCs* of the raw model. A per-fold AUC is usually a little higher than a
  pooled one, because pooling also penalises small differences in score level between folds.
- **0.864** (tuning table) is the same per-fold mean on a single repeat.

Quote 0.862 with its interval. The deployed setting is the best of the 12 tuning settings. It was chosen on the
same 679 loans, so the headline may be optimistic by at most the spread of the grid (about 0.01), well inside its
interval.

**Seed stability.** Retraining the final engine with a different random seed changes 1.8-2.8% of decisions, all
near the approval threshold (PD correlation 0.998). That is ordinary model variance, and in use it is far smaller
than the effect of recalibration on local outcomes.

**Confusion matrix at the decision** (flag = refer or decline; average per run of 679 loans): of 73 defaulters,
version 2 flags 68.3 (recall 93.6%) and approves 4.7; of 606 repayers it approves 254 (41.9%; version 1: 33.3%).
Precision of a flag is 16% because "refer" is a request for a human look, not a prediction of default; accuracy
is not used (approving everyone would be 89% "accurate"). **Calibration:** predicted and actual default rates agree
decile by decile (calibration error 0.019). **Main drivers** (mean |SHAP|): lowest balance vs income, bills paid
regularly, share of income that is regular, average balance vs EMI, income regularity, month-to-month income change.

## 3. Why rules + AI, and not AI alone

Same engine, same real loans, three designs (nested cross-validation):

| Design | Approved | Defaulted among approved | Referred | Declined |
|---|---|---|---|---|
| Rules only (version 1) | 30.8% | 3.3% | 51.1% | 18.1% |
| AI only (no rules at all) | 45.8% | 1.8% | 45.2% | 9.0% |
| **Rules + AI (version 2)** | 38.1% | 1.8% | 39.7% | 22.2% |

- **On risk, the AI makes most rules redundant.** The loans the behaviour rules stop (EMIs above 55% of income,
  3+ bounces, 5+ lenders) defaulted at 41%; the AI alone would approve only 3.9% of them, and none of those
  defaulted. Of the extra approvals of "AI only", 90% are applicants the **legal age rule** blocks.
- **But the AI cannot see what the rules see.** On the forged and misleading sample statements, the AI alone
  would **approve 5 of 7**: an edited salary (risk 1.2%), edited balances recalculated (1.4%), money passed back
  and forth (3.1%), own transfers inflating income (1.8%), income lower than declared (0.9%). A forged statement
  looks exactly like a good borrower to a model trained on genuine ones. The rules (fonts, printed totals,
  round-tripping, declared-vs-verified income) catch all of them.
- **Some decisions must not be a model's to make**: legal eligibility (age, KYC), borrower protection
  (lending within repayment capacity: banks set FOIR limits in their credit policies, and RBI's microfinance
  framework of 2022 caps loan repayments at 50% of household income; here the 55% EMI cap and take-home rule), and applicants
  unlike the training data.

**Conclusion:** the strongest design is a layered one -- rules for **legality, fraud and borrower protection**, the
AI for **risk** within those limits, and a human for what is uncertain -- the policy + validated model + human
oversight structure that RBI's FREE-AI framework and draft model-risk guidance describe.

## 4. Collateral and co-applicants

Lending against gold, a fixed deposit or property is the other main way people without a credit history get
loans in India. FAIRScore treats it the way a bank's risk framework does:

- **The statement tells whether the borrower can repay (probability of default); the collateral tells how much the
  bank loses if they do not (loss given default).** Expected loss = PD x LGD x amount.
- Collateral **lowers the LGD** (indicative: gold 10%, fixed deposit 3%, property 25% on the covered part, against
  65% unsecured), the **interest rate** and the expected loss. If it covers at least half the loan, the AI may
  approve a case it is unsure about **as long as the expected loss stays within the bank's appetite for an
  unsecured loan**, never a confident "will default" and never above 15% PD.
- Collateral **never rescues an unaffordable loan**: the 55% EMI cap and take-home rule apply first (tested on 150
  random applications with random collateral).
- The loan may not exceed the collateral's value x its LTV cap. Gold follows the **RBI Lending Against Gold and
  Silver Collateral Directions, 2025**: 85% up to Rs 2.5 lakh, 80% up to Rs 5 lakh, 75% above -- and those
  Directions themselves require a **repayment-capacity assessment above Rs 2.5 lakh**, which is what FAIRScore
  provides for people without a credit history. Fixed-deposit (90%) and property (60%) caps are lender settings.
- **Evidence that the borrower's own stake matters** (L&T, 116,950 Indian new-to-credit vehicle loans): default
  rises from 14.3% when the loan is at most 60% of the price to 29.7% at 85-90%. FAIRScore already checks the own
  contribution for purchase loans.
- A **co-applicant's income** (spouse, parent) counts towards affordability once verified; it lets a student
  without income be assessed and lifts the student cap.

Example (R04 shop owner, Rs 3 lakh): unsecured the AI is unsure (risk 6.4%) and refers; with gold worth Rs 6 lakh
the loss if he defaults falls from 65% to 10%, the expected loss is well within appetite, and the AI approves at a
lower rate (12.5% vs 13.7%). R02 (risk 40%) is not approved even with property worth Rs 50 lakh: collateral never rescues a confident default.

## 5. Deep learning on raw transactions (research module)

`src/train_sequence_model.py`: a GRU encoder reads every transaction as an event (category + scale-free amount,
balance, timing; `src/seq_events.py`). It is **pre-trained without labels** (contrastive, CoLES-style: two slices of
the same account map close together) on 4,421 Berka account histories (918,343 transactions; for borrowers only the
history before the loan, so no repayment behaviour leaks), then used on each loan's pre-loan statement.

| Model (nested CV) | AUC |
|---|---|
| Pre-trained embedding only, logistic regression (no feature engineering, no labels in pre-training) | 0.831 |
| Same GRU trained from scratch on the labels | 0.831 |
| Tabular engine inputs (CatBoost) | 0.872 |
| Hybrid: tabular + embedding | 0.876 |

Raw transactions alone reach the accuracy of a hand-built logistic model, but add little to the expert-guided
features on this data. It is **not used in live decisions**: Berka transactions are mostly cash, Indian statements
mostly UPI (a measured domain gap). The pipeline is dataset-agnostic, ready for a modern transaction data set
(AlfaBattle 2.0: 963,811 clients, card transactions before a loan, default labels).

## 6. Modern transactions at scale: AlfaBattle 2.0 (research)

AlfaBattle 2.0 (Alfa-Bank, 2021) holds card-transaction histories before a loan and whether it defaulted. Using the
first 6 training files: **132,242 clients, 3,108 defaults (2.35%)**, about 240 transactions each; a fixed 70/30 client
split (39,673 test clients), and each method trained on 1%, 5%, 20% and 100% of the labelled training clients
(`src/alfabattle.py`, `src/alfabattle_seq.py`; pre-training uses only training clients and no labels).

| Labelled training clients | 1% (23 defaults) | 5% (108) | 20% (420) | 100% (2,176) |
|---|---|---|---|---|
| A. LightGBM on 187 designed aggregates | 0.595 | 0.653 | 0.699 | **0.748** (0.731-0.763) |
| B. GRU from scratch | 0.565 | 0.570 | 0.634 | 0.695 |
| C. Pre-trained GRU, frozen embedding + logistic | 0.588 | 0.635 | 0.662 | 0.678 |
| C2. Pre-trained GRU, fine-tuned | 0.571 | 0.620 | 0.682 | 0.709 |
| D. LightGBM on aggregates + pre-trained embedding | **0.601** | **0.660** | **0.707** | 0.745 |

Findings (they agree with the Berka study in section 5):
1. **Self-supervised pre-training makes neural models about 4-5x more label-efficient**: fine-tuned with 5% of
   labels it reaches what training from scratch reaches with ~20%, and with 20% almost what scratch reaches with 100%.
2. **Gradient boosting on designed behaviour features remains the strongest single model** on tabularised credit
   data -- which is why the deployed engine is a monotone gradient-boosted model.
3. **The hybrid is best when labelled loans are scarce** (+0.006 to +0.008 AUC at 1-20% of labels): the regime of a
   lender starting to serve new-to-credit customers, or a pilot with few recorded defaults.
Limits: one split; a small GRU trained for 2-3 epochs on a laptop CPU (competition RNNs trained far longer reach
higher AUC); category codes are anonymised, so the model cannot be applied to Indian statements -- research evidence
for the method, not a live component.

## 7. Detecting manipulated statements (research module)

`src/statement_anomaly.py`: an Isolation Forest trained only on genuine statements (1,388 real Berka windows + 440
Indian-style) on 14 scale-free features (Benford fit, round amounts, credit concentration, late inflows, end-balance
jump, in/out pairs, ...). Tested on held-out statements with injected manipulations whose balances still add up:

| Manipulation | AUC | Caught at 5% false alarms |
|---|---|---|
| Window dressing (large credit just before applying) | 0.990 | 97% |
| Round-tripping (money in and straight back out) | 0.989 | 98% |
| Inflated income | 0.707 | 9% |
| Deleted spending | 0.696 | 15% |

Not used live yet: the seven real Indian statements all looked unusual to it (they differ from both training
sources). Live, the rule checks for large recent credits and money passed back and forth, the font / printed-total
tamper checks and the balance reconciliation stay in force.

## 8. Version 3.0: keeping the engine safe in a new market

See [FAIRSCORE_3.md](FAIRSCORE_3.md): the starter-loan ladder, the learning loop (day-0 estimates, shift-aware
conformal prediction, Bayesian recalibration, champion/challenger), exploration against selection bias, the
bureau layer, Account-Aggregator-first provenance and the proxy-fairness audit, each with its study and results.

## 9. Limits

- 679 labelled loans, 73 defaults, from a Czech bank in the 1990s: intervals are wide and calibration is to Berka.
- No public data links Indian statements to repayment: the PD is "as Berka borrowers with this pattern behaved".
  The engine therefore decides only for applicant types it has seen, and a pilot would recalibrate it
  (`docs/PILOT_PLAN.md`).
- Profit figures use simple stated assumptions (interest margin on 55% average outstanding, LGD 65%).
- **Reject inference:** like every credit model, it learns only from loans that were granted. Version 3 measures
  the cost (src/exploration_study.py) and adds logged learning offers so a lender can correct it; the correction
  itself needs the lender's outcomes.
- Version 2's lower default rate among approvals is directionally consistent (94% of resamples) but not
  significant at 95% on 679 loans; its higher approval rate and fewer referrals are significant.
