# FAIRScore: independent verification report (update 28, 31 Oct 2026)

A full audit of the system as installed: tests of every kind, a hunt for errors and bias, and the fixes made.
Everything here can be re-run (section 9). Numbers are from the run on 31 Oct 2026.

## 1. Verdict in one paragraph

The software is correct, robust and fair on everything that can be tested without a lender's data.
- Every test suite passes.
- The live system computes exactly the inputs the AI was trained on.
- The AI's risk always moves in the sensible direction.
- Names, gender, community and religion in a statement do not change any decision.
- Malicious files cannot crash it.

The audit found and fixed **four real problems**:
- a surname bias;
- a gaming route (money parked in the account just before applying);
- a crash on hostile files;
- an API that was open to other computers by default.

It also found the most important limitation for real use. On **real Indian statements the AI is outside the
experience of its Czech training data** (balances relative to income are far lower). It must not be trusted to
decline anyone until it is validated on Indian repayment outcomes, and it is now prevented from doing so
(section 6).

## 2. Test suites (all pass)

| Suite | What it covers | Result |
|---|---|---|
| tests/test_backend.py | units, policy, AI engine, collateral, bureau, starter ladder, learning loop, audit log, API end-to-end | 49/49 pass |
| tests/test_live_scoring.py | live scoring units | 14/14 pass |
| tests/test_invariants.py | 150 random applications x 21 statements: rules and AI never contradict each other | every invariant holds |
| tests/sample_cases.py | 21 sample statements (edited, forged, protected, odd layouts) read exactly and decided as expected | 21/21 pass |
| tests/realistic_statements.py | 11 realistic statements (salaried, gig, kirana, farmer, student...) | 11/11 pass (every transaction and balance reconciled) |
| tests/benchmark_layouts.py | 200 random bank layouts never seen before | 99.0% read exactly; the 2 misreads are caught by the balance check and referred (none silent) |
| **tests/test_audit.py (new)** | monotonicity, training-serving skew, counterfactual fairness, metamorphic relations, hostile input, governance | 6/6 pass (all 11 realistic statements) |

**On the project PC itself** (Windows, Python 3.14.0, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, catboost 1.2.10,
streamlit 1.61.1; the project's own venv), `tools\verify_on_pc.bat` ran the six suites above on 31 Oct 2026.
**All passed:**
- backend 49/49;
- live scoring 14/14;
- every invariant;
- samples 21/21;
- realistic statements 11/11;
- audit 6/6 (quick mode: 3-4 of the realistic statements per check; FULL=1 runs all 11).

The same code therefore behaves identically on the development container (Linux, Python 3.13) and on the PC.

## 3. Is the model right?

| Check | Method | Result |
|---|---|---|
| Leakage from the future | the statement used for each Berka loan ends the day before the loan date | none |
| Leakage inside evaluation | the whole recipe (model, calibration, conformal scores, threshold, unusual-applicant detector) is re-fitted in every outer fold | nested, correct |
| Training-serving skew | live app vs training rows on 76 Berka loans, all 24 inputs | identical (0 differences) |
| Monotonicity | each of 24 inputs swept over its range for 100 applicants | 0 violations; every direction makes lending sense |
| Calibration by segment | predicted vs actual default rate by employment type, gender, age group, term, loan size | the actual rate is inside the interval of the predicted rate in all 14 segments |
| Seed stability | final engine retrained with 3 seeds | PD correlation 0.998; 1.8-2.8% of decisions change, all near the threshold |
| Hyperparameter optimism | deployed setting is the best of 12, chosen on the same data | at most about 0.01 AUC (spread of the grid) |
| Three AUC figures | 0.862 pooled nested (headline), 0.870 mean per fold, 0.864 tuning | explained in AI_ENGINE.md; quote 0.862 (0.812-0.910) |

## 4. Is it biased?

**Counterfactual test: the same statement, only names or payees changed.** Run on all 11 realistic statements.
- Every person's name is replaced one-to-one by names from one group: Hindu, Muslim, Christian, Sikh,
  Dalit-associated surnames, tribal / North-East, women only, and surnames that look like lenders.
- Separately, shop payments go to a temple, a mosque, a church or a gurudwara instead.
- Decision, AI risk, FAIRScore, eligible amount and starter offer are compared with the original.

Result: **132 of 132 variants give an identical decision, AI risk, FAIRScore, eligible amount and starter offer**.

**Bug found and fixed.** A person-to-person transfer to someone surnamed **Muthoot** or **Manappuram** (also gold-loan
companies) was read as "borrowing from a lender". That lowered the FAIRScore by 37-43 points. The patterns now need
a company marker (finance, gold, loan...), as the Bajaj pattern already did. Unknown personal fields sent to the API
(e.g. caste, religion) are ignored and never stored.

**Proxy test (can the 24 inputs recover a protected attribute?).** Two probes, a linear one and a nonlinear
boosted one, are each compared with 200 label-shuffled runs. A Holm correction is applied over the 8 tests.

| Attribute | Linear probe AUC (chance) | Boosted probe AUC (chance) | Smallest p (Holm-corrected) | Reading |
|---|---|---|---|---|
| Region (Moravia vs Bohemia) | 0.491 (0.499) | 0.451 (0.503) | 0.55 (1.00) | no detectable trace |
| District income (poorer third) | 0.528 (0.495) | 0.514 (0.500) | 0.16 (1.00) | no detectable trace |
| Gender (female) | 0.478 (0.501) | 0.556 (0.506) | 0.05 (0.40) | no trace after correction; keep monitoring |
| Age under 30 | 0.426 (0.498) | 0.438 (0.497) | 0.96 (1.00) | no detectable trace |

"Chance" is the mean AUC of the same probe on shuffled labels. A cross-validated AUC with no signal sits a little
below 0.5, so 0.5 is the wrong reference. The p-values are one-sided: can the inputs recover the attribute *better*
than chance?
- The only raw signal is a weak nonlinear trace of gender, not significant after correcting for 8 tests.
- It does not turn into unfair outcomes: women repayers are approved slightly more often than men.

**Outcome parity on real loans.**
- Repayers approved: women 43.3%, men 40.5%.
- Every district-income group passes the four-fifths rule.
- Prague sits at 0.79 of the best-treated region (overlapping intervals; monitored).

## 5. Robustness, gaming and security

**Metamorphic tests** (15 statements). Every relation must hold:

| Relation | Result |
|---|---|
| 3 more EMI bounces never improve the decision or lower the risk | 15/15 |
| The same statement exported newest-first gives an identical result | 15/15 |
| Twice the money everywhere (statement, loan, price, savings) gives the same decision | 15/15 |
| A 25% higher salary never worsens the decision or raises the risk | 8/8 (statements with a salary) |
| Borrowed money parked in the account just before applying never helps | 15/15 (and the AI risk does not fall either) |

**Gaming route found and closed (window dressing).** A one-off deposit of three months' inflow, six days before
the statement end, lifted a case the AI was confident would default (risk 45%) to a risk of 22%.
- The rules caught the deposit, so the case was referred, not approved.
- But the deposit was counted as income, so affordability looked better too.

Now, when a large unexplained recent credit is still in the account, the **whole assessment runs twice**, with and
without that money, and the worse outcome stands. The officer sees the credit, the source-of-funds caution and why
the figures exclude it.

**Real outcomes support the rule.** On the 679 Berka loans, the 26 borrowers (3.8%) whose statements show this
pattern defaulted at **26.9%, against 10.1%** for the rest: odds ratio 3.3, Fisher exact p = 0.015. Version 2 had
auto-approved 5 such cases (pooled over 3 repeats), 3 of which defaulted.

**Hostile input through the API: 42 of 42 checks pass.** Covered:
- empty, text, image, truncated, corrupted and 16 MB files;
- fake Account Aggregator files;
- impossible ages, amounts and terms, NaN and Infinity;
- script injection;
- duplicate and too many statements.

Every one gets a clear 4xx message, never a crash. Two security fixes:
- **Crash fixed.** A deeply nested JSON / XML file crashed the server (500). XML with a DTD (billion-laughs or XXE
  attacks) is now refused outright, and absurd nesting is rejected with a clear message.
- **Closed by default.** Without `FAIRSCORE_API_KEY`, the API used to accept calls from any computer, and
  assessments hold personal data. It now answers only calls from the same computer until a key is set; `/health`
  stays open.

**Concurrency.** 24 simultaneous submissions: all succeeded, 24 unique references, all 24 stored in the audit log.

**Speed.** Median 0.3 s per full assessment (including SHAP, recourse, age-flip test and starter offer); the largest
statement (1,828 transactions) takes about 2 s.

**Static analysis** (ruff, pyflakes rules): no undefined names or real bugs; two loop-variable warnings checked and
harmless.

## 6. Real Indian statements: the finding that matters most

Seven real statements from Indian banks were run through the full system privately (no names, numbers or
transactions leave the machine).
- **Reading: every one read with 100% of rows reconciled and 0 balance breaks.**
- The AI, however, sees them as unlike its training data:
  - on these statements, 94-100% of training borrowers had fewer low-balance days;
  - average balance relative to the EMI sits in the bottom 0-5% of training borrowers;
  - typicality is 0.00-0.15 (the AI's own "how familiar is this" score).
- It would call 5 of 7 likely defaulters (risk 21-50%), mostly because of low balances, which are normal for UPI
  spend-through accounts.

Whether these people are risky cannot be known without Indian repayment outcomes. A Czech-trained model must not
turn Indian low-balance UPI users away on Czech norms: that would be the exclusion the project exists to prevent.

**Fix (governance rule, `[learning] ai_decline_needs_local_validation = true`).** Until the model-risk team
approves an adaptation fitted on local repayment outcomes:
- the AI may recommend approval, but never declines on its own;
- a confident "will default" becomes a referral with the AI's warning and reasons;
- hard-stop rules (bounces, gambling, unaffordable EMI, forged statements, legal age) still decline as before.

On Berka (the source market, where the AI is validated) the evaluated results are unchanged.

## 7. Is it publication level?

Yes, as an applied-ML / systems paper with honest limits. In its favour:
- **Evaluation:** nested evaluation, intervals and significance tests (DeLong, bootstrap, Fisher, permutation), 9
  baselines, ablations (rules vs AI-only, age, sequence model).
- **Uncertainty:** conformal guarantees checked empirically, including under 8 kinds of population shift.
- **Bias:** fairness and proxy audits with multiple-testing correction.
- **New mechanisms:** an exploration study for selection bias, and the window-dressing rule validated on real
  outcomes.
- **Reproducibility:** public data; fixed seeds; every number re-runs from code.

What a reviewer will still say, and what to write:
1. **No Indian outcome data.** The AI is trained and validated on 679 Czech loans from the 1990s. State it as the
   main threat to validity. Section 6 is evidence that it matters, and the learning loop is the design response.
2. **Small sample:** 73 defaults, wide intervals (AUC 0.81-0.91). Quote intervals everywhere.
3. **Label definition:** running loans without arrears count as good, and some may still default.
4. **Researcher degrees of freedom:** the expert scorecard and some rules were refined after seeing Berka results.
   Nested CV cannot remove this; only external validation can.
5. **Synthetic Indian test statements** prove the software works; they are not evidence of predictive accuracy.

Suitable targets: an applied venue (an IEEE / Springer conference on AI in finance, or a journal such as *Expert
Systems with Applications* if extended with Indian outcomes), or arXiv first.

## 8. Can it be used in the real world?

**As decision support for credit officers in a supervised pilot: yes.** The engineering is ready:
- statement reading;
- rules;
- audit trail;
- API with access control;
- fairness monitoring;
- drift monitoring and recalibration machinery.

**As an autonomous lender: not yet, and it now says so itself.** First:
1. a pilot that collects Indian repayment outcomes (docs/PILOT_PLAN.md);
2. recalibration and validation on them by the learning loop;
3. independent model validation per RBI's model-risk expectations;
4. DPDP-compliant consent and retention in the lender's systems;
5. live Account Aggregator integration through a licensed FIU.

## 9. Re-run

    python tests/test_backend.py        python tests/test_invariants.py     python tests/sample_cases.py
    python tests/realistic_statements.py python tests/test_live_scoring.py  python tests/benchmark_layouts.py
    python tests/test_audit.py           (FULL=1 for all realistic statements)
    python src/proxy_fairness.py         (proxy and outcome-parity audit, about 40 minutes)
