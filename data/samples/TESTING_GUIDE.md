# FAIRScore AI: sample PDF statements and how to test them

All statements are from a fictional "Demo Bank", are marked SAMPLE on every page, and use realistic
UPI / NEFT / ACH / NACH narrations. They cover twelve layouts modelled on how Indian banks really
format statements:

| Layout | Modelled on | What makes it hard |
|---|---|---|
| `grid` | ruled table | header repeated on each page |
| `plain` | unruled, HDFC-style | two-line headings, wrapped narrations, header only on page 1 |
| `signed` | one Amount column | "Dr"/"Cr" suffixes, dates like "05 Mar 2026" |
| `inr_unruled` | unruled, PSU-bank style | "INR 1,500.00", "-" for empty cells, narrations over 3-5 lines, opening/total lines |
| `paren_drcr` | ruled, one Amount column | "20.00(Dr)" / "239.80(Cr)", transaction-ID column |
| `summary_first` | SBI-style | a summary page first, headings the reader cannot recognise, two date columns |
| `serial_two_dates` | ICICI-style | serial-number column, value + transaction dates, "0.00" in empty cells |
| `newest_first` | Kotak-style | newest transaction first, "01-Apr-2026" dates, balance with "(Cr)" |
| `centred` | Bank of Baroda | date and amounts printed BETWEEN the two lines of each narration, balance "117,648.61 Cr" as two words, "Sep 01, 2026" period, page footers |
| `serial_bare_period` | Kotak (current) | "#" serial column, period printed alone ("01 Mar 2026 - 31 Aug 2026"), opening-balance row |
| `axis_branch` | Axis | branch-code column after the balance, opening-balance row |
| `trailing_value_date` | PNB / Canara | transaction number first, value date last, balance "12,345.00 Cr.", headings printed touching each other |
| `no_space_glyphs` | HDFC | no space characters in the PDF (words separated by small gaps), "From : .. To : ..", summary printed as a table |

The reader does not rely on column headings: it finds the date, narration and amount columns from
their content, and decides which column is debit, credit and balance by checking that every row
satisfies *balance = previous balance + credit - debit*. It was also verified on seven real statements from
six banks (Indian Bank, Union Bank in two different formats, SBI, Kotak, Bank of Baroda, HDFC; ruled and
unruled, 1 to 33 pages): every transaction read, every balance reconciled, and the opening / closing
balances and totals matched the figures the banks printed.

## Automatic check
From `D:\fairscore-ai` with the venv active, `python tools\final_check.py` runs everything below (plus the
invariant tests and the layout benchmark) and prints ALL CHECKS PASSED. Or one by one:

    python tests\sample_cases.py        -> 21/21 sample statements read exactly and behave as expected
    python tests\test_live_scoring.py   -> all passed
    python tests\test_backend.py        -> all passed (the API test is skipped unless requirements-api.txt is installed)

`sample_cases.py` also confirms each PDF is read **exactly**: same transactions and the same
income, spending, balances and flags as the source data (kept in `data/samples/csv/` for this check only).

## Manual check in the dashboard
Run `streamlit run app\dashboard.py`. For each case: upload the PDF, enter the password if one is
listed, fill the essential fields from the profile, add anything in "Also enter", click **Assess application**.

| Profile | Age | Employment type | Years in work | Housing | Dependents | Savings (₹) | Owns vehicle | Amount (₹) | Term | Purpose |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 41 | Government employee | 14 | Own home | 2 | 600000 | Yes | 600000 | 36 | Home repairs / renovation |
| **B** | 29 | Salaried (private sector) | 2 | Rented home | 1 | 25000 | No | 350000 | 30 | Used vehicle |
| **C** | 24 | Self-employed / business owner | 0.5 | Rented home | 0 | 2000 | No | 160000 | 36 | Business |

| # | PDF | Profile | Also enter | Expected | What it proves |
|---|-----|---------|------------|----------|----------------|
| 1 | 01_strong_salaried.pdf | A | – | **Approve** | Everything passes |
| 2 | 02_borderline_salaried.pdf | B | – | **Refer** (EMIs 48% of income) | Affordability limit; shows the affordable amount |
| 3 | 03_weak_overdraft_betting.pdf | C | – | **Decline** | Overdrafts, betting, irregular income |
| 4 | 04_unaffordable_request.pdf | A | Amount 2500000, Term 24 | **Decline** (Affordability) | A good customer can still ask for too much |
| 5 | 05_hidden_emi.pdf | B | Optional: Active loans + cards = 0 | **Refer** (Declaration) | Catches an undeclared loan |
| 6 | 06_income_lower_than_declared.pdf | B | Amount 150000; Optional: Declared income 42000 | **Refer** (Declared income) | Statement shows only ₹28,000 |
| 7 | 07_only_3_months.pdf | A | – | **Refer** (Statement history) | Needs 6 months |
| 8 | 08_emi_bounces.pdf | B | – | **Decline** (Account conduct) | Bounced EMIs; their refunds are not counted as income |
| 9 | 09_password_protected.pdf | A | Password **DEMO1234** | **Approve** | Opens protected PDFs (try without / wrong password too) |
| 10 | 10_amount_dr_cr_layout.pdf | B | – | **Refer** (Affordability) | Reads Amount + Dr/Cr layouts, dates like "05 Mar 2026" |
| 11 | 11_edited_salary.pdf | A | – | **Refer** (Statement integrity) | An edited salary breaks the running balance |
| 12 | 12_own_transfers_inflate_income.pdf | B | Amount 500000 | **Decline** (Affordability) | ₹40,000/month from own account is not income (verified income ₹32,000) |
| 13 | 13_salary_without_label.pdf | B | – | **Approve** | Salary credited as "NEFT CR-ACME TECHNOLOGIES" is still recognised |
| 14 | 14_centred_rows_layout.pdf | B | – | **Approve** | Bank-of-Baroda layout: each narration stays with its own transaction |
| 15 | 15_funded_from_own_accounts.pdf | B | – | **Refer** (Income source) | Money in comes only from the holder's own other accounts: not income |
| 16 | 16_branch_code_column.pdf | A | – | **Approve** | Axis layout with a branch-code column after the balance |
| 17 | 17_value_date_at_end.pdf | C | Years in work 3, Amount 120000, Term 24 | **Approve** (AI confident, risk ~3.5%) | PNB layout read exactly; approved subject to ITR / GST returns |
| 18 | 18_one_month_only.pdf | B | – | **Refer** (Statement history) | One month is a missing document, so refer (not decline) |
| 19 | 19_edited_balances_recalculated.pdf | A | – | **Refer** (Signs of editing) | A careful forgery (balances recalculated) caught by font and printed totals |
| 20 | 20_money_passing_through.pdf | C | Years in work 3, Amount 150000, Term 24 | **Refer** (Money passing through) | ₹85,000 in and ₹83,000 back to the same party each month is not income |
| 21 | 21_no_space_characters_layout.pdf | B | – | **Approve** | HDFC-style PDF with no space characters; salary naming the holder is income, the holder's own UPI transfer is not |

## Realistic statements (`data/samples/realistic/`)
The cases above are deliberately clean, one behaviour each. These eleven are as messy as real statements:
50-300 transactions a month, mostly small UPI payments, salaries that move between dates and are often not
labelled "salary", autopay subscriptions, cashback, family transfers both ways, ATM cash, card bills, charges and
interest, each printed in the layout style of a real Indian bank. They are **synthetic** (fictional bank, people,
employers and reference numbers, every page marked SAMPLE): present them as "synthetic statements modelled on real
bank layouts". `python tests\realistic_statements.py` regenerates them (latest 6 months) and checks every one.

Form inputs: Age, Employment type, Years in work, Housing, Dependents, Savings, Owns vehicle, Amount / Term,
Purpose (and, where given, Optional details -> "Price of the item being bought").

| # | PDF | Layout style | Applicant | Form inputs | Expected |
|---|---|---|---|---|---|
| R01 | R01_strong_it_professional.pdf | HDFC | IT professional, salary credited on the 1st by NEFT, rent, SIP, credit-card bill, 2-3 UPI payments a day; spends about three-quarters of pay and saves the rest | 31, Salaried (private sector), 5 yrs, Rented home, 1 dep., savings 400,000, vehicle No, ₹500,000 / 36 m, New vehicle; price 850,000 | **Approve** |
| R02 | R02_irregular_salary_small_firm.pdf | SBI / PSU | Salary from a small firm, unlabelled, paid between the 1st and the 8th; the latest month was paid only in part | 27, Salaried (private sector), 2 yrs, Rented home, 2 dep., savings 8,000, vehicle No, ₹120,000 / 24 m, Home appliances | **Refer** with an AI warning (risk ~41%; the AI does not decline on its own until validated on local outcomes) |
| R03 | R03_delivery_partner_weekly_payouts.pdf | Kotak | Gig delivery partner: weekly platform payouts that vary, tips, fuel almost every day, a bike EMI | 24, Freelancer / gig worker, 2.5 yrs, Rented home, 1 dep., savings 3,000, vehicle Yes, ₹60,000 / 18 m, Used vehicle | **Refer** |
| R04 | R04_kirana_shop_owner.pdf | Bank of Baroda | Grocery shop: dozens of customer UPI payments a day, weekly cash deposits, supplier payments, shop rent | 44, Self-employed / business owner, 12 yrs, Own home, 3 dep., savings 150,000, vehicle Yes, ₹300,000 / 36 m, Business | **Refer** (AI unsure: risk ~6%) + **starter loan ₹50,000 / 12 m approved** (3.0) |
| R05 | R05_salaried_with_stacked_loans.pdf | PNB / Canara | Steady salary, but two NBFC EMIs, a loan app, new app loans disbursed during the period and a card bill | 33, Salaried (private sector), 4 yrs, Rented home, 2 dep., savings 20,000, vehicle No, ₹250,000 / 36 m, Other | **Decline** |
| R06 | R06_stressed_account_bounces.pdf | Kotak (older format) | Low salary on irregular dates, two EMIs that bounce when the balance runs dry, minimum-balance charges, betting app | 29, Salaried (private sector), 3 yrs, Rented home, 2 dep., savings 0, vehicle No, ₹100,000 / 24 m, Other | **Decline** |
| R07 | R07_first_job_thin_file.pdf | Axis | First job: salary only in the last 4 months, pocket money from a parent before that, PG rent | 22, Salaried (private sector), 0.4 yrs, Rented home, 0 dep., savings 15,000, vehicle No, ₹50,000 / 12 m, Electronics (TV, phone, laptop); price 62,000 | **Refer** + **starter loan ₹24,000 / 12 m** (3.0) |
| R08 | R08_govt_teacher_home_loan.pdf | HDFC | Government salary, home-loan EMI, LIC, child's college fees, money passed back and forth with a relative | 46, Government employee, 18 yrs, Own home, 2 dep., savings 300,000, vehicle Yes, ₹200,000 / 24 m, Education | **Approve** |
| R09 | R09_freelancer_lumpy_income.pdf | SBI / PSU | Freelance designer: a few large client payments at irregular times, advance tax, software subscription | 35, Freelancer / gig worker, 6 yrs, Rented home, 1 dep., savings 250,000, vehicle No, ₹400,000 / 36 m, Business | **Refer** + starter ₹50,000 **suggested** (AI advisory; officer confirms) |
| R10 | R10_farmer_crops_and_dairy.pdf | SBI / PSU (rural branch) | Farmer: crop sales through APMC traders after harvest, milk bills from the dairy co-operative every 10 days, PM-KISAN credit, fertiliser and labour cash in the sowing months | 41, Farmer / agriculture, 15 yrs, Own home, 3 dep., savings 60,000, vehicle Yes, ₹200,000 / 24 m, Used vehicle | **Approve** |
| R11 | R11_student_with_stipend.pdf | Kotak | Student: internship stipend every month, small freelance jobs, pocket money from a parent (not counted) | 21, Student, 1 yrs, Living with family (no rent), 0 dep., savings 10,000, vehicle No, ₹60,000 / 12 m, Electronics (TV, phone, laptop); price 72,000 | **Approve** |

What each one shows: R01 a strong salaried customer approved through hundreds of everyday transactions (and the
loan-to-value check passing); R02 an unlabelled, shifting salary with a short latest month and a balance run to zero -> the AI is confident of default (risk about 41%) and refers it with a warning: until it is validated on local repayment outcomes it never declines on its own; R03 and R09
lumpy self-employed income -> refer for ITR / GST; R04 (the AI is not sure, risk about 6%, so an officer decides) a shop's hundreds of small collections recognised as
business income; R05 five lenders and loan apps in six months -> decline (loan stacking); R06 bounced EMIs,
minimum-balance charges and betting -> decline; R07 a new job (salary only in the last 4 months) -> refer, not
decline; R08 a government salary with a home loan and money passed back and forth with a relative (not income);
R10 a farmer with harvest sales and dairy income -> approve (seasonal income is normal for farming; land records
asked for); R11 a student with an internship stipend -> approve on her own income (pocket money is not counted).

Student / unemployed rule: choose Employment type "Student" with R11 (stipend) -> Approve; with R01 (a ₹1.18 lakh salary every month) -> Refer: "declared a student, but the statement shows a full-time salary" (confirm student status or re-assess as salaried). A student asking for more than 8x verified income, over ₹5 lakh or over 36 months -> Refer with a counter-offer inside those limits (config/policy.toml); with
15_funded_from_own_accounts.pdf (no income of their own) -> Refer for a co-applicant. "Unemployed" with R01 ->
Refer ("declared unemployed but income verified: confirm it"); with case 15 -> Decline.

Applicant profile moves the score: R01, ₹600,000 / 36 months, same statement each time (about 10% of the
points come from the form; the statement still decides most of the score, by design):

| Profile | FAIRScore |
|---|---|
| Student, 24, living with family, no dependents, savings 20,000, no vehicle | 795 |
| Salaried (private), living with family, same otherwise | 809 |
| Salaried (private), own home, savings 20,000 | 819 |
| Salaried (private), own home, savings 300,000, owns a vehicle | 834 |
| Government employee, own home, savings 300,000, owns a vehicle | 844 |
| Salaried (private), rented home, 4 dependents, no savings | 797 |

Every result also shows **Confidence** (High / Medium / Low) and a **Fairness check** line: the same
application re-decided at other ages must get the same recommendation (age is used only for eligibility).

Other checks:
- **Eligibility (the only use of age):** case 1 with Age 58 and Term 36 -> **Decline** (loan would end at 61,
  limit 60) with the suggestion of a 24-month term; Age 58, Amount 300000, Term 24 -> **Approve**.
- **Own contribution (loan-to-value):** case 1 with Purpose "New vehicle" and Optional "Price of the item" =
  1000000 -> **Approve** ("Own contribution: OK", loan 60% of the price); price 650000 -> **Refer** with a
  counter-offer of ₹5,52,000 (85% of the price); price 560000 -> **Decline** (the loan is larger than the price).
- **Officer decision:** after case 2, open History -> "Record a credit officer's decision", choose the
  reference, Approve or Decline, give a reason, Save. The row shows the officer's decision.
- **SHAP vs LIME:** after any assessment, open "How it decides" -> "Compare SHAP and LIME".
- **Nothing filled:** Assess → error listing the statement and the 10 essential fields.
- **Case 9 without the password / with a wrong one:** a clear message asking for the right password.
- **What-if:** after case 2, open "Try a different amount or term", enter 319000 (the counter-offer shown):
  it becomes **Approve**.
- **Account Aggregator data:** upload `data/samples/aa/01_strong_salaried.json` instead of the PDF (same
  profile A) -> the same result as case 1; `aa/15_funded_from_own_accounts.json` -> the same as case 15.
- **Two accounts together:** upload `14_centred_rows_layout.pdf` and `16_branch_code_column.pdf` together with
  profile A -> the checks show "Same account holder: 2 accounts assessed together" and income, EMIs and balances
  are combined over the common period. Uploading the same account twice (e.g. case 15 as PDF and as
  `aa/15_...json`) is refused instead of being counted twice.
- **Name check:** case 15 with Optional details -> "Name as on the bank account" = `Anaya V Desai` -> "Statement
  belongs to the applicant: OK"; any other name -> REVIEW. The name is not stored.
- **Credit appraisal memo and applicant letter:** the two download buttons at the end of any result.
- **Recording outcomes:** History -> "Record how a loan turned out" for an approved case.
- **Model monitoring:** History -> "Model monitoring" shows how far applicants' statement patterns are from the
  cash-flow model's training data (PSI).
- **History tab:** every assessment is listed with its reference number, the policy version, and the
  outcomes by age group (fairness monitor).
- **Your own bank's statements:** `python tools\check_statement.py <pdf or folder>` shows whether a format is
  read exactly (see docs/BACKEND.md).

Sample dates: the sample statements always cover the latest 6 complete months (statements older than 45 days
are referred, as a bank would ask for the latest months). Run `python tests\sample_cases.py` to refresh them.

## Version 3.0 in the samples
- **Starter-loan ladder:** R04, R07 and R09 show a starter offer under the AI card (the requested loan stays
  referred). R03 (the AI is confident of default) and R06 (a hard stop) get none.
- **Bureau layer:** take sample 01 and fill "6 · Credit bureau report": *Established file* + score 620 -> Decline;
  *Thin file* + score 640 -> Refer; *Established file* + score 780 + EMIs on the report ₹8,000 higher than the
  statement shows -> Refer, with the bureau EMIs used for affordability.

## What the system accepts
- A **text PDF e-statement** downloaded from net banking (ruled or plain layouts, any number of pages,
  password-protected or not), or **Account Aggregator** data (RBI AA framework, JSON / XML). Up to 4 accounts
  at once. Scanned images / photos are rejected with a clear message.
- Columns are recognised by name: Date / Txn Date / Value Date; Narration / Description / Particulars;
  Withdrawal / Debit + Deposit / Credit, or one Amount with Dr/Cr; Balance / Closing Balance.


## Version 2 (AI engine) in the samples
The AI decides for salaried, pension and business applicants; for students, farmers and gig workers it advises and the
lending policy decides. Each result shows who decided ("Decided by the AI engine" / "a lending rule" / "AI advisory"),
the AI's probability of default, its certainty, the SHAP reasons and what would change the decision.
Changed by version 2: **17_value_date_at_end -> Approve** (version 1 referred it only because foreign-data models
disagreed; the AI, trained on real outcomes, is confident at about 3.5% risk), **R02 -> Refer with an AI warning** (update 28: no AI-only declines before local validation), **R04 -> Refer**.
