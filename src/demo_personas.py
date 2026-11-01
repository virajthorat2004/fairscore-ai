"""
Demo applicants for the dashboard and for regression tests. Each persona has
an intake form and a 6-month bank statement generated deterministically
(fixed seed), so every demo run gives the same result.

    python src\\demo_personas.py      -> writes data/demo/<persona>_statement.csv
"""
import numpy as np
import pandas as pd

from config import PROJECT_ROOT

DEMO_DIR = PROJECT_ROOT / "data" / "demo"

PERSONAS = {
    "strong": {
        "title": "Strong applicant",
        "summary": "Government employee, 14 years in role, owns a home, steady salary, EMIs paid on time.",
        "applicant": {
            "age": 41, "monthly_income": 115000, "requested_amount": 600000,
            "requested_duration_months": 36, "total_savings": 600000, "dependents": 2,
            "employment_years": 14, "employment_type": "Government employee",
            "education": "Postgraduate / doctorate", "housing": "Own home", "owns_car": True,
            "purpose": "Home repairs / renovation", "active_credit_lines": 2, "property_loans": 1,
        },
    },
    "borderline": {
        "title": "Borderline applicant",
        "summary": "Private-sector salaried, 2 years in job, renting, one running EMI, modest savings.",
        "applicant": {
            "age": 29, "monthly_income": 42000, "requested_amount": 350000,
            "requested_duration_months": 30, "total_savings": 25000, "dependents": 1,
            "employment_years": 2.2, "employment_type": "Salaried (private sector)",
            "education": "Graduate", "housing": "Rented home", "owns_car": False,
            "purpose": "Used vehicle", "active_credit_lines": 1, "property_loans": 0,
        },
    },
    "weak": {
        "title": "Weak applicant",
        "summary": "Self-employed for 7 months, irregular income, frequent overdrafts, betting spends.",
        "applicant": {
            "age": 24, "monthly_income": 20000, "requested_amount": 160000,
            "requested_duration_months": 36, "total_savings": 2000, "dependents": 0,
            "employment_years": 0.6, "employment_type": "Self-employed / business owner",
            "education": "12th pass", "housing": "Rented home", "owns_car": False,
            "purpose": "Business", "active_credit_lines": 0, "property_loans": 0,
        },
    },
}


def _statement(persona, seed=7):
    rng = np.random.default_rng(seed)
    # the latest 6 complete months, so the demo statements are always 'recent'
    start = (pd.Timestamp.now().to_period("M") - 6).start_time
    days = pd.date_range(start, start + pd.DateOffset(months=6) - pd.Timedelta(days=1), freq="D")
    rows = []

    def add(d, desc, debit=None, credit=None):
        nonlocal balance
        balance += (credit or 0) - (debit or 0)
        rows.append([d.strftime("%Y-%m-%d"), desc, debit if debit else "",
                     credit if credit else "", round(balance, 2)])

    if persona == "strong":
        balance = 240000.0
        for d in days:
            if d.day == 1:
                add(d, "Salary credit - Govt", credit=115000)
            if d.day == 5:
                add(d, "Home loan EMI auto-debit", debit=24500)
            if d.day == 7:
                add(d, "SIP mutual fund", debit=10000)
            if d.day == 10:
                add(d, "Electricity bill", debit=int(rng.integers(1800, 3200)))
            if d.day == 12:
                add(d, "Credit card payment", debit=int(rng.integers(6000, 14000)))
            if rng.random() < 0.35:
                add(d, str(rng.choice(["Grocery store", "UPI transfer", "Pharmacy", "Restaurant"])),
                    debit=int(rng.integers(200, 2500)))

    elif persona == "borderline":
        balance = 18000.0
        for d in days:
            if d.day == 1:
                add(d, "Salary credit", credit=42000)
            if d.day == 3:
                add(d, "House rent", debit=11000)
            if d.day == 5:
                add(d, "EMI payment auto-debit", debit=6200)
            if d.day == 9:
                add(d, "Electricity bill", debit=int(rng.integers(700, 1500)))
            if rng.random() < 0.5:
                add(d, str(rng.choice(["Grocery store", "UPI transfer", "Restaurant", "Online shopping"])),
                    debit=int(rng.integers(150, 1600)))

    elif persona == "weak":
        balance = 3000.0
        for d in days:
            if rng.random() < 0.09:      # irregular gig income
                add(d, "UPI received - client", credit=int(rng.integers(3000, 9000)))
            if d.day == 4:
                add(d, "Room rent", debit=6500)
            if rng.random() < 0.10:
                add(d, str(rng.choice(["Online rummy", "Dream11 bet", "Casino app"])),
                    debit=int(rng.integers(300, 2500)))
            if rng.random() < 0.45:
                add(d, str(rng.choice(["UPI transfer", "Food delivery", "Mobile recharge", "Grocery store"])),
                    debit=int(rng.integers(100, 1200)))
            if balance < -2000 and rng.random() < 0.3:
                add(d, "Bank charges - negative balance", debit=590)
    else:
        raise ValueError(persona)

    return pd.DataFrame(rows, columns=["Date", "Description", "Debit", "Credit", "Balance"])


def statement_path(persona):
    return DEMO_DIR / f"{persona}_statement.csv"


def write_all():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for p in PERSONAS:
        df = _statement(p)
        df.to_csv(statement_path(p), index=False)
        print(f"Saved {statement_path(p).relative_to(PROJECT_ROOT)} -- {len(df)} transactions")


if __name__ == "__main__":
    write_all()
