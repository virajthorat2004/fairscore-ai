"""
Indian loan outcomes: the L&T Financial Services vehicle-loan data set (Kaggle, "L&T Vehicle Loan
Default Prediction"): 233,154 two-wheeler / vehicle loans disbursed in India between August and
October 2018, with the borrower's application details, the credit-bureau record where one exists,
and whether the borrower defaulted on the FIRST EMI (loan_default = 1).

About half the borrowers (116,950) have "No Bureau History Available": they are exactly the
new-to-credit population FAIRScore is built for, with real Indian repayment outcomes. The data has
no bank statements, so it cannot test the statement reader or the cash-flow model; it tests what
an Indian lender can learn about a new-to-credit applicant WITHOUT a statement, how fairly, and
which application facts matter (the loan-to-value ratio above all).

Data: download train.csv from Kaggle and copy it to data/raw/ltfs/train.csv.
"""
import numpy as np
import pandas as pd

from config import PROJECT_ROOT

LTFS_PATH = PROJECT_ROOT / "data" / "raw" / "ltfs" / "train.csv"
NTC_LABEL = "No Bureau History Available"
KYC = {"Aadhar_flag": "aadhaar", "PAN_flag": "pan", "VoterID_flag": "voter_id", "Driving_flag": "driving_licence",
       "Passport_flag": "passport"}
# Lender-operational identifiers: they carry dealer / branch / area effects of THIS lender only,
# do not exist for another lender's applicant, and can stand in for where someone lives.
LENDER_SPECIFIC = ["branch_id", "supplier_id", "manufacturer_id", "Employee_code_ID", "Current_pincode_ID"]


def _years(text):
    """'1yrs 11mon' -> 1.92"""
    m = pd.Series(text).astype(str).str.extract(r"(\d+)yrs\s*(\d+)mon").astype(float)
    return (m[0] + m[1] / 12).values


def load(path=LTFS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"L&T data not found at {path}: download train.csv from Kaggle "
                                f"('L&T Vehicle Loan Default Prediction') and copy it there.")
    d = pd.read_csv(path)
    dob = pd.to_datetime(d["Date.of.Birth"], format="%d-%m-%y", errors="coerce")
    dob = dob.where(dob.dt.year <= 2000, dob - pd.DateOffset(years=100))      # '01-01-84' is 1984, not 2084
    d["disbursal_date"] = pd.to_datetime(d["DisbursalDate"], format="%d-%m-%y")
    d["age"] = (d["disbursal_date"] - dob).dt.days / 365.25
    d["employment"] = d["Employment.Type"].fillna("Not stated")
    d["new_to_credit"] = d["PERFORM_CNS.SCORE.DESCRIPTION"].eq(NTC_LABEL)
    d["bureau_band"] = d["PERFORM_CNS.SCORE.DESCRIPTION"].str.extract(r"^([A-M])-")[0]
    d["bureau_scored"] = d["bureau_band"].notna()
    for col, name in KYC.items():
        d[name] = d[col].astype(int)
    d["kyc_documents"] = d[list(KYC)].sum(axis=1)
    d["own_contribution"] = 1 - d["ltv"] / 100                 # share of the vehicle price paid by the borrower
    d["credit_history_years"] = _years(d["CREDIT.HISTORY.LENGTH"])
    d["month"] = d["disbursal_date"].dt.to_period("M").astype(str)
    d["bad"] = d["loan_default"].astype(int)
    return d


def age_group(a):
    return "under 25" if a < 25 else "25-34" if a < 35 else "35-44" if a < 45 else "45-54" if a < 55 else "55 and over"
