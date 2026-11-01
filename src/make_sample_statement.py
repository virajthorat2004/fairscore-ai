import pandas as pd
import numpy as np
import os

np.random.seed(42)

start = pd.Timestamp("2026-01-01")
dates = pd.date_range(start, periods=180, freq="D")
rows = []
balance = 15000.0

for d in dates:
    # Salary credit on the 1st of each month
    if d.day == 1:
        balance += 42000
        rows.append([d, "Salary credit", "", 42000, balance])
    # Recurring EMI debit on the 5th
    if d.day == 5:
        balance -= 6200
        rows.append([d, "EMI payment auto-debit", 6200, "", balance])
    # Random daily spending
    if np.random.rand() < 0.5:
        spend = np.random.randint(100, 1500)
        balance -= spend
        rows.append([d, np.random.choice(["Grocery store", "UPI transfer", "Electricity bill", "Restaurant"]), spend, "", balance])

df = pd.DataFrame(rows, columns=["Date", "Description", "Debit", "Credit", "Balance"])
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/sample_statement.csv", index=False)
print(f"Saved data/raw/sample_statement.csv — {len(df)} transactions, {df['Date'].dt.to_period('M').nunique()} months")