#!/usr/bin/env python3
"""
gen_parquet.py — generate a sample transactions Parquet file.

Usage:
    python3 gen_parquet.py [rows] [output.parquet]
"""
import sys
import random
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
OUT = sys.argv[2] if len(sys.argv) > 2 else "transactions.parquet"

random.seed(1337)

CHANNELS = ["ach", "wire", "card_present", "card_not_present", "atm", "check", "zelle"]
MERCHANTS = ["Northgate Fuel", "Copperline Market", "Allegheny Hardware", "Tri-State Auto",
             "Blue Ridge Pharmacy", "Steel City Electronics", "Riverside Grocers",
             "Monongahela Wine", "Duquesne Dry Goods", "Ohio Valley Freight"]
MCC = ["5541", "5411", "5200", "5533", "5912", "5732", "5411", "5921", "5399", "4214"]
CITIES = [("Pittsburgh", "PA", 40.4406, -79.9959), ("Cleveland", "OH", 41.4993, -81.6944),
          ("Columbus", "OH", 39.9612, -82.9988), ("Buffalo", "NY", 42.8864, -78.8784),
          ("Baltimore", "MD", 39.2904, -76.6122), ("Newark", "NJ", 40.7357, -74.1724),
          ("Charlotte", "NC", 35.2271, -80.8431), ("Miami", "FL", 25.7617, -80.1918)]
STATUS = ["settled", "settled", "settled", "settled", "pending", "reversed", "declined"]

accounts = [f"ACCT-{100000 + i}" for i in range(2_000)]
customers = {a: f"CUST-{200000 + i}" for i, a in enumerate(accounts)}
# a small set of accounts that behave badly, for detection exercises
suspect = set(random.sample(accounts, 40))

start = datetime(2026, 1, 1, tzinfo=timezone.utc)
span = int(timedelta(days=180).total_seconds())

rows = {k: [] for k in [
    "transaction_id", "event_time", "account_id", "customer_id", "channel",
    "direction", "amount", "currency", "merchant_name", "merchant_category_code",
    "counterparty_account", "city", "state", "latitude", "longitude",
    "device_id", "ip_address", "status", "is_international", "risk_score",
]}

for i in range(ROWS):
    acct = random.choice(accounts)
    is_suspect = acct in suspect
    ts = start + timedelta(seconds=random.randint(0, span))

    if is_suspect and random.random() < 0.45:
        # structuring: repeated deposits just under the $10k CTR threshold
        amount = round(random.uniform(8800, 9950), 2)
        channel = random.choice(["wire", "ach", "atm"])
        direction = "credit"
        risk = round(random.uniform(0.55, 0.98), 3)
    else:
        amount = round(random.lognormvariate(3.6, 1.15), 2)
        channel = random.choice(CHANNELS)
        direction = random.choices(["debit", "credit"], weights=[0.78, 0.22])[0]
        risk = round(min(0.99, abs(random.gauss(0.12, 0.11))), 3)

    m_idx = random.randrange(len(MERCHANTS))
    city, state, lat, lon = random.choice(CITIES)

    rows["transaction_id"].append(f"TXN-{i:09d}")
    rows["event_time"].append(ts)
    rows["account_id"].append(acct)
    rows["customer_id"].append(customers[acct])
    rows["channel"].append(channel)
    rows["direction"].append(direction)
    rows["amount"].append(amount)
    rows["currency"].append("USD")
    rows["merchant_name"].append(MERCHANTS[m_idx])
    rows["merchant_category_code"].append(MCC[m_idx])
    rows["counterparty_account"].append(random.choice(accounts))
    rows["city"].append(city)
    rows["state"].append(state)
    rows["latitude"].append(lat + random.uniform(-0.08, 0.08))
    rows["longitude"].append(lon + random.uniform(-0.08, 0.08))
    rows["device_id"].append(f"DEV-{random.randrange(1, 4000):05d}")
    rows["ip_address"].append(f"{random.choice([10,172,192,203])}."
                              f"{random.randrange(0,255)}.{random.randrange(0,255)}."
                              f"{random.randrange(1,254)}")
    rows["status"].append(random.choice(STATUS))
    rows["is_international"].append(random.random() < 0.06)
    rows["risk_score"].append(risk)

schema = pa.schema([
    ("transaction_id", pa.string()),
    ("event_time", pa.timestamp("ms", tz="UTC")),
    ("account_id", pa.string()),
    ("customer_id", pa.string()),
    ("channel", pa.string()),
    ("direction", pa.string()),
    ("amount", pa.float64()),
    ("currency", pa.string()),
    ("merchant_name", pa.string()),
    ("merchant_category_code", pa.string()),
    ("counterparty_account", pa.string()),
    ("city", pa.string()),
    ("state", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("device_id", pa.string()),
    ("ip_address", pa.string()),
    ("status", pa.string()),
    ("is_international", pa.bool_()),
    ("risk_score", pa.float64()),
])

table = pa.Table.from_pydict(rows, schema=schema)
pq.write_table(table, OUT, compression="snappy", row_group_size=50_000)
print(f"wrote {OUT}: {table.num_rows:,} rows x {table.num_columns} cols")
