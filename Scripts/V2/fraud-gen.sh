cd /home/elastic/Fraud-Workshop/Scripts/V2
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy elasticsearch faker
python3 /home/elastic/Fraud-Workshop/Scripts/V2/wire-fraud.py
python3 /home/elastic/Fraud-Workshop/Scripts/V2/money-laundering.py
python3 /home/elastic/Fraud-Workshop/Scripts/V2/smurfing.py
python3 /home/elastic/Fraud-Workshop/Scripts/V2/brokerage_workshop.py
