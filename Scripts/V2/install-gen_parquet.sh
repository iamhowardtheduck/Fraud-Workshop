python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy elasticsearch faker pyarrow 
cd /home/elastic/ESQL-DataFederation/Scripts && ./.venv/bin/python gen_parquet.py
mc alias set local http://localhost:9000 minioadmin 'datafederation_hooray!'
mc admin info local
mc mb local/datasets
mc cp transactions.parquet local/datasets/transactions/
mc anonymous set download local/datasets
mc ls -r local/datasets/
mc alias list
mc ls -r local/datasets/
mc anonymous get local/datasets
