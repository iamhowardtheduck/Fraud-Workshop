#!/usr/bin/env bash
#
# setup-brokerage-enrichment.sh
# Loads the brokerage-workshop enrichment stack into Elasticsearch in the
# correct dependency order. Mirrors the bank-side Fraud-Workshop conventions.
#
# Usage:
#   ./setup-brokerage-enrichment.sh
#
# Override defaults with env vars:
#   ES_HOST=http://localhost:30920 ES_USER=fraud ES_PASS=hunter ./setup-brokerage-enrichment.sh
#
set -euo pipefail

ES_HOST="${ES_HOST:-http://localhost:30920}"
ES_USER="${ES_USER:-fraud}"
ES_PASS="${ES_PASS:-hunter}"
AUTH="-u ${ES_USER}:${ES_PASS}"
HDR="-H Content-Type:application/json"
NDHDR="-H Content-Type:application/x-ndjson"

# resolve script dir so it can run from anywhere
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { echo -e "\n>>> $*"; }

# 1) Load enrichment source data (the lookup table the policy reads from)
say "1/9 Loading enrich-brokers source data"
curl -s $AUTH $NDHDR -XPOST "${ES_HOST}/_bulk" \
  --data-binary "@${DIR}/Enrichment-Data/enrich-brokers.ndjson" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  errors:',d.get('errors'))"

curl -s $AUTH -XPOST "${ES_HOST}/enrich-brokers/_refresh" >/dev/null

# 2) Create the enrich policy
say "2/9 Creating enrich-brokers policy"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/_enrich/policy/enrich-brokers" \
  --data-binary "@${DIR}/Enrichment-Policies/enrich-brokers.json"; echo

# 3) Execute the policy (builds the .enrich-* system index). Required before
#    any pipeline can use it, and must be re-run if the source data changes.
say "3/9 Executing enrich-brokers policy"
curl -s $AUTH -XPOST "${ES_HOST}/_enrich/policy/enrich-brokers/_execute"; echo

# 4) Load the sub-pipeline that calls the enrich policy
say "4/9 Loading enrich-brokers sub-pipeline"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/_ingest/pipeline/enrich-brokers" \
  --data-binary "@${DIR}/Ingest-Pipelines/enrich-brokers.json"; echo

# 5) Load the main pipeline (references enrich-brokers + reuses bank sub-pipelines
#    enrich-accounts, enrich-intbank, enrich-austinswift, enrich-inbound, enrich-outbound)
say "5/9 Loading brokerage-detection-enrich main pipeline"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/_ingest/pipeline/brokerage-detection-enrich" \
  --data-binary "@${DIR}/Ingest-Pipelines/brokerage-detection-enrich.json"; echo

# 6) Load the index template (sets default_pipeline + logsdb mode)
say "6/9 Loading brokerage-workshop index template"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/_index_template/brokerage-workshop-logsdb" \
  --data-binary "@${DIR}/Index-Templates/brokerage-workshop-logsdb.json"; echo

# 7) Load the financial fraud analyst tools
say "7/9 Loading Fraud Agentic Tools"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/tools" \
  --data-binary "@${DIR}/Tools/account_profile.json"; echo  

curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/tools" \
  --data-binary "@${DIR}/Tools/fraud_smurfing_detection.json"; echo

curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/tools" \
  --data-binary "@${DIR}/Tools/geo_anomaly.json"; echo  

curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/tools" \
  --data-binary "@${DIR}/Tools/high_value_transactions.json"; echo  

curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/tools" \
  --data-binary "@${DIR}/Tools/risk_score_summary.json"; echo  

curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/tools" \
  --data-binary "@${DIR}/Tools/velocity_check.json"; echo

# 8) Load the finanicial fraud skill
say "8/9 Loading Financial Fraud Skills"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/skills" \
  --data-binary "@${DIR}/Skills/financial_fraud.json"; echo

# 9) Load the finanicial fraud skill
say "9/9 Loading Financial Fraud Agent"
curl -s $AUTH $HDR -XPUT "${ES_HOST}/api/agent_builder/agents" \
  --data-binary "@${DIR}/Skills/financial-fraud-analyst.json"; echo

say "Done. Now run:  python3 brokerage_workshop.py"
echo "    (template sets default_pipeline, so the script's pipeline= is belt-and-suspenders)"

