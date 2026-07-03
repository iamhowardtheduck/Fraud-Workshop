# Set up environment variables
echo 'ELASTICSEARCH_USERNAME=elastic' >> /root/.env
#echo -n 'ELASTICSEARCH_PASSWORD=' >> /root/.env
kubectl get secret elasticsearch-es-elastic-user -n default -o go-template='ELASTICSEARCH_PASSWORD={{.data.elastic | base64decode}}' >> /root/.env
echo '' >> /root/.env
echo 'ELASTICSEARCH_URL="http://localhost:30920"' >> /root/.env
echo 'KIBANA_URL="http://localhost:30002"' >> /root/.env
echo 'BUILD_NUMBER="10"' >> /root/.env
echo 'ELASTIC_VERSION="9.1.0"' >> /root/.env
echo 'ELASTIC_APM_SERVER_URL=http://apm.default.svc:8200' >> /root/.env
echo 'ELASTIC_APM_SECRET_TOKEN=pkcQROVMCzYypqXs0b' >> /root/.env

# Set up environment
export $(cat /root/.env | xargs)

BASE64=$(echo -n "elastic:${ELASTICSEARCH_PASSWORD}" | base64)
KIBANA_URL_WITHOUT_PROTOCOL=$(echo $KIBANA_URL | sed -e 's#http[s]\?://##g')

# Add sdg user with superuser role
curl -X POST "http://localhost:30920/_security/user/fraud" -H "Content-Type: application/json" -u "elastic:${ELASTICSEARCH_PASSWORD}" -d '{
  "password" : "hunter",
  "roles" : [ "superuser" ],
  "full_name" : "Fraud Hunter",
  "email" : "fraud-hunter@omnicorp.co"
}'

# Update existing elastic-llm.sh
#cp /root/Fraud-Workshop/Scripts/elastic-llm.sh /opt/workshops/elastic-llm.sh

# Install LLM Connector
#bash /opt/workshops/elastic-llm.sh -m gpt-4.1 -k false -d true 
#bash /opt/workshops/elastic-llm.sh -m gpt-4.1 -k false -d true
#bash /opt/workshops/elastic-llm.sh -m gpt-5.2 -k false -d true -n gpt5-connector -P curriculum-development

# Enable workflows
curl -X POST "http://localhost:30002/api/kibana/settings" -H "Content-Type: application/json" -H "kbn-xsrf: true" -H "x-elastic-internal-origin: featureflag" -u "fraud:hunter"  -d '{
    "changes": {
      "workflows:ui:enabled": true
    }
  }'

# Create 'sar-reports' ingest pipeline and index template
curl -X PUT "http://localhost:30920/_ingest/pipeline/sar-reports" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/sar-reports.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/brokerage-final" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/brokerage-final.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-brokers" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-brokers.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/brokerage-detection-enrich" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/brokerage-detection-enrich.json
curl -X POST "http://localhost:30920/_index_template/sar-reports" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/sar-reports.json
clear
echo
echo "Ingest pipelines loaded"
echo
clear

# Create fraud-workshop data views
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop*", "name": "Fraud Workshop", "timeFieldName": "@timestamp"  }}'  
#curl -X POST "http://localhost:30002/api/saved_objects/index-pattern" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-tsds*", "name": "Fraud-Workshop-TSDS", "timeFieldName": "@timestamp"  }}'  
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-money-laundering" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-money-laundering*", "name": "Money-Laundering", "timeFieldName": "@timestamp"  }}'  
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-wire-fraud" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-wire-fraud*", "name": "Wire-Fraud", "timeFieldName": "@timestamp"  }}' 
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-smurfing" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-smurfing*", "name": "Smurfing", "timeFieldName": "@timestamp"  }}'
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/sar-reports" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "sar-reports*", "name": "SAR Reports", "timeFieldName": "@timestamp"  }}'
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-brokerage" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "brokerage-workshop*", "name": "Brokerage Workshop", "timeFieldName": "@timestamp"  }}'
clear
echo
echo "Data Views loaded"
echo
clear


# Load saved-searches for assignment starts
curl -X POST "http://localhost:30002/api/saved_objects/_import" -H "kbn-xsrf: true" -u "fraud:hunter" -F "file=@/root/Fraud-Workshop/Saved-Searches/3-StartSavedSearches.ndjson"
clear
echo
echo "Saved searches loaded"
echo
clear

# Load DFA Workflow
curl -X POST "http://localhost:30002/api/workflows" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d @/root/Fraud-Workshop/Workflows/dfa-workflow.json

clear
echo
echo "Data Frame Analytics workflow loaded"
echo
clear

# Load High-Value Daily Aggregate Workflow
CONN_ID=$(curl -s "http://localhost:30002/api/actions/connectors" \
  -H "kbn-xsrf: true" -u "fraud:hunter" \
  | python3 -c 'import json,sys; print(next(c["id"] for c in json.load(sys.stdin) if c["name"]=="openai-connector"))')
echo "Resolved openai-connector -> $CONN_ID"

sed "s/CONNECTOR_ID_PLACEHOLDER/$CONN_ID/g" /root/Fraud-Workshop/Workflows/highvalue-workflow-body.json \
  | curl -X POST "http://localhost:30002/api/workflows" \
    -H "Content-Type: application/json" \
    -H "kbn-xsrf: true" \
    -u "fraud:hunter" \
    --data-binary @-
clear
echo
echo "High-Value Daily Aggregate workflow loaded"
echo
clear

# Load component templates
curl -X PUT "http://localhost:30920/_component_template/fraud-workshop-logsdb-mappings" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Component-Templates/fraud-workshop-logsdb-mappings.json

clear
echo
echo "Component Template loaded"
echo
clear

# Load index templates
curl -X POST "http://localhost:30920/_index_template/enrich-accounts" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-accounts.json
curl -X POST "http://localhost:30920/_index_template/enrich-austinbanks" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-austinbanks.json
curl -X POST "http://localhost:30920/_index_template/enrich-austinstores" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-austinstores.json
curl -X POST "http://localhost:30920/_index_template/enrich-intbank" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-intbank.json
curl -X POST "http://localhost:30920/_index_template/enrich-brokers" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-brokers.json
curl -X POST "http://localhost:30920/_index_template/fraud-workshop-logsdb" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/fraud-workshop-logsdb.json
curl -X POST "http://localhost:30920/_index_template/brokerage-workshop-logsdb" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/brokerage-workshop-logsdb.json

clear
echo
echo "Index templates loaded"
echo
clear
# Load enrichment data sources
# Legacy direct approach:
#curl -X POST "http://localhost:30920/enrich-accounts/_bulk" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Data/enrich-accounts.ndjson
#curl -X POST "http://localhost:30920/enrich-austinbanks/_bulk" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Data/enrich-austinbanks.ndjson
#curl -X POST "http://localhost:30920/enrich-austinstores/_bulk" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Data/enrich-austinstores.ndjson
#curl -X POST "http://localhost:30920/enrich-intbank/_bulk" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Data/enrich-intbank.ndjson

# New cleaner progress bar approach:
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:30920"
USER="fraud:hunter"
DATA_DIR="/root/Fraud-Workshop/Enrichment-Data"

declare -A SOURCES=(
  [enrich-accounts]="enrich-accounts.ndjson"
  [enrich-austinbanks]="enrich-austinbanks.ndjson"
  [enrich-austinstores]="enrich-austinstores.ndjson"
  [enrich-intbank]="enrich-intbank.ndjson"
  [sar-reports]="sar-reports.ndjson"
  [enrich-brokers]="enrich-brokers.ndjson"
)

for index in "${!SOURCES[@]}"; do
  file="${DATA_DIR}/${SOURCES[$index]}"
  output="bulk_${index}_response.json"

  echo "Uploading $file to index [$index]..."

  curl --progress-bar \
    -X POST "$BASE_URL/$index/_bulk" \
    -H "Content-Type: application/x-ndjson" \
    -u "$USER" \
    --data-binary "@$file" \
    -o "$output"

  echo "  --> Done. Response saved to $output"
  echo
done

clear
echo
echo "Enrichment data loaded"
echo
clear

# Create enrichment policies
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-accounts" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-accounts.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-austinbanks" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-austinbanks.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-austinstores" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-austinstores.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-austinswift" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-austinswift.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-inbounds" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-inbounds.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-intbank" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-intbank.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-outbounds" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-outbounds.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-brokers" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-brokers.json
clear
echo
echo "Enrichment policies loaded"
echo
clear
# Execute enrichment policies
curl -X POST "http://localhost:30920/_enrich/policy/enrich-accounts/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-austinbanks/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-austinstores/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-austinswift/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-inbounds/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-intbank/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-outbounds/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-brokers/_execute" -u "fraud:hunter"

clear
echo
echo "Enrichment policies executed"
echo
clear

# Create ingest pipelines
curl -X PUT "http://localhost:30920/_ingest/pipeline/atm-cleanup" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/atm-cleanup.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-accounts" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-accounts.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-austinbanks" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-austinbanks.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-austinstores" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-austinstores.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-austinswift" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-austinswift.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-inbound" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-inbound.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-intbank" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-intbank.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-outbound" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-outbound.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-outbounds" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-outbounds.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/enrich-brokers" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-brokers.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/fraud-detection-enrich" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/fraud-detection-enrich.json
curl -X PUT "http://localhost:30920/_ingest/pipeline/brokerage-detection-enrich" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/brokerage-detection-enrich.json

clear
echo
echo "Ingest pipelines loaded"
echo
clear

echo
echo "Deploying Agents"
echo

# Create Suspicious Activity Reporting Agent 
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:30002/api/agent_builder/agents"
USER="fraud:hunter"
DATA_DIR="/root/Fraud-Workshop/Agents"

declare -A SOURCES=(
  [SARA]="SARA.json"
  [Financial Fraud Analyst]="financial-fraud-analyst.json"
)

for index in "${!SOURCES[@]}"; do
  file="${DATA_DIR}/${SOURCES[$index]}"
  output="bulk_${index}_response.json"

  echo "Uploading $file to index [$index]..."

  curl --progress-bar \
    -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -H "kbn-xsrf: true" \
    -u "$USER" \
    -d "@$file" \
    -o "$output"

  echo "  --> Done. Response saved to $output"
  echo
done

# Tool creation

# Smurfing Detection
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_smurfing_detection",
  "type": "esql",
  "description": "Detects smurfing patterns by identifying accounts that split large transactions into multiple smaller ones to evade detection thresholds.",
  "tags": ["fraud", "smurfing", "aml", "transaction-splitting"],
  "configuration": {
    "query": """FROM fraud-* | WHERE event.type == "credit" AND event.amount > 0 AND event.amount < 3000 AND DATE_DIFF("days", transaction.date, NOW()) <= ?days | STATS small_deposits = COUNT(*), total_aggregated = SUM(event.amount), avg_deposit = AVG(event.amount) BY account.name | WHERE small_deposits >= 5 | SORT small_deposits DESC | LIMIT 50""",
    "params": {
      "days": {
        "type": "integer",
        "description": "Lookback window in days"
      }
    }
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {
      "days": {
        "type": "integer",
        "minimum": -9007199254740991,
        "maximum": 9007199254740991,
        "description": "Lookback window in days"
      }
    },
    "required": [
      "days"
    ],
    "description": "Parameters needed to execute the query"
  }
}'
# Veolicty Check
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_velocity_check",
  "type": "esql",
  "description": "Checks transaction velocity for a given account — counts how many transactions occurred within a rolling time window and flags accounts exceeding a velocity threshold.",
  "tags": ["fraud", "velocity", "aml", "real-time"],
  "configuration": {
    "query": """FROM fraud-* | WHERE event.amount > 0 AND DATE_DIFF("days", transaction.date, NOW()) <= ?days | STATS txn_count = COUNT(*), total_volume = SUM(event.amount), avg_txn = AVG(event.amount), max_txn = MAX(event.amount) BY account.name | WHERE txn_count >= 10 | SORT total_volume DESC | LIMIT 50""",
    "params": {
      "days": {
        "type": "integer",
        "description": "Lookback window in days"
      }
    }
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {
      "days": {
        "type": "integer",
        "minimum": -9007199254740991,
        "maximum": 9007199254740991,
        "description": "Lookback window in days"
      }
    },
    "required": [
      "days"
    ],
    "description": "Parameters needed to execute the query"
  }
}'

# Round Amount Transactions
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_round_amount_detection",
  "type": "esql",
  "description": "Detects suspicious ROUND-NUMBER transactions: transactions whose amount is an exact multiple of $1,000, a common indicator of artificial/laundering activity rather than organic spending. Returns account, count of round transactions and total. Use alongside other signals to corroborate laundering.",
  "tags": ["fraud", "high-value", "wire-fraud", "transactions"],
  "configuration": {
    "query": """FROM fraud-* | WHERE event.amount > 0 AND (event.amount % 1000) == 0 AND DATE_DIFF("days", transaction.date, NOW()) <= ?days | STATS round_txns = COUNT(*), total_round = SUM(event.amount) BY account.name | WHERE round_txns >= 3 | SORT round_txns DESC | LIMIT 50""",
    "params": {
      "days": {
        "type": "integer",
        "description": "Lookback window in days"
      }
    }
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {
      "days": {
        "type": "integer",
        "minimum": -9007199254740991,
        "maximum": 9007199254740991,
        "description": "Lookback window in days"
      }
    },
    "required": [
      "days"
    ],
    "description": "Parameters needed to execute the query"
  }
}'

# Layering Detection
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{
  "id": "fraud_layering_detection",
  "type": "esql",
  "description": "Builds a behavioral profile for a specific account: total transaction count, total volume, average amount, unique counterparties, and transaction type breakdown over a lookback period.",
  "tags": ["fraud", "account", "profiling", "behavioral-analysis"],
  "configuration": {
    "query": """FROM fraud-* | WHERE wire.direction == "outbound" | STATS wire_count = COUNT(*), total_wired = SUM(event.amount) BY account.name, wire.outbound.bank_name, wire.outbound.country | SORT total_wired DESC | LIMIT 50""",
    "params": {}
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {},
    "description": "Parameters needed to execute the query"
  }
}'

# Structuring Detection
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_structuring_detection",
  "type": "esql",
  "description": "Detects STRUCTURING / CTR-evasion: accounts with multiple transactions just below the $10,000 reporting threshold ($8,000-$9,999) over the last N days. Returns the account, count of near-threshold transactions, and total moved. High counts indicate deliberate structuring to avoid Currency Transaction Reports.",
  "tags": ["fraud", "geo-anomaly", "account-takeover", "international"],
  "configuration": {
    "query": """FROM fraud-* | WHERE event.type == "credit" AND event.amount >= 8000 AND event.amount < 10000 AND DATE_DIFF("days", transaction.date, NOW()) <= ?days | STATS near_threshold_txns = COUNT(*), total_amount = SUM(event.amount), min_amt = MIN(event.amount), max_amt = MAX(event.amount) BY account.name | WHERE near_threshold_txns >= 2 | SORT near_threshold_txns DESC | LIMIT 50""",
    "params": {
      "days": {
        "type": "integer",
        "description": "Lookback window in days (e.g. 7, 30, 90)"
      }
    }
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {
      "days": {
        "type": "integer",
        "minimum": -9007199254740991,
        "maximum": 9007199254740991,
        "description": "Lookback window in days (e.g. 7, 30, 90)"
      }
    },
    "required": [
      "days"
    ],
    "description": "Parameters needed to execute the query"
  }
}'
# High Value Daily Triage
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud-high-value-daily-triage",
  "type": "workflow",
  "description": "Welcome Fraud analysis to focus investigations",
  "tags": [
    "security",
    "investigation",
    "workflows"
  ],
  "configuration": {
    "workflow_id": "high-value-daily-aggregate-triage",
    "wait_for_completion": true
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {}
  }
}'

# Free-form fraud search
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_transaction_search",
  "type": "index_search",
  "description": "Free-form search over the fraud-* transaction indices. Use to investigate a specific account, name, merchant, wire counterparty, SWIFT/routing number, or time range once a suspicious pattern is identified by the detection tools, or to pull supporting raw transaction records.",
  "tags": [],
  "configuration": {
    "pattern": "fraud-*"
  },
  "readonly": false,
  "schema": {
    "type": "object",
    "properties": {
      "nlQuery": {
        "type": "string",
        "description": "A natural language query expressing the search request"
      }
    },
    "required": [
      "nlQuery"
    ]
  }
}'

curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_geo_anomaly",
  "type": "esql",
  "description": "Detects geographic anomalies by identifying accounts transacting from multiple countries within a short time window — a common indicator of account takeover or card fraud.",
  "tags": ["fraud", "geo-anomaly", "account-takeover", "international"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | STATS country_count=COUNT_DISTINCT(wire.outbound.country_code), tx_count=COUNT(*), total_amount=SUM(event.amount) BY account.name | WHERE country_count >= ?minCountries | SORT country_count DESC | LIMIT ?limit",
    "params": {
      "startTime": {
        "type": "date",
        "description": "Start of the time window in ISO 8601 format"
      },
      "endTime": {
        "type": "date",
        "description": "End of the time window in ISO 8601 format",
        "optional": true,
        "defaultValue": "now"
      },
      "minCountries": {
        "type": "integer",
        "description": "Minimum number of distinct countries to flag an account. Defaults to 2.",
        "optional": true,
        "defaultValue": 2
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return. Defaults to 25.",
        "optional": true,
        "defaultValue": 25
      }
    }
  }
}'


# Create Financial Fraud Skill
curl -X POST "http://localhost:30002/api/agent_builder/skills" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d @- <<'JSON'
{
  "id": "financial-fraud-analysis",
  "name": "Financial Fraud Analysis",
  "description": "Analyze fraud-* transaction data for AML red flags: structuring/CTR evasion, smurfing, layering, money laundering, mule-account funneling, and abnormal transaction velocity. Use when asked to hunt for suspicious transactions, score accounts for AML risk, investigate a specific account or counterparty, or explain which money-laundering typology a transaction pattern matches. Produces ranked investigative leads for human SAR review, not accusations.",
  "content": """# Financial Fraud Analysis

Analyze transaction data in the `fraud-*` indices for suspicious financial activity and turn statistical signals into ranked, human-reviewable investigative leads. Use this skill when asked to hunt for suspicious transactions, score accounts for AML risk, investigate a specific account or counterparty, or explain which AML typology a pattern matches.

## Scope and intent

Outputs are investigative LEADS, not accusations. Every flag is a statistical indicator requiring human review (e.g. a SAR/STR analyst decision), never proof of a crime. Do not present a flagged account as confirmed fraud, and always state the metrics and thresholds behind a finding so a human can judge it. Default the lookback window to 90 days when the user does not specify one.

## Data model

`fraud-*` documents are individual financial events. Banking and healthcare-provider (NPI) fields share the pattern; provider fields are null on banking events and should be ignored for AML work. Key fields:

- `event.amount` — transaction value (numeric)
- `event.type` — "credit" (money in) or "debit" (money out)
- `account.name` — account holder; the primary grouping key
- `account.type` — checking, savings, money market
- `account.checking` / `account.savings` / `account.moneymarket` — account numbers
- `transaction.date` — event timestamp used for lookback math
- `wire.direction` — inbound / outbound
- `wire.outbound.bank_name` / `wire.outbound.country` — wire destination
- `wire.inbound.bank_name` / `wire.inbound.swiftID` — wire origin
- `atm.deposit_amount` / `atm.withdrawal_amount` — ATM cash movement
- `pos.merchant_name` / `pos.geo_point` — point-of-sale context
- `risk.score` — pre-computed risk score, if populated

## Detection tools

- `fraud_structuring_detection(days)` — accounts with repeated credits in the $8,000-$9,999 band (just below the $10,000 CTR threshold).
- `fraud_smurfing_detection(days)` — accounts with many small deposits (under $3,000) aggregating to large sums.
- `fraud_layering_detection()` — outbound wires grouped by destination bank and country.
- `fraud_velocity_anomaly(days)` — accounts with abnormally high transaction count and total volume (mule/funnel signature).
- `fraud_round_amount_detection(days)` — high frequency of exact $1,000-multiple amounts.
- `platform.core.list_indices` / `platform.core.get_index_mapping` — confirm available fields when needed.

## Method

1. Map the question to a typology. "Breaking up cash" -> smurfing; "just under $10k" -> structuring; "money moving overseas" -> layering; "funnel/mule account" -> velocity. If the request is open-ended ("find anything suspicious"), run several detections and correlate.
2. Run the matching detection tool(s).
3. Correlate across typologies. Accounts that flag on more than one detection are the highest priority — a name appearing in both structuring and layering is a far stronger lead than either alone. Collect account.name from each detection and rank by how many distinct typologies each account appears in.
4. Pull supporting raw records before concluding, so the finding is evidence-backed, not just an aggregate count.
5. Report as a ranked lead list. For each suspect account give: name, typology(ies) matched, key metrics (counts, totals, min/max), a one-line rationale, and a qualitative risk level (High/Medium/Low). Close with the explicit caveat that these are leads for human investigation and SAR/STR review.

Threshold defaults (≥2 near-threshold txns for structuring, ≥5 small deposits for smurfing, ≥10 txns for velocity, ≥3 round txns) are tunable. State any threshold you change. The full typology playbook with the ES|QL behind each detection, false-positive notes, and tuning guidance is in the referenced AML Typology Playbook content.
""",
  "referenced_content": [
    {
      "name": "AML Typology Playbook",
      "relativePath": "./typologies.md",
      "content": """# AML Typology Playbook

The detection thresholds below are sensible defaults for synthetic workshop data. Tune them
to your data's magnitudes and your jurisdiction's reporting thresholds. Each detection
produces statistical leads for human review, not proof of wrongdoing.

All queries run against the `fraud-*` index pattern. Lookback is expressed with
`DATE_DIFF("days", transaction.date, NOW()) <= ?days` because ES|QL rejects parameterized
interval literals (`?days * 1 day` fails verification) — use `DATE_DIFF` instead.

---

## 1. Structuring (CTR evasion)

**Definition.** Deliberately keeping transactions just below the $10,000 Currency Transaction
Report threshold so each event escapes mandatory reporting. Also called "smurfing the
threshold," though it is distinct from smurfing proper (below).

**Indicators.** Multiple credits landing in the $8,000–$9,999 band, often within a short
window, frequently round-ish numbers a few hundred dollars under $10k.

**ES|QL.**

```esql
FROM fraud-*
| WHERE event.type == "credit" AND event.amount >= 8000 AND event.amount < 10000
    AND DATE_DIFF("days", transaction.date, NOW()) <= ?days
| STATS near_threshold_txns = COUNT(*), total_amount = SUM(event.amount),
        min_amt = MIN(event.amount), max_amt = MAX(event.amount) BY account.name
| WHERE near_threshold_txns >= 2
| SORT near_threshold_txns DESC
| LIMIT 50
```

**False positives.** A business with genuine large-but-sub-$10k recurring receipts (payroll
runs, rent rolls). Corroborate with regularity of timing and whether amounts cluster
suspiciously tight to $9,999.

**Tuning.** Raise the floor (e.g. $9,000) to reduce noise, or raise the minimum count to 3+
for higher-confidence-only output.

---

## 2. Smurfing

**Definition.** Breaking one large sum into many small deposits — often across multiple people
("smurfs"), accounts, or ATMs — to avoid attention. Distinct from structuring: structuring
hugs the reporting threshold, smurfing uses many *small* amounts that aggregate.

**Indicators.** A high count of small credits (each well under the threshold) that sum to a
large figure; deposits clustered in time or across many ATM locations.

**ES|QL.**

```esql
FROM fraud-*
| WHERE event.type == "credit" AND event.amount > 0 AND event.amount < 3000
    AND DATE_DIFF("days", transaction.date, NOW()) <= ?days
| STATS small_deposits = COUNT(*), total_aggregated = SUM(event.amount),
        avg_deposit = AVG(event.amount) BY account.name
| WHERE small_deposits >= 5
| SORT small_deposits DESC
| LIMIT 50
```

**False positives.** High-frequency small-ticket merchants (coffee shops, tips). Weight by
whether deposits are cash/ATM vs card settlement, and by how large the aggregate is.

**Tuning.** Lower the per-deposit ceiling for tighter focus on cash-like amounts; raise the
minimum deposit count to require a stronger pattern.

---

## 3. Layering

**Definition.** Moving illicit funds through a series of transfers — frequently cross-border
wires to intermediary banks — to distance the money from its source and obscure the audit
trail. The middle stage of the classic placement → layering → integration model.

**Indicators.** Outbound wires, especially to foreign banks, in volumes inconsistent with the
account's profile; funds arriving then leaving quickly; multiple destination institutions.

**ES|QL.**

```esql
FROM fraud-*
| WHERE wire.direction == "outbound"
| STATS wire_count = COUNT(*), total_wired = SUM(event.amount)
        BY account.name, wire.outbound.bank_name, wire.outbound.country
| SORT total_wired DESC
| LIMIT 50
```

**False positives.** Importers, treasury operations, expats with legitimate overseas
obligations. Corroborate with whether inbound credits shortly precede the outbound wires
(rapid pass-through is the stronger signal).

**Tuning.** Add a lookback `WHERE DATE_DIFF(...) <= ?days`, or filter to specific high-risk
destination countries.

---

## 4. Velocity anomaly (mule / funnel detection)

**Definition.** An account being used as a conduit shows transaction count and dollar volume
far above its peers — characteristic of money-mule and funnel accounts that receive and
immediately redistribute funds.

**Indicators.** High transaction count combined with high total volume in a short window; a
mix of many inbound credits and rapid outbound debits.

**ES|QL.**

```esql
FROM fraud-*
| WHERE event.amount > 0 AND DATE_DIFF("days", transaction.date, NOW()) <= ?days
| STATS txn_count = COUNT(*), total_volume = SUM(event.amount),
        avg_txn = AVG(event.amount), max_txn = MAX(event.amount) BY account.name
| WHERE txn_count >= 10
| SORT total_volume DESC
| LIMIT 50
```

**False positives.** Genuinely active accounts (small businesses, frequent traders). Compare
against the population median; an account 5–10x the median is more interesting than one merely
above the floor.

**Tuning.** Replace the fixed `>= 10` floor with a percentile-based cutoff once you know the
population distribution.

---

## 5. Round-amount transactions

**Definition.** Artificial money movement tends toward clean round figures (exact thousands)
because it is fabricated rather than organic spending. A supporting signal, rarely conclusive
on its own.

**Indicators.** A high frequency of transactions that are exact multiples of $1,000.

**ES|QL.**

```esql
FROM fraud-*
| WHERE event.amount > 0 AND (event.amount % 1000) == 0
    AND DATE_DIFF("days", transaction.date, NOW()) <= ?days
| STATS round_txns = COUNT(*), total_round = SUM(event.amount) BY account.name
| WHERE round_txns >= 3
| SORT round_txns DESC
| LIMIT 50
```

**False positives.** Round-number behavior is common in legitimate transfers, rent, and
savings. Use only to corroborate — never flag on round amounts alone.

**Tuning.** Tighten to $5,000 or $10,000 multiples for higher specificity.

---

## 6. Cross-typology correlation

The single most valuable step is not any one detection but the **overlap**. Run multiple
detections, collect the `account.name` from each, and rank accounts by how many distinct
typologies they appear in. An account that surfaces in structuring *and* layering *and*
velocity is a high-priority lead even if no single metric is extreme.

`scripts/fraud-detect.js scan` automates this: it runs every typology and prints a summary of
accounts that flag more than once, sorted by overlap count.

---

## Reporting template

For each suspect account, report:

- **Account:** holder name
- **Typologies matched:** e.g. structuring + velocity
- **Evidence:** the concrete metrics (counts, totals, min/max) from the detections
- **Rationale:** one line on why the pattern is suspicious
- **Risk:** High / Medium / Low (qualitative)

Always close with: these are statistical indicators requiring human investigation and
SAR/STR review, not determinations of criminal conduct.
"""
    }
  ],
  "tool_ids": [
    "fraud_structuring_detection",
    "fraud_smurfing_detection",
    "fraud_layering_detection",
    "fraud_velocity_anomaly",
    "fraud_round_amount_detection"
  ],
  "readonly": false,
  "experimental": false
}
JSON

curl -X POST "http://localhost:30002/api/agent_builder/agents" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d @- <<'JSON'
{
  "id": "financial-fraud-analyst",
  "name": "Financial Fraud Analyst",
  "description": "I can help you detect and investigate financial fraud — including smurfing, velocity abuse, geographic anomalies, high-value wire transfers, and account risk profiling.",
  "labels": ["fraud", "aml", "financial", "security"],
  "avatar_color": "#FF4444",
  "avatar_symbol": "FF",
  "configuration": {
    "instructions": "You are an expert financial fraud analyst. Use your available tools to investigate transaction data, identify suspicious patterns, and provide clear risk assessments with supporting evidence. Always cite specific data points such as amounts, counts, timeframes, and account IDs in your findings. Workflow: profile an account first (fraud_account_profile), then layer detectors (fraud_smurfing_detection, fraud_velocity_check, fraud_high_value_transactions, fraud_geo_anomaly), interpreting each against the baseline. Triage by risk using fraud_risk_score_summary, ranking by max risk and confirming with average risk. For ad-hoc analysis, generate queries with generate_esql and run them with execute_esql — never fabricate ES|QL. Check platform.core.cases for existing investigations before recommending escalation. No single signal is conclusive; strength comes from corroboration. Always set an explicit time window and recommend a clear escalation path for high-risk findings.",
    "tools": [
      {
        "tool_ids": [
          "platform.core.generate_esql",
          "platform.core.execute_esql",
          "platform.core.search",
          "platform.core.cases",
          "platform.core.list_indices",
          "platform.core.get_index_mapping",
          "platform.core.get_document_by_id",
          "platform.core.index_explorer",
          "fraud_geo_anomaly",
          "fraud_transaction_search"
        ]
      }
    ],
   "skill_ids": [ "financial-fraud-analysis", "graph-creation", "visualization-creation", "dashboard-management" ]
 }}
JSON

# Start data-gen installation
chmod +x /root/Fraud-Workshop/Scripts/fraud-gen.sh
bash /root/Fraud-Workshop/Scripts/fraud-gen.sh
