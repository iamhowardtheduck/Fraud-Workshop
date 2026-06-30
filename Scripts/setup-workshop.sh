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
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-brokerage" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "brokerage-workshops*", "name": "Brokerage Workshop", "timeFieldName": "@timestamp"  }}'
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
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "fraud_smurfing_detection",
  "type": "esql",
  "description": "Detects smurfing patterns by identifying accounts that split large transactions into multiple smaller ones to evade detection thresholds.",
  "tags": ["fraud", "smurfing", "aml", "transaction-splitting"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | WHERE amount < ?threshold | STATS tx_count=COUNT(*), total_amount=SUM(amount), avg_amount=AVG(amount), unique_recipients=COUNT_DISTINCT(recipient_account) BY account_id | WHERE tx_count >= ?minTransactions | SORT tx_count DESC | LIMIT ?limit",
    "params": {
      "startTime": {
        "type": "date",
        "description": "Start of the analysis window in ISO 8601 format (e.g. 2024-01-01T00:00:00Z)"
      },
      "endTime": {
        "type": "date",
        "description": "End of the analysis window in ISO 8601 format",
        "optional": true,
        "defaultValue": "now"
      },
      "threshold": {
        "type": "float",
        "description": "Transaction amount threshold below which smurfing is suspected. Defaults to 10000.",
        "optional": true,
        "defaultValue": 10000
      },
      "minTransactions": {
        "type": "integer",
        "description": "Minimum number of sub-threshold transactions to flag an account. Defaults to 3.",
        "optional": true,
        "defaultValue": 3
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
# Veolicty Check
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "fraud_velocity_check",
  "type": "esql",
  "description": "Checks transaction velocity for a given account — counts how many transactions occurred within a rolling time window and flags accounts exceeding a velocity threshold.",
  "tags": ["fraud", "velocity", "aml", "real-time"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime | WHERE account_id == ?accountId | STATS tx_count=COUNT(*), total_amount=SUM(amount), first_tx=MIN(@timestamp), last_tx=MAX(@timestamp), unique_recipients=COUNT_DISTINCT(recipient_account) BY account_id",
    "params": {
      "accountId": {
        "type": "string",
        "description": "The account ID to check velocity for"
      },
      "startTime": {
        "type": "date",
        "description": "Start of the lookback window. Defaults to last 24 hours.",
        "optional": true,
        "defaultValue": "now-24h"
      }
    }
  }
}'
# High Value Transactions
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "fraud_high_value_transactions",
  "type": "esql",
  "description": "Retrieves high-value transactions above a specified amount threshold within a time range. Useful for identifying large suspicious transfers, wire fraud, and outlier transactions.",
  "tags": ["fraud", "high-value", "wire-fraud", "transactions"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | WHERE amount >= ?minAmount | KEEP @timestamp, account_id, recipient_account, amount, transaction_type, merchant_category, country_code, risk_score | SORT amount DESC | LIMIT ?limit",
    "params": {
      "startTime": {
        "type": "date",
        "description": "Start of the time range in ISO 8601 format"
      },
      "endTime": {
        "type": "date",
        "description": "End of the time range in ISO 8601 format",
        "optional": true,
        "defaultValue": "now"
      },
      "minAmount": {
        "type": "float",
        "description": "Minimum transaction amount to include. Defaults to 50000.",
        "optional": true,
        "defaultValue": 50000
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return. Defaults to 50.",
        "optional": true,
        "defaultValue": 50
      }
    }
  }
}'
# Account Profile
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "fraud_account_profile",
  "type": "esql",
  "description": "Builds a behavioral profile for a specific account: total transaction count, total volume, average amount, unique counterparties, and transaction type breakdown over a lookback period.",
  "tags": ["fraud", "account", "profiling", "behavioral-analysis"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime | WHERE account_id == ?accountId | STATS tx_count=COUNT(*), total_volume=SUM(amount), avg_amount=AVG(amount), max_amount=MAX(amount), unique_recipients=COUNT_DISTINCT(recipient_account), unique_countries=COUNT_DISTINCT(country_code) BY transaction_type | SORT tx_count DESC",
    "params": {
      "accountId": {
        "type": "string",
        "description": "The account ID to profile"
      },
      "startTime": {
        "type": "date",
        "description": "Start of the lookback window. Defaults to last 30 days.",
        "optional": true,
        "defaultValue": "now-30d"
      }
    }
  }
}'

# Geo Anomaly
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "fraud_geo_anomaly",
  "type": "esql",
  "description": "Detects geographic anomalies by identifying accounts transacting from multiple countries within a short time window — a common indicator of account takeover or card fraud.",
  "tags": ["fraud", "geo-anomaly", "account-takeover", "international"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | STATS country_count=COUNT_DISTINCT(country_code), tx_count=COUNT(*), total_amount=SUM(amount) BY account_id | WHERE country_count >= ?minCountries | SORT country_count DESC | LIMIT ?limit",
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
# Risk Score
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "fraud_risk_score_summary",
  "type": "esql",
  "description": "Summarizes accounts by their average and maximum risk scores over a time period. Returns the top highest-risk accounts for triage and investigation prioritization.",
  "tags": ["fraud", "risk-score", "triage", "prioritization"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime | WHERE risk_score >= ?minRiskScore | STATS avg_risk=AVG(risk_score), max_risk=MAX(risk_score), tx_count=COUNT(*), total_amount=SUM(amount) BY account_id | SORT max_risk DESC | LIMIT ?limit",
    "params": {
      "startTime": {
        "type": "date",
        "description": "Start of the lookback window. Defaults to last 7 days.",
        "optional": true,
        "defaultValue": "now-7d"
      },
      "minRiskScore": {
        "type": "float",
        "description": "Minimum risk score threshold (0-100). Defaults to 70.",
        "optional": true,
        "defaultValue": 70
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of accounts to return. Defaults to 20.",
        "optional": true,
        "defaultValue": 20
      }
    }
  }
}'

# Create Financial Fraud Skill
curl -X POST "http://localhost:30002/api/agent_builder/skills" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d @- <<'JSON'
{
  "id": "fraud-analysis-skill",
  "name": "Fraud Analysis",
  "description": "Transaction-level AML and fraud detection: account profiling, structuring/smurfing, velocity, high-value transfers, and geographic anomaly screening with corroboration-based risk assessment.",
  "tool_ids": [
    "fraud_smurfing_detection",
    "fraud_velocity_check",
    "fraud_high_value_transactions",
    "fraud_account_profile",
    "fraud_geo_anomaly"
  ],
  "content": "# Financial Fraud Analysis\n\nAnalyze transaction data in the `fraud-*` & `brokerage-*` indices for suspicious financial activity and turn statistical signals into ranked, human-reviewable investigative leads. Use this skill when asked to hunt for suspicious transactions, score accounts for AML risk, investigate a specific account or counterparty, or explain which AML typology a pattern matches.\n\n## Scope and intent\n\nOutputs are investigative LEADS, not accusations. Every flag is a statistical indicator requiring human review (e.g. a SAR/STR analyst decision), never proof of a crime. Do not present a flagged account as confirmed fraud, and always state the metrics and thresholds behind a finding so a human can judge it. Default the lookback window to 90 days when the user does not specify one.\n\n## Data model\n\n`fraud-*` documents are individual financial events. Banking and healthcare-provider (NPI) fields share the pattern; provider fields are null on banking events and should be ignored for AML work. Key fields:\n\n- `event.amount` — transaction value (numeric)\n- `event.type` — "credit" (money in) or "debit" (money out)\n- `account.name` — account holder; the primary grouping key\n- `account.type` — checking, savings, money market\n- `account.checking` / `account.savings` / `account.moneymarket` — account numbers\n- `transaction.date` — event timestamp used for lookback math\n- `wire.direction` — inbound / outbound\n- `wire.outbound.bank_name` / `wire.outbound.country` — wire destination\n- `wire.inbound.bank_name` / `wire.inbound.swiftID` — wire origin\n- `atm.deposit_amount` / `atm.withdrawal_amount` — ATM cash movement\n- `pos.merchant_name` / `pos.geo_point` — point-of-sale context\n- `risk.score` — pre-computed risk score, if populated\n\n## Detection tools\n\n- `fraud_structuring_detection(days)` — accounts with repeated credits in the $8,000-$9,999 band (just below the $10,000 CTR threshold).\n- `fraud_smurfing_detection(days)` — accounts with many small deposits (under $3,000) aggregating to large sums.\n- `fraud_layering_detection()` — outbound wires grouped by destination bank and country.\n- `fraud_velocity_anomaly(days)` — accounts with abnormally high transaction count and total volume (mule/funnel signature).\n- `fraud_round_amount_detection(days)` — high frequency of exact $1,000-multiple amounts.\n- `platform.core.list_indices` / `platform.core.get_index_mapping` — confirm available fields when needed.\n\n## Method\n\n1. Map the question to a typology. "Breaking up cash" -> smurfing; "just under $10k" -> structuring; "money moving overseas" -> layering; "funnel/mule account" -> velocity. If the request is open-ended ("find anything suspicious"), run several detections and correlate.\n2. Run the matching detection tool(s).\n3. Correlate across typologies. Accounts that flag on more than one detection are the highest priority — a name appearing in both structuring and layering is a far stronger lead than either alone. Collect account.name from each detection and rank by how many distinct typologies each account appears in.\n4. Pull supporting raw records before concluding, so the finding is evidence-backed, not just an aggregate count.\n5. Report as a ranked lead list. For each suspect account give: name, typology(ies) matched, key metrics (counts, totals, min/max), a one-line rationale, and a qualitative risk level (High/Medium/Low). Close with the explicit caveat that these are leads for human investigation and SAR/STR review.\n\nThreshold defaults (≥2 near-threshold txns for structuring, ≥5 small deposits for smurfing, ≥10 txns for velocity, ≥3 round txns) are tunable. State any threshold you change. The full typology playbook with the ES|QL behind each detection, false-positive notes, and tuning guidance is in the referenced AML Typology Playbook content."
}
JSON

curl -X POST "http://localhost:30002/api/agent_builder/agents" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d @- <<'JSON'
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
          "fraud_smurfing_detection",
          "fraud_velocity_check",
          "fraud_high_value_transactions",
          "fraud_account_profile",
          "fraud_geo_anomaly",
          "fraud_risk_score_summary",
          "platform.core.generate_esql",
          "platform.core.execute_esql",
          "platform.core.search",
          "platform.core.cases",
          "platform.core.list_indices",
          "platform.core.get_index_mapping",
          "platform.core.get_document_by_id"
        ]
      }
    ]
  }
}
JSON

# Start data-gen installation
chmod +x /root/Fraud-Workshop/Scripts/fraud-gen.sh
bash /root/Fraud-Workshop/Scripts/fraud-gen.sh
