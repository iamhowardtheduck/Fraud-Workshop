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
curl -X POST "http://localhost:30920/_index_template/sar-reports" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/sar-reports.json

# Create fraud-workshop data views
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop*", "name": "Fraud Workshop", "timeFieldName": "@timestamp"  }}'  
#curl -X POST "http://localhost:30002/api/saved_objects/index-pattern" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-tsds*", "name": "Fraud-Workshop-TSDS", "timeFieldName": "@timestamp"  }}'  
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-money-laundering" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-money-laundering*", "name": "Money-Laundering", "timeFieldName": "@timestamp"  }}'  
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-wire-fraud" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-wire-fraud*", "name": "Wire-Fraud", "timeFieldName": "@timestamp"  }}' 
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop-smurfing" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop-smurfing*", "name": "Smurfing", "timeFieldName": "@timestamp"  }}'
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/sar-reports" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "sar-reports*", "name": "SAR Reports", "timeFieldName": "@timestamp"  }}'

# Load saved-searches for assignment starts
curl -X POST "http://localhost:30002/api/saved_objects/_import" -H "kbn-xsrf: true" -u fraud:hunter -F "file=@/root/Fraud-Workshop/Saved-Searches/3-StartSavedSearches.ndjson"


# Load component templates
curl -X PUT "http://localhost:30920/_component_template/fraud-workshop-logsdb-mappings" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Component-Templates/fraud-workshop-logsdb-mappings.json


# Load index templates
curl -X POST "http://localhost:30920/_index_template/enrich-accounts" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-accounts.json
curl -X POST "http://localhost:30920/_index_template/enrich-austinbanks" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-austinbanks.json
curl -X POST "http://localhost:30920/_index_template/enrich-austinstores" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-austinstores.json
curl -X POST "http://localhost:30920/_index_template/enrich-intbank" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-intbank.json
#curl -X POST "http://localhost:30920/_index_template/fraud-workshop-tsds" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/fraud-workshop-tsds.json
curl -X POST "http://localhost:30920/_index_template/fraud-workshop-logsdb" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/fraud-workshop-logsdb.json


echo
echo "Index templates loaded"
echo

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


echo
echo "Enrichment data loaded"
echo

# Create enrichment policies
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-accounts" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-accounts.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-austinbanks" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-austinbanks.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-austinstores" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-austinstores.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-austinswift" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-austinswift.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-inbounds" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-inbounds.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-intbank" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-intbank.json
curl -X PUT "http://localhost:30920/_enrich/policy/enrich-outbounds" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Enrichment-Policies/enrich-outbounds.json

echo
echo "Enrichment policies loaded"
echo

# Execute enrichment policies
curl -X POST "http://localhost:30920/_enrich/policy/enrich-accounts/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-austinbanks/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-austinstores/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-austinswift/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-inbounds/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-intbank/_execute" -u "fraud:hunter"
curl -X POST "http://localhost:30920/_enrich/policy/enrich-outbounds/_execute" -u "fraud:hunter"

echo
echo "Enrichment policies executed"
echo

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
curl -X PUT "http://localhost:30920/_ingest/pipeline/fraud-detection-enrich" -H "Content-Type: application/x-ndjson" -u "fraud:hunter" -d @/root/Fraud-Workshop/Ingest-Pipelines/fraud-detection-enrich.json


echo
echo "Ingest pipelines loaded"
echo

echo
echo "Deploying our Suspicous Activity Reporting Agent"
echo

# Create Suspicious Activity Reporting Agent 
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:30002/api/agent_builder/agents"
USER="fraud:hunter"
DATA_DIR="/root/Fraud-Workshop/Agents"

declare -A SOURCES=(
  [SARA]="SARA.json"
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
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_smurfing_detection",
  "type": "esql",
  "description": "Detects smurfing patterns by identifying accounts that split large transactions into multiple smaller ones to evade detection thresholds.",
  "tags": ["fraud", "smurfing", "aml", "transaction-splitting"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | WHERE amount < ?threshold | STATS tx_count=COUNT(*), total_amount=SUM(amount), avg_amount=AVG(amount), unique_recipients=COUNT_DISTINCT(recipient_account) BY account_id | WHERE tx_count >= ?minTransactions | SORT tx_count DESC | LIMIT ?limit",
    "params": {
      "startTime": {"type": "date", "description": "Start of the analysis window in ISO 8601 format"},
      "endTime": {"type": "date", "description": "End of the analysis window", "optional": true, "defaultValue": "now"},
      "threshold": {"type": "float", "description": "Amount threshold below which smurfing is suspected. Defaults to 10000.", "optional": true, "defaultValue": 10000},
      "minTransactions": {"type": "integer", "description": "Min sub-threshold transactions to flag. Defaults to 3.", "optional": true, "defaultValue": 3},
      "limit": {"type": "integer", "description": "Max results. Defaults to 25.", "optional": true, "defaultValue": 25}
    }
  }
}'

# Veolicty Check
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_velocity_check",
  "type": "esql",
  "description": "Checks transaction velocity for a given account within a rolling time window.",
  "tags": ["fraud", "velocity", "aml", "real-time"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime | WHERE account_id == ?accountId | STATS tx_count=COUNT(*), total_amount=SUM(amount), first_tx=MIN(@timestamp), last_tx=MAX(@timestamp), unique_recipients=COUNT_DISTINCT(recipient_account) BY account_id | EVAL velocity_per_hour = tx_count / DATE_DIFF(\"hours\", first_tx, last_tx)",
    "params": {
      "accountId": {"type": "string", "description": "The account ID to check velocity for"},
      "startTime": {"type": "date", "description": "Start of the lookback window. Defaults to last 24 hours.", "optional": true, "defaultValue": "now-24h"}
    }
  }
}'

# High Value Transactions
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_high_value_transactions",
  "type": "esql",
  "description": "Retrieves high-value transactions above a specified amount threshold within a time range.",
  "tags": ["fraud", "high-value", "wire-fraud", "transactions"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | WHERE amount >= ?minAmount | KEEP @timestamp, account_id, recipient_account, amount, transaction_type, merchant_category, country_code, risk_score | SORT amount DESC | LIMIT ?limit",
    "params": {
      "startTime": {"type": "date", "description": "Start of the time range in ISO 8601 format"},
      "endTime": {"type": "date", "description": "End of the time range", "optional": true, "defaultValue": "now"},
      "minAmount": {"type": "float", "description": "Minimum transaction amount. Defaults to 50000.", "optional": true, "defaultValue": 50000},
      "limit": {"type": "integer", "description": "Max results. Defaults to 50.", "optional": true, "defaultValue": 50}
    }
  }
}'

# Account Profile
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_account_profile",
  "type": "esql",
  "description": "Builds a behavioral profile for a specific account over a lookback period.",
  "tags": ["fraud", "account", "profiling", "behavioral-analysis"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime | WHERE account_id == ?accountId | STATS tx_count=COUNT(*), total_volume=SUM(amount), avg_amount=AVG(amount), max_amount=MAX(amount), unique_recipients=COUNT_DISTINCT(recipient_account), unique_countries=COUNT_DISTINCT(country_code) BY transaction_type | SORT tx_count DESC",
    "params": {
      "accountId": {"type": "string", "description": "The account ID to profile"},
      "startTime": {"type": "date", "description": "Start of the lookback window. Defaults to last 30 days.", "optional": true, "defaultValue": "now-30d"}
    }
  }
}'


# Geo Anomaly
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_geo_anomaly",
  "type": "esql",
  "description": "Detects accounts transacting from multiple countries within a short time window.",
  "tags": ["fraud", "geo-anomaly", "account-takeover", "international"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | STATS country_count=COUNT_DISTINCT(country_code), countries=VALUES(country_code), tx_count=COUNT(*), total_amount=SUM(amount) BY account_id | WHERE country_count >= ?minCountries | SORT country_count DESC | LIMIT ?limit",
    "params": {
      "startTime": {"type": "date", "description": "Start of the time window in ISO 8601 format"},
      "endTime": {"type": "date", "description": "End of the time window", "optional": true, "defaultValue": "now"},
      "minCountries": {"type": "integer", "description": "Min distinct countries to flag. Defaults to 2.", "optional": true, "defaultValue": 2},
      "limit": {"type": "integer", "description": "Max results. Defaults to 25.", "optional": true, "defaultValue": 25}
    }
  }
}'

# Risk Score
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_risk_score_summary",
  "type": "esql",
  "description": "Summarizes accounts by average and maximum risk scores over a time period.",
  "tags": ["fraud", "risk-score", "triage", "prioritization"],
  "configuration": {
    "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime | WHERE risk_score >= ?minRiskScore | STATS avg_risk=AVG(risk_score), max_risk=MAX(risk_score), tx_count=COUNT(*), total_amount=SUM(amount) BY account_id | SORT max_risk DESC | LIMIT ?limit",
    "params": {
      "startTime": {"type": "date", "description": "Start of the lookback window. Defaults to last 7 days.", "optional": true, "defaultValue": "now-7d"},
      "minRiskScore": {"type": "float", "description": "Minimum risk score threshold (0-100). Defaults to 70.", "optional": true, "defaultValue": 70},
      "limit": {"type": "integer", "description": "Max accounts. Defaults to 20.", "optional": true, "defaultValue": 20}
    }
  }
}'

# Create Financial Fraud Skill
curl -X POST "http://localhost:30002/api/agent_builder/skills" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "financial-fraud-analyst-skill",
  "name": "Financial Fraud Analysis",
  "description": "Core fraud detection and AML investigation skill covering smurfing, velocity, geo anomalies, high-value review, account profiling, and risk triage.",
  "content": "You are an expert financial fraud analyst specializing in Anti-Money Laundering (AML) and transaction fraud detection. Your responsibilities: (1) Smurfing Detection - identify accounts splitting large transactions to evade reporting thresholds using fraud_smurfing_detection. (2) Velocity Analysis - flag accounts with abnormally high transaction rates using fraud_velocity_check. (3) High-Value Transaction Review - surface large or unusual transfers using fraud_high_value_transactions. (4) Account Profiling - build behavioral baselines using fraud_account_profile. (5) Geographic Anomaly Detection - identify multi-country activity in short timeframes using fraud_geo_anomaly. (6) Risk Score Triage - prioritize investigations using fraud_risk_score_summary. Always provide clear risk assessments with supporting evidence. Cite specific transaction counts, amounts, and timeframes. Recommend escalation paths for high-risk findings.",
  "tool_ids": [
    "fraud_smurfing_detection",
    "fraud_velocity_check",
    "fraud_high_value_transactions",
    "fraud_account_profile",
    "fraud_geo_anomaly"
  ]
}'

# Create Financial Fraud Agent
curl -X POST "http://localhost:30002/api/agent_builder/agents" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -u "fraud:hunter" \
  -d '{
  "id": "financial-fraud-analyst",
  "name": "Financial Fraud Analyst",
  "description": "I can help you detect and investigate financial fraud — including smurfing, velocity abuse, geographic anomalies, high-value wire transfers, and account risk profiling.",
  "labels": ["fraud", "aml", "financial", "security"],
  "avatar_color": "#FF4444",
  "avatar_symbol": "FF",
  "configuration": {
    "instructions": "You are an expert financial fraud analyst. Use your available tools to investigate transaction data, identify suspicious patterns, and provide clear risk assessments with supporting evidence. Always cite specific data points such as amounts, counts, timeframes, and account IDs in your findings.",
    "tools": [
      {
        "tool_ids": [
          "fraud_smurfing_detection",
          "fraud_velocity_check",
          "fraud_high_value_transactions",
          "fraud_account_profile",
          "fraud_geo_anomaly",
          "fraud_risk_score_summary",
          "platform.core.search",
          "platform.core.list_indices",
          "platform.core.get_index_mapping",
          "platform.core.get_document_by_id"
        ]
      },
      {
        "skill_ids": [
          "financial-fraud-analyst-skill"
        ]
      }
    ]
  }
}'

#curl -X POST "http://localhost:30002/api/agent_builder/agents"  -H "Content-Type: application/json" -H "kbn-xsrf: true"  -u "fraud:hunter" -d @/root/Fraud-Workshop/Agents/SARA.json

# Start data-gen installation
chmod +x /root/Fraud-Workshop/Scripts/fraud-gen.sh
chmod +x /root/Fraud-Workshop/Scripts/setup-brokerage-enrichment.sh
bash /root/Fraud-Workshop/Scripts/fraud-gen.sh
