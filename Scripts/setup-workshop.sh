## Set up environment variables
echo 'ELASTICSEARCH_USERNAME=elastic' >> /root/.env
kubectl get secret elasticsearch-es-elastic-user -n default -o go-template='ELASTICSEARCH_PASSWORD={{.data.elastic | base64decode}}' >> /root/.env
echo '' >> /root/.env
echo 'ELASTICSEARCH_URL="http://localhost:30920"' >> /root/.env
echo 'KIBANA_URL="http://localhost:30002"' >> /root/.env
echo 'BUILD_NUMBER="10"' >> /root/.env
echo 'ELASTIC_VERSION="9.1.0"' >> /root/.env
echo 'ELASTIC_APM_SERVER_URL=http://apm.default.svc:8200' >> /root/.env
echo 'ELASTIC_APM_SECRET_TOKEN=pkcQROVMCzYypqXs0b' >> /root/.env

## Set up environment
export $(cat /root/.env | xargs)

BASE64=$(echo -n "elastic:${ELASTICSEARCH_PASSWORD}" | base64)
KIBANA_URL_WITHOUT_PROTOCOL=$(echo $KIBANA_URL | sed -e 's#http[s]\?://##g')

## Add sdg user with superuser role
curl -X POST "http://localhost:30920/_security/user/fraud" -H "Content-Type: application/json" -u "elastic:${ELASTICSEARCH_PASSWORD}" -d '{
  "password" : "hunter",
  "roles" : [ "superuser" ],
  "full_name" : "Fraud Hunter",
  "email" : "fraud-hunter@omnicorp.co"
}'

## Enable workflows
curl -X POST "http://localhost:30002/api/kibana/settings" -H "Content-Type: application/json" -H "kbn-xsrf: true" -H "x-elastic-internal-origin: featureflag" -u "fraud:hunter"  -d '{
    "changes": {
      "workflows:ui:enabled": true
    }
  }'

## Create 'sar-reports' ingest pipeline and index template
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

## Create fraud-workshop data views
curl -X POST "http://localhost:30002/api/saved_objects/index-pattern/fraud-workshop" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{ "attributes": { "title": "fraud-workshop*", "name": "Fraud Workshop", "timeFieldName": "@timestamp"  }}'  
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


## Load saved-searches for assignment starts
curl -X POST "http://localhost:30002/api/saved_objects/_import" -H "kbn-xsrf: true" -u "fraud:hunter" -F "file=@/root/Fraud-Workshop/Saved-Searches/3-StartSavedSearches.ndjson"
clear
echo
echo "Saved searches loaded"
echo
clear

## Load DFA Workflow
curl -X POST "http://localhost:30002/api/workflows" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d @/root/Fraud-Workshop/Workflows/dfa-workflow.json

clear
echo
echo "Data Frame Analytics workflow loaded"
echo
clear

## Load High-Value Daily Aggregate Workflow
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

## Load component templates
curl -X PUT "http://localhost:30920/_component_template/fraud-workshop-logsdb-mappings" -H "Content-Type: application/json" -u "fraud:hunter" -d @/root/Fraud-Workshop/Index-Templates/Component-Templates/fraud-workshop-logsdb-mappings.json

clear
echo
echo "Component Template loaded"
echo
clear

## Load index templates
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

## Create enrichment policies
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
## Execute enrichment policies
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

## Create ingest pipelines
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

## Tool creation
# Smurfing Detection
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_smurfing_detection",
  "type": "esql",
  "description": "Detects smurfing patterns by identifying accounts that split large transactions into multiple smaller ones to evade detection thresholds.",
  "tags": ["fraud", "smurfing", "aml", "transaction-splitting"],
  "configuration": {
    "query": "FROM fraud-* | WHERE event.type == \"credit\" AND event.amount > 0 AND event.amount < 3000 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS small_deposits = COUNT(*), total_aggregated = SUM(event.amount), avg_deposit = AVG(event.amount) BY account.name | WHERE small_deposits >= 5 | SORT small_deposits DESC | LIMIT 50",
    "params": {
      "days": {
        "type": "integer",
        "description": "Lookback window in days"
      }
    }
  }
}'
# Veolicty Check
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -u "fraud:hunter" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -d '{
    "id": "fraud_velocity_check",
    "type": "esql",
    "description": "Checks transaction velocity for a given account — counts how many transactions occurred within a rolling time window and flags accounts exceeding a velocity threshold.",
    "tags": [
      "fraud",
      "velocity",
      "aml",
      "real-time"
    ],
    "configuration": {
      "query": "FROM fraud-* | WHERE event.amount > 0 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS txn_count = COUNT(*), total_volume = SUM(event.amount), avg_txn = AVG(event.amount), max_txn = MAX(event.amount) BY account.name | WHERE txn_count >= 10 | SORT total_volume DESC | LIMIT 50",
      "params": {
        "days": {
          "type": "integer",
          "description": "Lookback window in days"
        }
      }
    }
  }'

# Round Amount Transactions
curl -X POST "http://localhost:30002/api/agent_builder/tools" \
  -u "fraud:hunter" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -d '{
    "id": "fraud_round_amount_detection",
    "type": "esql",
    "description": "Detects suspicious ROUND-NUMBER transactions: transactions whose amount is an exact multiple of $1,000, a common indicator of artificial/laundering activity rather than organic spending. Returns account, count of round transactions and total. Use alongside other signals to corroborate laundering.",
    "tags": [
      "fraud",
      "high-value",
      "wire-fraud",
      "transactions"
    ],
    "configuration": {
      "query": "FROM fraud-* | WHERE event.amount > 0 AND (event.amount % 1000) == 0 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS round_txns = COUNT(*), total_round = SUM(event.amount) BY account.name | WHERE round_txns >= 3 | SORT round_txns DESC | LIMIT 50",
      "params": {
        "days": {
          "type": "integer",
          "description": "Lookback window in days"
        }
      }
    }
  }'

# Layering Detection
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d '{
  "id": "fraud_layering_detection",
  "type": "esql",
  "description": "Builds a behavioral profile for a specific account: total transaction count, total volume, average amount, unique counterparties, and transaction type breakdown over a lookback period.",
  "tags": ["fraud", "account", "profiling", "behavioral-analysis"],
  "configuration": {
    "query": "FROM fraud-* | WHERE wire.direction == \"outbound\" | STATS wire_count = COUNT(*), total_wired = SUM(event.amount) BY account.name, wire.outbound.bank_name, wire.outbound.country | SORT total_wired DESC | LIMIT 50",
    "params": {}
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
    "query": "FROM fraud-* | WHERE event.type == \"credit\" AND event.amount >= 8000 AND event.amount < 10000 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS near_threshold_txns = COUNT(*), total_amount = SUM(event.amount), min_amt = MIN(event.amount), max_amt = MAX(event.amount) BY account.name | WHERE near_threshold_txns >= 2 | SORT near_threshold_txns DESC | LIMIT 50",
    "params": {
      "days": {
        "type": "integer",
        "description": "Lookback window in days (e.g. 7, 30, 90)"
      }
    }
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

## Create Suspicious Activity Reporting Agent 
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


## Start data-gen installation
chmod +x /root/Fraud-Workshop/Scripts/fraud-gen.sh
bash /root/Fraud-Workshop/Scripts/fraud-gen.sh

## Free-form fraud search
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_transaction_search",
  "type": "index_search",
  "description": "Free-form search over the fraud-* transaction indices. Use to investigate a specific account, name, merchant, wire counterparty, SWIFT/routing number, or time range once a suspicious pattern is identified by the detection tools, or to pull supporting raw transaction records.",
  "tags": [],
  "configuration": {
    "pattern": "fraud-*"
  }
}'

## Consistent amounts under SAR threshold search
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_consistent_amounts_under_sar_threshold",
  "type": "esql",
  "description": "Artificially consistent amounts just under our SAR threshold and assign scores with a case statement.",
  "tags": [
    "fraud",
    "transactions"
  ],
  "configuration": {
    "query": "FROM fraud-workshop-wire-fraud
| WHERE account.event == \"deposit\" AND deposit.type == \"cash\"
  AND event.amount > 8000 AND event.amount < 10000
| STATS
    deposit_count = COUNT(),
    avg_amount = AVG(event.amount),
    variance_amount = VARIANCE(event.amount),
    std_dev_amount = STD_DEV(event.amount),
    min_amount = MIN(event.amount),
    max_amount = MAX(event.amount),
    total_deposited = SUM(event.amount)
  BY account.name
| WHERE deposit_count >= 5 // ACCOUNTS WITH MULTIPLE LARGE DEPOSITS
| EVAL
    coefficient_of_variation = std_dev_amount / avg_amount,
    amount_range = max_amount - min_amount,
    variance_threshold = CASE(variance_amount < 10000, \"SUSPICIOUS\", \"NORMAL\"),
    avg_deviation_from_threshold = ABS(avg_amount - 9500) / 9500
| INLINE STATS
    overall_avg_variance = AVG(variance_amount),
    overall_std_variance = STD_DEV(variance_amount),
    median_deposit_count = PERCENTILE(deposit_count, 50)
| EVAL
    variance_z_score = (variance_amount - overall_avg_variance) / overall_std_variance,
    suspicion_score = CASE(
        variance_amount < 1000 AND avg_amount > 9000, 100, // HIGHLY SUSPICIOUS
        variance_amount < 5000 AND avg_amount > 8500, 75, // VERY SUSPICIOUS
        variance_amount < 10000 AND avg_amount > 8000, 50, // MODERATELY SUSPICIOUS
        25 // NORMAL ACTIVITY
    )
| WHERE suspicion_score >= 50
| SORT variance_amount ASC, coefficient_of_variation ASC",
    "params": {}
  }
}'

## Anomalous deposits as the bank is closing
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_deposit_timing_patterns",
  "type": "esql",
  "description": "Anomalous deposits occurring just as the bank would be closing",
  "tags": [
    "fraud",
    "transactions"
  ],
  "configuration": {
    "query": "FROM fraud-workshop-wire-fraud
| WHERE account.event == \"deposit\" AND event.amount > 8000
| EVAL
    hour_of_day = DATE_EXTRACT(\"HOUR_OF_DAY\", transaction.date),
    day_of_week = DATE_EXTRACT(\"DAY_OF_WEEK\", transaction.date),
    amount_bucket = ROUND(event.amount / 500) * 500
| STATS
    deposit_count = COUNT(),
    unique_hours = COUNT_DISTINCT(hour_of_day),
    avg_amount = AVG(event.amount),
    hour_variance = VARIANCE(hour_of_day),
    std_dev_hour = STD_DEV(hour_of_day),
    most_common_hour = MEDIAN(hour_of_day),
    peak_hour_deposits = SUM(CASE(hour_of_day == 16, 1, 0)),  // 4 PM deposits
    total_amount = SUM(event.amount)
  BY account.name
| WHERE deposit_count >= 5
| INLINE STATS
    avg_hour_variance = AVG(hour_variance),
    std_hour_variance = STD_DEV(hour_variance),
    avg_unique_hours = AVG(unique_hours),
    peak_hour_threshold = PERCENTILE(peak_hour_deposits, 90)
| EVAL
    hour_consistency_score = CASE(
        hour_variance < 1.0, 4,  // Very consistent timing
        hour_variance < 4.0, 3,  // Moderately consistent
        hour_variance < 9.0, 2,  // Somewhat consistent
        1  // Random timing
    ),
    peak_hour_ratio = peak_hour_deposits / deposit_count,
    timing_suspicion = CASE(
        peak_hour_ratio > 0.7 AND hour_variance < 2.0, \"CRITICAL\",
        peak_hour_ratio > 0.5 AND hour_variance < 4.0, \"HIGH\",
        peak_hour_ratio > 0.3, \"MODERATE\",
        \"LOW\"
    )
| SORT hour_variance ASC, peak_hour_ratio DESC",
    "params": {}
  }
}'

## Outbound wires occurring at the same time
curl -X POST "http://localhost:30002/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "fraud_coordinated_wire_transfers",
  "type": "esql",
  "description": "Outbound wires being coordinated at around the same time",
  "tags": [
    "fraud",
    "wire-fraud",
    "transactions"
  ],
  "configuration": {
    "query": "FROM fraud-workshop-wire-fraud
| WHERE account.event == \"wire\" AND wire.direction == \"outbound\"
  AND event.amount > 7000 AND event.amount < 10000
| EVAL
    wire_hour = DATE_EXTRACT(\"HOUR_OF_DAY\", transaction.date),
    wire_minute = DATE_EXTRACT(\"minute_of_hour\", transaction.date),
    wire_second = DATE_EXTRACT(\"second_of_minute\", transaction.date),
    precise_time = wire_hour * 3600 + wire_minute * 60 + wire_second
| STATS
    wire_count = COUNT(),
    avg_amount = AVG(event.amount),
    amount_variance = VARIANCE(event.amount),
    amount_std_dev = STD_DEV(event.amount),
    time_variance = VARIANCE(precise_time),
    time_std_dev = STD_DEV(precise_time),
    unique_banks = COUNT_DISTINCT(wire.outbound.bank_name),
    total_wired = SUM(event.amount),
    earliest_wire = MIN(transaction.date),
    latest_wire = MAX(transaction.date)
  BY account.name, wire.outbound.bank_name
| WHERE wire_count >= 3  // Multiple wires to same bank
| INLINE STATS
    avg_amount_variance = AVG(amount_variance),
    avg_time_variance = AVG(time_variance),
    suspicious_bank_threshold = PERCENTILE(wire_count, 95)
| EVAL
    amount_consistency = CASE(
        amount_variance < avg_amount_variance * 0.1, \"HIGHLY_CONSISTENT\",
        amount_variance < avg_amount_variance * 0.5, \"CONSISTENT\",
        \"VARIABLE\"
    ),
    timing_consistency = CASE(
        time_variance < 3600, \"SAME_HOUR\",  // Within same hour
        time_variance < 86400, \"SAME_DAY\",  // Within same day
        \"SPREAD_OUT\"
    ),
    coordination_score = wire_count * 10 +
                        CASE(amount_consistency == \"HIGHLY_CONSISTENT\", 50,
                             amount_consistency == \"CONSISTENT\", 25, 0) +
                        CASE(timing_consistency == \"SAME_HOUR\", 40,
                             timing_consistency == \"SAME_DAY\", 20, 0)
| WHERE coordination_score >= 80  // High coordination threshold
| SORT coordination_score DESC, amount_variance ASC",
    "params": {}
  }
}'


## Create Financial Fraud Skill
curl -X POST "http://localhost:30002/api/agent_builder/skills" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Skills/financial_fraud_analyst.json

## Create International Wire Fraud Skill
curl -X POST "http://localhost:30002/api/agent_builder/skills" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" --data-binary @/root/Fraud-Workshop/Skills/international_wire_fraud.json


## Create Financial Fraud Analyst Agent
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
   "skill_ids": [ "financial-fraud-analysis", "graph-creation", "visualization-creation", "dashboard-management" ],
   "enable_elastic_capabilities": true
 }}
JSON

clear

python3 /root/Fraud-Workshop/Scripts/brokerage_workshop.py

