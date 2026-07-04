#!/usr/bin/env bash
###############################################################################
# setup-workshop.sh  (progress-bar edition)
#
# Runs the Fraud Workshop setup with:
#   - an Overall progress bar (across all steps)
#   - a Step progress bar (across the commands within the current step)
#   - NO per-command output on screen
#   - ALL command output redirected to /root/Fraud-Workshop/setup-workshop.log
#
# Each step is a shell function whose name/description is shown on screen while
# the underlying curl/etc. commands run silently and log to the file.
###############################################################################

set -uo pipefail

LOG_DIR="/root/Fraud-Workshop"
LOG_FILE="${LOG_DIR}/setup-workshop.log"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"   # truncate/create fresh log

###############################################################################
# Environment (this block itself is logged, run before the stepped section)
###############################################################################
{
  echo 'ELASTICSEARCH_USERNAME=elastic' >> /root/.env
  kubectl get secret elasticsearch-es-elastic-user -n default \
    -o go-template='ELASTICSEARCH_PASSWORD={{.data.elastic | base64decode}}' >> /root/.env
  echo '' >> /root/.env
  echo 'ELASTICSEARCH_URL="http://localhost:30920"' >> /root/.env
  echo 'KIBANA_URL="http://localhost:30002"' >> /root/.env
  echo 'BUILD_NUMBER="10"' >> /root/.env
  echo 'ELASTIC_VERSION="9.1.0"' >> /root/.env
  echo 'ELASTIC_APM_SERVER_URL=http://apm.default.svc:8200' >> /root/.env
  echo 'ELASTIC_APM_SECRET_TOKEN=pkcQROVMCzYypqXs0b' >> /root/.env

  export $(cat /root/.env | xargs)
  BASE64=$(echo -n "elastic:${ELASTICSEARCH_PASSWORD}" | base64)
  KIBANA_URL_WITHOUT_PROTOCOL=$(echo "$KIBANA_URL" | sed -e 's#http[s]\?://##g')
} >> "$LOG_FILE" 2>&1

BASE_URL="http://localhost:30920"
KBN_URL="http://localhost:30002"
USER="fraud:hunter"

###############################################################################
# Progress-bar machinery
###############################################################################
# Total number of steps is computed from the STEPS array below.
STEP_TOTAL=0          # set after STEPS is defined
STEP_INDEX=0          # steps completed so far
STEP_CUR_TOTAL=1      # number of commands in the current step
STEP_CUR_DONE=0       # commands completed in current step
CUR_STEP_NAME=""

BAR_WIDTH=40

draw_bars() {
  # $1 = overall fraction 0..100 ; $2 = step fraction 0..100
  local overall=$1 step=$2
  local ofill=$(( overall * BAR_WIDTH / 100 ))
  local sfill=$(( step    * BAR_WIDTH / 100 ))
  local obar sbar
  obar=$(printf '%*s' "$ofill" '' | tr ' ' '#')
  obar=$(printf '%-*s' "$BAR_WIDTH" "$obar")
  sbar=$(printf '%*s' "$sfill" '' | tr ' ' '#')
  sbar=$(printf '%-*s' "$BAR_WIDTH" "$sbar")

  # Move cursor up 3 lines (after first draw) and redraw in place.
  printf '\r\033[K Overall : [%s] %3d%%\n'  "$obar" "$overall"
  printf '\r\033[K Step    : [%s] %3d%%\n'  "$sbar" "$step"
  printf '\r\033[K %-.60s'                  "$CUR_STEP_NAME"
  printf '\033[2A\r'   # move back up 2 lines so next redraw overwrites
}

refresh() {
  local overall_frac step_frac
  if (( STEP_TOTAL > 0 )); then
    overall_frac=$(( STEP_INDEX * 100 / STEP_TOTAL ))
  else
    overall_frac=0
  fi
  if (( STEP_CUR_TOTAL > 0 )); then
    step_frac=$(( STEP_CUR_DONE * 100 / STEP_CUR_TOTAL ))
  else
    step_frac=100
  fi
  draw_bars "$overall_frac" "$step_frac"
}

# run: execute a command silently, log stdout+stderr, advance the step bar.
run() {
  echo "+ $*" >> "$LOG_FILE"
  "$@" >> "$LOG_FILE" 2>&1
  local rc=$?
  echo "  [exit $rc]" >> "$LOG_FILE"
  (( STEP_CUR_DONE++ ))
  refresh
  return 0   # never abort the whole run on a single command failure
}

# run_sh: like run() but for a shell snippet needing pipes/redirection.
run_sh() {
  echo "+ (shell) $1" >> "$LOG_FILE"
  bash -c "$1" >> "$LOG_FILE" 2>&1
  local rc=$?
  echo "  [exit $rc]" >> "$LOG_FILE"
  (( STEP_CUR_DONE++ ))
  refresh
  return 0
}

# run_file: execute a heredoc-authored snippet from a temp file. This avoids the
# quote-escaping hell of embedding heredocs inside bash -c '...'. Callers write
# the snippet to $SNIPPET (a temp file) and then call run_file.
run_file() {
  local snippet="$1"
  echo "+ (file) $snippet" >> "$LOG_FILE"
  bash "$snippet" >> "$LOG_FILE" 2>&1
  local rc=$?
  echo "  [exit $rc]" >> "$LOG_FILE"
  rm -f "$snippet"
  (( STEP_CUR_DONE++ ))
  refresh
  return 0
}

# begin_step: called at the start of each step function.
#   $1 = human-readable step name
#   $2 = number of commands (run/run_sh calls) in this step
begin_step() {
  CUR_STEP_NAME="$1"
  STEP_CUR_TOTAL="$2"
  STEP_CUR_DONE=0
  echo "" >> "$LOG_FILE"
  echo "===== STEP $((STEP_INDEX+1))/${STEP_TOTAL}: $1 =====" >> "$LOG_FILE"
  refresh
}

end_step() {
  (( STEP_INDEX++ ))
  refresh
}

###############################################################################
# STEP FUNCTIONS
# Each begins with '## ' in its display name (per your step convention) and
# declares how many commands it contains so the step bar is accurate.
###############################################################################

step_add_fraud_user() {
  begin_step "## Add fraud user with superuser role" 1
  run curl -s -X POST "${BASE_URL}/_security/user/fraud" \
    -H "Content-Type: application/json" -u "elastic:${ELASTICSEARCH_PASSWORD}" -d '{
      "password" : "hunter",
      "roles" : [ "superuser" ],
      "full_name" : "Fraud Hunter",
      "email" : "fraud-hunter@omnicorp.co"
    }'
  end_step
}

step_enable_workflows() {
  begin_step "## Enable workflows UI" 1
  run curl -s -X POST "${KBN_URL}/api/kibana/settings" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" \
    -H "x-elastic-internal-origin: featureflag" -u "$USER" -d '{
      "changes": { "workflows:ui:enabled": true }
    }'
  end_step
}

step_ingest_pipelines_sar() {
  begin_step "## Create sar-reports ingest pipelines and index template" 5
  run curl -s -X PUT "${BASE_URL}/_ingest/pipeline/sar-reports" \
    -H "Content-Type: application/x-ndjson" -u "$USER" \
    -d @/root/Fraud-Workshop/Ingest-Pipelines/sar-reports.json
  run curl -s -X PUT "${BASE_URL}/_ingest/pipeline/brokerage-final" \
    -H "Content-Type: application/x-ndjson" -u "$USER" \
    -d @/root/Fraud-Workshop/Ingest-Pipelines/brokerage-final.json
  run curl -s -X PUT "${BASE_URL}/_ingest/pipeline/enrich-brokers" \
    -H "Content-Type: application/x-ndjson" -u "$USER" \
    -d @/root/Fraud-Workshop/Ingest-Pipelines/enrich-brokers.json
  run curl -s -X PUT "${BASE_URL}/_ingest/pipeline/brokerage-detection-enrich" \
    -H "Content-Type: application/x-ndjson" -u "$USER" \
    -d @/root/Fraud-Workshop/Ingest-Pipelines/brokerage-detection-enrich.json
  run curl -s -X POST "${BASE_URL}/_index_template/sar-reports" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/sar-reports.json
  end_step
}

step_data_views() {
  begin_step "## Create fraud-workshop data views" 6
  run curl -s -X POST "${KBN_URL}/api/saved_objects/index-pattern/fraud-workshop" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d '{ "attributes": { "title": "fraud-workshop*", "name": "Fraud Workshop", "timeFieldName": "@timestamp" }}'
  run curl -s -X POST "${KBN_URL}/api/saved_objects/index-pattern/fraud-workshop-money-laundering" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d '{ "attributes": { "title": "fraud-workshop-money-laundering*", "name": "Money-Laundering", "timeFieldName": "@timestamp" }}'
  run curl -s -X POST "${KBN_URL}/api/saved_objects/index-pattern/fraud-workshop-wire-fraud" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d '{ "attributes": { "title": "fraud-workshop-wire-fraud*", "name": "Wire-Fraud", "timeFieldName": "@timestamp" }}'
  run curl -s -X POST "${KBN_URL}/api/saved_objects/index-pattern/fraud-workshop-smurfing" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d '{ "attributes": { "title": "fraud-workshop-smurfing*", "name": "Smurfing", "timeFieldName": "@timestamp" }}'
  run curl -s -X POST "${KBN_URL}/api/saved_objects/index-pattern/sar-reports" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d '{ "attributes": { "title": "sar-reports*", "name": "SAR Reports", "timeFieldName": "@timestamp" }}'
  run curl -s -X POST "${KBN_URL}/api/saved_objects/index-pattern/fraud-workshop-brokerage" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d '{ "attributes": { "title": "brokerage-workshop*", "name": "Brokerage Workshop", "timeFieldName": "@timestamp" }}'
  end_step
}

step_saved_searches() {
  begin_step "## Load saved-searches for assignment starts" 1
  run curl -s -X POST "${KBN_URL}/api/saved_objects/_import" -H "kbn-xsrf: true" -u "$USER" \
    -F "file=@/root/Fraud-Workshop/Saved-Searches/3-StartSavedSearches.ndjson"
  end_step
}

step_dfa_workflow() {
  begin_step "## Load DFA workflow" 1
  run curl -s -X POST "${KBN_URL}/api/workflows" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d @/root/Fraud-Workshop/Workflows/dfa-workflow.json
  end_step
}

step_highvalue_workflow() {
  begin_step "## Load High-Value Daily Aggregate workflow" 1
  run_sh '
    CONN_ID=$(curl -s "http://localhost:30002/api/actions/connectors" \
      -H "kbn-xsrf: true" -u "fraud:hunter" \
      | python3 -c '"'"'import json,sys; print(next(c["id"] for c in json.load(sys.stdin) if c["name"]=="openai-connector"))'"'"')
    echo "Resolved openai-connector -> $CONN_ID"
    sed "s/CONNECTOR_ID_PLACEHOLDER/$CONN_ID/g" /root/Fraud-Workshop/Workflows/highvalue-workflow-body.json \
      | curl -s -X POST "http://localhost:30002/api/workflows" \
        -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
        --data-binary @-
  '
  end_step
}

step_component_templates() {
  begin_step "## Load component templates" 1
  run curl -s -X PUT "${BASE_URL}/_component_template/fraud-workshop-logsdb-mappings" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/Component-Templates/fraud-workshop-logsdb-mappings.json
  end_step
}

step_index_templates() {
  begin_step "## Load index templates" 7
  run curl -s -X POST "${BASE_URL}/_index_template/enrich-accounts" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-accounts.json
  run curl -s -X POST "${BASE_URL}/_index_template/enrich-austinbanks" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-austinbanks.json
  run curl -s -X POST "${BASE_URL}/_index_template/enrich-austinstores" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-austinstores.json
  run curl -s -X POST "${BASE_URL}/_index_template/enrich-intbank" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-intbank.json
  run curl -s -X POST "${BASE_URL}/_index_template/enrich-brokers" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/Enrichment-Index-Templates/enrich-brokers.json
  run curl -s -X POST "${BASE_URL}/_index_template/fraud-workshop-logsdb" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/fraud-workshop-logsdb.json
  run curl -s -X POST "${BASE_URL}/_index_template/brokerage-workshop-logsdb" \
    -H "Content-Type: application/json" -u "$USER" \
    -d @/root/Fraud-Workshop/Index-Templates/brokerage-workshop-logsdb.json
  end_step
}

step_enrichment_data() {
  # Bulk-load each enrichment source (progress bar suppressed; response logged).
  begin_step "## Load enrichment data sources" 6
  local DATA_DIR="/root/Fraud-Workshop/Enrichment-Data"
  for pair in \
    "enrich-accounts:enrich-accounts.ndjson" \
    "enrich-austinbanks:enrich-austinbanks.ndjson" \
    "enrich-austinstores:enrich-austinstores.ndjson" \
    "enrich-intbank:enrich-intbank.ndjson" \
    "sar-reports:sar-reports.ndjson" \
    "enrich-brokers:enrich-brokers.ndjson" ; do
    local index="${pair%%:*}"
    local fname="${pair##*:}"
    run curl -s -X POST "${BASE_URL}/${index}/_bulk" \
      -H "Content-Type: application/x-ndjson" -u "$USER" \
      --data-binary "@${DATA_DIR}/${fname}"
  done
  end_step
}

step_enrichment_policies() {
  begin_step "## Create enrichment policies" 8
  for p in enrich-accounts enrich-austinbanks enrich-austinstores enrich-austinswift \
           enrich-inbounds enrich-intbank enrich-outbounds enrich-brokers ; do
    run curl -s -X PUT "${BASE_URL}/_enrich/policy/${p}" \
      -H "Content-Type: application/x-ndjson" -u "$USER" \
      --data-binary "@/root/Fraud-Workshop/Enrichment-Policies/${p}.json"
  done
  end_step
}

step_execute_enrichment_policies() {
  begin_step "## Execute enrichment policies" 8
  for p in enrich-accounts enrich-austinbanks enrich-austinstores enrich-austinswift \
           enrich-inbounds enrich-intbank enrich-outbounds enrich-brokers ; do
    run curl -s -X POST "${BASE_URL}/_enrich/policy/${p}/_execute" -u "$USER"
  done
  end_step
}

step_create_ingest_pipelines() {
  begin_step "## Create ingest pipelines" 12
  for p in atm-cleanup enrich-accounts enrich-austinbanks enrich-austinstores \
           enrich-austinswift enrich-inbound enrich-intbank enrich-outbound \
           enrich-outbounds enrich-brokers fraud-detection-enrich brokerage-detection-enrich ; do
    run curl -s -X PUT "${BASE_URL}/_ingest/pipeline/${p}" \
      -H "Content-Type: application/x-ndjson" -u "$USER" \
      -d "@/root/Fraud-Workshop/Ingest-Pipelines/${p}.json"
  done
  end_step
}


step_data_generator() {
  begin_step "## Start data-gen installation" 4
  run python3 /root/Fraud-Workshop/Scripts/wire-fraud.py
  run python3 /root/Fraud-Workshop/Scripts/money-laundering.py
  run python3 /root/Fraud-Workshop/Scripts/smurfing.py
  run python3 /root/Fraud-Workshop/Scripts/brokerage_workshop.py
  end_step
}

step_deploy_agent_tools() {
  begin_step "## Deploy agent tools" 9
  local T="${KBN_URL}/api/agent_builder/tools"

  # The index_search tool 'fraud_transaction_search' validates that 'fraud-*'
  # resolves to real sources AT CREATION TIME. If the data generators haven't
  # produced/refreshed any fraud-* docs yet, creation 400s with "No sources
  # found", which then cascades into the agent's tool_ids validation failing.
  # Wait (up to ~60s) for fraud-* to exist and hold at least one document.
  run_sh '
    for i in $(seq 1 30); do
      curl -s -X POST "http://localhost:30920/fraud-*/_refresh" -u "fraud:hunter" >/dev/null 2>&1
      CNT=$(curl -s "http://localhost:30920/fraud-*/_count" -u "fraud:hunter" \
              | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"count\",0))" 2>/dev/null || echo 0)
      echo "fraud-* doc count: ${CNT} (attempt ${i})"
      if [ "${CNT:-0}" -gt 0 ]; then echo "fraud-* is ready"; break; fi
      sleep 2
    done
  '

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_smurfing_detection",
    "type": "esql",
    "description": "Detects smurfing patterns by identifying accounts that split large transactions into multiple smaller ones to evade detection thresholds.",
    "tags": ["fraud", "smurfing", "aml", "transaction-splitting"],
    "configuration": {
      "query": "FROM fraud-* | WHERE event.type == \"credit\" AND event.amount > 0 AND event.amount < 3000 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS small_deposits = COUNT(*), total_aggregated = SUM(event.amount), avg_deposit = AVG(event.amount) BY account.name | WHERE small_deposits >= 5 | SORT small_deposits DESC | LIMIT 50",
      "params": { "days": { "type": "integer", "description": "Lookback window in days" } }
    }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_velocity_check",
    "type": "esql",
    "description": "Checks transaction velocity for a given account — counts how many transactions occurred within a rolling time window and flags accounts exceeding a velocity threshold.",
    "tags": ["fraud", "velocity", "aml", "real-time"],
    "configuration": {
      "query": "FROM fraud-* | WHERE event.amount > 0 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS txn_count = COUNT(*), total_volume = SUM(event.amount), avg_txn = AVG(event.amount), max_txn = MAX(event.amount) BY account.name | WHERE txn_count >= 10 | SORT total_volume DESC | LIMIT 50",
      "params": { "days": { "type": "integer", "description": "Lookback window in days" } }
    }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_round_amount_detection",
    "type": "esql",
    "description": "Detects suspicious ROUND-NUMBER transactions: transactions whose amount is an exact multiple of $1,000, a common indicator of artificial/laundering activity rather than organic spending. Returns account, count of round transactions and total. Use alongside other signals to corroborate laundering.",
    "tags": ["fraud", "high-value", "wire-fraud", "transactions"],
    "configuration": {
      "query": "FROM fraud-* | WHERE event.amount > 0 AND (event.amount % 1000) == 0 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS round_txns = COUNT(*), total_round = SUM(event.amount) BY account.name | WHERE round_txns >= 3 | SORT round_txns DESC | LIMIT 50",
      "params": { "days": { "type": "integer", "description": "Lookback window in days" } }
    }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_layering_detection",
    "type": "esql",
    "description": "Builds a behavioral profile for a specific account: total transaction count, total volume, average amount, unique counterparties, and transaction type breakdown over a lookback period.",
    "tags": ["fraud", "account", "profiling", "behavioral-analysis"],
    "configuration": {
      "query": "FROM fraud-* | WHERE wire.direction == \"outbound\" | STATS wire_count = COUNT(*), total_wired = SUM(event.amount) BY account.name, wire.outbound.bank_name, wire.outbound.country | SORT total_wired DESC | LIMIT 50",
      "params": {}
    }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_structuring_detection",
    "type": "esql",
    "description": "Detects STRUCTURING / CTR-evasion: accounts with multiple transactions just below the $10,000 reporting threshold ($8,000-$9,999) over the last N days. Returns the account, count of near-threshold transactions, and total moved. High counts indicate deliberate structuring to avoid Currency Transaction Reports.",
    "tags": ["fraud", "geo-anomaly", "account-takeover", "international"],
    "configuration": {
      "query": "FROM fraud-* | WHERE event.type == \"credit\" AND event.amount >= 8000 AND event.amount < 10000 AND DATE_DIFF(\"days\", transaction.date, NOW()) <= ?days | STATS near_threshold_txns = COUNT(*), total_amount = SUM(event.amount), min_amt = MIN(event.amount), max_amt = MAX(event.amount) BY account.name | WHERE near_threshold_txns >= 2 | SORT near_threshold_txns DESC | LIMIT 50",
      "params": { "days": { "type": "integer", "description": "Lookback window in days (e.g. 7, 30, 90)" } }
    }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud-high-value-daily-triage",
    "type": "workflow",
    "description": "Welcome Fraud analysis to focus investigations",
    "tags": ["security", "investigation", "workflows"],
    "configuration": { "workflow_id": "high-value-daily-aggregate-triage", "wait_for_completion": true }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_transaction_search",
    "type": "index_search",
    "description": "Free-form search over the fraud-* transaction indices. Use to investigate a specific account, name, merchant, wire counterparty, SWIFT/routing number, or time range once a suspicious pattern is identified by the detection tools, or to pull supporting raw transaction records.",
    "tags": [],
    "configuration": { "pattern": "fraud-*" }
  }'

  run curl -s -X POST "$T" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" -d '{
    "id": "fraud_geo_anomaly",
    "type": "esql",
    "description": "Detects geographic anomalies by identifying accounts transacting from multiple countries within a short time window — a common indicator of account takeover or card fraud.",
    "tags": ["fraud", "geo-anomaly", "account-takeover", "international"],
    "configuration": {
      "query": "FROM fraud-workshop* | WHERE @timestamp >= ?startTime AND @timestamp <= ?endTime | STATS country_count=COUNT_DISTINCT(wire.outbound.country_code), tx_count=COUNT(*), total_amount=SUM(event.amount) BY account.name | WHERE country_count >= ?minCountries | SORT country_count DESC | LIMIT ?limit",
      "params": {
        "startTime": { "type": "date", "description": "Start of the time window in ISO 8601 format" },
        "endTime": { "type": "date", "description": "End of the time window in ISO 8601 format", "optional": true, "defaultValue": "now" },
        "minCountries": { "type": "integer", "description": "Minimum number of distinct countries to flag an account. Defaults to 2.", "optional": true, "defaultValue": 2 },
        "limit": { "type": "integer", "description": "Maximum number of results to return. Defaults to 25.", "optional": true, "defaultValue": 25 }
      }
    }
  }'
  end_step
}

step_create_fraud_skill_and_agent() {
  begin_step "## Create Financial Fraud skill and agent" 2

  # 1) Skill
  run curl -s -X POST "${KBN_URL}/api/agent_builder/skills" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    --data-binary @/root/Fraud-Workshop/Skills/financial_fraud_analyst.json

  # 2) Agent — authored to a temp file with a real heredoc, then executed.
  #    (Wrapping a heredoc inside bash -c mangles the quoting and silently fails,
  #    which is why the agent didn't get created before.)
  local snippet
  snippet="$(mktemp /tmp/ffa-agent.XXXXXX.sh)"
  cat > "$snippet" <<'SNIPPET'
curl -s -X POST "http://localhost:30002/api/agent_builder/agents" \
  -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d @- <<'JSON'
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
  }
}
JSON
SNIPPET
  run_file "$snippet"

  end_step
}

step_create_sar_agent() {
  begin_step "## Create Suspicious Activity Reporting agent (SARA)" 1
  run curl -s -X POST "${KBN_URL}/api/agent_builder/agents" \
    -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "$USER" \
    -d "@/root/Fraud-Workshop/Agents/SARA.json"
  end_step
}

###############################################################################
# STEP REGISTRY  — order matters
###############################################################################
STEPS=(
  step_add_fraud_user
  step_enable_workflows
  step_ingest_pipelines_sar
  step_data_views
  step_saved_searches
  step_dfa_workflow
  step_highvalue_workflow
  step_component_templates
  step_index_templates
  step_enrichment_data
  step_enrichment_policies
  step_execute_enrichment_policies
  step_create_ingest_pipelines
  step_data_generator
  step_deploy_agent_tools
  step_create_fraud_skill_and_agent
  step_create_sar_agent
)
STEP_TOTAL=${#STEPS[@]}

###############################################################################
# RUN
###############################################################################
clear
echo "Fraud Workshop setup starting..."
echo "All command output is being written to: ${LOG_FILE}"
echo
echo    # reserve 3 lines for the two bars + current-step label
echo
echo
printf '\033[3A'   # move cursor back up to the first reserved line

refresh   # initial draw (0% / 0%)

for step_fn in "${STEPS[@]}"; do
  "$step_fn"
done

# Force final 100% render and drop below the bars.
STEP_INDEX=$STEP_TOTAL
CUR_STEP_NAME="Complete"
refresh
printf '\033[3B\n'   # move cursor below the bar block

echo
echo "Setup complete. Full log: ${LOG_FILE}"
