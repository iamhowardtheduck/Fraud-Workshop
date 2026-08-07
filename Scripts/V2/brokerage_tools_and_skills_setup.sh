# brokerage_securities_layering tool
curl -X POST "https://kb.elastic.lab:443/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "brokerage_securities_layering",
  "type": "esql",
  "description": "Detects securities-based layering in brokerage accounts: large inbound wire, funds spread across securities purchases, full liquidation, then proceeds wired out. Returns per-account wired_in/bought/liquidated/wired_out totals, payout ratio, lifecycle span in days, servicing brokers, and whether a flagged broker was involved. Slow multi-day variant; for rapid same-day flips use brokerage_unexplained_wealth instead.",
  "tags": ["fraud", "brokerage", "layering"],
  "configuration": {
    "query": "FROM brokerage-workshop* | WHERE @timestamp >= ?start AND @timestamp < ?end | EVAL amt = TO_DOUBLE(event.amount), in_wire = CASE(account.event == \"wire\" AND wire.direction == \"inbound\", amt, 0.0), out_wire = CASE(account.event == \"wire\" AND wire.direction == \"outbound\", amt, 0.0), buy_amt = CASE(account.event == \"buy\", amt, 0.0), liq_amt = CASE(account.event == \"liquidation\", amt, 0.0) | STATS wired_in = SUM(in_wire), bought = SUM(buy_amt), liquidated = SUM(liq_amt), wired_out = SUM(out_wire), first_event = MIN(@timestamp), last_event = MAX(@timestamp), events = COUNT(*), brokers = VALUES(broker.name), flagged_broker_involved = MAX(CASE(broker.flagged == true, 1, 0)), symbols = VALUES(security.symbol) BY account.name | WHERE wired_in >= ?min_wired_in AND liquidated > 0 AND wired_out > 0 | EVAL payout_ratio = ROUND(wired_out / wired_in, 2), lifecycle_days = DATE_DIFF(\"days\", first_event, last_event) | WHERE lifecycle_days >= ?min_lifecycle_days | SORT wired_out DESC | LIMIT 25",
    "params": {
      "start": { "type": "date", "description": "Window start, ISO date. Workshop data always covers at least the trailing 9 days." },
      "end": { "type": "date", "description": "Window end (exclusive), ISO date. Typically tomorrow date for current data." },
      "min_wired_in": { "type": "integer", "description": "Minimum inbound wire total (USD) to qualify. Default 100000 unless the analyst specifies otherwise." },
      "min_lifecycle_days": { "type": "integer", "description": "Minimum days between first and last event; excludes rapid-flip accounts covered by the unexplained-wealth tool. Default 2." }
    }
  }
}'

# brokerage_wash_trading tool
curl -X POST "https://kb.elastic.lab:443/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "brokerage_wash_trading",
  "type": "esql",
  "description": "Detects wash trading rings: buy and sell legs sharing the same trade.id across two different accounts within seconds — coordinated volume with no net position change, typically in thin/microcap symbols. Returns per-symbol wash trade counts, total notional, the full participant ring, trading date range, and whether a flagged broker executed the trades. Highly specific: legitimate trading never shares a trade.id across accounts.",
  "tags": ["fraud", "brokerage", "wash-trading"],
  "configuration": {
    "query": "FROM brokerage-workshop* | WHERE @timestamp >= ?start AND @timestamp < ?end AND account.event IN (\"buy\", \"sell\") AND trade.id IS NOT NULL | STATS legs = COUNT(*), traders = COUNT_DISTINCT(account.name), pair = VALUES(account.name), symbol = VALUES(security.symbol), amt = MAX(TO_DOUBLE(event.amount)), span_sec = DATE_DIFF(\"seconds\", MIN(@timestamp), MAX(@timestamp)), flagged_broker = MAX(CASE(broker.flagged == true, 1, 0)), first_ts = MIN(@timestamp) BY trade.id | WHERE legs >= 2 AND traders >= 2 AND span_sec <= ?max_leg_seconds | STATS wash_trades = COUNT(*), total_notional = SUM(amt), participants = VALUES(pair), flagged_broker_involved = MAX(flagged_broker), first_trade = MIN(first_ts), last_trade = MAX(first_ts) BY symbol | SORT total_notional DESC | LIMIT 25",
    "params": {
      "start": { "type": "date", "description": "Window start, ISO date. Workshop data always covers at least the trailing 9 days." },
      "end": { "type": "date", "description": "Window end (exclusive), ISO date." },
      "max_leg_seconds": { "type": "integer", "description": "Maximum seconds between the buy and sell legs of a matched trade. Default 120 (generator pairs legs 2-90 seconds apart)." }
    }
  }
}'

# brokerage_unexplained_wealth
curl -X POST "https://kb.elastic.lab:443/api/agent_builder/tools" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" \
  -d '{
  "id": "brokerage_unexplained_wealth",
  "type": "esql",
  "description": "Detects unexplained-wealth rapid flips: large inbound wire, near-immediate liquidation (within a configurable hold time), and proceeds wired out at a high payout ratio. Returns per-account amounts, hold time in hours, payout ratio, servicing brokers, and flagged-broker involvement. Rapid variant of the layering pattern; for slow multi-day layering use brokerage_securities_layering.",
  "tags": ["fraud", "brokerage", "unexplained-wealth"],
  "configuration": {
    "query": "FROM brokerage-workshop* | WHERE @timestamp >= ?start AND @timestamp < ?end | EVAL amt = TO_DOUBLE(event.amount), in_amt = CASE(account.event == \"wire\" AND wire.direction == \"inbound\", amt, 0.0), liq_amt = CASE(account.event == \"liquidation\", amt, 0.0), out_amt = CASE(account.event == \"wire\" AND wire.direction == \"outbound\", amt, 0.0), in_ts = CASE(account.event == \"wire\" AND wire.direction == \"inbound\", @timestamp, null), liq_ts = CASE(account.event == \"liquidation\", @timestamp, null) | STATS wired_in = SUM(in_amt), liquidated = SUM(liq_amt), wired_out = SUM(out_amt), first_inbound = MIN(in_ts), liq_time = MAX(liq_ts), brokers = VALUES(broker.name), flagged_broker_involved = MAX(CASE(broker.flagged == true, 1, 0)) BY account.name | WHERE wired_in > 0 AND liquidated > 0 AND wired_out > 0 | EVAL hold_hours = DATE_DIFF(\"hours\", first_inbound, liq_time), payout_ratio = ROUND(wired_out / wired_in, 2) | WHERE hold_hours >= 0 AND hold_hours <= ?max_hold_hours | SORT wired_in DESC | LIMIT 25",
    "params": {
      "start": { "type": "date", "description": "Window start, ISO date. Workshop data always covers at least the trailing 9 days." },
      "end": { "type": "date", "description": "Window end (exclusive), ISO date." },
      "max_hold_hours": { "type": "integer", "description": "Maximum hours between inbound wire and liquidation to qualify as a rapid flip. Default 36 (generator liquidates within 36 hours)." }
    }
  }
}'

# Create Securities Irregularities Skill
curl -X POST "https://kb.elastic.lab:443/api/agent_builder/skills" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" --data-binary @/home/elastic/Fraud-Workshop/Skills/securities_irregularities.json

## Create Securities Irregularities Agent
curl -X POST "https://kb.elastic.lab:443/api/agent_builder/agents" -H "Content-Type: application/json" -H "kbn-xsrf: true" -u "fraud:hunter" -d @- <<'JSON'
{
  "id": "securities-irregularities-agentt",
  "name": "Financial Fraud Analyst",
  "description": "I can help you detect and investigate financial fraud — including smurfing, velocity abuse, geographic anomalies, high-value wire transfers, and account risk profiling.",
  "labels": ["fraud", "brokerage", "financial", "security"],
  "avatar_color": "#FF4444",
  "avatar_symbol": "FF",
  "configuration": {
    "instructions": "You are a securities fraud analyst specializing in brokerage account irregularities in the brokerage-workshop* indices. Your capabilities and rules of engagement are defined by your securities_irregularities skill: three detection tools (securities layering, wash trading, unexplained wealth) and one case-escalation tool with side effects.\n\nMethod: for any inquiry about suspicious brokerage activity, run the relevant detections per the skill's selection rules over an explicit time window (workshop data covers at least the trailing 9 days - compute the window, never guess). Interpret results against the skill's signatures, rank findings by notional and indicator strength, and report a ranked list: account, scenario matched, key metrics, brokers involved (flag compliance-flagged brokers prominently), one-line rationale, and a qualitative risk level.\n\nWhen an investigation concludes with concrete findings, offer to open a case; on the analyst's confirmation - or when they asked up front for a case - call securities_case_escalation exactly once with your top 3-5 findings per the skill's escalation format. Check platform.core.cases first and mention any existing case that appears to cover the same accounts rather than duplicating it.\n\nFindings are investigative leads for human review, never accusations - phrase rationales as \"consistent with\" or \"exhibits indicators associated with\"",
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
          "platform.core.index_explorer"
        ]
      }
    ],
   "skill_ids": [ "securities_irregularities", "graph-creation", "visualization-creation", "dashboard-management", "fraud_rate_and_report" ],
   "enable_elastic_capabilities": false
 }}
JSON
