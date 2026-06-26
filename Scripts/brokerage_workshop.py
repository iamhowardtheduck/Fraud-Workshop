#!/usr/bin/env python3
"""
AML Brokerage/Investment Fraud Workshop Generator - Single File Version
Parallel to fraud_workshop.py, targeting investment banks & brokerages
(Vanguard / E*TRADE style) with much higher dollar values.

Run from:        /root/Fraud-Workshop
Elasticsearch:   http://localhost:30920
Index:           brokerage-workshop
Credentials:     fraud/hunter
Workers:         16, Events: 10,000
Business Hours:  7 AM - 9 PM (7x volume)

Embedded fraud scenarios:
  1. Securities-based layering  (large inbound wire -> buy securities -> liquidate -> wire out)
  2. Wash trading               (coordinated buy/sell between linked accounts, no net position change)
  3. Unexplained wealth         (rapid in/out: wire in -> immediate liquidation -> wire out to high-risk bank)
"""

import json
import random
import time
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- dependency checks -----------------------------------------------------
missing_packages = []
try:
    import pandas as pd
except ImportError:
    missing_packages.append('pandas')

try:
    import numpy as np
except ImportError:
    missing_packages.append('numpy')

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    missing_packages.append('elasticsearch')

if missing_packages:
    print("Missing required packages. Install with:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)


# ---- configuration ---------------------------------------------------------
@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration - hard-coded for the workshop box."""
    host: str = "http://localhost:30920"
    index_name: str = "brokerage-workshop"        # <-- new separate index
    username: str = "fraud"
    password: str = "hunter"
    workers: int = 16
    events_per_day: int = 10000
    pipeline: str = "brokerage-detection-enrich"   # parallel enrich pipeline
    verify_certs: bool = False
    timeout: int = 30


@dataclass
class BrokerageFraudConfig:
    """Fraud parameters tuned for brokerage-scale dollar values."""

    # ---- Scenario 1: Securities-based layering -----------------------------
    # large inbound wire, then spread into securities, then liquidate + wire out
    layering_inbound_min: float = 500_000.0
    layering_inbound_max: float = 2_000_000.0
    layering_accounts: int = 3
    layering_days_span: int = 7
    layering_payout_ratio: float = 0.93   # fraction wired out after fees/slippage

    # ---- Scenario 2: Wash trading ------------------------------------------
    # coordinated buy/sell among linked accounts -> volume w/o net position
    wash_accounts: int = 4
    wash_trade_min: float = 50_000.0
    wash_trade_max: float = 400_000.0
    wash_rounds: int = 12                 # buy/sell pairs across the window
    wash_days_span: int = 7

    # ---- Scenario 3: Unexplained wealth / rapid in-out ---------------------
    uw_inbound_min: float = 250_000.0
    uw_inbound_max: float = 1_500_000.0
    uw_hold_hours_max: int = 36           # liquidated fast
    uw_payout_ratio: float = 0.97

    # ---- noise event distribution (must sum to <= 1.0; purchase = remainder)
    buy_percentage: float = 0.34
    sell_percentage: float = 0.30
    dividend_percentage: float = 0.08
    fee_percentage: float = 0.08
    wire_percentage: float = 0.12
    # remainder -> 'transfer' (ACAT/journal) events
    international_wire_percentage: float = 0.12

    # ---- business hours (markets skew to trading hours, kept 7-21 for noise)
    business_start_hour: int = 7
    business_end_hour: int = 21
    business_hours_multiplier: float = 7.0


# ---- securities universe (symbol, type, rough price band) ------------------
EQUITIES = [
    ("AAPL", "equity", 170, 230), ("MSFT", "equity", 380, 470),
    ("AMZN", "equity", 150, 210), ("NVDA", "equity", 95, 150),
    ("GOOGL", "equity", 150, 200), ("TSLA", "equity", 170, 360),
    ("JPM", "equity", 190, 270), ("BRK.B", "equity", 400, 480),
]
ETFS = [
    ("VOO", "etf", 480, 560), ("VTI", "etf", 250, 300),
    ("SPY", "etf", 520, 600), ("QQQ", "etf", 440, 530),
    ("BND", "etf", 70, 76),
]
# thinly-traded / microcap-ish names handy for wash-trade scenarios
THIN = [
    ("ZXCO", "equity", 2, 9), ("BLTX", "equity", 1, 6),
    ("QNTM", "equity", 3, 12), ("HLIO", "equity", 4, 15),
]
ALL_SECURITIES = EQUITIES + ETFS + THIN


def pick_security(pool=None):
    sym, stype, lo, hi = random.choice(pool or ALL_SECURITIES)
    return sym, stype, round(random.uniform(lo, hi), 2)


# ---- event schema (superset of the bank schema) ----------------------------
@dataclass
class TransactionEvent:
    """Reuses the bank fields and extends with brokerage-specific ones.

    Reused from fraud_workshop.py:
      accountID, event_amount, event_type, account_type, account_event,
      transaction_date, timestamp, wire_direction, intbankID, txbankId, addressId
    New for brokerage:
      security_symbol, security_type, quantity, price_per_unit,
      settlement_date, counterparty_account, brokerID, tradeID
    """
    accountID: int
    event_amount: float
    event_type: str            # debit or credit
    account_type: str          # brokerage, ira, roth_ira, margin, money market
    account_event: str         # buy, sell, dividend, fee, wire, transfer, liquidation
    transaction_date: str
    timestamp: str

    # reused optional enrichment IDs (same semantics as bank version)
    wire_direction: Optional[str] = None
    intbankID: Optional[int] = None
    txbankId: Optional[int] = None
    addressId: Optional[int] = None

    # new brokerage fields
    security_symbol: Optional[str] = None
    security_type: Optional[str] = None
    quantity: Optional[float] = None
    price_per_unit: Optional[float] = None
    settlement_date: Optional[str] = None
    counterparty_account: Optional[int] = None
    brokerID: Optional[int] = None
    tradeID: Optional[str] = None


class BusinessHoursGenerator:
    """Identical weighting logic to the bank generator."""

    def __init__(self, config: BrokerageFraudConfig):
        self.config = config

    def get_weighted_hour(self) -> int:
        hours = list(range(24))
        weights = []
        for hour in hours:
            if self.config.business_start_hour <= hour < self.config.business_end_hour:
                weights.append(self.config.business_hours_multiplier)
            else:
                weights.append(1.0)
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(hours, weights=weights)[0]

    def generate_business_weighted_datetime(self, base_date: datetime, days_back_range: int = 8) -> datetime:
        days_back = random.randint(0, days_back_range)
        hour = self.get_weighted_hour()
        return (base_date - timedelta(days=days_back)).replace(
            hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
        )


# ---- Elasticsearch ingest (unchanged structure, new mapping) ---------------
class ElasticsearchIngester:
    def __init__(self, es_config: ElasticsearchConfig):
        self.config = es_config
        self.es = self._create_elasticsearch_client()

    def _create_elasticsearch_client(self) -> Optional[Elasticsearch]:
        try:
            client_config = {
                'hosts': [self.config.host],
                'verify_certs': self.config.verify_certs,
                'ssl_show_warn': False,
                'request_timeout': self.config.timeout,
            }
            if self.config.username and self.config.password:
                try:
                    client_config['basic_auth'] = (self.config.username, self.config.password)
                    return Elasticsearch(**client_config)
                except TypeError:
                    client_config['http_auth'] = (self.config.username, self.config.password)
                    return Elasticsearch(**client_config)
            return Elasticsearch(**client_config)
        except Exception as e:
            logger.error(f"Failed to create Elasticsearch client: {e}")
            return None

    def test_connection(self) -> bool:
        if not self.es:
            logger.error("Elasticsearch client not initialized")
            return False
        try:
            info = self.es.info()
            if hasattr(info, 'body'):
                version = info.body.get('version', {}).get('number', 'unknown')
            else:
                version = info.get('version', {}).get('number', 'unknown')
            logger.info(f"Connected to Elasticsearch at {self.config.host}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Index: {self.config.index_name}")
            logger.info(f"   Workers: {self.config.workers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            logger.error(f"   Host: {self.config.host}")
            logger.error(f"   Credentials: {self.config.username} / {self.config.password}")
            return False

    def create_index_if_not_exists(self):
        """Let the index template own mapping/pipeline/mode.

        The brokerage-workshop-logsdb index template (loaded by
        setup-brokerage-enrichment.sh) defines the mapping, sets
        default_pipeline=brokerage-detection-enrich, and puts the index in
        logsdb + synthetic _source mode. We must NOT create a competing manual
        mapping here, or the flat raw fields collide with the template's dotted
        fields and synthetic-source rules -> bulk 500s.

        If a matching template exists, do nothing and let the first write
        auto-create the index from it. If no template exists, fall back to a
        minimal index so the script still works standalone.
        """
        if not self.es:
            return
        try:
            if self.es.indices.exists(index=self.config.index_name):
                logger.info(f"Index '{self.config.index_name}' already exists")
                return

            # Is there an index template matching our index name?
            has_template = False
            try:
                resp = self.es.indices.simulate_index_template(name=self.config.index_name)
                body = resp.body if hasattr(resp, 'body') else resp
                tmpl = body.get('template', {}) if isinstance(body, dict) else {}
                # a real match returns mappings/settings under 'template'
                has_template = bool(tmpl.get('mappings') or tmpl.get('settings'))
            except Exception:
                has_template = False

            if has_template:
                logger.info(f"Index template matches '{self.config.index_name}'; "
                            f"letting first write auto-create the index from it")
                return

            # Standalone fallback: minimal index, dynamic mapping, no synthetic source
            logger.info(f"No matching template; creating minimal fallback index "
                        f"'{self.config.index_name}'")
            self.es.indices.create(
                index=self.config.index_name,
                body={"settings": {"number_of_shards": 1, "number_of_replicas": 0}},
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")

    def bulk_index_events(self, events: List[dict], chunk_size: int = 500) -> tuple:
        if not self.es or not events:
            return 0, len(events) if events else 0

        def generate_docs():
            for event in events:
                doc = {'_index': self.config.index_name, '_source': event}
                if self.config.pipeline:
                    doc['pipeline'] = self.config.pipeline
                yield doc

        try:
            # raise_on_error=False so we can SEE why docs fail instead of a
            # generic "N document(s) failed to index" exception.
            success_count, errors = bulk(
                self.es, generate_docs(),
                chunk_size=chunk_size, request_timeout=self.config.timeout,
                max_retries=3, initial_backoff=2, max_backoff=600,
                raise_on_error=False, raise_on_exception=False,
            )
            failed_count = len(errors) if errors else 0
            if failed_count:
                # log the first few distinct reasons so the cause is obvious
                seen = set()
                for err in (errors or [])[:50]:
                    info = err.get('index', err.get('create', {})) if isinstance(err, dict) else {}
                    reason = ''
                    if isinstance(info, dict):
                        e = info.get('error', {})
                        reason = f"{e.get('type','?')}: {e.get('reason','?')}" if isinstance(e, dict) else str(e)
                    if reason and reason not in seen:
                        seen.add(reason)
                        logger.error(f"  reject reason: {reason}")
                logger.error(f"Bulk: {success_count} ok, {failed_count} failed "
                             f"({len(seen)} distinct reason(s) above)")
            return success_count, failed_count
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return 0, len(events)


# ---- fraud + noise generation ----------------------------------------------
class BrokerageDataGenerator:
    def __init__(self, fraud_config: BrokerageFraudConfig, es_config: ElasticsearchConfig):
        self.fraud_config = fraud_config
        self.es_config = es_config
        self.business_hours = BusinessHoursGenerator(fraud_config)
        self.ingester = ElasticsearchIngester(es_config)
        self._trade_seq = 0

    def _next_trade_id(self) -> str:
        self._trade_seq += 1
        return f"T{datetime.now().strftime('%y%j')}{self._trade_seq:07d}"

    def _settle(self, ts: datetime, days: int = 2) -> str:
        # T+1 for equities is current US standard; keep configurable
        return (ts + timedelta(days=days)).strftime('%Y-%m-%d')

    # ---- Scenario 1: securities-based layering -----------------------------
    def generate_layering_scenario(self) -> List[TransactionEvent]:
        evts = []
        base = datetime.now()
        accounts = [random.randint(1, 35000) for _ in range(self.fraud_config.layering_accounts)]
        broker = random.randint(1, 40)
        logger.info("Layering scenario:")
        logger.info(f"   Accounts: {accounts}  Broker: {broker}")

        for acct in accounts:
            # 1) large inbound wire
            inbound = round(random.uniform(self.fraud_config.layering_inbound_min,
                                           self.fraud_config.layering_inbound_max), 2)
            t0 = self.business_hours.generate_business_weighted_datetime(
                base - timedelta(days=self.fraud_config.layering_days_span), 0)
            evts.append(TransactionEvent(
                accountID=acct, event_amount=inbound, event_type='credit',
                account_type='brokerage', account_event='wire',
                transaction_date=t0.strftime('%Y-%m-%d'), timestamp=t0.isoformat(),
                wire_direction='inbound', intbankID=random.randint(1, 25), brokerID=broker,
            ))

            # 2) spread across several securities buys over the window
            remaining = inbound
            n_buys = random.randint(4, 8)
            for i in range(n_buys):
                chunk = round(remaining * random.uniform(0.10, 0.30), 2) if i < n_buys - 1 else round(remaining * 0.9, 2)
                if chunk < 1000:
                    continue
                remaining -= chunk
                sym, stype, px = pick_security(EQUITIES + ETFS)
                qty = round(chunk / px, 4)
                day_off = random.randint(0, self.fraud_config.layering_days_span - 1)
                tb = self.business_hours.generate_business_weighted_datetime(base - timedelta(days=day_off), 0)
                evts.append(TransactionEvent(
                    accountID=acct, event_amount=chunk, event_type='debit',
                    account_type='brokerage', account_event='buy',
                    transaction_date=tb.strftime('%Y-%m-%d'), timestamp=tb.isoformat(),
                    security_symbol=sym, security_type=stype, quantity=qty, price_per_unit=px,
                    settlement_date=self._settle(tb), brokerID=broker, tradeID=self._next_trade_id(),
                ))

            # 3) liquidate everything near the end of the window
            tl = self.business_hours.generate_business_weighted_datetime(base - timedelta(days=1), 0)
            sym, stype, px = pick_security(EQUITIES + ETFS)
            liq_amt = round(inbound * 0.98, 2)
            evts.append(TransactionEvent(
                accountID=acct, event_amount=liq_amt, event_type='credit',
                account_type='brokerage', account_event='liquidation',
                transaction_date=tl.strftime('%Y-%m-%d'), timestamp=tl.isoformat(),
                security_symbol=sym, security_type=stype, quantity=round(liq_amt / px, 4),
                price_per_unit=px, settlement_date=self._settle(tl),
                brokerID=broker, tradeID=self._next_trade_id(),
            ))

            # 4) wire proceeds out to a high-risk foreign bank
            tw = self.business_hours.generate_business_weighted_datetime(base, 0)
            evts.append(TransactionEvent(
                accountID=acct, event_amount=round(inbound * self.fraud_config.layering_payout_ratio, 2),
                event_type='debit', account_type='brokerage', account_event='wire',
                transaction_date=tw.strftime('%Y-%m-%d'), timestamp=tw.isoformat(),
                wire_direction='outbound', intbankID=random.randint(1, 25), brokerID=broker,
            ))

        logger.info(f"   Generated {len(evts)} layering events")
        return evts

    # ---- Scenario 2: wash trading ------------------------------------------
    def generate_wash_trading_scenario(self) -> List[TransactionEvent]:
        evts = []
        base = datetime.now()
        accounts = [random.randint(1, 35000) for _ in range(self.fraud_config.wash_accounts)]
        broker = random.randint(1, 40)
        sym, stype, px = pick_security(THIN)  # thin name = easier to move
        logger.info("Wash trading scenario:")
        logger.info(f"   Accounts: {accounts}  Symbol: {sym}  Broker: {broker}")

        for _ in range(self.fraud_config.wash_rounds):
            buyer, seller = random.sample(accounts, 2)
            amt = round(random.uniform(self.fraud_config.wash_trade_min,
                                       self.fraud_config.wash_trade_max), 2)
            px_round = round(px * random.uniform(0.97, 1.06), 2)
            qty = round(amt / px_round, 4)
            day_off = random.randint(0, self.fraud_config.wash_days_span - 1)
            ts = self.business_hours.generate_business_weighted_datetime(base - timedelta(days=day_off), 0)
            tid = self._next_trade_id()

            # buy side
            evts.append(TransactionEvent(
                accountID=buyer, event_amount=amt, event_type='debit',
                account_type='brokerage', account_event='buy',
                transaction_date=ts.strftime('%Y-%m-%d'), timestamp=ts.isoformat(),
                security_symbol=sym, security_type=stype, quantity=qty, price_per_unit=px_round,
                settlement_date=self._settle(ts), counterparty_account=seller,
                brokerID=broker, tradeID=tid,
            ))
            # matching sell side, seconds later, same tradeID linkage
            ts2 = ts + timedelta(seconds=random.randint(2, 90))
            evts.append(TransactionEvent(
                accountID=seller, event_amount=amt, event_type='credit',
                account_type='brokerage', account_event='sell',
                transaction_date=ts2.strftime('%Y-%m-%d'), timestamp=ts2.isoformat(),
                security_symbol=sym, security_type=stype, quantity=qty, price_per_unit=px_round,
                settlement_date=self._settle(ts2), counterparty_account=buyer,
                brokerID=broker, tradeID=tid,
            ))

        logger.info(f"   Generated {len(evts)} wash-trade events")
        return evts

    # ---- Scenario 3: unexplained wealth / rapid in-out ---------------------
    def generate_unexplained_wealth_scenario(self) -> List[TransactionEvent]:
        evts = []
        base = datetime.now()
        acct = random.randint(1, 35000)
        broker = random.randint(1, 40)
        inbound = round(random.uniform(self.fraud_config.uw_inbound_min,
                                       self.fraud_config.uw_inbound_max), 2)
        logger.info("Unexplained wealth scenario:")
        logger.info(f"   Account: {acct}  Inbound: ${inbound:,.2f}  Broker: {broker}")

        t0 = self.business_hours.generate_business_weighted_datetime(base - timedelta(days=2), 0)
        evts.append(TransactionEvent(
            accountID=acct, event_amount=inbound, event_type='credit',
            account_type='brokerage', account_event='wire',
            transaction_date=t0.strftime('%Y-%m-%d'), timestamp=t0.isoformat(),
            wire_direction='inbound', intbankID=random.randint(1, 25), brokerID=broker,
        ))
        # token buy to look like investing
        sym, stype, px = pick_security(EQUITIES + ETFS)
        t1 = t0 + timedelta(hours=random.randint(1, 5))
        evts.append(TransactionEvent(
            accountID=acct, event_amount=round(inbound * 0.99, 2), event_type='debit',
            account_type='brokerage', account_event='buy',
            transaction_date=t1.strftime('%Y-%m-%d'), timestamp=t1.isoformat(),
            security_symbol=sym, security_type=stype, quantity=round(inbound * 0.99 / px, 4),
            price_per_unit=px, settlement_date=self._settle(t1),
            brokerID=broker, tradeID=self._next_trade_id(),
        ))
        # rapid liquidation
        t2 = t1 + timedelta(hours=random.randint(2, self.fraud_config.uw_hold_hours_max))
        evts.append(TransactionEvent(
            accountID=acct, event_amount=round(inbound * 0.985, 2), event_type='credit',
            account_type='brokerage', account_event='liquidation',
            transaction_date=t2.strftime('%Y-%m-%d'), timestamp=t2.isoformat(),
            security_symbol=sym, security_type=stype, quantity=round(inbound * 0.985 / px, 4),
            price_per_unit=px, settlement_date=self._settle(t2),
            brokerID=broker, tradeID=self._next_trade_id(),
        ))
        # wire out to high-risk bank
        t3 = t2 + timedelta(hours=random.randint(1, 8))
        evts.append(TransactionEvent(
            accountID=acct, event_amount=round(inbound * self.fraud_config.uw_payout_ratio, 2),
            event_type='debit', account_type='brokerage', account_event='wire',
            transaction_date=t3.strftime('%Y-%m-%d'), timestamp=t3.isoformat(),
            wire_direction='outbound', intbankID=random.randint(1, 25), brokerID=broker,
        ))
        logger.info(f"   Generated {len(evts)} unexplained-wealth events")
        return evts

    # ---- noise -------------------------------------------------------------
    def generate_daily_noise_events(self, events_per_worker: int) -> List[TransactionEvent]:
        noise = []
        base = datetime.now()
        fc = self.fraud_config

        buy_n = int(events_per_worker * fc.buy_percentage)
        sell_n = int(events_per_worker * fc.sell_percentage)
        div_n = int(events_per_worker * fc.dividend_percentage)
        fee_n = int(events_per_worker * fc.fee_percentage)
        wire_n = int(events_per_worker * fc.wire_percentage)
        transfer_n = events_per_worker - (buy_n + sell_n + div_n + fee_n + wire_n)

        def ts():
            return self.business_hours.generate_business_weighted_datetime(base)

        # buys
        for _ in range(buy_n):
            t = ts(); sym, stype, px = pick_security()
            amt = round(random.uniform(500, 75000), 2)
            noise.append(TransactionEvent(
                accountID=random.randint(1, 35000), event_amount=amt, event_type='debit',
                account_type=random.choice(['brokerage', 'ira', 'roth_ira', 'margin']),
                account_event='buy', transaction_date=t.strftime('%Y-%m-%d'), timestamp=t.isoformat(),
                security_symbol=sym, security_type=stype, quantity=round(amt / px, 4),
                price_per_unit=px, settlement_date=self._settle(t),
                brokerID=random.randint(1, 40), tradeID=self._next_trade_id(),
            ))
        # sells
        for _ in range(sell_n):
            t = ts(); sym, stype, px = pick_security()
            amt = round(random.uniform(500, 75000), 2)
            noise.append(TransactionEvent(
                accountID=random.randint(1, 35000), event_amount=amt, event_type='credit',
                account_type=random.choice(['brokerage', 'ira', 'roth_ira', 'margin']),
                account_event='sell', transaction_date=t.strftime('%Y-%m-%d'), timestamp=t.isoformat(),
                security_symbol=sym, security_type=stype, quantity=round(amt / px, 4),
                price_per_unit=px, settlement_date=self._settle(t),
                brokerID=random.randint(1, 40), tradeID=self._next_trade_id(),
            ))
        # dividends
        for _ in range(div_n):
            t = ts(); sym, stype, px = pick_security()
            noise.append(TransactionEvent(
                accountID=random.randint(1, 35000), event_amount=round(random.uniform(5, 2500), 2),
                event_type='credit', account_type=random.choice(['brokerage', 'ira', 'roth_ira']),
                account_event='dividend', transaction_date=t.strftime('%Y-%m-%d'), timestamp=t.isoformat(),
                security_symbol=sym, security_type=stype, brokerID=random.randint(1, 40),
            ))
        # fees
        for _ in range(fee_n):
            t = ts()
            noise.append(TransactionEvent(
                accountID=random.randint(1, 35000), event_amount=round(random.uniform(0.50, 49.99), 2),
                event_type='debit', account_type=random.choice(['brokerage', 'ira', 'roth_ira', 'margin']),
                account_event='fee', transaction_date=t.strftime('%Y-%m-%d'), timestamp=t.isoformat(),
                brokerID=random.randint(1, 40),
            ))
        # wires (incl. some international decoys, larger than retail)
        for _ in range(wire_n):
            t = ts()
            etype = random.choice(['debit', 'credit'])
            intl = random.random() < fc.international_wire_percentage
            e = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1000, 250000), 2),
                event_type=etype, account_type=random.choice(['brokerage', 'ira', 'margin']),
                account_event='wire', transaction_date=t.strftime('%Y-%m-%d'), timestamp=t.isoformat(),
                wire_direction='outbound' if etype == 'debit' else 'inbound',
                brokerID=random.randint(1, 40),
            )
            if intl:
                e.intbankID = random.randint(1, 700)
            elif random.random() < 0.5:
                e.addressId = random.randint(1, 750)
            else:
                e.txbankId = random.randint(1, 30)
            noise.append(e)
        # transfers (ACAT / journal between accounts)
        for _ in range(transfer_n):
            t = ts()
            noise.append(TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1000, 150000), 2),
                event_type=random.choice(['debit', 'credit']),
                account_type=random.choice(['brokerage', 'ira', 'roth_ira', 'margin']),
                account_event='transfer', transaction_date=t.strftime('%Y-%m-%d'),
                timestamp=t.isoformat(), counterparty_account=random.randint(1, 35000),
                brokerID=random.randint(1, 40),
            ))
        return noise

    # ---- worker / threaded driver ------------------------------------------
    def generate_and_ingest_worker(self, worker_id: int, events_per_worker: int) -> dict:
        noise = self.generate_daily_noise_events(events_per_worker)
        out = []
        for e in noise:
            d = {k: v for k, v in asdict(e).items() if v is not None}
            out.append(d)
        ok, failed = self.ingester.bulk_index_events(out)
        logger.info(f"Worker {worker_id}: {ok} indexed, {failed} failed")
        return {'worker_id': worker_id, 'generated': len(out), 'indexed': ok, 'failed': failed}

    def generate_and_ingest_threaded(self, total_events: int, num_workers: int) -> dict:
        logger.info("Starting threaded brokerage generation:")
        logger.info(f"   Total events: {total_events:,}  Workers: {num_workers}")

        events_per_worker = total_events // num_workers
        remaining = total_events % num_workers
        results = {'total_generated': 0, 'total_indexed': 0, 'total_failed': 0, 'workers': []}

        # embed all three fraud scenarios first
        fraud = []
        fraud += self.generate_layering_scenario()
        fraud += self.generate_wash_trading_scenario()
        fraud += self.generate_unexplained_wealth_scenario()
        fraud_docs = [{k: v for k, v in asdict(e).items() if v is not None} for e in fraud]
        f_ok, f_failed = self.ingester.bulk_index_events(fraud_docs)
        logger.info(f"Fraud scenarios: {f_ok} indexed, {f_failed} failed")
        results['total_indexed'] += f_ok
        results['total_failed'] += f_failed
        results['total_generated'] += len(fraud_docs)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for wid in range(num_workers):
                n = events_per_worker + (1 if wid < remaining else 0)
                futures.append(executor.submit(self.generate_and_ingest_worker, wid, n))
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    results['workers'].append(r)
                    results['total_generated'] += r['generated']
                    results['total_indexed'] += r['indexed']
                    results['total_failed'] += r['failed']
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
        return results

    def save_to_files(self, events: List[dict], output_dir: str = ".") -> tuple:
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jf = os.path.join(output_dir, f"brokerage_events_{stamp}.json")
        with open(jf, 'w') as f:
            json.dump(events, f, indent=2, default=str)
        logger.info(f"Saved JSON: {jf}")
        cf = os.path.join(output_dir, f"brokerage_events_{stamp}.csv")
        pd.DataFrame(events).to_csv(cf, index=False)
        logger.info(f"Saved CSV: {cf}")
        return jf, cf


def main():
    print("BROKERAGE / INVESTMENT FRAUD WORKSHOP")
    print("=" * 60)
    print(f"Running from: {os.getcwd()}")
    print(f"Elasticsearch: http://localhost:30920")
    print(f"Index: brokerage-workshop")
    print(f"User: fraud | Workers: 16 | Events: 10,000")
    print(f"Business Hours: 7:00 - 21:00 (7x volume)")
    print("Scenarios: layering | wash trading | unexplained wealth")
    print("=" * 60)

    fraud_config = BrokerageFraudConfig()
    es_config = ElasticsearchConfig()
    gen = BrokerageDataGenerator(fraud_config, es_config)

    print("\nTesting Elasticsearch connection...")
    if not gen.ingester.test_connection():
        print("Cannot connect to Elasticsearch.")
        choice = input("\nContinue without Elasticsearch (files only)? (y/N): ").lower()
        if choice != 'y':
            return
        print("\nGenerating to files only...")
        start = time.time()
        all_events = []
        scenario = (gen.generate_layering_scenario()
                    + gen.generate_wash_trading_scenario()
                    + gen.generate_unexplained_wealth_scenario()
                    + gen.generate_daily_noise_events(es_config.events_per_day))
        for e in scenario:
            all_events.append({k: v for k, v in asdict(e).items() if v is not None})
        random.shuffle(all_events)
        jf, cf = gen.save_to_files(all_events)
        print(f"\nGenerated {len(all_events):,} events in {time.time()-start:.2f}s")
        print(f"Files: {jf}, {cf}")
        return

    gen.ingester.create_index_if_not_exists()

    print("\nStarting brokerage fraud data generation...")
    start = time.time()
    results = gen.generate_and_ingest_threaded(es_config.events_per_day, es_config.workers)
    dur = time.time() - start

    print("\n" + "=" * 60)
    print("WORKSHOP DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Events Generated: {results['total_generated']:,}")
    print(f"Successfully Indexed: {results['total_indexed']:,}")
    print(f"Failed: {results['total_failed']:,}")
    print(f"Duration: {dur:.2f}s  ({results['total_generated']/dur:.2f} ev/s)")
    print(f"\nIndex: {es_config.index_name}  Host: {es_config.host}")
    if results['total_failed']:
        print(f"\n{results['total_failed']} events failed to index")
    else:
        print("\nAll events successfully indexed.")
    print(f"\nStart detecting fraud in index '{es_config.index_name}'!")


if __name__ == "__main__":
    main()
