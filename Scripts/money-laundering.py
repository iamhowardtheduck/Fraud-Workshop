#!/usr/bin/env python3
"""
AML Fraud Workshop Generator - Single File Version
Run from: /root/Fraud-Workshop
Your Elasticsearch: http://localhost:30920
Index: fraud-workshop
Credentials: fraud/hunter
Workers: 16, Events: 10,000
Business Hours: 7 AM - 9 PM (7x volume)
"""

import json
import random
import time
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import required packages
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
    print("âŒ Missing required packages. Install with:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Your specific configuration
@dataclass
class ElasticsearchConfig:
    """Your Elasticsearch configuration - hard-coded"""
    host: str = "http://localhost:30920"
    index_name: str = "fraud-workshop"
    username: str = "fraud"
    password: str = "hunter"
    workers: int = 16
    events_per_day: int = 10000
    pipeline: str = "fraud-detection-enrich"
    verify_certs: bool = False
    timeout: int = 30

@dataclass
class FraudConfig:
    """Fraud configuration with Austin, TX banking hours"""
    # Money laundering parameters
    ml_cash_deposits_min: float = 9000.0
    ml_cash_deposits_max: float = 9999.99
    ml_checking_accounts: int = 3
    ml_savings_accounts: int = 5
    ml_days_span: int = 5
    
    # Event distribution
    deposit_percentage: float = 0.05
    fee_percentage: float = 0.10
    wire_percentage: float = 0.20
    withdrawal_percentage: float = 0.25
    purchase_percentage: float = 0.40
    international_wire_percentage: float = 0.10
    
    # Austin, TX banking hours: 9 AM - 6 PM peak, with activity until 9 PM
    banking_start_hour: int = 9
    banking_end_hour: int = 18  # 6 PM
    activity_end_hour: int = 21  # 9 PM
    peak_hour: int = 12  # Noon peak
    
    # Time range configuration
    days_back_min: int = 1  # NOW-1d (fraud events end)
    days_back_max: int = 8  # NOW-8d (events start)
    austin_tz: str = "America/Chicago"

@dataclass
class TransactionEvent:
    """Transaction event with ID fields for enrichment"""
    accountID: int
    event_amount: float
    event_type: str  # debit or credit
    account_type: str  # checking, savings, money market
    account_event: str  # deposit, fee, wire, withdrawal, purchase
    transaction_date: str  # Format: 2025-11-22T13:47:15.984Z
    timestamp: str  # Format: 2025-11-22T13:47:15.984Z
    
    # Optional ID fields for enrichment
    deposit_type: Optional[str] = None
    wire_direction: Optional[str] = None
    posID: Optional[int] = None
    txbankId: Optional[int] = None
    addressId: Optional[int] = None
    intbankID: Optional[int] = None

class AustinTimeGenerator:
    """Generates timestamps for Austin, TX timezone with banking hour distribution"""
    
    def __init__(self, config: FraudConfig):
        self.config = config
        self.austin_tz = ZoneInfo(config.austin_tz)
        
    def get_weighted_hour(self) -> int:
        """Get hour with realistic banking distribution for Austin, TX"""
        hours = list(range(24))
        weights = []
        
        for hour in hours:
            if 9 <= hour <= 18:  # 9 AM - 6 PM banking hours
                # Peak at noon (12), higher during banking hours
                if hour == 12:  # Noon peak
                    weight = 10.0
                elif hour in [11, 13]:  # Around noon
                    weight = 8.0
                elif hour in [10, 14, 15, 16]:  # Active banking hours
                    weight = 6.0
                elif hour in [9, 17, 18]:  # Start/end of banking
                    weight = 4.0
                else:
                    weight = 3.0
            elif 19 <= hour <= 21:  # 7 PM - 9 PM tapering off
                weight = 2.0
            elif 22 <= hour <= 23 or 0 <= hour <= 6:  # Night hours
                weight = 0.5
            else:  # 7 AM - 8 AM early morning
                weight = 1.0
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return random.choices(hours, weights=weights)[0]
    
    def generate_austin_datetime(self, base_date: datetime = None) -> datetime:
        """Generate datetime for Austin, TX in the last week (NOW-8d to NOW-1d for fraud events)"""
        if base_date is None:
            base_date = datetime.now(self.austin_tz)
        
        # For fraud events: NOW-8d to NOW-1d
        # For regular events: NOW-8d to NOW
        days_back = random.randint(self.config.days_back_min, self.config.days_back_max)
        
        weighted_hour = self.get_weighted_hour()
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        random_microseconds = random.randint(0, 999) * 1000  # Convert to microseconds for .984 format
        
        # Create Austin time
        austin_time = (base_date - timedelta(days=days_back)).replace(
            hour=weighted_hour, 
            minute=random_minutes, 
            second=random_seconds,
            microsecond=random_microseconds
        )
        
        # Convert to UTC for Zulu time
        utc_time = austin_time.astimezone(ZoneInfo('UTC'))
        return utc_time
    
    def generate_fraud_datetime(self, base_date: datetime = None) -> datetime:
        """Generate datetime specifically for fraud events (NOW-8d to NOW-1d)"""
        if base_date is None:
            base_date = datetime.now(self.austin_tz)
        
        # Fraud events should be in the past week but not today
        days_back = random.randint(1, 7)  # 1 to 7 days back
        
        weighted_hour = self.get_weighted_hour()
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        random_microseconds = random.randint(0, 999) * 1000
        
        # Create Austin time
        austin_time = (base_date - timedelta(days=days_back)).replace(
            hour=weighted_hour, 
            minute=random_minutes, 
            second=random_seconds,
            microsecond=random_microseconds
        )
        
        # Convert to UTC for Zulu time
        utc_time = austin_time.astimezone(ZoneInfo('UTC'))
        return utc_time
    
    def format_zulu_time(self, dt: datetime) -> str:
        """Format datetime as Zulu time: 2025-11-22T13:47:15.984Z"""
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

class ElasticsearchIngester:
    """Handles direct ingestion to your Elasticsearch"""
    
    def __init__(self, es_config: ElasticsearchConfig):
        self.config = es_config
        self.es = self._create_elasticsearch_client()
        
    def _create_elasticsearch_client(self) -> Optional[Elasticsearch]:
        """Create Elasticsearch client for your environment"""
        try:
            # Configuration for your Elasticsearch setup
            client_config = {
                'hosts': [self.config.host],
                'verify_certs': self.config.verify_certs,
                'ssl_show_warn': False,
                'request_timeout': self.config.timeout
            }
            
            # Add your authentication
            if self.config.username and self.config.password:
                try:
                    # Try new auth format first (ES 8+)
                    client_config['basic_auth'] = (self.config.username, self.config.password)
                    return Elasticsearch(**client_config)
                except TypeError:
                    # Fallback to older auth format (ES 7)
                    client_config['http_auth'] = (self.config.username, self.config.password)
                    return Elasticsearch(**client_config)
            
            return Elasticsearch(**client_config)
            
        except Exception as e:
            logger.error(f"âŒ Failed to create Elasticsearch client: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to your Elasticsearch"""
        if not self.es:
            logger.error("âŒ Elasticsearch client not initialized")
            return False
            
        try:
            info = self.es.info()
            # Handle both old and new ES client response formats
            if hasattr(info, 'body'):
                version = info.body.get('version', {}).get('number', 'unknown')
            else:
                version = info.get('version', {}).get('number', 'unknown')
            
            logger.info(f"âœ… Connected to Elasticsearch at {self.config.host}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Index: {self.config.index_name}")
            logger.info(f"   Workers: {self.config.workers}")
            return True
        except Exception as e:
            logger.error(f"âŒ Failed to connect to Elasticsearch: {e}")
            logger.error(f"   Host: {self.config.host}")
            logger.error(f"   Credentials: {self.config.username} / {self.config.password}")
            return False
    
    def create_index_if_not_exists(self):
        """Create your fraud-workshop index with proper mapping"""
        if not self.es:
            return
            
        try:
            if self.es.indices.exists(index=self.config.index_name):
                logger.info(f"ðŸ“‹ Index '{self.config.index_name}' already exists")
                return
            
            mapping = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "accountID": {"type": "integer"},
                        "event_amount": {"type": "float"},
                        "event_type": {"type": "keyword"},
                        "account_type": {"type": "keyword"},
                        "account_event": {"type": "keyword"},
                        "transaction_date": {"type": "date"},
                        "timestamp": {"type": "date"},
                        "deposit_type": {"type": "keyword"},
                        "wire_direction": {"type": "keyword"},
                        "posID": {"type": "integer"},
                        "txbankId": {"type": "integer"},
                        "addressId": {"type": "integer"},
                        "intbankID": {"type": "integer"}
                    }
                }
            }
            
            self.es.indices.create(index=self.config.index_name, body=mapping)
            logger.info(f"âœ… Created index '{self.config.index_name}'")
            
        except Exception as e:
            logger.error(f"âŒ Failed to create index: {e}")
    
    def bulk_index_events(self, events: List[dict], chunk_size: int = 500) -> tuple:
        """Bulk index events to your Elasticsearch"""
        if not self.es or not events:
            return 0, len(events) if events else 0
        
        def generate_docs():
            for event in events:
                doc = {
                    '_index': self.config.index_name,
                    '_source': event
                }
                # Add pipeline if configured
                if self.config.pipeline:
                    doc['pipeline'] = self.config.pipeline
                yield doc
        
        try:
            success_count, failed_items = bulk(
                self.es,
                generate_docs(),
                chunk_size=chunk_size,
                request_timeout=self.config.timeout,
                max_retries=3,
                initial_backoff=2,
                max_backoff=600
            )
            return success_count, len(failed_items) if failed_items else 0
            
        except Exception as e:
            logger.error(f"âŒ Bulk indexing failed: {e}")
            return 0, len(events)

class FraudDataGenerator:
    """AML fraud generator configured for Austin, TX environment"""
    
    def __init__(self, fraud_config: FraudConfig, es_config: ElasticsearchConfig):
        self.fraud_config = fraud_config
        self.es_config = es_config
        self.time_generator = AustinTimeGenerator(fraud_config)
        self.ingester = ElasticsearchIngester(es_config)
        
    def generate_money_laundering_scenario(self) -> List[TransactionEvent]:
        """Generate money laundering scenario with your business hours"""
        ml_events = []
        base_date = datetime.now()
        
        # Target accounts for money laundering
        target_accounts = {
            'checking': [random.randint(1, 35000) for _ in range(self.fraud_config.ml_checking_accounts)],
            'savings': [random.randint(1, 35000) for _ in range(self.fraud_config.ml_savings_accounts)]
        }
        
        logger.info(f"ðŸŽ¯ Money Laundering Scenario:")
        logger.info(f"   Checking accounts: {target_accounts['checking']}")
        logger.info(f"   Savings accounts: {target_accounts['savings']}")
        
        # Generate structured deposits over time span
        for day in range(self.fraud_config.ml_days_span):
            deposit_date_base = base_date - timedelta(days=day)
            
            for account_id in target_accounts['checking'] + target_accounts['savings']:
                if random.random() < 0.8:  # 80% chance of deposit per day
                    amount = round(random.uniform(
                        self.fraud_config.ml_cash_deposits_min, 
                        self.fraud_config.ml_cash_deposits_max
                    ), 2)
                    
                    # Generate fraud datetime (ensures it's in the past week but not today)
                    timestamp = self.time_generator.generate_fraud_datetime()
                    zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
                    account_type = 'checking' if account_id in target_accounts['checking'] else 'savings'
                    
                    event = TransactionEvent(
                        accountID=account_id,
                        event_amount=amount,
                        event_type='credit',
                        account_type=account_type,
                        account_event='deposit',
                        transaction_date=zulu_timestamp,
                        timestamp=zulu_timestamp,
                        deposit_type='cash'
                    )
                    ml_events.append(event)
        
        # Generate wire transfer out to Chinese bank
        all_accounts = target_accounts['checking'] + target_accounts['savings']
        source_account = random.choice(all_accounts)
        total_deposited = sum(e.event_amount for e in ml_events if e.accountID == source_account)
        wire_amount = round(total_deposited * 0.95, 2) if total_deposited > 0 else round(random.uniform(50000, 100000), 2)
        chinese_bank_id = random.randint(1, 25)  # Chinese banks are IDs 1-25
        
        # Wire happens after deposits (but still in fraud timeframe)
        wire_timestamp = self.time_generator.generate_fraud_datetime()
        wire_zulu = self.time_generator.format_zulu_time(wire_timestamp)
        
        wire_event = TransactionEvent(
            accountID=source_account,
            event_amount=wire_amount,
            event_type='debit',
            account_type='checking',
            account_event='wire',
            transaction_date=wire_zulu,
            timestamp=wire_zulu,
            wire_direction='outbound',
            intbankID=chinese_bank_id
        )
        ml_events.append(wire_event)
        
        logger.info(f"ðŸ’° Generated {len(ml_events)} money laundering events")
        logger.info(f"ðŸ¦ Wire transfer: ${wire_amount:,.2f} to Chinese bank ID {chinese_bank_id}")
        
        return ml_events
    
    def generate_daily_noise_events(self, events_per_worker: int) -> List[TransactionEvent]:
        """Generate noise events with Austin time distribution"""
        noise_events = []
        
        # Calculate event distribution
        deposit_count = int(events_per_worker * self.fraud_config.deposit_percentage)
        fee_count = int(events_per_worker * self.fraud_config.fee_percentage)
        wire_count = int(events_per_worker * self.fraud_config.wire_percentage)
        withdrawal_count = int(events_per_worker * self.fraud_config.withdrawal_percentage)
        purchase_count = events_per_worker - (deposit_count + fee_count + wire_count + withdrawal_count)
        
        # Generate all event types with your business hours weighting
        
        # Deposits
        for _ in range(deposit_count):
            deposit_types = ['cash'] * 90 + ['check'] * 8 + ['money_order'] * 2
            timestamp = self.time_generator.generate_austin_datetime()
            zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1.00, 9599.99), 2),
                event_type='credit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='deposit',
                transaction_date=zulu_timestamp,
                timestamp=zulu_timestamp,
                deposit_type=random.choice(deposit_types)
            )
            noise_events.append(event)
        
        # Fees
        for _ in range(fee_count):
            timestamp = self.time_generator.generate_austin_datetime()
            zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1.50, 15.00), 2),
                event_type='debit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='fee',
                transaction_date=zulu_timestamp,
                timestamp=zulu_timestamp
            )
            noise_events.append(event)
        
        # Wire transfers
        for _ in range(wire_count):
            event_type = random.choice(['debit', 'credit'])
            is_international = random.random() < self.fraud_config.international_wire_percentage
            timestamp = self.time_generator.generate_austin_datetime()
            zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(0, 9599.99), 2),
                event_type=event_type,
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='wire',
                transaction_date=zulu_timestamp,
                timestamp=zulu_timestamp,
                wire_direction='outbound' if event_type == 'debit' else 'inbound'
            )
            
            if is_international:
                event.intbankID = random.randint(1, 700)
            else:
                if random.random() < 0.5:
                    event.addressId = random.randint(1, 750)
                else:
                    event.txbankId = random.randint(1, 30)
            
            noise_events.append(event)
        
        # Withdrawals
        for _ in range(withdrawal_count):
            timestamp = self.time_generator.generate_austin_datetime()
            zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(20.00, 9599.99), 2),
                event_type='debit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='withdrawal',
                transaction_date=zulu_timestamp,
                timestamp=zulu_timestamp
            )
            noise_events.append(event)
        
        # Purchases
        for _ in range(purchase_count):
            timestamp = self.time_generator.generate_austin_datetime()
            zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1.00, 9599.99), 2),
                event_type='debit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='purchase',
                transaction_date=zulu_timestamp,
                timestamp=zulu_timestamp,
                posID=random.randint(1, 10500)
            )
            noise_events.append(event)
        
        return noise_events
    
    def generate_and_ingest_worker(self, worker_id: int, events_per_worker: int) -> dict:
        """Worker function for threaded generation and ingestion"""
        worker_events = []
        
        # Generate events for this worker
        noise_events = self.generate_daily_noise_events(events_per_worker)
        
        # Convert to dictionaries and clean
        for event in noise_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            worker_events.append(event_dict)
        
        # Ingest to your Elasticsearch
        success_count, failed_count = self.ingester.bulk_index_events(worker_events)
        logger.info(f"ðŸ”§ Worker {worker_id}: {success_count} indexed, {failed_count} failed")
        return {
            'worker_id': worker_id,
            'generated': len(worker_events),
            'indexed': success_count,
            'failed': failed_count
        }
    
    def generate_and_ingest_threaded(self, total_events: int, num_workers: int) -> dict:
        """Generate events using multiple threads and ingest to your Elasticsearch"""
        logger.info(f"🚀 Starting threaded generation with Austin, TX configuration:")
        logger.info(f"   Total events: {total_events:,}")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Banking hours: {self.fraud_config.banking_start_hour}:00 - {self.fraud_config.banking_end_hour}:00 (peak activity)")
        logger.info(f"   Peak hour: {self.fraud_config.peak_hour}:00 (noon)")
        logger.info(f"   Activity until: {self.fraud_config.activity_end_hour}:00 (9 PM)")
        logger.info(f"   Event timeframe: NOW-{self.fraud_config.days_back_max}d to NOW")
        logger.info(f"   Fraud events: NOW-{self.fraud_config.days_back_max}d to NOW-{self.fraud_config.days_back_min}d")
        logger.info(f"   Timezone: {self.fraud_config.austin_tz}")
        
        # Calculate events per worker
        events_per_worker = total_events // num_workers
        remaining_events = total_events % num_workers
        
        results = {
            'total_generated': 0,
            'total_indexed': 0,
            'total_failed': 0,
            'workers': []
        }
        
        # Generate money laundering scenario first
        logger.info("ðŸš¨ Generating money laundering scenario...")
        ml_events = self.generate_money_laundering_scenario()
        
        # Convert ML events and ingest
        ml_events_dict = []
        for event in ml_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            ml_events_dict.append(event_dict)
        
        ml_success, ml_failed = self.ingester.bulk_index_events(ml_events_dict)
        logger.info(f"ðŸ’° ML Events: {ml_success} indexed to '{self.es_config.index_name}', {ml_failed} failed")
        results['total_indexed'] += ml_success
        results['total_failed'] += ml_failed
        results['total_generated'] += len(ml_events_dict)
        
        # Generate noise events using your 16 workers
        logger.info(f"ðŸ“Š Generating noise events with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                worker_events = events_per_worker
                if worker_id < remaining_events:
                    worker_events += 1
                
                future = executor.submit(self.generate_and_ingest_worker, worker_id, worker_events)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    worker_result = future.result()
                    results['workers'].append(worker_result)
                    results['total_generated'] += worker_result['generated']
                    results['total_indexed'] += worker_result['indexed']
                    results['total_failed'] += worker_result['failed']
                except Exception as e:
                    logger.error(f"âŒ Worker failed: {e}")
        
        return results
    
    def save_to_files(self, events: List[dict], output_dir: str = "."):
        """Save events to JSON and CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = os.path.join(output_dir, f"fraud_events_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(events, f, indent=2, default=str)
        logger.info(f"ðŸ’¾ Saved JSON: {json_file}")
        
        # Save CSV
        csv_file = os.path.join(output_dir, f"fraud_events_{timestamp}.csv")
        df = pd.DataFrame(events)
        df.to_csv(csv_file, index=False)
        logger.info(f"ðŸ’¾ Saved CSV: {csv_file}")
        
        return json_file, csv_file

def main():
    """Main function with Austin, TX timezone configuration"""
    print("ðŸ’° AML FRAUD WORKSHOP - YOUR CONFIGURATION")
    print("=" * 60)
    print(f"ðŸ“ Running from: {os.getcwd()}")
    print(f"ðŸ” Elasticsearch: http://localhost:30920")
    print(f"ðŸ“Š Index: fraud-workshop")
    print(f"ðŸ‘¤ User: fraud")
    print(f"ðŸ”§ Workers: 16")
    print(f"ðŸ“ˆ Events: 10,000")
    print(f"â° Business Hours: 7:00 - 21:00 (7x volume)")
    print(f"ðŸŒ™ Off Hours: 22:00 - 06:59 (1x volume)")
    print("=" * 60)
    
    # Your configurations
    fraud_config = FraudConfig()
    es_config = ElasticsearchConfig()
    
    # Initialize generator
    generator = FraudDataGenerator(fraud_config, es_config)
    
    # Test connection to your Elasticsearch
    print("\nðŸ” Testing Elasticsearch connection...")
    if not generator.ingester.test_connection():
        print("âŒ Cannot connect to your Elasticsearch")
        print(f"   Host: {es_config.host}")
        print(f"   Username: {es_config.username}")
        print(f"   Password: {es_config.password}")
        print("\nðŸ’¡ Options:")
        print("   1. Check if Elasticsearch is running at the configured host")
        print("   2. Verify credentials are correct")
        print("   3. Generate data to files only (without Elasticsearch)")
        
        choice = input("\nContinue without Elasticsearch? (y/N): ").lower()
        if choice != 'y':
            return
        
        # Generate without Elasticsearch
        print("\nðŸš€ Generating fraud data to files only...")
        start_time = time.time()
        
        # Generate events
        ml_events = generator.generate_money_laundering_scenario()
        noise_events = generator.generate_daily_noise_events(es_config.events_per_day)
        
        all_events = []
        for event in ml_events + noise_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            all_events.append(event_dict)
        
        random.shuffle(all_events)
        
        # Save to files
        json_file, csv_file = generator.save_to_files(all_events)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nâœ… Generated {len(all_events):,} events in {duration:.2f} seconds")
        print(f"ðŸ“ Files saved: {json_file}, {csv_file}")
        return
    
    # Create your index
    generator.ingester.create_index_if_not_exists()
    
    # Generate and ingest with your settings
    print(f"\nðŸš€ Starting fraud data generation...")
    start_time = time.time()
    
    results = generator.generate_and_ingest_threaded(
        total_events=es_config.events_per_day,
        num_workers=es_config.workers
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Display results
    print(f"\n" + "=" * 60)
    print("ðŸ“Š WORKSHOP DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Events Generated: {results['total_generated']:,}")
    print(f"Successfully Indexed: {results['total_indexed']:,}")
    print(f"Failed: {results['total_failed']:,}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Events/second: {results['total_generated']/duration:.2f}")
    print(f"\nðŸŽ¯ Your Elasticsearch Info:")
    print(f"   Index: {es_config.index_name}")
    print(f"   Host: {es_config.host}")
    print(f"   Events: {results['total_indexed']:,}")
    
    if results['total_failed'] > 0:
        print(f"\nâš ï¸  {results['total_failed']} events failed to index")
    else:
        print("\nâœ… All events successfully indexed to your Elasticsearch!")
    
    print(f"\nðŸ•µï¸ Start detecting fraud in index '{es_config.index_name}'!")

if __name__ == "__main__":
    main()
