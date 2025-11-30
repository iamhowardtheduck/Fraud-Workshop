#!/usr/bin/env python3
"""
AML Fraud Workshop Generator - Layering Fraud Scenario
Money laundering through account hopping/layering detection exercise
Run from: /root/Fraud-Workshop
Your Elasticsearch: http://localhost:30920
Index: fraud-workshop
Credentials: fraud/hunter
Workers: 16, Events: 100,000
Business Hours: 7 AM - 9 PM (7x volume)
"""

import json
import random
import time
import sys
import os
import urllib3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

# Suppress SSL warnings for Elasticsearch connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    index_name: str = "fraud-workshop-money-laundering"
    username: str = "fraud"
    password: str = "hunter"
    workers: int = 16
    events_per_day: int = 100000
    pipeline: str = "fraud-detection-enrich"
    verify_certs: bool = False
    timeout: int = 30

@dataclass
class FraudConfig:
    """Austin, TX fraud configuration for layering scenario"""
    # Layering scenario parameters
    initial_deposit_amount: float = 25000.00
    target_accounts: List[int] = None
    
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
    
    def __post_init__(self):
        if self.target_accounts is None:
            # Layering chain: 32687 â†’ 16384 â†’ 8192 â†’ 4096 â†’ 2048 (all within 1-35000)
            self.target_accounts = [32687, 16384, 8192, 4096, 2048]

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
    to_account: Optional[int] = None  # For internal transfers

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
    
    def generate_specific_datetime(self, days_back: int, base_date: datetime = None) -> datetime:
        """Generate datetime for a specific day back with banking hours weighting"""
        if base_date is None:
            base_date = datetime.now(self.austin_tz)
        
        weighted_hour = self.get_weighted_hour()
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        random_microseconds = random.randint(0, 999) * 1000
        
        # Create Austin time for specific day
        target_date = base_date - timedelta(days=days_back)
        austin_time = target_date.replace(
            hour=weighted_hour,
            minute=random_minutes,
            second=random_seconds,
            microsecond=random_microseconds
        )
        
        return austin_time
    
    def generate_random_datetime(self, base_date: datetime = None) -> datetime:
        """Generate random datetime within the fraud timeframe"""
        if base_date is None:
            base_date = datetime.now(self.austin_tz)
        
        # Random day within the fraud timeframe
        days_back = random.randint(self.config.days_back_min, self.config.days_back_max)
        return self.generate_specific_datetime(days_back, base_date)

class ElasticsearchIngester:
    """Handles Elasticsearch ingestion with proper error handling"""
    
    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self.es = None
        self._connect()
    
    def _connect(self):
        """Connect to your Elasticsearch cluster"""
        try:
            self.es = Elasticsearch(
                [self.config.host],
                basic_auth=(self.config.username, self.config.password),
                verify_certs=self.config.verify_certs,
                request_timeout=self.config.timeout,
                retry_on_timeout=True,
                max_retries=3
            )
            logger.info(f"ðŸ“¡ Connected to Elasticsearch: {self.config.host}")
        except Exception as e:
            logger.error(f"âŒ Elasticsearch connection failed: {e}")
            self.es = None
    
    def test_connection(self) -> bool:
        """Test connection to your Elasticsearch"""
        if not self.es:
            return False
        try:
            info = self.es.info()
            logger.info(f"âœ… Elasticsearch connection successful: {info['version']['number']}")
            return True
        except Exception as e:
            logger.error(f"âŒ Elasticsearch test failed: {e}")
            return False
    
    def create_index_if_not_exists(self):
        """Create the fraud workshop index with mapping"""
        if not self.es:
            logger.warning("âš ï¸ No Elasticsearch connection")
            return
        
        try:
            if self.es.indices.exists(index=self.config.index_name):
                logger.info(f"ðŸ“‹ Index '{self.config.index_name}' already exists")
                return
            
            # Index mapping for fraud detection
            mapping = {
                "mappings": {
                    "properties": {
                        "accountID": {"type": "keyword"},
                        "event_amount": {"type": "double"},
                        "event_type": {"type": "keyword"},
                        "account_type": {"type": "keyword"},
                        "account_event": {"type": "keyword"},
                        "transaction_date": {"type": "date"},
                        "timestamp": {"type": "date"},
                        "deposit_type": {"type": "keyword"},
                        "wire_direction": {"type": "keyword"},
                        "posID": {"type": "keyword"},
                        "txbankId": {"type": "keyword"},
                        "addressId": {"type": "keyword"},
                        "intbankID": {"type": "keyword"},
                        "to_account": {"type": "keyword"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "1s"
                }
            }
            
            self.es.indices.create(index=self.config.index_name, body=mapping)
            logger.info(f"ðŸ“‹ Created index: '{self.config.index_name}'")
        except Exception as e:
            logger.error(f"âŒ Index creation failed: {e}")
    
    def bulk_index_events(self, events: List[dict]) -> tuple:
        """Bulk index events to your Elasticsearch"""
        if not self.es:
            logger.warning("âš ï¸ No Elasticsearch connection - skipping ingestion")
            return 0, len(events)
        
        try:
            actions = []
            for event in events:
                action = {
                    "_index": self.config.index_name,
                    "_source": event
                }
                actions.append(action)
            
            success_count, failed_items = bulk(
                self.es,
                actions,
                index=self.config.index_name,
                chunk_size=1000,
                request_timeout=60,
                max_retries=3,
                initial_backoff=2,
                max_backoff=600
            )
            
            failed_count = len(failed_items) if failed_items else 0
            return success_count, failed_count
        
        except Exception as e:
            logger.error(f"âŒ Bulk indexing failed: {e}")
            return 0, len(events)

class FraudDataGenerator:
    """Generates Austin, TX fraud scenarios with money laundering patterns"""
    
    def __init__(self, fraud_config: FraudConfig, es_config: ElasticsearchConfig):
        self.fraud_config = fraud_config
        self.es_config = es_config
        self.time_generator = AustinTimeGenerator(fraud_config)
        self.ingester = ElasticsearchIngester(es_config)
    
    def to_zulu_timestamp(self, austin_dt: datetime) -> str:
        """Convert Austin datetime to Zulu format string"""
        # Convert to UTC for Zulu time
        utc_dt = austin_dt.astimezone(ZoneInfo("UTC"))
        return utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z'
    
    def generate_money_laundering_scenario(self) -> List[TransactionEvent]:
        """
        Generate money laundering scenario: Large cash deposit followed by layering
        
        SCENARIO: Money laundering through account hopping/layering
        - Day 1 (NOW-8d): Large cash deposit $25K into Account 32687 (under SAR threshold)
        - Day 2-5: Systematic layering through 5 accounts to obscure source
        - Day 7 (NOW-2d): International wire to Chinese bank
        """
        events = []
        base_date = datetime.now(ZoneInfo(self.fraud_config.austin_tz))
        
        logger.info("ðŸ” FRAUD SCENARIO: Money Laundering through Account Layering")
#        logger.info(f"   Initial deposit: ${self.fraud_config.initial_deposit_amount:,.2f}")
#        logger.info(f"   Layering accounts: {self.fraud_config.target_accounts}")
#        logger.info(f"   Timeframe: {self.fraud_config.days_back_max} days back to {self.fraud_config.days_back_min} day back")
        
        # Day 1 (NOW-8d): Large structured cash deposit
        day_1_time = self.time_generator.generate_specific_datetime(8, base_date)
        events.append(TransactionEvent(
            accountID=self.fraud_config.target_accounts[0],  # 32687
            event_amount=self.fraud_config.initial_deposit_amount,
            event_type='credit',
            account_type='checking',
            account_event='deposit',
            transaction_date=self.to_zulu_timestamp(day_1_time),
            timestamp=self.to_zulu_timestamp(day_1_time),
            deposit_type='cash'
        ))
        
        # Day 2-5: Layering transfers between accounts
        current_amount = self.fraud_config.initial_deposit_amount
        for day in range(2, 6):  # Days 2-5 (4 transfers through 5 accounts)
            from_account = self.fraud_config.target_accounts[day - 2]
            to_account = self.fraud_config.target_accounts[day - 1]
            
            # Subtract small fee for transfer
            fee_amount = round(random.uniform(15.00, 15.00), 2)
            transfer_amount = round(current_amount - fee_amount, 2)
            
            day_time = self.time_generator.generate_specific_datetime(8 - day + 1, base_date)
            
            # Withdrawal from source account
            events.append(TransactionEvent(
                accountID=from_account,
                event_amount=transfer_amount,
                event_type='debit',
                account_type='checking',
                account_event='wire',
                transaction_date=self.to_zulu_timestamp(day_time),
                timestamp=self.to_zulu_timestamp(day_time),
                wire_direction='outgoing_domestic',
                to_account=to_account
            ))
            
            # Fee for transfer
            events.append(TransactionEvent(
                accountID=from_account,
                event_amount=fee_amount,
                event_type='debit',
                account_type='checking',
                account_event='fee',
                transaction_date=self.to_zulu_timestamp(day_time),
                timestamp=self.to_zulu_timestamp(day_time)
            ))
            
            # Deposit to target account (few minutes later)
            deposit_time = day_time + timedelta(minutes=random.randint(5, 15))
            events.append(TransactionEvent(
                accountID=to_account,
                event_amount=transfer_amount,
                event_type='credit',
                account_type='checking',
                account_event='wire',
                transaction_date=self.to_zulu_timestamp(deposit_time),
                timestamp=self.to_zulu_timestamp(deposit_time),
                wire_direction='incoming_domestic'
            ))
            
            current_amount = transfer_amount
        
        # Day 7 (NOW-2d): International wire to Chinese bank
        final_day_time = self.time_generator.generate_specific_datetime(2, base_date)
        final_account = self.fraud_config.target_accounts[-1]  # 2048
        
        # International wire fee
        intl_fee = round(random.uniform(45.00, 45.00), 2)
        final_amount = round(current_amount - intl_fee, 2)
        
        # International wire transfer
        events.append(TransactionEvent(
            accountID=final_account,
            event_amount=final_amount,
            event_type='debit',
            account_type='checking',
            account_event='wire',
            transaction_date=self.to_zulu_timestamp(final_day_time),
            timestamp=self.to_zulu_timestamp(final_day_time),
            wire_direction='outgoing_international',
            intbankID=random.choice([8001, 8002, 8003, 8004, 8005])  # Chinese banks
        ))
        
        # International wire fee
        events.append(TransactionEvent(
            accountID=final_account,
            event_amount=intl_fee,
            event_type='debit',
            account_type='checking',
            account_event='fee',
            transaction_date=self.to_zulu_timestamp(final_day_time),
            timestamp=self.to_zulu_timestamp(final_day_time)
        ))
        
        logger.info(f"   Generated {len(events)} money laundering events")
#        logger.info(f"   Final amount wired: ${final_amount:,.2f}")
        
        return events
    
    def generate_daily_noise_events(self, num_events: int) -> List[TransactionEvent]:
        """Generate noise events to hide fraud patterns"""
        noise_events = []
        
        for _ in range(num_events):
            # Random account in Austin market range
            account_id = random.randint(1, 35000)
            
            # Skip fraud accounts to avoid confusion
            if account_id in self.fraud_config.target_accounts:
                account_id = random.randint(35001, 50000)
            
            # Random timestamp in the full range
            event_time = self.time_generator.generate_random_datetime()
            zulu_timestamp = self.to_zulu_timestamp(event_time)
            
            # Generate random transaction types based on distribution
            event_type_rand = random.random()
            
            if event_type_rand < self.fraud_config.deposit_percentage:
                # Deposit
                event = TransactionEvent(
                    accountID=account_id,
                    event_amount=round(random.uniform(50.00, 9999.99), 2),
                    event_type='credit',
                    account_type=random.choice(['checking', 'savings']),
                    account_event='deposit',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    deposit_type=random.choice(['check', 'cash', 'direct_deposit'])
                )
            elif event_type_rand < (self.fraud_config.deposit_percentage + self.fraud_config.fee_percentage):
                # Fee
                event = TransactionEvent(
                    accountID=account_id,
                    event_amount=round(random.uniform(5.00, 45.00), 2),
                    event_type='debit',
                    account_type=random.choice(['checking', 'savings', 'money market']),
                    account_event='fee',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp
                )
            elif event_type_rand < (self.fraud_config.deposit_percentage + self.fraud_config.fee_percentage + self.fraud_config.wire_percentage):
                # Wire transfer
                event = TransactionEvent(
                    accountID=account_id,
                    event_amount=round(random.uniform(100.00, 15000.00), 2),
                    event_type=random.choice(['debit', 'credit']),
                    account_type='checking',
                    account_event='wire',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    wire_direction=random.choice(['incoming_domestic', 'outgoing_domestic']),
                    txbankId=random.randint(1, 30)
                )
            elif event_type_rand < (self.fraud_config.deposit_percentage + self.fraud_config.fee_percentage + 
                                   self.fraud_config.wire_percentage + self.fraud_config.withdrawal_percentage):
                # Withdrawal
                event = TransactionEvent(
                    accountID=account_id,
                    event_amount=round(random.uniform(20.00, 800.00), 2),
                    event_type='debit',
                    account_type=random.choice(['checking', 'savings']),
                    account_event='withdrawal',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    posID=random.randint(1, 13000)  # ATM ID range
                )
            else:
                # Purchase
                event = TransactionEvent(
                    accountID=account_id,
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
        logger.info(f"ðŸš€ Starting threaded generation with Austin, TX configuration:")
        logger.info(f"   Total events: {total_events:,}")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Banking hours: {self.fraud_config.banking_start_hour}:00 - {self.fraud_config.banking_end_hour}:00 (peak activity)")
        logger.info(f"   Peak hour: {self.fraud_config.peak_hour}:00 (noon)")
 #       logger.info(f"   Activity until: {self.fraud_config.activity_end_hour}:00 (9 PM)")
        logger.info(f"   Event timeframe: NOW-{self.fraud_config.days_back_max}d to NOW")
 #       logger.info(f"   Fraud events: NOW-{self.fraud_config.days_back_max}d to NOW-{self.fraud_config.days_back_min}d")
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
        logger.info(f"ðŸ”€ Generating noise events with {num_workers} workers...")
        
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
    print("ðŸ¦ AML FRAUD WORKSHOP - YOUR CONFIGURATION")
    print("=" * 60)
    print(f"ðŸ“ Running from: {os.getcwd()}")
    print(f"ðŸ” Elasticsearch: http://localhost:30920")
    print(f"ðŸ“Š Index: fraud-workshop-money-laundering")
    print(f"ðŸ‘¤ User: fraud")
    print(f"âš¡ Workers: 16")
    print(f"ðŸ“ˆ Events: 100,000")
    print(f"ðŸ•˜ Business Hours: 9:00 - 18:00 (peak activity)")
    print(f"ðŸ•˜ Extended Hours: 19:00 - 21:00 (reduced activity)")
    print("=" * 60)
    
    # Your configurations
    fraud_config = FraudConfig()
    es_config = ElasticsearchConfig()
    
    # Initialize generator
    generator = FraudDataGenerator(fraud_config, es_config)
    
    # Test connection to your Elasticsearch
    print("\nðŸ”Œ Testing Elasticsearch connection...")
    if not generator.ingester.test_connection():
        print("âŒ Cannot connect to your Elasticsearch")
        print(f"   Host: {es_config.host}")
        print(f"   Username: {es_config.username}")
        print(f"   Password: {es_config.password}")
        print("\nðŸ“‹ Options:")
        print("   1. Check if Elasticsearch is running at the configured host")
        print("   2. Verify credentials are correct")
        print("   3. Generate data to files only (without Elasticsearch)")
        
        choice = input("\nContinue without Elasticsearch? (y/N): ").lower()
        if choice != 'y':
            return
        
        # Generate without Elasticsearch
        print("\nðŸ“ Generating fraud data to files only...")
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
        print(f"ðŸ“„ Files saved: {json_file}, {csv_file}")
        return
    
    # Create your index
    generator.ingester.create_index_if_not_exists()
    
    # Generate and ingest with your settings
    print(f"\nðŸŽ¯ Starting fraud data generation...")
    start_time = time.time()
    
    results = generator.generate_and_ingest_threaded(
        total_events=es_config.events_per_day,
        num_workers=es_config.workers
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Display results
    print(f"\n" + "=" * 60)
    print("FRAUD WORKSHOP MONEY LAUNDERING DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nðŸ“ˆ Generation Statistics:")
    print(f"   Total Generated: {results['total_generated']:,} events")
    print(f"   Total Indexed: {results['total_indexed']:,} events")
    print(f"   Total Failed: {results['total_failed']:,} events")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Rate: {results['total_generated']/duration:,.0f} events/second")


if __name__ == "__main__":
    main()
