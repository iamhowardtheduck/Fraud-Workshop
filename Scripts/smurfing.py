#!/usr/bin/env python3
"""
AML Fraud Workshop Generator - ATM Structuring Fraud Scenario
Structured cash deposits through ATMs with disabled cameras
"""

import json
import random
import time
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
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
    print("⚠ Missing required packages. Install with:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Your specific configuration
@dataclass
class ElasticsearchConfig:
    """Your Elasticsearch configuration - hard-coded"""
    host: str = "https://fraud-workshop-5b63ee.es.us-west2.gcp.elastic-cloud.com:443"
    index_name: str = "fraud-workshop-atm"
    username: str = "fraudster"
    password: str = "flt72100"
    workers: int = 24
    events_per_day: int = 100000
    pipeline: str = "fraud-detection-enrich"
    verify_certs: bool = False
    timeout: int = 30

@dataclass
class FraudConfig:
    """Austin, TX fraud configuration for ATM structuring scenario"""
    # ATM structuring parameters
    fraud_accounts: List[int] = None
    fraud_pos_ids: List[int] = None
    fraud_deposit_amount: float = 500.00
    fraud_start_hour: int = 9  # 9 AM
    fraud_end_hour: int = 21   # 9 PM
    fraud_interval_minutes: int = 30  # Every 30 minutes
    
    # Date range configuration
    fraud_start_days_back: int = 8  # Start 8 days ago
    fraud_end_days_back: int = 1    # End 1 day ago
    
    # Event distribution for noise
    deposit_percentage: float = 0.30
    inquiry_percentage: float = 0.10
    status_check_percentage: float = 0.02
    withdrawal_percentage: float = 0.58
    
    # Austin, TX timezone
    austin_tz: str = "America/Chicago"
    
    def __post_init__(self):
        if self.fraud_accounts is None:
            # 10 accounts for structured deposits
            self.fraud_accounts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        if self.fraud_pos_ids is None:
            # ATMs with disabled cameras
            self.fraud_pos_ids = [6, 7, 16, 18, 37, 49, 60, 61, 64, 68, 74, 78, 87, 91, 94, 100, 103, 106, 107, 111, 113, 116, 123, 124, 126]

@dataclass
class ATMTransactionEvent:
    """ATM transaction event with comprehensive fields"""
    accountID: int
    event_amount: float
    event_type: str  # debit or credit
    account_type: str  # checking, savings, money market
    account_event: str  # deposit, withdrawal, inquiry, fee
    transaction_date: str  # Format: 2025-11-22T13:47:15.984Z
    timestamp: str  # Format: 2025-11-22T13:47:15.984Z
    posID: int
    business_hours: bool
    
    # ATM specific fields
    atm_event: str  # deposit, withdrawal, inquiry, status_check, LOCKOUT
    atm_camera_enabled: bool
    atm_online: bool = True
    atm_pin_attempts: int = 1
    
    # ATM deposit fields
    atm_deposit_type: Optional[str] = None
    atm_deposit_amount: Optional[int] = None
    
    # ATM withdrawal fields
    atm_withdrawal_source: Optional[str] = None
    atm_withdrawal_amount: Optional[int] = None
    
    # ATM inquiry fields
    atm_inquiry_type: Optional[str] = None
    
    # ATM status fields
    atm_paper_jam: Optional[bool] = None
    atm_available_amount: Optional[int] = None
    atm_available_10: Optional[int] = None
    atm_available_20: Optional[int] = None
    atm_available_50: Optional[int] = None
    atm_available_100: Optional[int] = None

class AustinTimeGenerator:
    """Generates timestamps for Austin, TX timezone"""
    
    def __init__(self, config: FraudConfig):
        self.config = config
        self.austin_tz = ZoneInfo(config.austin_tz)
        
    def generate_fraud_timestamp(self, hour: int, minute: int, days_back: int = 7) -> datetime:
        """Generate specific timestamp for fraud events"""
        base_date = datetime.now(self.austin_tz)
        fraud_date = (base_date - timedelta(days=days_back)).replace(
            hour=hour,
            minute=minute,
            second=random.randint(0, 59),
            microsecond=random.randint(0, 999) * 1000
        )
        
        # Convert to UTC for Zulu time
        utc_time = fraud_date.astimezone(ZoneInfo('UTC'))
        return utc_time
    
    def generate_random_timestamp(self, days_back_range: int = 8) -> datetime:
        """Generate random timestamp for noise events"""
        base_date = datetime.now(self.austin_tz)
        days_back = random.randint(0, days_back_range)
        
        random_hour = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        random_microseconds = random.randint(0, 999) * 1000
        
        austin_time = (base_date - timedelta(days=days_back)).replace(
            hour=random_hour, 
            minute=random_minutes, 
            second=random_seconds,
            microsecond=random_microseconds
        )
        
        # Convert to UTC for Zulu time
        utc_time = austin_time.astimezone(ZoneInfo('UTC'))
        return utc_time
    
    def is_business_hours(self, dt: datetime) -> bool:
        """Check if timestamp is within business hours (9 AM - 9 PM CST)"""
        austin_dt = dt.astimezone(self.austin_tz)
        return 9 <= austin_dt.hour <= 21
    
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
            client_config = {
                'hosts': [self.config.host],
                'verify_certs': self.config.verify_certs,
                'ssl_show_warn': False,
                'request_timeout': self.config.timeout
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
            logger.error(f"⚠ Failed to create Elasticsearch client: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to your Elasticsearch"""
        if not self.es:
            return False
        
        try:
            info = self.es.info()
            logger.info(f"✅ Connected to Elasticsearch {info.get('version', {}).get('number', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"⚠ Connection test failed: {e}")
            return False
    
    def create_index_if_not_exists(self) -> None:
        """Create index with mapping if it doesn't exist"""
        if not self.es:
            return
        
        try:
            if self.es.indices.exists(index=self.config.index_name):
                logger.info(f"📊 Index '{self.config.index_name}' already exists")
                return
            
            # ATM transaction mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "accountID": {"type": "integer"},
                        "event_amount": {"type": "float"},
                        "event_type": {"type": "keyword"},
                        "account_type": {"type": "keyword"},
                        "account_event": {"type": "keyword"},
                        "transaction_date": {"type": "date"},
                        "timestamp": {"type": "date"},
                        "posID": {"type": "integer"},
                        "business_hours": {"type": "boolean"},
                        "atm_event": {"type": "keyword"},
                        "atm_camera_enabled": {"type": "boolean"},
                        "atm_online": {"type": "boolean"},
                        "atm_pin_attempts": {"type": "integer"},
                        "atm_deposit_type": {"type": "keyword"},
                        "atm_deposit_amount": {"type": "integer"},
                        "atm_withdrawal_source": {"type": "keyword"},
                        "atm_withdrawal_amount": {"type": "integer"},
                        "atm_inquiry_type": {"type": "keyword"},
                        "atm_paper_jam": {"type": "boolean"},
                        "atm_available_amount": {"type": "integer"},
                        "atm_available_10": {"type": "integer"},
                        "atm_available_20": {"type": "integer"},
                        "atm_available_50": {"type": "integer"},
                        "atm_available_100": {"type": "integer"}
                    }
                }
            }
            
            self.es.indices.create(index=self.config.index_name, body=mapping)
            logger.info(f"✅ Created index '{self.config.index_name}'")
            
        except Exception as e:
            logger.error(f"⚠ Failed to create index: {e}")
    
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
                if self.config.pipeline:
                    doc['pipeline'] = self.config.pipeline
                yield doc
        
        try:
            success_count, failed_items = bulk(
                self.es,
                generate_docs(),
                chunk_size=chunk_size,
                request_timeout=self.config.timeout
            )
            return success_count, len(failed_items)
        except Exception as e:
            logger.error(f"⚠ Bulk indexing failed: {e}")
            return 0, len(events)

class ATMFraudGenerator:
    """Austin, TX ATM fraud pattern generator"""
    
    def __init__(self, fraud_config: FraudConfig, es_config: ElasticsearchConfig):
        self.fraud_config = fraud_config
        self.time_generator = AustinTimeGenerator(fraud_config)
        self.ingester = ElasticsearchIngester(es_config)
    
    def generate_atm_structuring_scenario(self) -> List[ATMTransactionEvent]:
        """Generate ATM structuring fraud scenario across multiple days"""
        atm_fraud_events = []
        
        # Generate fraud events for each day in the range
        for days_back in range(self.fraud_config.fraud_end_days_back, 
                             self.fraud_config.fraud_start_days_back + 1):
            
            # Generate structured deposits every 30 minutes from 9 AM to 9 PM for this day
            current_hour = self.fraud_config.fraud_start_hour
            current_minute = 0
            account_index = 0
            pos_index = 0
            
            while current_hour <= self.fraud_config.fraud_end_hour:
                # Select account (cycle through)
                account_id = self.fraud_config.fraud_accounts[account_index % len(self.fraud_config.fraud_accounts)]
                
                # Select POS ID ensuring no consecutive use
                pos_id = self.fraud_config.fraud_pos_ids[pos_index % len(self.fraud_config.fraud_pos_ids)]
                
                # Generate timestamp for this specific day
                timestamp = self.time_generator.generate_fraud_timestamp(current_hour, current_minute, days_back)
                zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
                
                # Create structured deposit event
                fraud_event = ATMTransactionEvent(
                    accountID=account_id,
                    event_amount=self.fraud_config.fraud_deposit_amount,
                    event_type='credit',
                    account_type='checking',
                    account_event='deposit',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    posID=pos_id,
                    business_hours=True,  # Always during business hours
                    atm_event='deposit',
                    atm_camera_enabled=False,  # Cameras disabled at fraud ATMs
                    atm_online=True,
                    atm_pin_attempts=1,
                    atm_deposit_type='cash',
                    atm_deposit_amount=500
                )
                
                atm_fraud_events.append(fraud_event)
                
                # Move to next time slot
                current_minute += self.fraud_config.fraud_interval_minutes
                if current_minute >= 60:
                    current_minute = 0
                    current_hour += 1
                
                # Cycle through accounts and POS IDs
                account_index += 1
                pos_index += 1
                
                # Break if we've reached end hour for this day
                if current_hour > self.fraud_config.fraud_end_hour:
                    break
        
        return atm_fraud_events
    
    def generate_withdrawal_amount(self) -> int:
        """Generate realistic withdrawal amounts"""
        amounts = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 300, 320, 400, 500]
        weights = [20, 15, 12, 10, 8, 6, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1]
        return random.choices(amounts, weights=weights)[0]
    
    def generate_deposit_amount(self) -> float:
        """Generate realistic deposit amounts"""
        # Mix of common amounts
        if random.random() < 0.6:  # 60% common amounts
            amounts = [20.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0]
            return random.choice(amounts)
        else:  # 40% random amounts
            return round(random.uniform(25.0, 2500.0), 2)
    
    def generate_atm_noise_events(self, count: int) -> List[ATMTransactionEvent]:
        """Generate legitimate ATM noise events"""
        events = []
        legitimate_accounts = list(range(1, 5001))  # 5000 legitimate accounts
        legitimate_pos_ids = list(range(1, 129))  # 128 ATM locations
        
        for _ in range(count):
            # Select random legitimate account and ATM
            account_id = random.choice(legitimate_accounts)
            pos_id = random.choice(legitimate_pos_ids)
            
            # Generate random timestamp
            timestamp = self.time_generator.generate_random_timestamp()
            zulu_timestamp = self.time_generator.format_zulu_time(timestamp)
            business_hours = self.time_generator.is_business_hours(timestamp)
            
            # Determine event type
            rand = random.random()
            if rand < self.fraud_config.withdrawal_percentage:
                # Withdrawal event
                withdrawal_amount = self.generate_withdrawal_amount()
                event = ATMTransactionEvent(
                    accountID=account_id,
                    event_amount=withdrawal_amount,
                    event_type='debit',
                    account_type=random.choice(['checking', 'savings']),
                    account_event='withdrawal',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    posID=pos_id,
                    business_hours=business_hours,
                    atm_event='withdrawal',
                    atm_camera_enabled=random.choice([True, False]),
                    atm_online=True,
                    atm_pin_attempts=random.choices([1, 2, 3], weights=[85, 12, 3])[0],
                    atm_withdrawal_source=random.choice(['checking', 'savings']),
                    atm_withdrawal_amount=withdrawal_amount
                )
            
            elif rand < (self.fraud_config.withdrawal_percentage + self.fraud_config.deposit_percentage):
                # Deposit event
                deposit_amount = self.generate_deposit_amount()
                event = ATMTransactionEvent(
                    accountID=account_id,
                    event_amount=deposit_amount,
                    event_type='credit',
                    account_type=random.choice(['checking', 'savings']),
                    account_event='deposit',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    posID=pos_id,
                    business_hours=business_hours,
                    atm_event='deposit',
                    atm_camera_enabled=random.choice([True, False]),
                    atm_online=True,
                    atm_pin_attempts=random.choices([1, 2], weights=[90, 10])[0],
                    atm_deposit_type=random.choices(['cash', 'check'], weights=[70, 30])[0],
                    atm_deposit_amount=int(deposit_amount)
                )
            
            elif rand < (self.fraud_config.withdrawal_percentage + 
                        self.fraud_config.deposit_percentage + 
                        self.fraud_config.inquiry_percentage):
                # Inquiry event
                event = ATMTransactionEvent(
                    accountID=account_id,
                    event_amount=0.0,
                    event_type='inquiry',
                    account_type=random.choice(['checking', 'savings']),
                    account_event='inquiry',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    posID=pos_id,
                    business_hours=business_hours,
                    atm_event='inquiry',
                    atm_camera_enabled=random.choice([True, False]),
                    atm_online=True,
                    atm_pin_attempts=1,
                    atm_inquiry_type=random.choice(['balance', 'recent_transactions', 'mini_statement'])
                )
            
            else:
                # Status check event
                event = ATMTransactionEvent(
                    accountID=account_id,
                    event_amount=0.0,
                    event_type='status',
                    account_type=random.choice(['checking', 'savings']),
                    account_event='status_check',
                    transaction_date=zulu_timestamp,
                    timestamp=zulu_timestamp,
                    posID=pos_id,
                    business_hours=business_hours,
                    atm_event='status_check',
                    atm_camera_enabled=random.choice([True, False]),
                    atm_online=random.choices([True, False], weights=[95, 5])[0],
                    atm_pin_attempts=0,
                    atm_paper_jam=random.choices([False, True], weights=[98, 2])[0],
                    atm_available_amount=random.randint(50000, 200000),
                    atm_available_10=random.randint(100, 500),
                    atm_available_20=random.randint(200, 1000),
                    atm_available_50=random.randint(50, 300),
                    atm_available_100=random.randint(100, 800)
                )
            
            events.append(event)
        
        return events
    
    def generate_and_ingest_worker(self, worker_id: int, events_per_worker: int) -> dict:
        """Generate events for a single worker thread"""
        worker_events = []
        
        # Generate events for this worker
        noise_events = self.generate_atm_noise_events(events_per_worker)
        
        # Convert to dictionaries and clean
        for event in noise_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            worker_events.append(event_dict)
        
        # Ingest to your Elasticsearch
        success_count, failed_count = self.ingester.bulk_index_events(worker_events)
        logger.info(f"🔧 Worker {worker_id}: {success_count} indexed, {failed_count} failed")
        return {
            'worker_id': worker_id,
            'generated': len(worker_events),
            'indexed': success_count,
            'failed': failed_count
        }
    
    def generate_and_ingest_threaded(self, total_events: int, num_workers: int) -> dict:
        """Generate events using multiple threads and ingest to your Elasticsearch"""
        logger.info(f"🏧 Starting ATM fraud generation with Austin, TX configuration:")
        logger.info(f"   Total events: {total_events:,}")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Fraud timeframe: NOW-{self.fraud_config.fraud_start_days_back}d to NOW-{self.fraud_config.fraud_end_days_back}d from {self.fraud_config.fraud_start_hour}:00-{self.fraud_config.fraud_end_hour}:00 (every {self.fraud_config.fraud_interval_minutes}min)")
        logger.info(f"   Fraud accounts: {len(self.fraud_config.fraud_accounts)} accounts")
        logger.info(f"   Fraud ATMs: {len(self.fraud_config.fraud_pos_ids)} ATMs")
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
        
        # Generate ATM structuring scenario first (no obvious logging)
        atm_fraud_events = self.generate_atm_structuring_scenario()
        
        # Convert ATM fraud events and ingest
        atm_fraud_events_dict = []
        for event in atm_fraud_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            atm_fraud_events_dict.append(event_dict)
        
        atm_success, atm_failed = self.ingester.bulk_index_events(atm_fraud_events_dict)
        results['total_indexed'] += atm_success
        results['total_failed'] += atm_failed
        results['total_generated'] += len(atm_fraud_events_dict)
        
        # Generate noise events using workers
        logger.info(f"📊 Generating noise events with {num_workers} workers...")
        
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
                    logger.error(f"⚠ Worker failed: {e}")
        
        return results
    
    def save_to_files(self, events: List[dict], output_dir: str = "."):
        """Save events to JSON and CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = os.path.join(output_dir, f"atm_fraud_events_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(events, f, indent=2, default=str)
        logger.info(f"💾 Saved JSON: {json_file}")
        
        # Save CSV
        csv_file = os.path.join(output_dir, f"atm_fraud_events_{timestamp}.csv")
        df = pd.DataFrame(events)
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Saved CSV: {csv_file}")
        
        return json_file, csv_file

def main():
    """Main function for ATM fraud scenario"""
    print("🏧 AML FRAUD WORKSHOP - ATM STRUCTURING SCENARIO")
    print("=" * 60)
    print(f"📍 Running from: {os.getcwd()}")
    print(f"🔗 Elasticsearch: http://localhost:30920")
    print(f"📊 Index: fraud-workshop-atm")
    print(f"👤 User: fraud")
    print(f"🔧 Workers: 16")
    print(f"📈 Events: 10,000")
    print(f"🏧 Fraud Pattern: Structured $500 deposits every 30min")
    print(f"📅 Fraud Timing: NOW-8d to NOW-1d, 9:00-21:00 (business hours)")
    print(f"👥 Fraud Accounts: 2,4,6,8,10,12,14,16,18,20")
    print(f"📹 Camera Status: Disabled at fraud ATMs")
    print(f"🕐 Format: 2025-11-22T13:47:15.984Z (Zulu)")
    print("=" * 60)
    
    # Your configurations
    fraud_config = FraudConfig()
    es_config = ElasticsearchConfig()
    
    # Initialize generator
    generator = ATMFraudGenerator(fraud_config, es_config)
    
    # Test connection to your Elasticsearch
    print("\n🔍 Testing Elasticsearch connection...")
    if not generator.ingester.test_connection():
        print("⚠ Cannot connect to your Elasticsearch")
        print(f"   Host: {es_config.host}")
        print(f"   Username: {es_config.username}")
        print(f"   Password: {es_config.password}")
        print("\n💡 Options:")
        print("   1. Check if Elasticsearch is running at the configured host")
        print("   2. Verify credentials are correct")
        print("   3. Generate data to files only (without Elasticsearch)")
        
        choice = input("\nContinue without Elasticsearch? (y/N): ").lower()
        if choice != 'y':
            return
        
        # Generate without Elasticsearch
        print("\n🚀 Generating ATM fraud data to files only...")
        start_time = time.time()
        
        # Generate events
        atm_fraud_events = generator.generate_atm_structuring_scenario()
        noise_events = generator.generate_atm_noise_events(es_config.events_per_day)
        
        all_events = []
        for event in atm_fraud_events + noise_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            all_events.append(event_dict)
        
        random.shuffle(all_events)
        
        # Save to files
        json_file, csv_file = generator.save_to_files(all_events)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Generated {len(all_events):,} events in {duration:.2f} seconds")
        print(f"📁 Files saved: {json_file}, {csv_file}")
        return
    
    # Create your index
    generator.ingester.create_index_if_not_exists()
    
    # Generate and ingest with your settings
    print(f"\n🚀 Starting ATM fraud data generation...")
    start_time = time.time()
    
    results = generator.generate_and_ingest_threaded(
        total_events=es_config.events_per_day,
        num_workers=es_config.workers
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Display results
    print(f"\n" + "=" * 60)
    print("📊 ATM FRAUD WORKSHOP COMPLETE")
    print("=" * 60)
    print(f"Total Events Generated: {results['total_generated']:,}")
    print(f"Successfully Indexed: {results['total_indexed']:,}")
    print(f"Failed: {results['total_failed']:,}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Events/second: {results['total_generated']/duration:.2f}")
    print(f"\n🎯 Your Elasticsearch Info:")
    print(f"   Index: {es_config.index_name}")
    print(f"   Host: {es_config.host}")
    print(f"   Events: {results['total_indexed']:,}")
    
    if results['total_failed'] > 0:
        print(f"\n⚠️ {results['total_failed']} events failed to index")
    else:
        print("\n✅ All events successfully indexed to your Elasticsearch!")
    
if __name__ == "__main__":
    main()
clear
####    
    
    # Display results
    print(f"\n" + "=" * 60)
    print("📊 FRAUD WORKSHOP SETUP COMPLETE")
    print("=" * 60)       
    print(f"\n🕵️ Start detecting fraud!")
    print(f"\n🔍 DETECTION CHALLENGES:")
    print(f"\n Challenge 1 - Wire Fraud")
    print(f"   - Idenitfy account clustering and temporal pattern recognition")
    print(f"   - Structural amount analysis for SAR avoidance")
    print(f"   - Wire correlation to an overseas bank")
    print(f"\n Challenge 2 - Money Laundering")
    print(f"   - Identify layering technique to obscure money trail")
    print(f"   - Find the large SAR worthy cash deposit into an account")
    print(f"   - Trace the money through account hops")
    print(f"   - Notice the progression across 5 consecutive days")
    print(f"\n Challenge 3 - Smurfing")
    print(f"   • Find the structured deposits every 30 minutes over the course of a week")
    print(f"   • Identify across all 10 smurfs/accounts")
    print(f"   • Correlate with ATM condition")
    print(f"   • Recognize this as ATM-based structuring")


if __name__ == "__main__":
    main()
