#!/bin/bash

# AML Fraud Workshop - Root Installation Script
# Configured for your specific Elasticsearch environment
# Run from: /root/Fraud-Workshop/Scripts/install_aml_generator.sh

set -e  # Exit on any error

# Set non-interactive mode for package installations
export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none
export NEEDRESTART_MODE=a

# Configure dpkg to use old config files by default (auto-accept)
export DEBIAN_PRIORITY=critical
export DEBCONF_NONINTERACTIVE_SEEN=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration - Your Specific Settings
WORKSHOP_USER="aml-workshop"
WORKSHOP_DIR="/opt/aml-fraud-workshop"
PYTHON_VERSION="3.10"

# Elasticsearch Configuration - Your Environment
ES_HOST="http://localhost:30920"
ES_USERNAME="fraud"
ES_PASSWORD="hunter"
ES_INDEX="fraud-workshop"
ES_WORKERS="16"
ES_EVENTS="10000"

# Business Hours Configuration - Your Requirements
BUSINESS_START="7"      # 7 AM
BUSINESS_END="21"       # 9 PM
BUSINESS_MULTIPLIER="7" # 7x volume during business hours
# 9:01 PM to 6:59 AM will have 1x volume (off-hours events)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE} AML Fraud Workshop Installation${NC}"
    echo -e "${PURPLE} Root Installation Mode${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

# Check if running as root (required for this version)
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        print_status "You are running from: /root/Fraud-Workshop/Scripts/"
        print_status "Please run as: sudo ./install_aml_generator.sh"
        exit 1
    fi
    print_success "Running as root - proceeding with installation"
}

# Setup non-interactive environment
setup_noninteractive_environment() {
    print_status "Configuring non-interactive installation mode..."
    
    # Prevent interactive prompts during package installation
    export DEBIAN_FRONTEND=noninteractive
    export APT_LISTCHANGES_FRONTEND=none
    export NEEDRESTART_MODE=a
    export DEBIAN_PRIORITY=critical
    export DEBCONF_NONINTERACTIVE_SEEN=true
    export UCF_FORCE_CONFFOLD=1
    export UCF_FORCE_CONFFNEW=0
    
    # Configure debconf for non-interactive mode
    echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
    
    # Configure dpkg to automatically keep existing config files
    echo 'DPkg::options { "--force-confdef"; "--force-confold"; }' > /etc/apt/apt.conf.d/99local-force-confold
    
    # Configure APT to be non-interactive
    cat > /etc/apt/apt.conf.d/99local-noninteractive << 'APTEOF'
APT::Get::Assume-Yes "true";
APT::Get::Fix-Broken "true";
APT::Get::Force-Yes "true";
Dpkg::Use-Pty "false";
Dpkg::Options {
    "--force-confdef";
    "--force-confold";
}
APTEOF
    
    # Configure needrestart to restart services automatically
    if [ -f /etc/needrestart/needrestart.conf ]; then
        sed -i 's/#$nrconf{restart} = '"'"'i'"'"';/$nrconf{restart} = '"'"'a'"'"';/' /etc/needrestart/needrestart.conf
    fi
    
    print_success "Non-interactive mode configured (will auto-accept defaults)"
}

# Check Ubuntu version
check_ubuntu_version() {
    if ! command -v lsb_release &> /dev/null; then
        print_error "Cannot determine Ubuntu version"
        exit 1
    fi

    UBUNTU_VERSION=$(lsb_release -rs)
    print_status "Detected Ubuntu version: $UBUNTU_VERSION"

    if [[ ! "$UBUNTU_VERSION" =~ ^(20\.04|22\.04|24\.04) ]]; then
        print_warning "This script is tested on Ubuntu 20.04, 22.04, and 24.04"
        print_status "Your version ($UBUNTU_VERSION) may work but is not officially supported"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages (non-interactive mode)..."
    
    # Set additional non-interactive options
    export UCF_FORCE_CONFFOLD=1
    export UCF_FORCE_CONFFNEW=0
    
    # Configure dpkg to keep old config files automatically
    echo 'DPkg::options { "--force-confdef"; "--force-confold"; }' > /etc/apt/apt.conf.d/local
    
    # Update with non-interactive flags
    apt update -qq
    apt upgrade -y -qq \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        -o APT::Get::Assume-Yes=true \
        -o APT::Get::Fix-Broken=true \
        -o APT::Get::Force-Yes=true \
        -o Dpkg::Use-Pty=0
    
    print_success "System packages updated (auto-accepted defaults)"
}

# Install essential packages
install_essentials() {
    print_status "Installing essential packages (non-interactive mode)..."
    
    apt install -y -qq \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        curl \
        wget \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        lsb-release \
        unzip \
        git \
        tree \
        jq \
        nano \
        vim \
        htop \
        build-essential
        
    print_success "Essential packages installed (auto-accepted defaults)"
}

# Install Python and pip
install_python() {
    print_status "Installing Python $PYTHON_VERSION and pip (non-interactive mode)..."
    
    # Add deadsnakes PPA for latest Python versions (non-interactive)
    add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    apt update -qq
    
    # Install Python and related packages (non-interactive)
    apt install -y -qq \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        python3-setuptools \
        python3-wheel
    
    # Update alternatives
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 > /dev/null 2>&1
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 > /dev/null 2>&1
    
    print_success "Python $PYTHON_VERSION installed (auto-accepted defaults)"
}

# Create workshop user
create_workshop_user() {
    print_status "Creating workshop user: $WORKSHOP_USER"
    
    if id "$WORKSHOP_USER" &>/dev/null; then
        print_status "User $WORKSHOP_USER already exists"
    else
        useradd -m -s /bin/bash -G sudo $WORKSHOP_USER
        echo "$WORKSHOP_USER:aml-workshop-2024" | chpasswd
        print_success "Workshop user created: $WORKSHOP_USER"
    fi
}

# Create workshop directory structure
create_workshop_directory() {
    print_status "Creating workshop directory: $WORKSHOP_DIR"
    
    mkdir -p $WORKSHOP_DIR/{src,data,config,logs,output,notebooks}
    chown -R $WORKSHOP_USER:$WORKSHOP_USER $WORKSHOP_DIR
    chmod -R 755 $WORKSHOP_DIR
    
    print_success "Workshop directory created"
}

# Install Python packages
install_python_packages() {
    print_status "Installing Python packages for fraud generation with Elasticsearch support..."
    
    # Create virtual environment as workshop user
    sudo -u $WORKSHOP_USER python3 -m venv $WORKSHOP_DIR/venv
    
    # Install packages in virtual environment
    sudo -u $WORKSHOP_USER $WORKSHOP_DIR/venv/bin/pip install --upgrade pip
    sudo -u $WORKSHOP_USER $WORKSHOP_DIR/venv/bin/pip install \
        pandas==2.1.4 \
        numpy==1.24.4 \
        python-dateutil==2.8.2 \
        requests==2.31.0 \
        elasticsearch==8.11.3 \
        jupyter==1.0.0 \
        matplotlib==3.8.2 \
        seaborn==0.13.0 \
        plotly==5.18.0 \
        scikit-learn==1.3.2
    
    print_success "Python packages installed"
}

# Create fraud generator with your specific configuration
create_fraud_generator_files() {
    print_status "Creating fraud generator files with your Elasticsearch configuration..."
    
    # Create the main fraud generator with your specific settings
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/src/fraud_workshop_generator.py > /dev/null <<EOF
#!/usr/bin/env python3
"""
AML Fraud Workshop Generator - Configured for Your Environment
Elasticsearch: $ES_HOST
Index: $ES_INDEX
Business Hours: ${BUSINESS_START}:00 AM - ${BUSINESS_END}:00 PM (${BUSINESS_MULTIPLIER}x volume)
"""

import json
import random
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import Elasticsearch
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    print("⚠️  Elasticsearch not installed. Run: pip install elasticsearch")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ElasticsearchConfig:
    """Your Elasticsearch configuration"""
    host: str = "$ES_HOST"
    index_name: str = "$ES_INDEX"
    username: str = "$ES_USERNAME"
    password: str = "$ES_PASSWORD"
    workers: int = $ES_WORKERS
    events_per_day: int = $ES_EVENTS
    pipeline: str = "fraud-detection-enrich"
    verify_certs: bool = False
    timeout: int = 30

@dataclass
class TransactionEvent:
    """Transaction event with ID fields for enrichment"""
    accountID: int
    event_amount: float
    event_type: str  # debit or credit
    account_type: str  # checking, savings, money market
    account_event: str  # deposit, fee, wire, withdrawal, purchase
    transaction_date: str
    timestamp: str
    
    # Optional ID fields for enrichment
    deposit_type: Optional[str] = None
    wire_direction: Optional[str] = None
    posID: Optional[int] = None
    txbankId: Optional[int] = None
    addressId: Optional[int] = None
    intbankID: Optional[int] = None

@dataclass
class FraudConfig:
    """Fraud configuration with your business hours"""
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
    
    # Your specific business hours: 7 AM - 9 PM with 7x volume
    business_start_hour: int = $BUSINESS_START
    business_end_hour: int = $BUSINESS_END
    business_hours_multiplier: float = $BUSINESS_MULTIPLIER

class BusinessHoursGenerator:
    """Generates timestamps with your specific business hours weighting"""
    
    def __init__(self, config: FraudConfig):
        self.config = config
    
    def get_weighted_hour(self) -> int:
        """Get hour with 7x volume during 7 AM - 9 PM, 1x during 9:01 PM - 6:59 AM"""
        hours = list(range(24))
        weights = []
        
        for hour in hours:
            if self.config.business_start_hour <= hour < self.config.business_end_hour:
                # 7 AM - 9 PM: 7x volume
                weights.append(self.config.business_hours_multiplier)
            else:
                # 9:01 PM - 6:59 AM: 1x volume (still events, just fewer)
                weights.append(1.0)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return random.choices(hours, weights=weights)[0]
    
    def generate_business_weighted_datetime(self, base_date: datetime, days_back_range: int = 8) -> datetime:
        """Generate datetime with your business hours weighting"""
        days_back = random.randint(0, days_back_range)
        weighted_hour = self.get_weighted_hour()
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        
        return (base_date - timedelta(days=days_back)).replace(
            hour=weighted_hour, minute=random_minutes, second=random_seconds
        )

class ElasticsearchIngester:
    """Handles direct ingestion to your Elasticsearch"""
    
    def __init__(self, es_config: ElasticsearchConfig):
        self.config = es_config
        self.es = None
        if ES_AVAILABLE:
            self.es = self._create_elasticsearch_client()
        
    def _create_elasticsearch_client(self) -> Optional[Elasticsearch]:
        """Create Elasticsearch client for your environment"""
        if not ES_AVAILABLE:
            return None
            
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
            logger.error(f"❌ Failed to create Elasticsearch client: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to your Elasticsearch"""
        if not self.es:
            logger.error("❌ Elasticsearch client not initialized")
            return False
            
        try:
            info = self.es.info()
            # Handle both old and new ES client response formats
            if hasattr(info, 'body'):
                version = info.body.get('version', {}).get('number', 'unknown')
            else:
                version = info.get('version', {}).get('number', 'unknown')
            
            logger.info(f"✅ Connected to your Elasticsearch at {self.config.host}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Index: {self.config.index_name}")
            logger.info(f"   Workers: {self.config.workers}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            logger.error(f"   Check if Elasticsearch is running at {self.config.host}")
            logger.error(f"   Check credentials: {self.config.username} / {self.config.password}")
            return False
    
    def create_index_if_not_exists(self):
        """Create your fraud-workshop index with proper mapping"""
        if not self.es:
            return
            
        try:
            if self.es.indices.exists(index=self.config.index_name):
                logger.info(f"📋 Index '{self.config.index_name}' already exists")
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
            logger.info(f"✅ Created index '{self.config.index_name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to create index: {e}")
    
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
            logger.error(f"❌ Bulk indexing failed: {e}")
            return 0, len(events)

class FraudDataGenerator:
    """AML fraud generator configured for your environment"""
    
    def __init__(self, fraud_config: FraudConfig, es_config: Optional[ElasticsearchConfig] = None):
        self.fraud_config = fraud_config
        self.es_config = es_config
        self.business_hours = BusinessHoursGenerator(fraud_config)
        self.ingester = ElasticsearchIngester(es_config) if es_config else None
        
    def generate_money_laundering_scenario(self) -> List[TransactionEvent]:
        """Generate money laundering scenario with your business hours"""
        ml_events = []
        base_date = datetime.now()
        
        # Target accounts for money laundering
        target_accounts = {
            'checking': [random.randint(1, 35000) for _ in range(self.fraud_config.ml_checking_accounts)],
            'savings': [random.randint(1, 35000) for _ in range(self.fraud_config.ml_savings_accounts)]
        }
        
        logger.info(f"🎯 Money Laundering Scenario:")
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
                    
                    # Use your business hours weighting
                    timestamp = self.business_hours.generate_business_weighted_datetime(deposit_date_base, 0)
                    account_type = 'checking' if account_id in target_accounts['checking'] else 'savings'
                    
                    event = TransactionEvent(
                        accountID=account_id,
                        event_amount=amount,
                        event_type='credit',
                        account_type=account_type,
                        account_event='deposit',
                        transaction_date=timestamp.strftime('%Y-%m-%d'),
                        timestamp=timestamp.isoformat(),
                        deposit_type='cash'
                    )
                    ml_events.append(event)
        
        # Generate wire transfer out to Chinese bank
        all_accounts = target_accounts['checking'] + target_accounts['savings']
        source_account = random.choice(all_accounts)
        total_deposited = sum(e.event_amount for e in ml_events if e.accountID == source_account)
        wire_amount = round(total_deposited * 0.95, 2) if total_deposited > 0 else round(random.uniform(50000, 100000), 2)
        chinese_bank_id = random.randint(1, 25)  # Chinese banks are IDs 1-25
        
        wire_date = base_date + timedelta(days=1)
        wire_timestamp = self.business_hours.generate_business_weighted_datetime(wire_date, 0)
        
        wire_event = TransactionEvent(
            accountID=source_account,
            event_amount=wire_amount,
            event_type='debit',
            account_type='checking',
            account_event='wire',
            transaction_date=wire_timestamp.strftime('%Y-%m-%d'),
            timestamp=wire_timestamp.isoformat(),
            wire_direction='outbound',
            intbankID=chinese_bank_id
        )
        ml_events.append(wire_event)
        
        logger.info(f"💰 Generated {len(ml_events)} money laundering events")
        logger.info(f"🏦 Wire transfer: \${wire_amount:,.2f} to Chinese bank ID {chinese_bank_id}")
        
        return ml_events
    
    def generate_daily_noise_events(self, events_per_worker: int) -> List[TransactionEvent]:
        """Generate noise events with your business hours weighting"""
        noise_events = []
        base_date = datetime.now()
        
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
            timestamp = self.business_hours.generate_business_weighted_datetime(base_date)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1.00, 9599.99), 2),
                event_type='credit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='deposit',
                transaction_date=timestamp.strftime('%Y-%m-%d'),
                timestamp=timestamp.isoformat(),
                deposit_type=random.choice(deposit_types)
            )
            noise_events.append(event)
        
        # Fees
        for _ in range(fee_count):
            timestamp = self.business_hours.generate_business_weighted_datetime(base_date)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1.50, 15.00), 2),
                event_type='debit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='fee',
                transaction_date=timestamp.strftime('%Y-%m-%d'),
                timestamp=timestamp.isoformat()
            )
            noise_events.append(event)
        
        # Wire transfers
        for _ in range(wire_count):
            event_type = random.choice(['debit', 'credit'])
            is_international = random.random() < self.fraud_config.international_wire_percentage
            timestamp = self.business_hours.generate_business_weighted_datetime(base_date)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(0, 9599.99), 2),
                event_type=event_type,
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='wire',
                transaction_date=timestamp.strftime('%Y-%m-%d'),
                timestamp=timestamp.isoformat(),
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
            timestamp = self.business_hours.generate_business_weighted_datetime(base_date)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(20.00, 9599.99), 2),
                event_type='debit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='withdrawal',
                transaction_date=timestamp.strftime('%Y-%m-%d'),
                timestamp=timestamp.isoformat()
            )
            noise_events.append(event)
        
        # Purchases
        for _ in range(purchase_count):
            timestamp = self.business_hours.generate_business_weighted_datetime(base_date)
            
            event = TransactionEvent(
                accountID=random.randint(1, 35000),
                event_amount=round(random.uniform(1.00, 9599.99), 2),
                event_type='debit',
                account_type=random.choice(['checking', 'savings', 'money market']),
                account_event='purchase',
                transaction_date=timestamp.strftime('%Y-%m-%d'),
                timestamp=timestamp.isoformat(),
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
        
        # Ingest to your Elasticsearch if configured
        if self.ingester:
            success_count, failed_count = self.ingester.bulk_index_events(worker_events)
            logger.info(f"🔧 Worker {worker_id}: {success_count} indexed, {failed_count} failed")
            return {
                'worker_id': worker_id,
                'generated': len(worker_events),
                'indexed': success_count,
                'failed': failed_count
            }
        
        return {
            'worker_id': worker_id,
            'generated': len(worker_events),
            'indexed': 0,
            'failed': 0
        }
    
    def generate_and_ingest_threaded(self, total_events: int, num_workers: int) -> dict:
        """Generate events using multiple threads and ingest to your Elasticsearch"""
        logger.info(f"🚀 Starting threaded generation with your configuration:")
        logger.info(f"   Total events: {total_events:,}")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Business hours: {self.fraud_config.business_start_hour}:00 - {self.fraud_config.business_end_hour}:00 ({self.fraud_config.business_hours_multiplier}x volume)")
        logger.info(f"   Off hours: {self.fraud_config.business_end_hour+1}:00 - {self.fraud_config.business_start_hour-1}:59 (1x volume)")
        
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
        logger.info("🚨 Generating money laundering scenario...")
        ml_events = self.generate_money_laundering_scenario()
        
        # Convert ML events and ingest
        ml_events_dict = []
        for event in ml_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            ml_events_dict.append(event_dict)
        
        if self.ingester:
            ml_success, ml_failed = self.ingester.bulk_index_events(ml_events_dict)
            logger.info(f"💰 ML Events: {ml_success} indexed to '{self.es_config.index_name}', {ml_failed} failed")
            results['total_indexed'] += ml_success
            results['total_failed'] += ml_failed
        
        results['total_generated'] += len(ml_events_dict)
        
        # Generate noise events using your 16 workers
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
                    logger.error(f"❌ Worker failed: {e}")
        
        return results

def main():
    """Main function with your specific configuration"""
    print("💰 AML FRAUD WORKSHOP - YOUR CONFIGURATION")
    print("=" * 60)
    print(f"🔍 Elasticsearch: {ElasticsearchConfig().host}")
    print(f"📊 Index: {ElasticsearchConfig().index_name}")
    print(f"👤 User: {ElasticsearchConfig().username}")
    print(f"🔧 Workers: {ElasticsearchConfig().workers}")
    print(f"📈 Events: {ElasticsearchConfig().events_per_day:,}")
    print(f"⏰ Business Hours: {FraudConfig().business_start_hour}:00 - {FraudConfig().business_end_hour}:00 ({FraudConfig().business_hours_multiplier}x volume)")
    print("=" * 60)
    
    # Your configurations
    fraud_config = FraudConfig()
    es_config = ElasticsearchConfig()
    
    # Initialize generator
    generator = FraudDataGenerator(fraud_config, es_config)
    
    # Test connection to your Elasticsearch
    if not generator.ingester.test_connection():
        print("❌ Cannot connect to your Elasticsearch")
        print(f"   Host: {es_config.host}")
        print(f"   Username: {es_config.username}")
        print(f"   Password: {es_config.password}")
        print("   Please check your Elasticsearch service and configuration")
        return
    
    # Create your index
    generator.ingester.create_index_if_not_exists()
    
    # Generate and ingest with your settings
    print(f"\\n🚀 Starting fraud data generation for your workshop...")
    start_time = time.time()
    
    results = generator.generate_and_ingest_threaded(
        total_events=es_config.events_per_day,
        num_workers=es_config.workers
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Display results
    print(f"\\n" + "=" * 60)
    print("📊 WORKSHOP DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Events Generated: {results['total_generated']:,}")
    print(f"Successfully Indexed: {results['total_indexed']:,}")
    print(f"Failed: {results['total_failed']:,}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Events/second: {results['total_generated']/duration:.2f}")
    print(f"\\n🎯 Your Elasticsearch Info:")
    print(f"   Index: {es_config.index_name}")
    print(f"   Host: {es_config.host}")
    print(f"   Events: {results['total_indexed']:,}")
    
    if results['total_failed'] > 0:
        print(f"\\n⚠️  {results['total_failed']} events failed to index")
    else:
        print("\\n✅ All events successfully indexed to your Elasticsearch!")
    
    print(f"\\n🕵️ Start detecting fraud in index '{es_config.index_name}'!")

if __name__ == "__main__":
    main()
EOF

    # Create a quick run script with your settings
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/src/run_workshop.py > /dev/null <<EOF
#!/usr/bin/env python3
"""
Quick run script for your AML Fraud Workshop
"""

from fraud_workshop_generator import FraudDataGenerator, FraudConfig, ElasticsearchConfig

def main():
    print("🎯 AML FRAUD WORKSHOP - QUICK START")
    print("=" * 40)
    print("Using your configuration:")
    print("  • Elasticsearch: $ES_HOST")
    print("  • Index: $ES_INDEX")
    print("  • Workers: $ES_WORKERS")
    print("  • Events: $ES_EVENTS")
    print("  • Business Hours: ${BUSINESS_START}:00 - ${BUSINESS_END}:00 (${BUSINESS_MULTIPLIER}x volume)")
    print()
    
    # Your predefined configuration
    fraud_config = FraudConfig()
    es_config = ElasticsearchConfig()
    
    # Run generator
    generator = FraudDataGenerator(fraud_config, es_config)
    
    # Test connection
    if generator.ingester.test_connection():
        print("🚀 Generating fraud data...")
        results = generator.generate_and_ingest_threaded(
            es_config.events_per_day,
            es_config.workers
        )
        print(f"✅ Complete! {results['total_indexed']:,} events in '{es_config.index_name}'")
    else:
        print("❌ Connection failed. Check Elasticsearch service.")

if __name__ == "__main__":
    main()
EOF

    # Create configuration test script
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/src/test_config.py > /dev/null <<EOF
#!/usr/bin/env python3
"""
Test script for your Elasticsearch configuration
"""

from fraud_workshop_generator import ElasticsearchConfig, ElasticsearchIngester

def main():
    print("🔍 TESTING YOUR ELASTICSEARCH CONFIGURATION")
    print("=" * 50)
    
    config = ElasticsearchConfig()
    print(f"Host: {config.host}")
    print(f"Index: {config.index_name}")
    print(f"Username: {config.username}")
    print(f"Password: {config.password}")
    print(f"Workers: {config.workers}")
    print(f"Events: {config.events_per_day}")
    print()
    
    print("Testing connection...")
    ingester = ElasticsearchIngester(config)
    
    if ingester.test_connection():
        print("✅ Connection successful!")
        print("Creating index if needed...")
        ingester.create_index_if_not_exists()
        print("✅ Ready for fraud data generation!")
    else:
        print("❌ Connection failed!")
        print("Check:")
        print(f"  1. Elasticsearch running at {config.host}")
        print(f"  2. Credentials: {config.username} / {config.password}")
        print(f"  3. Network connectivity")

if __name__ == "__main__":
    main()
EOF

    # Create startup script
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/start_fraud_workshop.sh > /dev/null <<EOF
#!/bin/bash

echo "🎯 AML FRAUD WORKSHOP - YOUR ENVIRONMENT"
echo "========================================"
echo "Elasticsearch: $ES_HOST"
echo "Index: $ES_INDEX"
echo "Workers: $ES_WORKERS workers"
echo "Events: $ES_EVENTS events"
echo "Business Hours: ${BUSINESS_START}:00 AM - ${BUSINESS_END}:00 PM (${BUSINESS_MULTIPLIER}x volume)"
echo "Off Hours: ${BUSINESS_END}:01 PM - $(printf "%02d" $((BUSINESS_START-1))):59 AM (1x volume)"
echo "========================================"
echo

# Activate virtual environment
source $WORKSHOP_DIR/venv/bin/activate
cd $WORKSHOP_DIR/src

echo "Available commands:"
echo "  1. python test_config.py           - Test Elasticsearch connection"
echo "  2. python run_workshop.py          - Generate fraud data with your config"
echo "  3. python fraud_workshop_generator.py - Full featured generator"
echo
echo "Quick start: python run_workshop.py"
echo

# Start bash in the environment
bash
EOF

    chmod +x $WORKSHOP_DIR/start_fraud_workshop.sh
    
    print_success "Fraud generator configured for your environment"
}

# Create documentation with your settings
create_documentation() {
    print_status "Creating documentation with your configuration..."
    
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/README.md > /dev/null <<EOF
# AML Fraud Workshop - Your Configuration

## Your Elasticsearch Setup

- **Host**: $ES_HOST
- **Index**: $ES_INDEX
- **Username**: $ES_USERNAME
- **Password**: $ES_PASSWORD
- **Workers**: $ES_WORKERS
- **Events**: $ES_EVENTS

## Your Business Hours

- **Business Hours**: ${BUSINESS_START}:00 AM - ${BUSINESS_END}:00 PM
- **Business Volume**: ${BUSINESS_MULTIPLIER}x normal volume
- **Off Hours**: ${BUSINESS_END}:01 PM - $(printf "%02d" $((BUSINESS_START-1))):59 AM
- **Off Hours Volume**: 1x normal volume (events still occur)

## Quick Start

1. **Switch to workshop user**:
   \`\`\`bash
   su - aml-workshop
   \`\`\`

2. **Start workshop environment**:
   \`\`\`bash
   $WORKSHOP_DIR/start_fraud_workshop.sh
   \`\`\`

3. **Test your Elasticsearch**:
   \`\`\`bash
   python test_config.py
   \`\`\`

4. **Generate fraud data**:
   \`\`\`bash
   python run_workshop.py
   \`\`\`

## Generated Data

Your fraud workshop will generate:

- **Money Laundering Events**: Structured cash deposits (\$9,000-\$9,999) followed by wire transfers to Chinese banks
- **Realistic Noise**: Business hours weighted transactions across all event types
- **$ES_EVENTS total events** distributed across **$ES_WORKERS workers**
- **Real-time ingestion** to your **$ES_INDEX** index

## Business Hours Distribution

With your ${BUSINESS_MULTIPLIER}x multiplier:

- **${BUSINESS_START}:00 AM - ${BUSINESS_END}:00 PM**: ~${BUSINESS_MULTIPLIER}x more transactions (peak business activity)
- **${BUSINESS_END}:01 PM - $(printf "%02d" $((BUSINESS_START-1))):59 AM**: Normal volume (overnight/early morning activity)

This creates a realistic banking transaction pattern where most activity occurs during business hours but some transactions still happen 24/7.

## Files

- \`fraud_workshop_generator.py\` - Main generator with your config
- \`run_workshop.py\` - Quick start script
- \`test_config.py\` - Connection test
- \`start_fraud_workshop.sh\` - Workshop startup script
EOF
    
    print_success "Documentation created with your settings"
}

# Configure system services
setup_system() {
    print_status "Configuring system for root installation..."
    
    # Set proper ownership
    chown -R $WORKSHOP_USER:$WORKSHOP_USER $WORKSHOP_DIR
    chmod -R 755 $WORKSHOP_DIR
    
    # Make scripts executable
    chmod +x $WORKSHOP_DIR/start_fraud_workshop.sh
    chmod +x $WORKSHOP_DIR/src/*.py
    
    print_success "System configured for workshop user"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check Python environment
    if [ -f "$WORKSHOP_DIR/venv/bin/python" ]; then
        print_success "Python virtual environment created"
        python_version=$(sudo -u $WORKSHOP_USER $WORKSHOP_DIR/venv/bin/python --version)
        print_status "Python version: $python_version"
        
        # Test package imports
        if sudo -u $WORKSHOP_USER $WORKSHOP_DIR/venv/bin/python -c "import pandas, numpy, elasticsearch; print('✅ Core packages available')" 2>/dev/null; then
            print_success "Core Python packages installed"
        else
            print_warning "Some Python packages may be missing"
        fi
    else
        print_error "Python virtual environment not found"
    fi
    
    # Check workshop files
    if [ -f "$WORKSHOP_DIR/src/fraud_workshop_generator.py" ]; then
        print_success "Your fraud generator files created"
    else
        print_warning "Generator files not found"
    fi
    
    print_success "Installation verification complete"
}

# Display completion information
display_completion_info() {
    print_header
    print_success "AML Fraud Workshop installation completed!"
    echo ""
    echo -e "${CYAN}📋 Your Configuration:${NC}"
    echo "   🔍 Elasticsearch: $ES_HOST"
    echo "   📊 Index: $ES_INDEX"  
    echo "   👤 Username: $ES_USERNAME"
    echo "   🔧 Workers: $ES_WORKERS"
    echo "   📈 Events: $ES_EVENTS"
    echo "   ⏰ Business Hours: ${BUSINESS_START}:00 AM - ${BUSINESS_END}:00 PM (${BUSINESS_MULTIPLIER}x volume)"
    echo "   🌙 Off Hours: Events still generated at 1x volume"
    echo "   🤖 Installation: Non-interactive (auto-accepted package defaults)"
    echo ""
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    echo "   1. Switch to workshop user:"
    echo "      su - $WORKSHOP_USER"
    echo ""
    echo "   2. Start workshop:"
    echo "      $WORKSHOP_DIR/start_fraud_workshop.sh"
    echo ""
    echo "   3. Test Elasticsearch:"
    echo "      python test_config.py"
    echo ""
    echo "   4. Generate fraud data:"
    echo "      python run_workshop.py"
    echo ""
    echo -e "${CYAN}📁 Workshop Directory:${NC}"
    echo "   📂 $WORKSHOP_DIR/src/         - Your generator scripts"
    echo "   📂 $WORKSHOP_DIR/config/      - Configuration files"
    echo "   📂 $WORKSHOP_DIR/logs/        - Log files"
    echo ""
    echo -e "${CYAN}🔐 Access:${NC}"
    echo "   👤 Workshop user: $WORKSHOP_USER"
    echo "   🔑 Password: aml-workshop-2024"
    echo ""
    echo -e "${GREEN}✅ Ready to generate AML fraud data for your Elasticsearch!${NC}"
}

# Main installation function
main() {
    print_header
    print_status "Installing AML Fraud Workshop with your specific configuration..."
    print_status "Running from: /root/Fraud-Workshop/Scripts/"
    print_status "Mode: Non-interactive (auto-accepts package defaults)"
    echo ""
    
    # Pre-flight checks
    check_root
    setup_noninteractive_environment
    check_ubuntu_version
    
    # System setup
    update_system
    install_essentials
    install_python
    
    # Workshop setup
    create_workshop_user
    create_workshop_directory
    install_python_packages
    create_fraud_generator_files
    create_documentation
    setup_system
    
    # Verification
    verify_installation
    
    # Cleanup non-interactive configurations
    cleanup_noninteractive_environment
    
    # Completion
    display_completion_info
}

# Cleanup non-interactive environment
cleanup_noninteractive_environment() {
    print_status "Cleaning up non-interactive configuration..."
    
    # Remove APT configuration files we created
    rm -f /etc/apt/apt.conf.d/99local-force-confold
    rm -f /etc/apt/apt.conf.d/99local-noninteractive
    
    # Reset debconf to interactive mode
    echo 'debconf debconf/frontend select Dialog' | debconf-set-selections
    
    # Unset environment variables
    unset DEBIAN_FRONTEND
    unset APT_LISTCHANGES_FRONTEND
    unset NEEDRESTART_MODE
    unset DEBIAN_PRIORITY
    unset DEBCONF_NONINTERACTIVE_SEEN
    unset UCF_FORCE_CONFFOLD
    unset UCF_FORCE_CONFFNEW
    
    print_success "Interactive mode restored for future package installations"
}

# Error handling
trap 'print_error "Installation failed at line $LINENO. Check output above."' ERR

# Run main installation
main

exit 0
