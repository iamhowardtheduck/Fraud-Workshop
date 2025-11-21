#!/bin/bash

# AML Fraud Workshop - Data Generator Only Installation Script
# Lightweight setup for fraud data generation without Elasticsearch
# Compatible with Ubuntu 20.04, 22.04, and 24.04

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
WORKSHOP_USER="aml-workshop"
WORKSHOP_DIR="/opt/aml-fraud-workshop"
PYTHON_VERSION="3.10"

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
    echo -e "${PURPLE} AML Fraud Data Generator Installation${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        print_status "Please run as a regular user with sudo privileges"
        exit 1
    fi
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
    print_status "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    print_success "System packages updated"
}

# Install essential packages
install_essentials() {
    print_status "Installing essential packages..."
    sudo apt install -y \
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
        htop
    print_success "Essential packages installed"
}

# Install Python and pip
install_python() {
    print_status "Installing Python $PYTHON_VERSION and pip..."
    
    # Add deadsnakes PPA for latest Python versions
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    
    # Install Python and related packages
    sudo apt install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        python3-setuptools \
        python3-wheel
    
    # Update alternatives
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
    
    print_success "Python $PYTHON_VERSION installed"
}

# Create workshop user
create_workshop_user() {
    print_status "Creating workshop user: $WORKSHOP_USER"
    
    if id "$WORKSHOP_USER" &>/dev/null; then
        print_status "User $WORKSHOP_USER already exists"
    else
        sudo useradd -m -s /bin/bash -G sudo $WORKSHOP_USER
        echo "$WORKSHOP_USER:aml-workshop-2024" | sudo chpasswd
        print_success "Workshop user created: $WORKSHOP_USER"
    fi
}

# Create workshop directory structure
create_workshop_directory() {
    print_status "Creating workshop directory: $WORKSHOP_DIR"
    
    sudo mkdir -p $WORKSHOP_DIR/{src,data,config,logs,output,notebooks}
    sudo chown -R $WORKSHOP_USER:$WORKSHOP_USER $WORKSHOP_DIR
    sudo chmod -R 755 $WORKSHOP_DIR
    
    print_success "Workshop directory created"
}

# Install Python packages
install_python_packages() {
    print_status "Installing Python packages for fraud generation..."
    
    # Create virtual environment
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

# Create fraud generator files
create_fraud_generator_files() {
    print_status "Creating fraud generator files..."
    
    # Create main fraud generator script
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/src/lightweight_fraud_generator.py > /dev/null <<'EOF'
#!/usr/bin/env python3
"""
AML Fraud Data Generator - Standalone Version
Generates realistic money laundering scenarios and banking transaction data
"""

import json
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Configuration for fraud generation"""
    # Money laundering parameters
    ml_cash_deposits_min: float = 9000.0
    ml_cash_deposits_max: float = 9999.99
    ml_checking_accounts: int = 3
    ml_savings_accounts: int = 5
    ml_days_span: int = 5
    
    # Event distribution
    events_per_day: int = 10000
    deposit_percentage: float = 0.05
    fee_percentage: float = 0.10
    wire_percentage: float = 0.20
    withdrawal_percentage: float = 0.25
    purchase_percentage: float = 0.40
    international_wire_percentage: float = 0.10
    
    # Business hours
    business_start_hour: int = 8
    business_end_hour: int = 18
    business_hours_multiplier: float = 3.0

class BusinessHoursGenerator:
    """Generates timestamps with business hours weighting"""
    
    def __init__(self, config: FraudConfig):
        self.config = config
    
    def get_weighted_hour(self) -> int:
        """Get hour with business hours weighting"""
        hours = list(range(24))
        weights = []
        
        for hour in hours:
            if self.config.business_start_hour <= hour < self.config.business_end_hour:
                weights.append(self.config.business_hours_multiplier)
            else:
                weights.append(1.0)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return random.choices(hours, weights=weights)[0]
    
    def generate_business_weighted_datetime(self, base_date: datetime, days_back_range: int = 8) -> datetime:
        """Generate datetime with business hours weighting"""
        days_back = random.randint(0, days_back_range)
        weighted_hour = self.get_weighted_hour()
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        
        return (base_date - timedelta(days=days_back)).replace(
            hour=weighted_hour, minute=random_minutes, second=random_seconds
        )

class FraudDataGenerator:
    """Main fraud data generator"""
    
    def __init__(self, config: FraudConfig):
        self.config = config
        self.business_hours = BusinessHoursGenerator(config)
        
    def generate_money_laundering_scenario(self) -> List[TransactionEvent]:
        """Generate money laundering scenario"""
        ml_events = []
        base_date = datetime.now()
        
        # Target accounts
        target_accounts = {
            'checking': [random.randint(1, 35000) for _ in range(self.config.ml_checking_accounts)],
            'savings': [random.randint(1, 35000) for _ in range(self.config.ml_savings_accounts)]
        }
        
        logger.info(f"🎯 Money Laundering Accounts:")
        logger.info(f"   Checking: {target_accounts['checking']}")
        logger.info(f"   Savings: {target_accounts['savings']}")
        
        # Generate deposits over time span
        for day in range(self.config.ml_days_span):
            deposit_date_base = base_date - timedelta(days=day)
            
            for account_id in target_accounts['checking'] + target_accounts['savings']:
                if random.random() < 0.8:  # 80% chance
                    amount = round(random.uniform(
                        self.config.ml_cash_deposits_min, 
                        self.config.ml_cash_deposits_max
                    ), 2)
                    
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
        
        # Generate wire transfer
        all_accounts = target_accounts['checking'] + target_accounts['savings']
        source_account = random.choice(all_accounts)
        wire_amount = round(sum(e.event_amount for e in ml_events if e.accountID == source_account) * 0.95, 2)
        chinese_bank_id = random.randint(1, 25)
        
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
        logger.info(f"🏦 Wire: ${wire_amount:,.2f} to Chinese bank {chinese_bank_id}")
        
        return ml_events
    
    def generate_noise_events(self, events_count: int) -> List[TransactionEvent]:
        """Generate noise events"""
        noise_events = []
        base_date = datetime.now()
        
        # Calculate distribution
        deposit_count = int(events_count * self.config.deposit_percentage)
        fee_count = int(events_count * self.config.fee_percentage)
        wire_count = int(events_count * self.config.wire_percentage)
        withdrawal_count = int(events_count * self.config.withdrawal_percentage)
        purchase_count = events_count - (deposit_count + fee_count + wire_count + withdrawal_count)
        
        logger.info(f"📊 Generating {events_count} noise events:")
        logger.info(f"   {deposit_count} deposits, {fee_count} fees, {wire_count} wires")
        logger.info(f"   {withdrawal_count} withdrawals, {purchase_count} purchases")
        
        # Generate deposits
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
        
        # Generate fees
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
        
        # Generate wires
        for _ in range(wire_count):
            event_type = random.choice(['debit', 'credit'])
            is_international = random.random() < self.config.international_wire_percentage
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
        
        # Generate withdrawals
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
        
        # Generate purchases
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
    
    def generate_all_events(self) -> List[dict]:
        """Generate complete dataset"""
        all_events = []
        
        # Generate fraud scenario
        ml_events = self.generate_money_laundering_scenario()
        
        # Generate noise
        noise_events = self.generate_noise_events(self.config.events_per_day)
        
        # Combine and convert
        all_events.extend(ml_events)
        all_events.extend(noise_events)
        random.shuffle(all_events)
        
        # Convert to dictionaries
        events_dict = []
        for event in all_events:
            event_dict = asdict(event)
            event_dict = {k: v for k, v in event_dict.items() if v is not None}
            events_dict.append(event_dict)
        
        return events_dict

def export_to_json(events: List[dict], filename: str):
    """Export events to JSON file"""
    with open(filename, 'w') as f:
        json.dump(events, f, indent=2, default=str)
    logger.info(f"📁 Exported {len(events):,} events to {filename}")

def export_to_csv(events: List[dict], filename: str):
    """Export events to CSV file"""
    import pandas as pd
    df = pd.DataFrame(events)
    df.to_csv(filename, index=False)
    logger.info(f"📊 Exported {len(events):,} events to {filename}")

def main():
    """Main function"""
    print("💰 AML FRAUD DATA GENERATOR")
    print("=" * 50)
    
    # Configuration
    config = FraudConfig(
        events_per_day=10000,
        ml_cash_deposits_min=9000.0,
        ml_cash_deposits_max=9999.99
    )
    
    # Generate data
    generator = FraudDataGenerator(config)
    start_time = time.time()
    
    events = generator.generate_all_events()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Export files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = f"../output/fraud_events_{timestamp}.json"
    csv_file = f"../output/fraud_events_{timestamp}.csv"
    
    export_to_json(events, json_file)
    export_to_csv(events, csv_file)
    
    # Summary
    ml_deposits = [e for e in events if e.get('deposit_type') == 'cash' and e.get('event_amount', 0) >= 9000]
    wire_transfers = [e for e in events if e.get('account_event') == 'wire' and e.get('intbankID')]
    
    print("\n" + "=" * 50)
    print("📊 GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total Events: {len(events):,}")
    print(f"Money Laundering Deposits: {len(ml_deposits)}")
    print(f"International Wires: {len(wire_transfers)}")
    if ml_deposits:
        print(f"ML Amount: ${sum(e['event_amount'] for e in ml_deposits):,.2f}")
    print(f"Generation Time: {duration:.2f} seconds")
    print(f"Events/Second: {len(events)/duration:.2f}")
    print(f"\n📁 Files created:")
    print(f"   JSON: {json_file}")
    print(f"   CSV: {csv_file}")

if __name__ == "__main__":
    main()
EOF

    # Create simple configuration script
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/src/generate_fraud_data.py > /dev/null <<'EOF'
#!/usr/bin/env python3
"""
Simple fraud data generation script with configuration options
"""

from lightweight_fraud_generator import FraudDataGenerator, FraudConfig, export_to_json, export_to_csv
from datetime import datetime
import os

def main():
    print("🎯 AML FRAUD DATA GENERATOR")
    print("=" * 40)
    
    # Get user preferences
    print("\n📋 Configuration:")
    
    try:
        events_count = int(input("Number of events to generate [10000]: ") or "10000")
        difficulty = input("Difficulty level (easy/medium/hard) [medium]: ") or "medium"
    except ValueError:
        events_count = 10000
        difficulty = "medium"
    
    # Configure based on difficulty
    if difficulty.lower() == "easy":
        config = FraudConfig(
            events_per_day=events_count,
            ml_checking_accounts=2,
            ml_savings_accounts=3,
            ml_cash_deposits_min=9500.0,  # More obvious amounts
            business_hours_multiplier=2.0
        )
    elif difficulty.lower() == "hard":
        config = FraudConfig(
            events_per_day=events_count,
            ml_checking_accounts=5,
            ml_savings_accounts=7,
            ml_cash_deposits_min=8000.0,  # Lower amounts, harder to detect
            ml_cash_deposits_max=9800.0,
            business_hours_multiplier=4.0
        )
    else:  # medium
        config = FraudConfig(events_per_day=events_count)
    
    print(f"\n🚀 Generating {events_count:,} events ({difficulty} difficulty)...")
    
    # Generate data
    generator = FraudDataGenerator(config)
    events = generator.generate_all_events()
    
    # Create output directory
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = f"{output_dir}/aml_fraud_{difficulty}_{timestamp}.json"
    csv_file = f"{output_dir}/aml_fraud_{difficulty}_{timestamp}.csv"
    
    export_to_json(events, json_file)
    export_to_csv(events, csv_file)
    
    # Analysis summary
    ml_deposits = [e for e in events if e.get('deposit_type') == 'cash' and e.get('event_amount', 0) >= 8000]
    
    print(f"\n✅ Generation Complete!")
    print(f"   Total Events: {len(events):,}")
    print(f"   Suspicious Deposits: {len(ml_deposits)}")
    print(f"   Files: {json_file}, {csv_file}")

if __name__ == "__main__":
    main()
EOF

    # Create analysis script
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/src/analyze_fraud_data.py > /dev/null <<'EOF'
#!/usr/bin/env python3
"""
Simple fraud data analysis script
"""

import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_latest_data():
    """Load the most recent fraud data file"""
    import os
    import glob
    
    # Find latest JSON file
    json_files = glob.glob("../output/aml_fraud_*.json")
    if not json_files:
        json_files = glob.glob("../output/fraud_events_*.json")
    
    if not json_files:
        print("❌ No fraud data files found in ../output/")
        print("   Run generate_fraud_data.py first")
        return None
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📊 Loading data from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        events = json.load(f)
    
    return pd.DataFrame(events)

def detect_money_laundering(df):
    """Detect money laundering patterns"""
    print("\n🔍 MONEY LAUNDERING DETECTION")
    print("=" * 40)
    
    # Find suspicious deposits
    suspicious = df[
        (df['account_event'] == 'deposit') &
        (df['deposit_type'] == 'cash') &
        (df['event_amount'] >= 8000) &
        (df['event_amount'] < 10000)
    ]
    
    if suspicious.empty:
        print("❌ No suspicious deposits found")
        return
    
    print(f"🚨 Found {len(suspicious)} suspicious cash deposits")
    print(f"💰 Total amount: ${suspicious['event_amount'].sum():,.2f}")
    
    # Group by account
    account_analysis = suspicious.groupby('accountID').agg({
        'event_amount': ['count', 'sum', 'mean'],
        'timestamp': ['min', 'max']
    }).round(2)
    
    account_analysis.columns = ['deposit_count', 'total_amount', 'avg_amount', 'first_deposit', 'last_deposit']
    
    # Find high-risk accounts
    high_risk = account_analysis[account_analysis['deposit_count'] >= 3].sort_values('total_amount', ascending=False)
    
    print(f"\n🎯 High-Risk Accounts (3+ deposits):")
    print("-" * 50)
    for account_id, row in high_risk.head(10).iterrows():
        print(f"Account {account_id:5d}: {row['deposit_count']} deposits, ${row['total_amount']:8,.2f}")
    
    # Find wire transfers
    wires = df[
        (df['account_event'] == 'wire') &
        (df['wire_direction'] == 'outbound') &
        (df['intbankID'].notna())
    ]
    
    if not wires.empty:
        print(f"\n🏦 International Wire Transfers: {len(wires)}")
        print(f"💸 Total wired: ${wires['event_amount'].sum():,.2f}")
        
        # Check for correlated accounts
        suspicious_accounts = set(suspicious['accountID'].unique())
        wire_accounts = set(wires['accountID'].unique())
        correlated = suspicious_accounts.intersection(wire_accounts)
        
        if correlated:
            print(f"\n🚨 CORRELATED ACCOUNTS (deposits + wires): {len(correlated)}")
            for account_id in list(correlated)[:5]:
                deposits_total = suspicious[suspicious['accountID'] == account_id]['event_amount'].sum()
                wires_total = wires[wires['accountID'] == account_id]['event_amount'].sum()
                print(f"   Account {account_id}: ${deposits_total:,.2f} deposited → ${wires_total:,.2f} wired")

def create_visualizations(df):
    """Create fraud detection visualizations"""
    print("\n📊 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Transaction amounts by type
    df.boxplot(column='event_amount', by='account_event', ax=axes[0,0])
    axes[0,0].set_title('Transaction Amounts by Type')
    axes[0,0].set_xlabel('Transaction Type')
    axes[0,0].set_ylabel('Amount ($)')
    
    # 2. Hourly transaction distribution
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly_counts = df['hour'].value_counts().sort_index()
    axes[0,1].bar(hourly_counts.index, hourly_counts.values)
    axes[0,1].set_title('Transaction Distribution by Hour')
    axes[0,1].set_xlabel('Hour of Day')
    axes[0,1].set_ylabel('Transaction Count')
    
    # 3. Suspicious deposit amounts
    suspicious = df[
        (df['account_event'] == 'deposit') &
        (df['deposit_type'] == 'cash') &
        (df['event_amount'] >= 8000)
    ]
    
    if not suspicious.empty:
        axes[1,0].hist(suspicious['event_amount'], bins=20, alpha=0.7, color='red')
        axes[1,0].axvline(x=10000, color='black', linestyle='--', label='$10K Threshold')
        axes[1,0].set_title('Suspicious Cash Deposit Amounts')
        axes[1,0].set_xlabel('Amount ($)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
    
    # 4. Account type distribution
    account_type_counts = df['account_type'].value_counts()
    axes[1,1].pie(account_type_counts.values, labels=account_type_counts.index, autopct='%1.1f%%')
    axes[1,1].set_title('Distribution by Account Type')
    
    plt.tight_layout()
    plt.savefig('../output/fraud_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Visualizations saved to ../output/fraud_analysis.png")

def main():
    """Main analysis function"""
    print("🕵️ AML FRAUD DATA ANALYSIS")
    print("=" * 40)
    
    # Load data
    df = load_latest_data()
    if df is None:
        return
    
    print(f"📊 Loaded {len(df):,} transaction events")
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"   Total amount: ${df['event_amount'].sum():,.2f}")
    print(f"   Unique accounts: {df['accountID'].nunique():,}")
    print(f"   Event types: {', '.join(df['account_event'].unique())}")
    
    # Detect fraud
    detect_money_laundering(df)
    
    # Create visualizations
    try:
        create_visualizations(df)
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
EOF
    
    # Create workshop startup script
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/start_generator.sh > /dev/null <<'EOF'
#!/bin/bash

WORKSHOP_DIR="/opt/aml-fraud-workshop"

echo "🎯 AML Fraud Data Generator"
echo "=========================="
echo ""

# Activate virtual environment
source $WORKSHOP_DIR/venv/bin/activate

cd $WORKSHOP_DIR/src

echo "Available commands:"
echo "  1. python generate_fraud_data.py    - Interactive data generation"
echo "  2. python lightweight_fraud_generator.py - Full featured generator"
echo "  3. python analyze_fraud_data.py     - Analyze generated data"
echo ""
echo "📁 Output files will be saved to: $WORKSHOP_DIR/output/"
echo ""

# Start interactive shell
bash
EOF

    chmod +x $WORKSHOP_DIR/start_generator.sh
    
    # Create sample configuration
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/config/generator_config.py > /dev/null <<'EOF'
#!/usr/bin/env python3
"""
AML Fraud Generator Configuration Templates
"""

# Workshop Size Configurations
WORKSHOP_SMALL = {
    'events_per_day': 5000,
    'ml_checking_accounts': 2,
    'ml_savings_accounts': 3,
    'difficulty': 'easy'
}

WORKSHOP_MEDIUM = {
    'events_per_day': 15000,
    'ml_checking_accounts': 3,
    'ml_savings_accounts': 5,
    'difficulty': 'medium'
}

WORKSHOP_LARGE = {
    'events_per_day': 50000,
    'ml_checking_accounts': 5,
    'ml_savings_accounts': 7,
    'difficulty': 'hard'
}

# Fraud Scenario Templates
BASIC_STRUCTURING = {
    'ml_cash_deposits_min': 9000.0,
    'ml_cash_deposits_max': 9999.0,
    'ml_days_span': 5
}

ADVANCED_LAYERING = {
    'ml_cash_deposits_min': 8000.0,
    'ml_cash_deposits_max': 9800.0,
    'ml_days_span': 10,
    'business_hours_multiplier': 4.0
}
EOF
    
    print_success "Fraud generator files created"
}

# Create documentation
create_documentation() {
    print_status "Creating documentation..."
    
    sudo -u $WORKSHOP_USER tee $WORKSHOP_DIR/README.md > /dev/null <<'EOF'
# AML Fraud Data Generator

## Quick Start

1. **Activate Environment**:
   ```bash
   source /opt/aml-fraud-workshop/venv/bin/activate
   ```

2. **Generate Fraud Data**:
   ```bash
   cd /opt/aml-fraud-workshop/src
   python generate_fraud_data.py
   ```

3. **Analyze Data**:
   ```bash
   python analyze_fraud_data.py
   ```

## Workshop Structure

```
/opt/aml-fraud-workshop/
├── src/                    # Generator scripts
├── output/                 # Generated data files  
├── config/                 # Configuration templates
├── notebooks/              # Jupyter notebooks
├── venv/                   # Python virtual environment
└── logs/                   # Log files
```

## Features

- **Money Laundering Scenarios**: Structured deposits + wire transfers
- **Business Hours Weighting**: Realistic transaction timing
- **Multiple Output Formats**: JSON, CSV, Elasticsearch bulk
- **Configurable Complexity**: Easy, Medium, Hard scenarios
- **Analysis Tools**: Automated fraud detection

## Usage Examples

### Interactive Generation
```bash
python generate_fraud_data.py
# Follow prompts for configuration
```

### Programmatic Generation
```python
from lightweight_fraud_generator import FraudDataGenerator, FraudConfig

config = FraudConfig(
    events_per_day=10000,
    ml_checking_accounts=3,
    ml_savings_accounts=5
)

generator = FraudDataGenerator(config)
events = generator.generate_all_events()
```

### Analysis
```bash
python analyze_fraud_data.py
# Automatically analyzes latest generated data
```

## Generated Data

Each event contains:
- `accountID`: Customer account (1-35,000)
- `event_amount`: Transaction amount  
- `event_type`: debit/credit
- `account_event`: deposit/wire/purchase/fee/withdrawal
- `timestamp`: Business hours weighted
- ID fields for enrichment (posID, intbankID, etc.)

## Money Laundering Pattern

- **Structured Deposits**: $9,000-$9,999 cash deposits
- **Multiple Accounts**: 3-8 target accounts
- **Time Distribution**: 5-day window
- **Wire Transfer**: Single large transfer to Chinese bank
- **Hidden Signal**: Embedded in realistic transaction noise
EOF
    
    print_success "Documentation created"
}

# Configure system services
setup_system() {
    print_status "Configuring system..."
    
    # Create workshop service for easy startup
    sudo tee /etc/systemd/system/aml-generator.service > /dev/null <<EOF
[Unit]
Description=AML Fraud Data Generator Service
After=multi-user.target

[Service]
Type=oneshot
User=$WORKSHOP_USER
Group=$WORKSHOP_USER
WorkingDirectory=$WORKSHOP_DIR
ExecStart=$WORKSHOP_DIR/start_generator.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable aml-generator
    
    print_success "System services configured"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check Python environment
    if [ -f "$WORKSHOP_DIR/venv/bin/python" ]; then
        print_success "Python virtual environment created"
        python_version=$($WORKSHOP_DIR/venv/bin/python --version)
        print_status "Python version: $python_version"
        
        # Test package imports
        if $WORKSHOP_DIR/venv/bin/python -c "import pandas, numpy; print('✅ Core packages available')" 2>/dev/null; then
            print_success "Core Python packages installed"
        else
            print_warning "Some Python packages may be missing"
        fi
    else
        print_error "Python virtual environment not found"
    fi
    
    # Check workshop files
    if [ -f "$WORKSHOP_DIR/src/lightweight_fraud_generator.py" ]; then
        print_success "Fraud generator files created"
    else
        print_warning "Generator files not found"
    fi
    
    # Check directories
    for dir in src output config notebooks logs; do
        if [ -d "$WORKSHOP_DIR/$dir" ]; then
            print_success "Directory $dir exists"
        else
            print_warning "Directory $dir missing"
        fi
    done
    
    print_success "Installation verification complete"
}

# Display completion information
display_completion_info() {
    print_header
    print_success "AML Fraud Data Generator installation completed!"
    echo ""
    echo -e "${CYAN}📋 Installation Summary:${NC}"
    echo "   🖥️  Ubuntu $(lsb_release -rs) with updates"
    echo "   🐍 Python $PYTHON_VERSION with virtual environment"
    echo "   📊 Data science packages (pandas, numpy, matplotlib)"
    echo "   👤 Workshop user: $WORKSHOP_USER"
    echo "   📁 Workshop directory: $WORKSHOP_DIR"
    echo ""
    echo -e "${CYAN}🚀 Getting Started:${NC}"
    echo "   1. Switch to workshop user:"
    echo "      su - $WORKSHOP_USER"
    echo ""
    echo "   2. Start generator environment:"
    echo "      $WORKSHOP_DIR/start_generator.sh"
    echo ""
    echo "   3. Generate fraud data:"
    echo "      cd $WORKSHOP_DIR/src"
    echo "      source ../venv/bin/activate"
    echo "      python generate_fraud_data.py"
    echo ""
    echo -e "${CYAN}📁 Directory Structure:${NC}"
    echo "   📂 $WORKSHOP_DIR/src/         - Generator scripts"
    echo "   📂 $WORKSHOP_DIR/output/      - Generated data files"
    echo "   📂 $WORKSHOP_DIR/config/      - Configuration templates"
    echo "   📂 $WORKSHOP_DIR/notebooks/   - Analysis notebooks"
    echo ""
    echo -e "${CYAN}🔐 Access Information:${NC}"
    echo "   👤 Workshop user: $WORKSHOP_USER"
    echo "   🔑 Password: aml-workshop-2024"
    echo ""
    echo -e "${GREEN}✅ Ready to generate AML fraud detection data!${NC}"
}

# Main installation function
main() {
    print_header
    print_status "Starting AML Fraud Data Generator installation..."
    echo ""
    
    # Pre-flight checks
    check_root
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
    
    # Completion
    display_completion_info
}

# Error handling
trap 'print_error "Installation failed at line $LINENO. Check output above."' ERR

# Confirmation prompt
echo -e "${YELLOW}⚠️  This script will install the AML Fraud Data Generator.${NC}"
echo -e "${YELLOW}   It will install Python, create users, and setup the workshop environment.${NC}"
echo ""
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installation cancelled"
    exit 0
fi

# Run installation
main

exit 0
