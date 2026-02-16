# =============================================================================
# DATA GENERATION PIPELINE
# Optimized with Vectorization for Performance
# =============================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
PROJECT_DIR = "/Users/philipowusu/Development/recommendation_system"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "customers_with_truth.csv")
OUTPUT_TRANSACTIONS = os.path.join(DATA_DIR, "transactions.csv")
OUTPUT_INTERACTIONS = os.path.join(DATA_DIR, "offer_interactions.csv")

# Configuration
SERVICES = [
    {'service_id': 'SRV001', 'name': 'Oil Change', 'base_price': 49.99, 'category': 'Maintenance'},
    {'service_id': 'SRV002', 'name': 'Tire Rotation', 'base_price': 29.99, 'category': 'Maintenance'},
    {'service_id': 'SRV003', 'name': 'Brake Inspection', 'base_price': 39.99, 'category': 'Maintenance'},
    {'service_id': 'SRV004', 'name': 'Full Detail', 'base_price': 149.99, 'category': 'Cosmetic'},
    {'service_id': 'SRV005', 'name': 'Windshield Repair', 'base_price': 79.99, 'category': 'Repair'},
    {'service_id': 'SRV006', 'name': 'Battery Replacement', 'base_price': 129.99, 'category': 'Repair'},
    {'service_id': 'SRV007', 'name': 'AC Service', 'base_price': 89.99, 'category': 'Maintenance'},
    {'service_id': 'SRV008', 'name': 'Transmission Flush', 'base_price': 179.99, 'category': 'Maintenance'},
    {'service_id': 'SRV009', 'name': 'Wheel Alignment', 'base_price': 99.99, 'category': 'Maintenance'},
    {'service_id': 'SRV010', 'name': 'Engine Diagnostic', 'base_price': 69.99, 'category': 'Diagnostic'},
]

OFFERS = [
    {'offer_id': 'OFF001', 'name': 'Free Oil Change', 'type': 'Free Service', 'value': 49.99},
    {'offer_id': 'OFF002', 'name': '20% Off Any Service', 'type': 'Discount', 'value': 0.20},
    {'offer_id': 'OFF003', 'name': 'Loyalty Points 2X', 'type': 'Points', 'value': 2.0},
    {'offer_id': 'OFF004', 'name': '$25 Off Next Visit', 'type': 'Credit', 'value': 25.00},
    {'offer_id': 'OFF005', 'name': 'Free Tire Rotation', 'type': 'Free Service', 'value': 29.99},
    {'offer_id': 'OFF006', 'name': 'Winter Package Deal', 'type': 'Bundle', 'value': 50.00},
    {'offer_id': 'OFF007', 'name': 'Refer a Friend $50', 'type': 'Referral', 'value': 50.00},
    {'offer_id': 'OFF008', 'name': 'Birthday Special 30%', 'type': 'Discount', 'value': 0.30},
]

LOCATIONS = ['Edmonton South', 'Edmonton North', 'Calgary Downtown', 'Calgary NE',
             'Red Deer', 'Lethbridge', 'Vancouver', 'Surrey', 'Winnipeg', 'Saskatoon']


def generate_transactions_vectorized(customer_ids):
    """Generates transactions using numpy vectorization instead of loops."""
    print("Generating transactions (Vectorized)...")
    
    n_customers = len(customer_ids)
    
    # Assign segments
    segments = np.random.choice(['occasional', 'regular', 'frequent', 'vip'], 
                              size=n_customers, p=[0.35, 0.40, 0.18, 0.07])
    
    # Vectorized counts
    counts = np.zeros(n_customers, dtype=int)
    counts[segments == 'occasional'] = np.random.randint(1, 3, size=(segments == 'occasional').sum())
    counts[segments == 'regular'] = np.random.randint(4, 10, size=(segments == 'regular').sum())
    counts[segments == 'frequent'] = np.random.randint(10, 18, size=(segments == 'frequent').sum())
    counts[segments == 'vip'] = np.random.randint(18, 30, size=(segments == 'vip').sum())
    
    # Repeat customer IDs
    txn_cust_ids = np.repeat(customer_ids, counts)
    total_txns = len(txn_cust_ids)
    
    # Generate dates
    base_date = np.datetime64('2024-01-01')
    days_offset = np.random.randint(0, 730, size=total_txns)
    txn_dates = base_date - days_offset.astype('timedelta64[D]')
    
    # Generate services
    srv_indices = np.random.randint(0, len(SERVICES), size=total_txns)
    srv_ids = [SERVICES[i]['service_id'] for i in srv_indices]
    srv_names = [SERVICES[i]['name'] for i in srv_indices]
    srv_cats = [SERVICES[i]['category'] for i in srv_indices]
    srv_prices = np.array([SERVICES[i]['base_price'] for i in srv_indices])
    
    # Add noise to price
    price_noise = np.random.uniform(0.9, 1.1, size=total_txns)
    final_prices = np.round(srv_prices * price_noise, 2)
    
    # Locations
    txn_locs = np.random.choice(LOCATIONS, size=total_txns)
    
    df = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(total_txns)],
        'customer_id': txn_cust_ids,
        'service_id': srv_ids,
        'service_name': srv_names,
        'service_category': srv_cats,
        'amount': final_prices,
        'location': txn_locs,
        'transaction_date': txn_dates
    })
    
    print(f"  Generated {len(df):,} transactions")
    return df

def generate_offers_vectorized(customer_ids, df_transactions):
    """Generates offer interactions with signal injection."""
    print("Generating offers (Vectorized)...")
    
    # Calculate simple features for signal
    cust_stats = df_transactions.groupby('customer_id').agg(
        frequency=('transaction_id', 'count'),
        monetary_avg=('amount', 'mean')
    )
    
    # Align to customer_ids order
    cust_stats = cust_stats.reindex(customer_ids).fillna(0)
    
    # Vectorized offer counts
    freqs = cust_stats['frequency'].values
    n_offers = np.zeros(len(customer_ids), dtype=int)
    
    # Logic: More frequent = more offers
    n_offers = np.where(freqs >= 12, np.random.randint(8, 15, size=len(freqs)),
               np.where(freqs >= 6, np.random.randint(5, 10, size=len(freqs)),
               np.where(freqs >= 3, np.random.randint(3, 7, size=len(freqs)),
                                    np.random.randint(2, 5, size=len(freqs)))))
    
    # Repeat for interactions
    int_cust_ids = np.repeat(customer_ids, n_offers)
    total_ints = len(int_cust_ids)
    
    # Assign offers randomly
    off_indices = np.random.randint(0, len(OFFERS), size=total_ints)
    
    # Calculate probabilities (Vectorized Signal Injection)
    # This replaces the slow if/else logic with matrix operations
    
    # Base probabilities
    probs = np.random.uniform(0.05, 0.15, size=total_ints)
    
    # Inject signal from frequency (broadcasted)
    cust_freq_map = cust_stats['frequency'].to_dict()
    current_freqs = np.array([cust_freq_map[c] for c in int_cust_ids])
    
    # Boost probability for frequent customers
    probs += (current_freqs * 0.02) 
    
    # Clip
    probs = np.clip(probs, 0.0, 0.8)
    
    # Simulate outcomes
    opened = np.random.random(total_ints) < (0.3 + (current_freqs * 0.01))
    clicked = (np.random.random(total_ints) < 0.4) & opened
    redeemed = (np.random.random(total_ints) < probs) & clicked
    
    df = pd.DataFrame({
        'interaction_id': [f'INT{i:08d}' for i in range(total_ints)],
        'customer_id': int_cust_ids,
        'offer_id': [OFFERS[i]['offer_id'] for i in off_indices],
        'offer_name': [OFFERS[i]['name'] for i in off_indices],
        'offer_type': [OFFERS[i]['type'] for i in off_indices],
        'offer_value': [OFFERS[i]['value'] for i in off_indices],
        'sent_date': '2024-01-01', # Placeholder
        'opened': opened.astype(int),
        'clicked': clicked.astype(int),
        'redeemed': redeemed.astype(int)
    })
    
    print(f"  Generated {len(df):,} interactions with {df['redeemed'].mean():.2%} redemption rate")
    return df

def main():
    print("="*60)
    print("Running Optimized Data Generation")
    print("="*60)
    
    if not os.path.exists(CUSTOMERS_FILE):
        print("Customers file not found.")
        return

    df_customers = pd.read_csv(CUSTOMERS_FILE)
    customer_ids = df_customers['true_customer_id'].unique()
    
    df_trans = generate_transactions_vectorized(customer_ids)
    df_offers = generate_offers_vectorized(customer_ids, df_trans)
    
    df_trans.to_csv(OUTPUT_TRANSACTIONS, index=False)
    df_offers.to_csv(OUTPUT_INTERACTIONS, index=False)
    print("Done!")

if __name__ == "__main__":
    main()