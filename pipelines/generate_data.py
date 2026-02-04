"""Generate synthetic transaction and offer interaction data."""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

PROJECT_DIR = os.environ.get(
    "RECOMMENDATION_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

np.random.seed(42)
random.seed(42)

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


def generate_transactions(customer_ids: list) -> pd.DataFrame:
    """Generate service transaction data."""
    segments = np.random.choice(['regular', 'occasional', 'frequent'], size=len(customer_ids), p=[0.60, 0.25, 0.15])
    transactions = []
    tid = 0

    for idx, cid in enumerate(customer_ids):
        n = {'occasional': np.random.randint(1, 3), 'regular': np.random.randint(2, 8), 'frequent': np.random.randint(6, 20)}[segments[idx]]
        pref_loc = random.choice(LOCATIONS)
        pref_cat = random.choice(['Maintenance', 'Repair', 'Cosmetic', 'Diagnostic'])

        for _ in range(n):
            date = datetime(2024, 1, 1) - timedelta(days=random.randint(0, 1095))
            services = [s for s in SERVICES if s['category'] == pref_cat] if random.random() < 0.6 else SERVICES
            svc = random.choice(services) if services else random.choice(SERVICES)

            transactions.append({
                'transaction_id': f"TXN{tid:08d}",
                'customer_id': cid,
                'service_id': svc['service_id'],
                'service_name': svc['name'],
                'service_category': svc['category'],
                'amount': round(svc['base_price'] * random.uniform(0.85, 1.15), 2),
                'location': pref_loc if random.random() < 0.7 else random.choice(LOCATIONS),
                'transaction_date': date.strftime('%Y-%m-%d'),
                'transaction_time': f"{random.randint(8,18):02d}:{random.randint(0,59):02d}:00"
            })
            tid += 1

    return pd.DataFrame(transactions)


def generate_offers(customer_ids: list) -> pd.DataFrame:
    """Generate offer interaction data."""
    interactions = []
    iid = 0

    for cid in customer_ids:
        for _ in range(random.randint(2, 10)):
            offer = random.choice(OFFERS)
            date = datetime(2024, 1, 1) - timedelta(days=random.randint(0, 730))
            opened = random.random() < 0.40
            clicked = opened and random.random() < 0.30
            redeemed = clicked and random.random() < 0.25

            interactions.append({
                'interaction_id': f"INT{iid:08d}",
                'customer_id': cid,
                'offer_id': offer['offer_id'],
                'offer_name': offer['name'],
                'offer_type': offer['type'],
                'offer_value': offer['value'],
                'sent_date': date.strftime('%Y-%m-%d'),
                'opened': int(opened),
                'clicked': int(clicked),
                'redeemed': int(redeemed)
            })
            iid += 1

    return pd.DataFrame(interactions)


def main():
    customers_file = os.path.join(DATA_DIR, "customers_with_truth.csv")
    if not os.path.exists(customers_file):
        raise FileNotFoundError("customers_with_truth.csv not found")

    customer_ids = pd.read_csv(customers_file)['true_customer_id'].unique().tolist()

    transactions = generate_transactions(customer_ids)
    offers = generate_offers(customer_ids)

    transactions.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
    offers.to_csv(os.path.join(DATA_DIR, "offer_interactions.csv"), index=False)

    print(f"Generated {len(transactions):,} transactions and {len(offers):,} offer interactions")


if __name__ == "__main__":
    main()
