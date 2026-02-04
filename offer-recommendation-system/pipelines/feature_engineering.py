"""Create customer features from transaction and offer data."""
import pandas as pd
import numpy as np
import os

PROJECT_DIR = os.environ.get(
    "RECOMMENDATION_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REFERENCE_DATE = pd.to_datetime('2024-01-01')


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Recency, Frequency, Monetary features."""
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    rfm = df.groupby('customer_id').agg(
        last_purchase=('transaction_date', 'max'),
        first_purchase=('transaction_date', 'min'),
        frequency=('transaction_id', 'count'),
        monetary_total=('amount', 'sum'),
        monetary_avg=('amount', 'mean'),
        monetary_min=('amount', 'min'),
        monetary_max=('amount', 'max'),
        monetary_std=('amount', 'std')
    ).round(2)

    rfm['recency_days'] = (REFERENCE_DATE - rfm['last_purchase']).dt.days
    rfm['first_purchase_days'] = (REFERENCE_DATE - rfm['first_purchase']).dt.days
    rfm['customer_tenure_days'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
    rfm = rfm.drop(columns=['last_purchase', 'first_purchase'])
    rfm['monetary_std'] = rfm['monetary_std'].fillna(0)
    rfm['avg_days_between_purchases'] = (rfm['customer_tenure_days'] / rfm['frequency'].clip(lower=1)).round(1)
    rfm['is_one_time_buyer'] = (rfm['frequency'] == 1).astype(int)

    return rfm


def create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create service preference features."""
    diversity = df.groupby('customer_id').agg({
        'service_id': 'nunique',
        'service_category': 'nunique',
        'location': 'nunique',
    }).rename(columns={
        'service_id': 'unique_service_count',
        'service_category': 'unique_service_category',
        'location': 'unique_location'
    })

    favorite = df.groupby('customer_id')['service_category'].agg(
        lambda x: x.value_counts().index[0]
    ).rename('favorite_category')

    spending = df.groupby(['customer_id', 'service_category'])['amount'].sum().unstack(fill_value=0)
    pct = spending.div(spending.sum(axis=1), axis=0).round(3)
    pct.columns = [f'pct_spent_{col.lower()}' for col in pct.columns]

    return diversity.join([favorite, pct])


def create_offer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create offer response features."""
    features = df.groupby('customer_id').agg(
        total_offer_received=('interaction_id', 'count'),
        total_opens=('opened', 'sum'),
        open_rate=('opened', 'mean'),
        total_clicks=('clicked', 'sum'),
        click_rate=('clicked', 'mean'),
        total_redemptions=('redeemed', 'sum'),
        redemption_rate=('redeemed', 'mean')
    ).round(3)

    engagement = df.groupby(['customer_id', 'offer_type'])['clicked'].sum().unstack(fill_value=0)
    favorite = engagement.idxmax(axis=1).rename('favorite_offer_type')

    redemptions = df[df['redeemed'] == 1].groupby(['customer_id', 'offer_type']).size().unstack(fill_value=0)
    redemptions.columns = [f'redemptions_{col.lower().replace(" ", "_")}' for col in redemptions.columns]

    return features.join([favorite, redemptions]).fillna(0)


def main():
    df_transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df_offers = pd.read_csv(os.path.join(DATA_DIR, "offer_interactions.csv"))

    rfm = create_rfm_features(df_transactions)
    service = create_service_features(df_transactions)
    offer = create_offer_features(df_offers)

    features = rfm.join(service).join(offer).reset_index().rename(columns={'index': 'customer_id'})
    features[features.select_dtypes(include=[np.number]).columns] = features.select_dtypes(include=[np.number]).fillna(0)

    features.to_csv(os.path.join(DATA_DIR, "customer_features.csv"), index=False)
    print(f"Created {len(features):,} customer profiles with {len(features.columns)} features")


if __name__ == "__main__":
    main()
