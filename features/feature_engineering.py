"""
Feature Engineering Pipeline for Churn Prevention System.

This module handles:
1. Data cleaning (remove invalid records)
2. RFM feature construction (Recency, Frequency, Monetary)
3. Behavioral feature engineering
4. Churn label generation (90-day window)
5. User-item interaction dataset creation
6. Time-based train/validation/test split
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_FEATURES,
    CHURN_WINDOW_DAYS,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    RANDOM_SEED,
)


def load_raw_data(filepath=None):
    """Load raw Online Retail II data from Excel."""
    if filepath is None:
        # Find the data file
        xlsx_files = list(DATA_RAW_DIR.glob("*.xlsx"))
        csv_files = list(DATA_RAW_DIR.glob("*.csv"))
        if xlsx_files:
            filepath = xlsx_files[0]
        elif csv_files:
            filepath = csv_files[0]
        else:
            raise FileNotFoundError(f"No data files found in {DATA_RAW_DIR}")

    print(f"Loading data from: {filepath}")
    if str(filepath).endswith(".xlsx"):
        # Read both sheets (Year 2009-2010, Year 2010-2011)
        df1 = pd.read_excel(filepath, sheet_name=0)
        df2 = pd.read_excel(filepath, sheet_name=1)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = pd.read_csv(filepath)

    print(f"  Raw records: {len(df):,}")
    return df


def clean_data(df):
    """
    Clean transactional data:
    - Remove rows with null CustomerID
    - Remove cancelled/returned orders (InvoiceNo starting with 'C')
    - Remove records with negative or zero Quantity
    - Remove records with negative or zero Price
    - Standardize column names
    """
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename for consistency
    rename_map = {
        "customer_id": "customer_id",
        "invoice": "invoice_no",
        "invoiceno": "invoice_no",
        "invoicedate": "invoice_date",
        "invoice_date": "invoice_date",
        "stockcode": "stock_code",
        "stock_code": "stock_code",
        "quantity": "quantity",
        "price": "price",
        "unitprice": "price",
        "description": "description",
        "country": "country",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"  Before cleaning: {len(df):,} records")

    # Remove null CustomerID
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)

    # Remove cancelled orders (invoice starts with 'C')
    df["invoice_no"] = df["invoice_no"].astype(str)
    df = df[~df["invoice_no"].str.startswith("C")]

    # Remove invalid quantity and price
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]

    # Parse dates
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # Add total amount
    df["total_amount"] = df["quantity"] * df["price"]

    print(f"  After cleaning: {len(df):,} records")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    print(f"  Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")

    return df


def create_time_splits(df):
    """
    Create time-based train/validation/test splits.

    Strategy:
    - observation_end = max(invoice_date) - CHURN_WINDOW_DAYS
      (we need CHURN_WINDOW_DAYS after to create labels)
    - Split the observation period into train/val periods
    - Test labels come from the final CHURN_WINDOW_DAYS
    """
    max_date = df["invoice_date"].max()
    min_date = df["invoice_date"].min()
    total_days = (max_date - min_date).days

    # We need the last CHURN_WINDOW_DAYS for test label generation
    observation_end = max_date - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    # Split observation period
    observation_days = (observation_end - min_date).days
    train_end = min_date + pd.Timedelta(days=int(observation_days * 0.7))
    val_end = observation_end

    splits = {
        "train_start": min_date,
        "train_end": train_end,
        "val_start": train_end,
        "val_end": val_end,
        "test_label_start": val_end,
        "test_label_end": max_date,
        "observation_end": observation_end,
    }

    print(f"\n  Time Splits:")
    print(f"    Train: {splits['train_start'].date()} to {splits['train_end'].date()}")
    print(f"    Val:   {splits['val_start'].date()} to {splits['val_end'].date()}")
    print(f"    Test label window: {splits['test_label_start'].date()} to {splits['test_label_end'].date()}")

    return splits


def build_rfm_features(df, reference_date):
    """
    Build RFM features for each customer relative to reference_date.

    Returns DataFrame with columns:
    - recency: days since last purchase
    - frequency: number of unique transactions
    - monetary: total spending
    """
    customer_features = df.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        first_purchase=("invoice_date", "min"),
        frequency=("invoice_no", "nunique"),
        monetary=("total_amount", "sum"),
        n_items=("quantity", "sum"),
        n_unique_products=("stock_code", "nunique"),
        n_records=("invoice_no", "count"),
    ).reset_index()

    customer_features["recency"] = (
        reference_date - customer_features["last_purchase"]
    ).dt.days

    return customer_features


def build_behavioral_features(df, customer_features):
    """
    Build behavioral features:
    - avg_basket_size: average items per transaction
    - avg_purchase_interval: average days between purchases
    - product_diversity: unique products / total items
    - avg_quantity_per_txn: average quantity per transaction
    - return_rate: (placeholder, set to 0 since we removed returns)
    - days_as_customer: duration from first to last purchase
    """
    # Average basket size (items per transaction)
    basket_sizes = df.groupby(["customer_id", "invoice_no"])["quantity"].sum().reset_index()
    avg_basket = basket_sizes.groupby("customer_id")["quantity"].mean().reset_index()
    avg_basket.columns = ["customer_id", "avg_basket_size"]

    # Average purchase interval
    purchase_dates = df.groupby(["customer_id", "invoice_no"])["invoice_date"].min().reset_index()
    purchase_dates = purchase_dates.sort_values(["customer_id", "invoice_date"])

    def calc_avg_interval(group):
        dates = group["invoice_date"].sort_values()
        if len(dates) < 2:
            return 0
        diffs = dates.diff().dropna().dt.days
        return diffs.mean()

    intervals = purchase_dates.groupby("customer_id").apply(calc_avg_interval).reset_index()
    intervals.columns = ["customer_id", "avg_purchase_interval"]

    # Merge behavioral features
    customer_features = customer_features.merge(avg_basket, on="customer_id", how="left")
    customer_features = customer_features.merge(intervals, on="customer_id", how="left")

    # Product diversity
    customer_features["product_diversity"] = (
        customer_features["n_unique_products"] / customer_features["n_items"].clip(lower=1)
    )

    # Avg quantity per transaction
    customer_features["avg_quantity_per_txn"] = (
        customer_features["n_items"] / customer_features["frequency"].clip(lower=1)
    )

    # Return rate (set to 0 since we removed returns in cleaning)
    customer_features["return_rate"] = 0.0

    # Days as customer
    customer_features["days_as_customer"] = (
        customer_features["last_purchase"] - customer_features["first_purchase"]
    ).dt.days

    return customer_features


def create_churn_labels(customer_features, df, splits):
    """
    Create churn labels based on the defined window.
    A customer is churned if they have NO purchase in the test label window.
    """
    label_start = splits["test_label_start"]
    label_end = splits["test_label_end"]

    # Get customers who made purchases in the label window
    future_purchases = df[
        (df["invoice_date"] > label_start) & (df["invoice_date"] <= label_end)
    ]["customer_id"].unique()

    # Label: 1 = churned, 0 = retained
    customer_features["churn"] = (~customer_features["customer_id"].isin(future_purchases)).astype(int)

    churn_rate = customer_features["churn"].mean()
    print(f"\n  Churn labeling (window={CHURN_WINDOW_DAYS} days):")
    print(f"    Total customers: {len(customer_features):,}")
    print(f"    Churned: {customer_features['churn'].sum():,} ({churn_rate:.1%})")
    print(f"    Retained: {(1 - customer_features['churn']).sum():,.0f} ({1-churn_rate:.1%})")

    return customer_features


def create_interaction_dataset(df):
    """
    Create user-item interaction dataset for recommendation models.
    Each row represents a (customer_id, stock_code) pair with interaction strength.
    """
    interactions = df.groupby(["customer_id", "stock_code"]).agg(
        n_purchases=("invoice_no", "nunique"),
        total_quantity=("quantity", "sum"),
        total_amount=("total_amount", "sum"),
        last_purchase=("invoice_date", "max"),
        first_purchase=("invoice_date", "min"),
    ).reset_index()

    # Create implicit rating (log-scaled purchase count)
    interactions["rating"] = np.log1p(interactions["n_purchases"])

    print(f"\n  Interaction dataset:")
    print(f"    Records: {len(interactions):,}")
    print(f"    Unique users: {interactions['customer_id'].nunique():,}")
    print(f"    Unique items: {interactions['stock_code'].nunique():,}")
    print(f"    Sparsity: {1 - len(interactions) / (interactions['customer_id'].nunique() * interactions['stock_code'].nunique()):.4%}")

    return interactions


def create_item_metadata(df):
    """Create item metadata from transaction data."""
    items = df.groupby("stock_code").agg(
        description=("description", "first"),
        avg_price=("price", "mean"),
        total_sold=("quantity", "sum"),
        n_customers=("customer_id", "nunique"),
        n_transactions=("invoice_no", "nunique"),
    ).reset_index()

    return items


def create_bgnbd_dataset(df, reference_date):
    """
    Create dataset in the format required by BG/NBD model (lifetimes library).

    Returns DataFrame with columns:
    - frequency: number of repeat purchases (excluding first)
    - recency: time between first and last purchase (in days)
    - T: time between first purchase and reference_date (in days)
    - monetary_value: average order value (excluding first purchase)
    """
    from lifetimes.utils import summary_data_from_transaction_data

    summary = summary_data_from_transaction_data(
        transactions=df,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        monetary_value_col="total_amount",
        observation_period_end=reference_date,
    )

    print(f"\n  BG/NBD dataset:")
    print(f"    Customers: {len(summary):,}")
    print(f"    Avg frequency: {summary['frequency'].mean():.2f}")
    print(f"    Avg recency: {summary['recency'].mean():.1f} days")
    print(f"    Avg T: {summary['T'].mean():.1f} days")

    return summary


def run_pipeline():
    """Execute the full feature engineering pipeline."""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Step 1: Load and clean data
    print("\n[1/7] Loading raw data...")
    df = load_raw_data()

    print("\n[2/7] Cleaning data...")
    df = clean_data(df)

    # Step 2: Time splits
    print("\n[3/7] Creating time-based splits...")
    splits = create_time_splits(df)

    # Filter training data (everything up to val_end for feature construction)
    df_train = df[df["invoice_date"] <= splits["val_end"]].copy()

    # Step 3: Build features
    print("\n[4/7] Building RFM features...")
    reference_date = splits["val_end"]
    customer_features = build_rfm_features(df_train, reference_date)

    print("\n[5/7] Building behavioral features...")
    customer_features = build_behavioral_features(df_train, customer_features)

    # Step 4: Create churn labels
    print("\n[6/7] Creating churn labels...")
    customer_features = create_churn_labels(customer_features, df, splits)

    # Step 5: Create interaction dataset
    print("\n[7/7] Creating interaction & BG/NBD datasets...")
    interactions = create_interaction_dataset(df_train)
    item_metadata = create_item_metadata(df_train)
    bgnbd_data = create_bgnbd_dataset(df_train, reference_date)

    # Add churn labels to BG/NBD data
    churn_labels = customer_features[["customer_id", "churn"]].set_index("customer_id")
    bgnbd_data = bgnbd_data.join(churn_labels, how="inner")

    # Scale features for churn model
    feature_cols = [c for c in ALL_FEATURES if c in customer_features.columns]
    scaler = StandardScaler()
    customer_features[feature_cols] = customer_features[feature_cols].fillna(0)
    customer_features_scaled = customer_features.copy()
    customer_features_scaled[feature_cols] = scaler.fit_transform(
        customer_features[feature_cols]
    )

    # Save everything
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    # Save processed datasets
    customer_features.to_parquet(DATA_PROCESSED_DIR / "customer_features.parquet", index=False)
    customer_features_scaled.to_parquet(DATA_PROCESSED_DIR / "customer_features_scaled.parquet", index=False)
    interactions.to_parquet(DATA_PROCESSED_DIR / "interactions.parquet", index=False)
    item_metadata.to_parquet(DATA_PROCESSED_DIR / "item_metadata.parquet", index=False)
    bgnbd_data.to_parquet(DATA_PROCESSED_DIR / "bgnbd_summary.parquet")
    df_train.to_parquet(DATA_PROCESSED_DIR / "transactions_clean.parquet", index=False)

    # Save splits info
    splits_df = pd.DataFrame([{k: str(v) for k, v in splits.items()}])
    splits_df.to_csv(DATA_PROCESSED_DIR / "splits.csv", index=False)

    # Save scaler
    import joblib
    joblib.dump(scaler, DATA_PROCESSED_DIR / "scaler.joblib")

    print(f"\n  All files saved to: {DATA_PROCESSED_DIR}")
    print("  Files:")
    for f in sorted(DATA_PROCESSED_DIR.glob("*")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    {f.name} ({size_mb:.2f} MB)")

    print("\n✅ Feature engineering pipeline completed!")
    return customer_features, interactions, bgnbd_data, splits


if __name__ == "__main__":
    run_pipeline()
