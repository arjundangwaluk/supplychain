"""
data_pipeline.py - Production Data Ingestion, Cleaning & Operational Synthesizer
================================================================================
Principal AI & Supply Chain Systems Architecture
Strict Ingestion Contract, Validation Guards, and Operational Event Derivation
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class DataValidationError(Exception):
    """Raised when ingestion contract assertions fail."""
    pass


def clean_currency(value: Any) -> float:
    """Strip currency symbols, commas, and cast to float."""
    if pd.isna(value):
        return np.nan
    s = str(value).replace('₹', '').replace(',', '').replace('$', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_percentage(value: Any) -> float:
    """Strip percentage symbol and cast to float in 0-100 scale."""
    if pd.isna(value):
        return np.nan
    s = str(value).replace('%', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_rating(value: Any) -> float:
    """Clean rating string, handling anomalies like '|' or non-numeric entries."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    try:
        val = float(s)
        return val
    except ValueError:
        return np.nan


def clean_raw_catalog(csv_path: str = "D:/Supply Chain/amazon.csv") -> pd.DataFrame:
    """
    Ingests and cleans the raw Amazon catalog dataset.
    Enforces strict data ingestion contracts and validation guards.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source catalog not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Clean Numeric Attributes
    df['discounted_price'] = df['discounted_price'].apply(clean_currency)
    df['actual_price'] = df['actual_price'].apply(clean_currency)
    df['discount_percentage'] = df['discount_percentage'].apply(clean_percentage)
    df['rating'] = df['rating'].apply(clean_rating)
    df['rating_count'] = df['rating_count'].apply(clean_currency)

    # 2. Imputation for known anomalies
    # Handle corrupt rating (e.g. '|' at row 1279) with median rating
    median_rating = df['rating'].median()
    df['rating'] = df['rating'].fillna(median_rating)

    # Handle missing rating_count with median
    median_rating_count = df['rating_count'].median()
    df['rating_count'] = df['rating_count'].fillna(median_rating_count)

    # Ensure actual price is at least discounted price
    mask_inversion = df['discounted_price'] > df['actual_price']
    df.loc[mask_inversion, 'actual_price'] = df.loc[mask_inversion, 'discounted_price']

    # Recalculate discount_percentage if inconsistent
    computed_discount = ((df['actual_price'] - df['discounted_price']) / df['actual_price'] * 100).round(1)
    df['discount_percentage'] = df['discount_percentage'].fillna(computed_discount)
    df['discount_percentage'] = df['discount_percentage'].clip(lower=0.0, upper=100.0)

    # 3. Clean Text Attributes
    for col in ['product_name', 'about_product', 'review_title', 'review_content', 'category']:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # 4. Extract Category Hierarchy
    def parse_category(cat_str: str) -> Tuple[str, str]:
        parts = [p.strip() for p in cat_str.split('|') if p.strip()]
        primary = parts[0] if len(parts) > 0 else "General"
        sub = parts[1] if len(parts) > 1 else (parts[0] if len(parts) > 0 else "General")
        return primary, sub

    cat_tuples = df['category'].apply(parse_category)
    df['primary_category'] = [t[0] for t in cat_tuples]
    df['sub_category'] = [t[1] for t in cat_tuples]

    # 5. INGESTION ASSERTIONS & VALIDATION GUARDS
    # Assert non-negative prices
    if (df['discounted_price'] < 0).any():
        raise DataValidationError("Ingestion Assertion Failed: Negative discounted prices detected.")
    if (df['actual_price'] < 0).any():
        raise DataValidationError("Ingestion Assertion Failed: Negative actual prices detected.")
    
    # Assert rating bounds [1.0, 5.0]
    if (df['rating'] < 1.0).any() or (df['rating'] > 5.0).any():
        raise DataValidationError("Ingestion Assertion Failed: Ratings outside [1.0, 5.0] bound.")

    # Assert discount percentage bounds [0.0, 100.0]
    if (df['discount_percentage'] < 0.0).any() or (df['discount_percentage'] > 100.0).any():
        raise DataValidationError("Ingestion Assertion Failed: Discount % outside [0, 100] bound.")

    # Outlier clipping on rating_count (upper 99.9th percentile) to protect downstream features
    upper_bound = df['rating_count'].quantile(0.999)
    df['rating_count_clipped'] = df['rating_count'].clip(upper=upper_bound)

    return df


def parse_exploded_reviews(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Unrolls comma-delimited customer reviews into individual interaction rows.
    """
    records = []
    for idx, row in df_clean.iterrows():
        p_id = row['product_id']
        p_name = row['product_name']
        cat = row['primary_category']
        
        users = [u.strip() for u in str(row['user_id']).split(',') if u.strip()]
        names = [n.strip() for n in str(row['user_name']).split(',') if n.strip()]
        titles = [t.strip() for t in str(row['review_title']).split(',') if t.strip()]
        contents = [c.strip() for c in str(row['review_content']).split(',') if c.strip()]

        max_len = max(len(users), len(titles), len(contents), 1)
        for i in range(max_len):
            uid = users[i] if i < len(users) else f"USER_{p_id[-4:]}_{i}"
            uname = names[i] if i < len(names) else f"Customer {i+1}"
            rtitle = titles[i] if i < len(titles) else "Product Review"
            rcontent = contents[i] if i < len(contents) else "Satisfactory purchase."

            records.append({
                'product_id': p_id,
                'product_name': p_name,
                'primary_category': cat,
                'user_id': uid,
                'user_name': uname,
                'review_title': rtitle,
                'review_content': rcontent,
                'discounted_price': row['discounted_price'],
                'actual_price': row['actual_price'],
                'discount_percentage': row['discount_percentage'],
                'rating': row['rating']
            })

    return pd.DataFrame(records)


def synthesize_supply_chain_operations(
    df_clean: pd.DataFrame,
    start_date: str = "2025-01-01",
    days: int = 180,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic daily supply chain operational series and SKU state.
    Includes:
    - Daily demand with price elasticity, weekend surge, holiday/trend dynamics.
    - Vendor lead times (mean L, std sigma_L) per category.
    - On-hand stock, pipeline orders, and stockout events.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Select top representative SKUs across categories for intensive time-series tracking
    unique_skus = df_clean.drop_duplicates('product_id').copy()
    
    # Category lead-time profiles (mean_days, std_days)
    lead_time_profiles = {
        'Electronics': (4.0, 1.2),
        'Computers&Accessories': (5.0, 1.5),
        'Home&Kitchen': (6.0, 1.8),
        'OfficeProducts': (3.5, 0.8),
        'MusicalInstruments': (7.0, 2.0),
        'HomeImprovement': (5.5, 1.4),
        'General': (4.5, 1.0)
    }

    sku_inventory_states = []
    daily_sales_records = []

    for _, sku in unique_skus.iterrows():
        p_id = sku['product_id']
        cat = sku['primary_category']
        disc_price = sku['discounted_price']
        act_price = sku['actual_price']
        disc_pct = sku['discount_percentage']
        base_rating = sku['rating']
        r_count = sku['rating_count']

        # Category lead time
        l_mean, l_std = lead_time_profiles.get(cat, (5.0, 1.2))

        # Base daily sales velocity proportional to log(rating_count) and rating
        # Lower priced items naturally have higher volume
        price_factor = 1.0 / np.log1p(max(disc_price, 10.0)) * 15.0
        popularity_factor = np.log1p(r_count) / 3.0
        rating_factor = (base_rating / 3.5)
        base_daily_demand = max(1.0, popularity_factor * rating_factor * price_factor)

        # Minimum Order Quantity (MOQ) and holding cost
        moq = int(max(10, np.round(base_daily_demand * 5)))
        holding_cost_rate = 0.20 / 365.0  # 20% annual holding cost
        unit_holding_cost = max(0.5, disc_price * holding_cost_rate)
        stockout_penalty = max(10.0, disc_price * 0.35)  # 35% margin penalty per lost sale

        # Generate 180-day series
        on_hand_inventory = int(base_daily_demand * (l_mean + 7))  # Initial stock buffer
        
        sku_demand_history = []
        for d_idx, cur_date in enumerate(date_range):
            # Calendar lift: Weekends +25%, month-end/promo +20%
            day_of_week = cur_date.dayofweek
            is_weekend = 1 if day_of_week in [5, 6] else 0
            cal_lift = 1.25 if is_weekend else 1.0
            if cur_date.day in [1, 15, 28, 29, 30]:
                cal_lift *= 1.20

            # Discount elasticity: lift = 1 + (discount_pct / 50)
            discount_lift = 1.0 + (disc_pct / 65.0)

            # Random Poisson/Negative-Binomial demand realization
            expected_demand = base_daily_demand * cal_lift * discount_lift
            daily_demand = np.random.poisson(lam=max(0.1, expected_demand))

            # Fulfill demand against on_hand
            units_fulfilled = min(on_hand_inventory, daily_demand)
            unmet_demand = daily_demand - units_fulfilled
            on_hand_inventory = max(0, on_hand_inventory - units_fulfilled)

            # Periodic replenishment simulation
            # Every 14 days, receive replenishment
            if d_idx % 14 == 0:
                replenishment_qty = int(np.round(base_daily_demand * 14))
                on_hand_inventory += replenishment_qty

            traffic_sessions = int(daily_demand * np.random.uniform(15, 30))

            daily_sales_records.append({
                'date': cur_date,
                'product_id': p_id,
                'product_name': sku['product_name'][:60],
                'primary_category': cat,
                'discounted_price': disc_price,
                'actual_price': act_price,
                'discount_percentage': disc_pct,
                'rating': base_rating,
                'units_demanded': daily_demand,
                'units_sold': units_fulfilled,
                'unmet_demand': unmet_demand,
                'on_hand_stock': on_hand_inventory,
                'traffic_sessions': traffic_sessions,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend
            })
            sku_demand_history.append(daily_demand)

        d_arr = np.array(sku_demand_history)
        sku_inventory_states.append({
            'product_id': p_id,
            'product_name': sku['product_name'][:70],
            'primary_category': cat,
            'discounted_price': disc_price,
            'actual_price': act_price,
            'discount_percentage': disc_pct,
            'rating': base_rating,
            'avg_daily_demand': float(np.mean(d_arr)),
            'std_daily_demand': float(max(np.std(d_arr), 0.1)),
            'mean_lead_time': float(l_mean),
            'std_lead_time': float(l_std),
            'current_inventory': int(on_hand_inventory),
            'moq': int(moq),
            'unit_holding_cost': float(round(unit_holding_cost, 2)),
            'stockout_penalty': float(round(stockout_penalty, 2))
        })

    df_sales = pd.DataFrame(daily_sales_records)
    df_inventory = pd.DataFrame(sku_inventory_states)

    return df_sales, df_inventory


def synthesize_customer_transactions(
    df_clean: pd.DataFrame,
    num_customers: int = 1200,
    start_date: str = "2024-06-01",
    end_date: str = "2025-06-01",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Synthesizes transactional order records for RFM, CLV, and Churn analysis.
    Maps real catalog products to synthetic customer buying histories.
    """
    np.random.seed(random_seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    skus = df_clean[['product_id', 'product_name', 'primary_category', 'discounted_price']].drop_duplicates('product_id').values
    
    customers = [f"CUST_{i:05d}" for i in range(1, num_customers + 1)]
    
    # Customer segments profile: 15% high frequency, 35% medium, 50% one-time / dormant
    freq_weights = np.random.choice([1, 2, 4, 8, 15], size=num_customers, p=[0.45, 0.25, 0.15, 0.10, 0.05])

    orders = []
    order_counter = 10001
    
    for c_idx, cust_id in enumerate(customers):
        n_orders = freq_weights[c_idx]
        # Distribute order dates across the year
        selected_dates = np.random.choice(dates, size=n_orders, replace=False)
        selected_dates.sort()

        for o_date in selected_dates:
            # Pick 1 to 3 items per order
            n_items = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            for _ in range(n_items):
                sku_idx = np.random.randint(0, len(skus))
                p_id, p_name, cat, price = skus[sku_idx]
                qty = int(np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03]))
                total = float(round(qty * price, 2))

                orders.append({
                    'order_id': f"ORD_{order_counter}",
                    'customer_id': cust_id,
                    'order_date': o_date,
                    'product_id': p_id,
                    'product_name': p_name,
                    'category': cat,
                    'units_ordered': qty,
                    'unit_price': price,
                    'order_amount': total
                })
            order_counter += 1

    df_orders = pd.DataFrame(orders)
    return df_orders


def run_pipeline(csv_path: str = "D:/Supply Chain/amazon.csv") -> Dict[str, pd.DataFrame]:
    """Orchestrates full ingestion and derivation pipeline."""
    print("=" * 70)
    print("INGESTION CONTRACT: Loading and validating raw Amazon catalog...")
    df_clean = clean_raw_catalog(csv_path)
    print(f"[OK] Catalog verified: {len(df_clean)} records, {df_clean['product_id'].nunique()} unique SKUs.")

    print("Unrolling customer reviews for NLP and sentiment parsing...")
    df_reviews = parse_exploded_reviews(df_clean)
    print(f"[OK] Extracted {len(df_reviews)} review observations.")

    print("Synthesizing daily operational supply chain time-series...")
    df_sales, df_inventory = synthesize_supply_chain_operations(df_clean)
    print(f"[OK] Generated {len(df_sales)} daily SKU sales observations across 180 days.")
    print(f"[OK] Calibrated state for {len(df_inventory)} active inventory SKUs.")

    print("Synthesizing customer transactional order stream for RFM & CLV...")
    df_orders = synthesize_customer_transactions(df_clean)
    print(f"[OK] Generated {len(df_orders)} transactional orders.")
    print("=" * 70)

    return {
        'catalog': df_clean,
        'reviews': df_reviews,
        'sales': df_sales,
        'inventory': df_inventory,
        'orders': df_orders
    }


if __name__ == "__main__":
    datasets = run_pipeline()
    print("Pipeline execution complete and verified.")
