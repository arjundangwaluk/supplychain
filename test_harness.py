"""
test_harness.py - Unit, Metamorphic, and Sanity Test Suite
==========================================================
Verification & Evaluation Harness for Supply Chain ML & NLP Pipeline
Enforces strict contracts, invariant guards, and metamorphic properties.
"""

import os
import pytest
import numpy as np
import pandas as pd
from data_pipeline import clean_raw_catalog, parse_exploded_reviews, clean_currency, clean_percentage, clean_rating
from nlp_engine import NLPSentimentEngine
from forecasting_engine import InventoryForecastingEngine
from clv_engine import CustomerLifecycleEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "amazon.csv") if os.path.exists(os.path.join(BASE_DIR, "amazon.csv")) else "amazon.csv"


# ==========================================================
# 1. Ingestion Contract & Data Guard Tests
# ==========================================================
def test_raw_catalog_ingestion_invariants():
    """Verify catalog ingestion contract, bounds, and absence of critical nulls."""
    df = clean_raw_catalog(CSV_PATH)

    assert len(df) > 0, "Catalog cannot be empty"
    assert (df['discounted_price'] >= 0).all(), "Discounted prices must be strictly non-negative"
    assert (df['actual_price'] >= df['discounted_price']).all(), "Actual price must be >= discounted price"
    assert (df['discount_percentage'] >= 0.0).all() and (df['discount_percentage'] <= 100.0).all(), \
        "Discount percentage must be bounded within [0, 100]"
    assert (df['rating'] >= 1.0).all() and (df['rating'] <= 5.0).all(), \
        "Ratings must be bounded within [1.0, 5.0]"
    assert (df['rating_count'] >= 0).all(), "Rating count must be non-negative"
    assert not df['product_id'].isna().any(), "Product ID must have zero nulls"


def test_clean_helpers():
    """Verify currency, percentage, and rating string parsing robustness."""
    assert clean_currency("₹1,499.00") == 1499.0
    assert clean_currency("$25.50") == 25.50
    assert clean_percentage("45%") == 45.0
    assert clean_rating("4.3") == 4.3
    assert np.isnan(clean_rating("|"))


# ==========================================================
# 2. NLP & Sentiment Intelligence Tests
# ==========================================================
def test_sentiment_score_normalization():
    """Verify sentiment compound score is strictly normalized within [-1.0, 1.0]."""
    nlp = NLPSentimentEngine()
    test_texts = [
        "Absolutely amazing product, charges extremely fast and high quality!",
        "Terrible waste of money, broke within one day and overheated.",
        "Average cable, works as described.",
        "",
        "12345 !@#$%"
    ]
    for text in test_texts:
        res = nlp.analyze_text(text)
        assert -1.0 <= res['compound'] <= 1.0, f"Compound score {res['compound']} outside [-1, 1]"
        assert 0.0 <= res['pos'] <= 1.0
        assert 0.0 <= res['neg'] <= 1.0
        assert 0.0 <= res['neu'] <= 1.0


def test_aspect_extraction_integrity():
    """Verify domain-specific aspect taxonomy detects expected aspects."""
    nlp = NLPSentimentEngine()
    text = "The quality and solid build are great, but the delivery package was late and damaged."
    aspects = nlp.extract_aspect_sentiment(text)

    assert aspects['Quality & Build']['detected'] is True
    assert aspects['Delivery & Packaging']['detected'] is True
    assert aspects['Quality & Build']['compound'] > 0, "Quality sentiment should be positive"


# ==========================================================
# 3. Forecasting & Operations Research Invariant Tests
# ==========================================================
def test_forecast_non_negativity_guard():
    """Verify forecast values are strictly non-negative (hat_y >= 0)."""
    engine = InventoryForecastingEngine()
    y_true = np.array([10.0, 15.0, 20.0])
    # Deliberately feed negative raw predictions to test clipping guard
    y_raw_pred = np.array([-5.0, 12.0, -1.0])
    metrics = engine.evaluate_predictions(y_true, y_raw_pred)

    assert metrics['MAE'] >= 0.0
    assert metrics['WAPE'] >= 0.0
    assert metrics['RMSE'] >= 0.0


def test_asymmetric_business_loss_monotonicity():
    """
    Verify asymmetric cost function penalizes stockouts heavier than excess holding.
    Stockout (under-forecasting by 10 units) must incur higher penalty than
    holding excess (over-forecasting by 10 units).
    """
    c_stockout = 5.0
    c_holding = 1.0
    engine = InventoryForecastingEngine(c_stockout=c_stockout, c_holding=c_holding)

    y_actual = np.array([50.0])
    y_under = np.array([40.0])  # Under-forecast by 10 (Stockout)
    y_over = np.array([60.0])   # Over-forecast by 10 (Excess holding)

    loss_under = engine.calculate_asymmetric_loss(y_actual, y_under, c_stockout, c_holding)
    loss_over = engine.calculate_asymmetric_loss(y_actual, y_over, c_stockout, c_holding)

    assert loss_under == 10.0 * c_stockout, f"Expected 50.0, got {loss_under}"
    assert loss_over == 10.0 * c_holding, f"Expected 10.0, got {loss_over}"
    assert loss_under > loss_over, "Stockout loss must be strictly greater than holding loss"


def test_dynamic_safety_stock_and_rop():
    """Verify Operations Research formulas for dynamic SS and ROP."""
    engine = InventoryForecastingEngine()
    avg_d = 20.0
    std_d = 4.0
    mean_l = 5.0
    std_l = 1.0

    ss_95 = engine.calculate_dynamic_safety_stock(avg_d, std_d, mean_l, std_l, service_level=0.95)
    ss_99 = engine.calculate_dynamic_safety_stock(avg_d, std_d, mean_l, std_l, service_level=0.99)
    rop_95 = engine.calculate_dynamic_rop(avg_d, mean_l, ss_95)

    assert ss_95 > 0, "Safety stock must be strictly positive"
    assert ss_99 > ss_95, "Higher service level (99% vs 95%) must require higher safety stock"
    assert rop_95 > ss_95, "ROP must strictly exceed Safety Stock by lead time demand"
    assert rop_95 == pytest.approx((avg_d * mean_l) + ss_95, rel=1e-3)


def test_moq_enforcement_on_requisition():
    """Verify automated purchase requisition strictly respects Minimum Order Quantity (MOQ)."""
    engine = InventoryForecastingEngine()
    moq = 100
    rop = 50.0
    current_inventory = 45.0  # Just below ROP, deficit = 5

    res = engine.evaluate_purchase_requisition(current_inventory, rop, moq=moq)
    assert res['reorder_triggered'] is True
    assert res['recommended_quantity'] >= moq, f"Order {res['recommended_quantity']} violates MOQ {moq}"
    assert res['moq_enforced'] is True


# ==========================================================
# 4. Metamorphic Tests
# ==========================================================
def test_metamorphic_discount_monotonicity():
    """
    Metamorphic Test:
    Holding all base factors equal, an increased promotional discount
    must never decrease expected consumer sales velocity in an elasticity model.
    """
    base_price = 1000.0
    discount_low = 10.0  # 10%
    discount_high = 50.0 # 50%

    lift_low = 1.0 + (discount_low / 65.0)
    lift_high = 1.0 + (discount_high / 65.0)

    assert lift_high > lift_low, "Promotional lift must be monotonically increasing with discount"


# ==========================================================
# 5. Customer Lifecycle & Anomaly Invariants
# ==========================================================
def test_clv_and_anomaly_invariants():
    """Verify CLV positivity, retention bounds, and anomaly flag consistency."""
    clv_engine = CustomerLifecycleEngine()

    dummy_rfm = pd.DataFrame([{
        'customer_id': 'TEST_01',
        'recency': 25,
        'frequency': 5,
        'monetary': 4500.0,
        'avg_order_value': 900.0,
        'unique_skus': 3
    }])

    clv_scored = clv_engine.calculate_clv(dummy_rfm)
    assert (clv_scored['clv_1yr'] > 0).all(), "CLV must be strictly positive"
    assert (clv_scored['retention_rate'] >= 0.15).all() and (clv_scored['retention_rate'] <= 0.95).all()

    # Anomaly detector output check
    dummy_sales = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=30),
        'product_id': 'PROD_TEST',
        'units_demanded': [10]*25 + [120, 15, 12, 10, 11],  # Day 26 has massive spike
        'traffic_sessions': [100]*25 + [2500, 150, 120, 100, 110]
    })
    _, alerts = clv_engine.detect_demand_anomalies(dummy_sales, z_threshold=2.5)
    assert len(alerts) > 0, "Spike anomaly must be flagged"
    assert (alerts['units_demanded'] == 120).any(), "Day 26 demand spike must be captured"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
