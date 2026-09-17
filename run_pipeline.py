"""
run_pipeline.py - Unified Master Orchestrator
==============================================
Executes the full pipeline:
1. Data Ingestion & Sanitization
2. Engine A: Aspect-Based NLP & Sentiment Analysis
3. Engine B: Inventory Regressors & Operations Research
4. Engine C: RFM Segmentation, CLV, Churn & Demand Surge Anomaly Detection
"""

import sys
import time
from data_pipeline import run_pipeline
from nlp_engine import NLPSentimentEngine
from forecasting_engine import InventoryForecastingEngine
from clv_engine import CustomerLifecycleEngine


def main():
    start_time = time.time()
    print("=" * 80)
    print("AMAZON SUPPLY CHAIN AI & OPERATIONS ORCHESTRATOR")
    print("=" * 80)

    # 1. Ingestion & Operations Derivation
    print("\n[STEP 1/4] Running Data Ingestion & Synthetic Operations Derivation...")
    datasets = run_pipeline("amazon.csv")
    catalog = datasets['catalog']
    reviews = datasets['reviews']
    sales = datasets['sales']
    inventory = datasets['inventory']
    orders = datasets['orders']

    # 2. Engine A: NLP Voice-of-Customer
    print("\n[STEP 2/4] Running Engine A: Aspect NLP & Sentiment Intelligence...")
    nlp = NLPSentimentEngine()
    processed_reviews = nlp.process_reviews_dataframe(reviews, sample_size=3000)
    sku_sentiment = nlp.aggregate_sku_sentiment(processed_reviews)
    corr_report = nlp.correlate_sentiment_with_velocity(sku_sentiment, inventory)
    print(f"  -> Processed {len(processed_reviews)} reviews.")
    print(f"  -> Sentiment-Velocity Correlation: r = {corr_report['pearson_r']} (p = {corr_report['pearson_p_value']})")
    print(f"  -> Result: {corr_report['interpretation']}")

    # 3. Engine B: Predictive Forecasting & Operations Research
    print("\n[STEP 3/4] Running Engine B: Inventory Forecasting & OR Logic...")
    forecast_engine = InventoryForecastingEngine(c_stockout=5.0, c_holding=1.0)
    sample_skus = inventory['product_id'].head(30).tolist()
    sub_sales = sales[sales['product_id'].isin(sample_skus)].copy()
    df_featured = forecast_engine.build_features(sub_sales, sku_sentiment)
    train_results = forecast_engine.train_models(df_featured, test_days=28)

    print("  -> Model Evaluation Metrics (Out-of-Time Test Set):")
    for model_name, m in train_results['metrics'].items():
        print(f"     * {model_name:26s} | WAPE: {m['WAPE_Pct']}% | MAE: {m['MAE']} | Asymmetric Loss: ${m['Asymmetric_Loss']}")

    # 4. Engine C: Customer Lifecycle & Risk
    print("\n[STEP 4/4] Running Engine C: Customer Lifecycle, CLV, Churn & Anomaly Surges...")
    clv_engine = CustomerLifecycleEngine()
    rfm = clv_engine.compute_rfm_metrics(orders)
    rfm_clv = clv_engine.calculate_clv(rfm)
    churn_results = clv_engine.train_churn_classifier(rfm_clv)
    sales_scored, alerts = clv_engine.detect_demand_anomalies(sales, z_threshold=2.5)

    print(f"  -> RFM Segments: {len(rfm_clv)} customer profiles calibrated.")
    print(f"  -> Churn Classifier (Random Forest): Accuracy = {churn_results['metrics']['Random Forest']['Accuracy']*100:.1f}%, ROC-AUC = {churn_results['metrics']['Random Forest']['ROC_AUC']:.3f}")
    print(f"  -> Anomaly Surges Detected: {len(alerts)} alerts generated.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()
