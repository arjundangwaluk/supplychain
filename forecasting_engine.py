"""
forecasting_engine.py - Supply Chain Inventory & Predictive Forecasting Engine
==============================================================================
Engine B: SKU Demand Regressors, Operations Research Optimization (SS, ROP, MOQ)
and Asymmetric Business Loss Evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from scipy.stats import norm
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


class InventoryForecastingEngine:
    """
    Production-grade time series feature engineering, machine learning regressors,
    evaluator metrics (WAPE, Asymmetric Loss), and dynamic Operations Research logic.
    """

    SERVICE_LEVEL_Z_TABLE = {
        0.80: 0.8416,
        0.85: 1.0364,
        0.90: 1.2816,
        0.95: 1.6449,
        0.98: 2.0537,
        0.99: 2.3263,
        0.999: 3.0902
    }

    def __init__(self, c_stockout: float = 5.0, c_holding: float = 1.0):
        self.c_stockout = c_stockout
        self.c_holding = c_holding
        self.lgb_model = None
        self.xgb_model = None
        self.feature_cols = []

    # ==========================================
    # 1. Feature Engineering Pipeline
    # ==========================================
    def build_features(
        self,
        df_sales: pd.DataFrame,
        df_sku_sentiment: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Engineers temporal, lag, rolling, pricing, and NLP features across SKU panels.
        """
        df = df_sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

        # Merge sentiment score if available
        if df_sku_sentiment is not None and 'mean_sentiment' in df_sku_sentiment.columns:
            df = df.merge(
                df_sku_sentiment[['product_id', 'mean_sentiment']],
                on='product_id',
                how='left'
            )
            df['mean_sentiment'] = df['mean_sentiment'].fillna(0.0)
        else:
            df['mean_sentiment'] = 0.0

        # Calendar features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month

        # Price elasticity & discount lift feature
        # discount_lift = (actual_price - discounted_price) / actual_price
        df['discount_lift'] = ((df['actual_price'] - df['discounted_price']) / (df['actual_price'] + 1e-5)).clip(0.0, 1.0)
        df['log_price'] = np.log1p(df['discounted_price'])

        # Lags and Rolling Windows grouped by product_id
        grouped = df.groupby('product_id')['units_demanded']
        
        df['lag_1'] = grouped.shift(1)
        df['lag_7'] = grouped.shift(7)
        df['lag_14'] = grouped.shift(14)
        
        # Rolling statistics (using shift(1) to avoid lookahead data leakage)
        df['rolling_mean_7'] = grouped.shift(1).rolling(7, min_periods=1).mean()
        df['rolling_std_7'] = grouped.shift(1).rolling(7, min_periods=1).std().fillna(0.0)
        df['rolling_median_7'] = grouped.shift(1).rolling(7, min_periods=1).median()
        df['rolling_mean_14'] = grouped.shift(1).rolling(14, min_periods=1).mean()
        df['rolling_mean_28'] = grouped.shift(1).rolling(28, min_periods=1).mean()

        # Fill early warmup lags with backfill/median
        for col in ['lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_median_7', 'rolling_mean_14', 'rolling_mean_28']:
            df[col] = df[col].fillna(df['units_demanded'])
        df['rolling_std_7'] = df['rolling_std_7'].fillna(0.0)

        self.feature_cols = [
            'discount_lift', 'log_price', 'rating', 'mean_sentiment',
            'dayofweek', 'is_weekend', 'day', 'month',
            'lag_1', 'lag_7', 'lag_14',
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_mean_28'
        ]

        return df

    # ==========================================
    # 2. Evaluation Harness Metrics
    # ==========================================
    @staticmethod
    def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(y)."""
        denom = np.sum(y_true)
        if denom == 0:
            return 0.0
        return float(np.sum(np.abs(y_true - y_pred)) / denom)

    @classmethod
    def calculate_asymmetric_loss(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        c_stockout: float = 5.0,
        c_holding: float = 1.0
    ) -> float:
        """
        Supply Chain Business Loss:
        Penalizes stockouts (under-forecasting) heavier than excess inventory holding.
        Cost = sum( C_stockout * max(y - y_hat, 0) + C_holding * max(y_hat - y, 0) )
        """
        error = y_true - y_pred
        stockout_loss = np.maximum(error, 0.0) * c_stockout
        holding_loss = np.maximum(-error, 0.0) * c_holding
        total_loss = float(np.sum(stockout_loss + holding_loss))
        return total_loss

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates comprehensive forecast and operational cost metrics."""
        # Enforce non-negativity drift guard
        y_pred_clipped = np.maximum(y_pred, 0.0)

        mae = float(mean_absolute_error(y_true, y_pred_clipped))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
        wape = self.calculate_wape(y_true, y_pred_clipped)
        asym_loss = self.calculate_asymmetric_loss(
            y_true, y_pred_clipped, self.c_stockout, self.c_holding
        )

        return {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'WAPE': round(wape, 4),
            'WAPE_Pct': round(wape * 100, 2),
            'Asymmetric_Loss': round(asym_loss, 2)
        }

    # ==========================================
    # 3. Model Training & Forecasting
    # ==========================================
    def train_models(
        self,
        df_featured: pd.DataFrame,
        test_days: int = 30
    ) -> Dict[str, Any]:
        """
        Trains LightGBM, XGBoost, and Rolling Median baseline on train split,
        evaluating performance on the out-of-time test horizon.
        """
        max_date = df_featured['date'].max()
        split_date = max_date - pd.Timedelta(days=test_days)

        train_df = df_featured[df_featured['date'] <= split_date].copy()
        test_df = df_featured[df_featured['date'] > split_date].copy()

        X_train = train_df[self.feature_cols]
        y_train = train_df['units_demanded'].values
        X_test = test_df[self.feature_cols]
        y_test = test_df['units_demanded'].values

        # Baseline: Rolling 7-day Median
        y_pred_baseline = np.maximum(test_df['rolling_median_7'].values, 0.0)
        baseline_metrics = self.evaluate_predictions(y_test, y_pred_baseline)

        # LightGBM Regressor
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.lgb_model.fit(X_train, y_train)
        y_pred_lgb = np.maximum(self.lgb_model.predict(X_test), 0.0)
        lgb_metrics = self.evaluate_predictions(y_test, y_pred_lgb)

        # XGBoost Regressor
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        y_pred_xgb = np.maximum(self.xgb_model.predict(X_test), 0.0)
        xgb_metrics = self.evaluate_predictions(y_test, y_pred_xgb)

        # Feature Importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'LGB_Importance': self.lgb_model.feature_importances_,
            'XGB_Importance': self.xgb_model.feature_importances_
        }).sort_values(by='LGB_Importance', ascending=False)

        return {
            'metrics': {
                'Baseline (Rolling Median)': baseline_metrics,
                'LightGBM': lgb_metrics,
                'XGBoost': xgb_metrics
            },
            'test_df': test_df,
            'y_test': y_test,
            'y_pred_lgb': y_pred_lgb,
            'y_pred_xgb': y_pred_xgb,
            'y_pred_baseline': y_pred_baseline,
            'feature_importance': importance_df
        }

    # ==========================================
    # 4. Operations Research & Inventory Logic
    # ==========================================
    @classmethod
    def calculate_dynamic_safety_stock(
        cls,
        avg_demand: float,
        std_demand: float,
        lead_time_mean: float,
        lead_time_std: float,
        service_level: float = 0.95
    ) -> float:
        """
        Calculates dynamic safety stock taking both demand variability and
        vendor lead time variability into account:
        SS = Z * sqrt( L * sigma_d^2 + d^2 * sigma_L^2 )
        """
        z = norm.ppf(service_level)
        variance_lead_time_demand = (lead_time_mean * (std_demand ** 2)) + ((avg_demand ** 2) * (lead_time_std ** 2))
        sigma_ltd = np.sqrt(max(0.0001, variance_lead_time_demand))
        safety_stock = z * sigma_ltd
        return float(max(1.0, round(safety_stock, 2)))

    @classmethod
    def calculate_dynamic_rop(
        cls,
        avg_demand: float,
        lead_time_mean: float,
        safety_stock: float
    ) -> float:
        """
        Calculates dynamic Reorder Point:
        ROP = (d * L) + SS
        """
        expected_demand_in_lead = avg_demand * lead_time_mean
        rop = expected_demand_in_lead + safety_stock
        return float(round(rop, 2))

    @classmethod
    def compute_stockout_risk_score(
        cls,
        current_inventory: float,
        avg_demand: float,
        std_demand: float,
        lead_time_mean: float,
        lead_time_std: float
    ) -> float:
        """
        Probability of stockout during replenishment cycle:
        P(Demand in Lead Time > Current Inventory) = 1 - Phi( (Stock - d*L) / sigma_LTD )
        """
        expected_demand = avg_demand * lead_time_mean
        variance_ltd = (lead_time_mean * (std_demand ** 2)) + ((avg_demand ** 2) * (lead_time_std ** 2))
        sigma_ltd = np.sqrt(max(0.0001, variance_ltd))

        z_score = (current_inventory - expected_demand) / sigma_ltd
        stockout_prob = 1.0 - norm.cdf(z_score)
        return float(round(np.clip(stockout_prob, 0.0, 1.0), 4))

    @classmethod
    def evaluate_purchase_requisition(
        cls,
        current_inventory: float,
        rop: float,
        moq: int = 50,
        eoq_buffer: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Automated Purchase Requisition Trigger:
        Triggers when current inventory <= ROP.
        Enforces Minimum Order Quantity (MOQ) constraint strictly:
        Order Qty = max(MOQ, ceil(ROP - Current Inventory + EOQ))
        """
        trigger = bool(current_inventory <= rop)
        if trigger:
            deficit = max(0.0, rop - current_inventory)
            buffer_qty = eoq_buffer if eoq_buffer is not None else rop * 0.5
            recommended_order = int(np.ceil(deficit + buffer_qty))
            final_order_qty = int(max(moq, recommended_order))
            status = "CRITICAL_ORDER_REQUIRED" if current_inventory < (rop * 0.5) else "REORDER_TRIGGERED"
        else:
            final_order_qty = 0
            status = "HEALTHY_INVENTORY"

        return {
            'reorder_triggered': trigger,
            'status': status,
            'recommended_quantity': final_order_qty,
            'moq_enforced': bool(final_order_qty == moq and trigger),
            'deficit_to_rop': float(round(max(0.0, rop - current_inventory), 2))
        }


if __name__ == "__main__":
    from data_pipeline import clean_raw_catalog, parse_exploded_reviews, synthesize_supply_chain_operations
    from nlp_engine import NLPSentimentEngine

    print("Running Forecasting Engine verification...")
    catalog = clean_raw_catalog()
    reviews = parse_exploded_reviews(catalog)
    df_sales, df_inventory = synthesize_supply_chain_operations(catalog)

    nlp = NLPSentimentEngine()
    processed_reviews = nlp.process_reviews_dataframe(reviews, sample_size=1000)
    sku_sentiment = nlp.aggregate_sku_sentiment(processed_reviews)

    # Pick top 20 representative SKUs for fast demonstration
    top_skus = df_inventory['product_id'].head(20).tolist()
    sub_sales = df_sales[df_sales['product_id'].isin(top_skus)].copy()

    engine = InventoryForecastingEngine(c_stockout=5.0, c_holding=1.0)
    df_featured = engine.build_features(sub_sales, sku_sentiment)
    train_results = engine.train_models(df_featured, test_days=30)

    print("\n[OK] Model Benchmark Metrics (30-day out-of-time test horizon):")
    for model_name, m in train_results['metrics'].items():
        print(f"  - {model_name:28s} | WAPE: {m['WAPE_Pct']}% | MAE: {m['MAE']} | Asym Loss: ${m['Asymmetric_Loss']}")

    # Operations Research verification
    sample_sku = df_inventory.iloc[0]
    ss = engine.calculate_dynamic_safety_stock(
        sample_sku['avg_daily_demand'], sample_sku['std_daily_demand'],
        sample_sku['mean_lead_time'], sample_sku['std_lead_time'], service_level=0.95
    )
    rop = engine.calculate_dynamic_rop(
        sample_sku['avg_daily_demand'], sample_sku['mean_lead_time'], ss
    )
    risk = engine.compute_stockout_risk_score(
        sample_sku['current_inventory'], sample_sku['avg_daily_demand'], sample_sku['std_daily_demand'],
        sample_sku['mean_lead_time'], sample_sku['std_lead_time']
    )
    req = engine.evaluate_purchase_requisition(
        sample_sku['current_inventory'], rop, sample_sku['moq']
    )

    print(f"\n[OK] Operations Research Calibration for SKU: {sample_sku['product_id']}")
    print(f"  - Daily Demand: {sample_sku['avg_daily_demand']:.2f} ± {sample_sku['std_daily_demand']:.2f}")
    print(f"  - Vendor Lead Time: {sample_sku['mean_lead_time']:.1f} ± {sample_sku['std_lead_time']:.1f} days")
    print(f"  - Dynamic Safety Stock (95% SL): {ss}")
    print(f"  - Dynamic Reorder Point (ROP): {rop}")
    print(f"  - Current On-Hand: {sample_sku['current_inventory']}")
    print(f"  - Stockout Risk Score: {risk * 100:.2f}%")
    print(f"  - Requisition Action: {req['status']} (Order Qty: {req['recommended_quantity']}, MOQ: {sample_sku['moq']})")
