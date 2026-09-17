"""
clv_engine.py - Customer Lifecycle & Risk Analytics Engine
==========================================================
Engine C: RFM Segmentation, Probabilistic Customer Lifetime Value (CLV),
Machine Learning Churn Classification, and Demand Surge Anomaly Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class CustomerLifecycleEngine:
    """
    Engine C: RFM segmentation, probabilistic CLV estimation,
    churn classification, and operational surge anomaly detection.
    """

    def __init__(self, dormancy_threshold_days: int = 90, margin_rate: float = 0.22, discount_rate: float = 0.10):
        self.dormancy_threshold_days = dormancy_threshold_days
        self.margin_rate = margin_rate
        self.discount_rate = discount_rate
        self.churn_classifier = None
        self.scaler = StandardScaler()
        self.isolation_forest = None

    # ==========================================
    # 1. RFM Segmentation
    # ==========================================
    def compute_rfm_metrics(
        self,
        df_orders: pd.DataFrame,
        snapshot_date: Any = None
    ) -> pd.DataFrame:
        """
        Calculates Recency, Frequency, and Monetary value per customer.
        Assigns quintile scores and business segment labels.
        """
        df = df_orders.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])

        if snapshot_date is None:
            snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(snapshot_date)

        # Aggregate customer orders
        rfm = df.groupby('customer_id').agg(
            recency=('order_date', lambda d: (snapshot_date - d.max()).days),
            frequency=('order_id', 'nunique'),
            monetary=('order_amount', 'sum'),
            avg_order_value=('order_amount', 'mean'),
            first_order_date=('order_date', 'min'),
            last_order_date=('order_date', 'max'),
            unique_skus=('product_id', 'nunique'),
            primary_category=('category', lambda c: c.mode()[0] if not c.empty else 'General')
        ).reset_index()

        # Recency score: Lower recency days = higher score (5 is best)
        # Frequency score: Higher orders = higher score (5 is best)
        # Monetary score: Higher spend = higher score (5 is best)
        rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

        rfm['rfm_score'] = (
            rfm['r_score'].astype(str) +
            rfm['f_score'].astype(str) +
            rfm['m_score'].astype(str)
        )

        # Assign enterprise customer segments
        def assign_segment(row):
            r = row['r_score']
            f = row['f_score']
            m = row['m_score']

            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'Recent Promising'
            elif r == 3 and f <= 2:
                return 'Potential Loyalist'
            elif r <= 2 and f >= 3:
                return 'At Risk / Need Attention'
            elif r <= 2 and f <= 2:
                return 'Dormant / Inactive'
            else:
                return 'General Active'

        rfm['customer_segment'] = rfm.apply(assign_segment, axis=1)
        return rfm

    # ==========================================
    # 2. Probabilistic CLV Estimation
    # ==========================================
    def calculate_clv(self, df_rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates expected Customer Lifetime Value (CLV) using probabilistic retention:
        Retention rate r is inversely related to Recency and proportional to Frequency.
        CLV = AOV * Annual_Frequency * margin * [ r * (1 + d) / (1 + d - r) ]
        """
        df = df_rfm.copy()

        # Empirical retention proxy: bounded in [0.15, 0.92]
        # Active recent customers have high retention; dormant have low retention
        r_factor = 1.0 / (1.0 + np.exp((df['recency'] - 60) / 30.0))
        freq_factor = np.clip(df['frequency'] / 5.0, 0.4, 1.2)
        df['retention_rate'] = np.clip(r_factor * freq_factor, 0.15, 0.92)

        # Annualized frequency
        annual_freq = np.clip(df['frequency'] * 1.5, 1.0, 24.0)

        # Expected CLV formula
        clv_multiplier = (df['retention_rate'] * (1 + self.discount_rate)) / (
            1 + self.discount_rate - df['retention_rate']
        )
        df['annual_spend_forecast'] = df['avg_order_value'] * annual_freq
        df['clv_1yr'] = (df['annual_spend_forecast'] * self.margin_rate).round(2)
        df['clv_lifetime'] = (df['annual_spend_forecast'] * self.margin_rate * clv_multiplier).round(2)

        # Categorize CLV tiers
        clv_quantiles = df['clv_lifetime'].quantile([0.70, 0.90]).values
        def clv_tier(val):
            if val >= clv_quantiles[1]:
                return 'Tier 1 - Platinum (High Value)'
            elif val >= clv_quantiles[0]:
                return 'Tier 2 - Gold'
            else:
                return 'Tier 3 - Standard'

        df['clv_tier'] = df['clv_lifetime'].apply(clv_tier)
        return df

    # ==========================================
    # 3. Churn Risk Classification
    # ==========================================
    def train_churn_classifier(
        self,
        df_rfm: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Trains Logistic Regression and Random Forest classifiers to predict
        customer churn probability based on behavioral RFM features.
        """
        df = df_rfm.copy()

        # Define churn target: Customer inactive for > dormancy_threshold_days
        df['is_churned'] = (df['recency'] > self.dormancy_threshold_days).astype(int)

        feature_cols = ['recency', 'frequency', 'monetary', 'avg_order_value', 'unique_skus']
        X = df[feature_cols].copy()
        y = df['is_churned'].values

        # Stratified train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model 1: Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=500)
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)
        y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

        # Model 2: Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

        # Evaluate models
        def compute_metrics(y_true, y_pred, y_prob):
            return {
                'Accuracy': round(accuracy_score(y_true, y_pred), 4),
                'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
                'F1_Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
                'ROC_AUC': round(roc_auc_score(y_true, y_prob), 4)
            }

        lr_metrics = compute_metrics(y_test, y_pred_lr, y_prob_lr)
        rf_metrics = compute_metrics(y_test, y_pred_rf, y_prob_rf)

        # Score full dataset with Random Forest
        df['churn_probability'] = rf_model.predict_proba(X)[:, 1].round(4)
        df['churn_risk_level'] = pd.cut(
            df['churn_probability'],
            bins=[-0.01, 0.30, 0.70, 1.0],
            labels=['Low Risk', 'Moderate Risk', 'Critical Risk']
        )

        self.churn_classifier = rf_model

        return {
            'metrics': {
                'Logistic Regression': lr_metrics,
                'Random Forest': rf_metrics
            },
            'scored_customers': df,
            'feature_importance': pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        }

    # ==========================================
    # 4. Traffic & Demand Surge Anomaly Detection
    # ==========================================
    def detect_demand_anomalies(
        self,
        df_sales: pd.DataFrame,
        z_threshold: float = 2.5,
        contamination: float = 0.03
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detects demand spikes and traffic surges using rolling Z-scores
        and an Isolation Forest multi-variate detector.
        """
        df = df_sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

        # 1. Fast Rolling Z-Score on units_demanded
        grouped = df.groupby('product_id')['units_demanded']
        rolling_mean = grouped.transform(lambda s: s.rolling(14, min_periods=3).mean())
        rolling_std = grouped.transform(lambda s: s.rolling(14, min_periods=3).std()).fillna(1.0)
        rolling_std = rolling_std.replace(0.0, 1.0)

        df['z_score_demand'] = ((df['units_demanded'] - rolling_mean) / rolling_std).round(2)
        df['is_z_anomaly'] = (df['z_score_demand'] > z_threshold).astype(int)

        # 2. Isolation Forest on units_demanded and traffic_sessions
        iso_features = df[['units_demanded', 'traffic_sessions']].copy()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        # Isolation Forest returns -1 for anomalies, 1 for normal
        iso_preds = self.isolation_forest.fit_predict(iso_features)
        df['is_iso_anomaly'] = (iso_preds == -1).astype(int)

        # Combined Anomaly Flag: Surge if both or high z-score
        df['is_surge_anomaly'] = (
            (df['is_z_anomaly'] == 1) | ((df['is_iso_anomaly'] == 1) & (df['units_demanded'] > rolling_mean))
        ).astype(int)

        # Generate Actionable Alert Log
        alerts_df = df[df['is_surge_anomaly'] == 1].copy()
        alerts_df['severity'] = np.where(
            alerts_df['z_score_demand'] > 4.0,
            'CRITICAL_SURGE',
            'ELEVATED_SURGE'
        )
        alerts_df['recommended_action'] = np.where(
            alerts_df['severity'] == 'CRITICAL_SURGE',
            'Expedite Emergency PO & Reserve Dedicated Warehouse Lane',
            'Trigger Buffer Stock Check & Monitor Daily Reorder Point'
        )

        return df, alerts_df


if __name__ == "__main__":
    from data_pipeline import clean_raw_catalog, synthesize_customer_transactions, synthesize_supply_chain_operations

    print("Running Customer Lifecycle & Risk Analytics Engine (Engine C)...")
    catalog = clean_raw_catalog()
    orders = synthesize_customer_transactions(catalog)
    sales, _ = synthesize_supply_chain_operations(catalog)

    clv_engine = CustomerLifecycleEngine()
    rfm_df = clv_engine.compute_rfm_metrics(orders)
    rfm_clv_df = clv_engine.calculate_clv(rfm_df)
    churn_results = clv_engine.train_churn_classifier(rfm_clv_df)

    print("[OK] RFM & Customer Segments:")
    print(rfm_clv_df['customer_segment'].value_counts())

    print("\n[OK] Churn Classification Benchmark:")
    for model_name, met in churn_results['metrics'].items():
        print(f"  - {model_name:22s} | F1: {met['F1_Score']} | ROC-AUC: {met['ROC_AUC']} | Acc: {met['Accuracy']}")

    print("\n[OK] Running Traffic & Demand Surge Anomaly Detection...")
    sales_scored, alerts = clv_engine.detect_demand_anomalies(sales)
    print(f"[OK] Total Anomaly Surge Events Detected: {len(alerts)} out of {len(sales_scored)} records.")
    print(alerts[['date', 'product_id', 'units_demanded', 'z_score_demand', 'severity', 'recommended_action']].head(3))
