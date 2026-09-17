"""
nlp_engine.py - Natural Language & Sentiment Intelligence Pipeline
==================================================================
Engine A: Aspect-Based Voice-of-Customer & Sales Correlation Harness
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class NLPSentimentEngine:
    """
    Extracts sentiment polarity, compound metrics, and aspect-based insights
    from customer review titles and content using VADER.
    """

    # Domain-specific aspect taxonomy for e-commerce and consumer tech
    ASPECT_TAXONOMY = {
        'Quality & Build': [
            'quality', 'build', 'material', 'finish', 'sturdy', 'plastic', 'broken',
            'premium', 'solid', 'cheap material', 'flimsy', 'craftsmanship'
        ],
        'Price & Value': [
            'price', 'cost', 'worth', 'value', 'cheap', 'expensive', 'money',
            'deal', 'affordable', 'budget', 'overpriced', 'investment'
        ],
        'Delivery & Packaging': [
            'delivery', 'packaging', 'package', 'box', 'courier', 'shipped',
            'fast delivery', 'late', 'damaged box', 'arrival', 'shipping'
        ],
        'Performance & Functionality': [
            'performance', 'speed', 'battery', 'charge', 'charging', 'fast', 'slow',
            'working', 'durability', 'connection', 'cable', 'heat', 'sound', 'lag'
        ]
    }

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Calibrate domain-specific supply chain & consumer lexicon updates
        self.analyzer.lexicon.update({
            'unbreakable': 2.8,
            'sturdy': 2.4,
            'durable': 2.5,
            'flimsy': -2.6,
            'defective': -3.2,
            'overheating': -2.8,
            'laggy': -2.2,
            'value for money': 2.9,
            'waste of money': -3.2,
            'broke quickly': -3.0
        })

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Calculates VADER polarity scores for a given text."""
        if not text or not isinstance(text, str):
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        return self.analyzer.polarity_scores(text)

    def extract_aspect_sentiment(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Segments text by aspect keywords and evaluates aspect-specific sentiment.
        """
        text_lower = text.lower()
        aspect_results = {}

        # Split into sentences or clauses for local aspect context
        clauses = re.split(r'[.,;!?|\n]+', text_lower)

        for aspect_name, keywords in self.ASPECT_TAXONOMY.items():
            relevant_clauses = [
                c.strip() for c in clauses
                if any(kw in c for kw in keywords)
            ]
            if relevant_clauses:
                joined_clauses = ". ".join(relevant_clauses)
                scores = self.analyzer.polarity_scores(joined_clauses)
                aspect_results[aspect_name] = {
                    'detected': True,
                    'compound': float(scores['compound']),
                    'clause_count': len(relevant_clauses),
                    'sentiment': 'Positive' if scores['compound'] >= 0.05 else (
                        'Negative' if scores['compound'] <= -0.05 else 'Neutral'
                    )
                }
            else:
                aspect_results[aspect_name] = {
                    'detected': False,
                    'compound': 0.0,
                    'clause_count': 0,
                    'sentiment': 'Not Mentioned'
                }

        return aspect_results

    def process_reviews_dataframe(self, df_reviews: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
        """
        Enriches review records with compound polarity, classification, and aspects.
        """
        df = df_reviews.copy()
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).copy()

        # Combine title and content for robust context
        combined_texts = (df['review_title'].fillna('') + ". " + df['review_content'].fillna('')).tolist()

        compound_scores = []
        sentiment_labels = []
        quality_scores = []
        price_scores = []
        delivery_scores = []
        performance_scores = []

        for txt in combined_texts:
            pol = self.analyze_text(txt)
            compound = float(pol['compound'])
            compound_scores.append(compound)

            if compound >= 0.05:
                label = 'Positive'
            elif compound <= -0.05:
                label = 'Negative'
            else:
                label = 'Neutral'
            sentiment_labels.append(label)

            aspects = self.extract_aspect_sentiment(txt)
            quality_scores.append(aspects['Quality & Build']['compound'])
            price_scores.append(aspects['Price & Value']['compound'])
            delivery_scores.append(aspects['Delivery & Packaging']['compound'])
            performance_scores.append(aspects['Performance & Functionality']['compound'])

        df['sentiment_compound'] = compound_scores
        df['sentiment_label'] = sentiment_labels
        df['aspect_quality'] = quality_scores
        df['aspect_price'] = price_scores
        df['aspect_delivery'] = delivery_scores
        df['aspect_performance'] = performance_scores

        return df

    def aggregate_sku_sentiment(self, df_processed_reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates review sentiment to the SKU level.
        """
        agg = df_processed_reviews.groupby('product_id').agg(
            mean_sentiment=('sentiment_compound', 'mean'),
            pct_positive=('sentiment_label', lambda s: (s == 'Positive').mean() * 100),
            pct_negative=('sentiment_label', lambda s: (s == 'Negative').mean() * 100),
            mean_aspect_quality=('aspect_quality', 'mean'),
            mean_aspect_price=('aspect_price', 'mean'),
            mean_aspect_delivery=('aspect_delivery', 'mean'),
            mean_aspect_performance=('aspect_performance', 'mean'),
            review_sample_count=('sentiment_compound', 'count')
        ).reset_index()

        # Scale compound strictly between [-1.0, 1.0]
        agg['mean_sentiment'] = agg['mean_sentiment'].clip(lower=-1.0, upper=1.0)
        return agg

    @staticmethod
    def correlate_sentiment_with_velocity(
        df_sku_sentiment: pd.DataFrame,
        df_inventory_state: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Engine A Correlation Harness:
        Quantifies the statistical relationship between customer sentiment and SKU sales velocity.
        Computes Pearson r, Spearman rho, p-values, and linear regression slope.
        """
        merged = pd.merge(
            df_sku_sentiment,
            df_inventory_state[['product_id', 'avg_daily_demand', 'rating', 'discount_percentage']],
            on='product_id',
            how='inner'
        )

        if len(merged) < 10:
            return {
                'status': 'Insufficient overlapping data',
                'sample_size': len(merged)
            }

        # Filter out zero-variance if present
        x = merged['mean_sentiment'].values
        y = merged['avg_daily_demand'].values

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)

        interpretation = (
            "Statistically Significant Positive Elasticity (p < 0.05)"
            if pearson_p < 0.05 and pearson_r > 0 else
            "Inconclusive / Neutral Relationship"
        )

        return {
            'sample_size': len(merged),
            'pearson_r': float(round(pearson_r, 4)),
            'pearson_p_value': float(round(pearson_p, 6)),
            'spearman_rho': float(round(spearman_rho, 4)),
            'spearman_p_value': float(round(spearman_p, 6)),
            'regression_slope': float(round(slope, 4)),
            'regression_intercept': float(round(intercept, 4)),
            'r_squared': float(round(r_val**2, 4)),
            'interpretation': interpretation,
            'merged_data': merged
        }


if __name__ == "__main__":
    from data_pipeline import clean_raw_catalog, parse_exploded_reviews, synthesize_supply_chain_operations

    print("Executing Engine A (NLP & Voice-of-Customer Pipeline)...")
    catalog = clean_raw_catalog()
    reviews = parse_exploded_reviews(catalog)
    _, inventory = synthesize_supply_chain_operations(catalog)

    nlp = NLPSentimentEngine()
    processed_reviews = nlp.process_reviews_dataframe(reviews, sample_size=2000)
    sku_sentiment = nlp.aggregate_sku_sentiment(processed_reviews)
    corr_report = nlp.correlate_sentiment_with_velocity(sku_sentiment, inventory)

    print("[OK] Processed Reviews Sample:")
    print(processed_reviews[['product_id', 'sentiment_compound', 'sentiment_label']].head(3))
    print("\n[OK] Correlation Harness Results:")
    print(f"Sample Size: {corr_report['sample_size']}")
    print(f"Pearson r: {corr_report['pearson_r']} (p = {corr_report['pearson_p_value']})")
    print(f"Spearman rho: {corr_report['spearman_rho']} (p = {corr_report['spearman_p_value']})")
    print(f"Interpretation: {corr_report['interpretation']}")
