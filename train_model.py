"""
Model Training Script for 2026 Global Trend Forecaster
This script loads the dataset, trains models, and saves them for the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_data(csv_path="trending_topics_2026.csv"):
    """Load the trending topics dataset."""
    print(f"📊 Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
    return df


def preprocess_data(df):
    """Preprocess the data and create features."""
    print("\n🔧 Preprocessing data...")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Encode categorical variables
    platform_encoder = LabelEncoder()
    region_encoder = LabelEncoder()
    sentiment_encoder = LabelEncoder()
    
    data['platform_encoded'] = platform_encoder.fit_transform(data['platform'])
    data['region_encoded'] = region_encoder.fit_transform(data['region'])
    data['sentiment_encoded'] = sentiment_encoder.fit_transform(data['sentiment'])
    
    print(f"✅ Data preprocessing complete")
    print(f"   Platforms: {list(platform_encoder.classes_)}")
    print(f"   Regions: {list(region_encoder.classes_)}")
    print(f"   Sentiments: {list(sentiment_encoder.classes_)}")
    
    return data, platform_encoder, region_encoder, sentiment_encoder


def create_vectorizer(texts):
    """Create TF-IDF vectorizer for text."""
    print("\n📝 Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_text = vectorizer.fit_transform(texts)
    print(f"✅ Vectorizer created with {X_text.shape[1]} features")
    return vectorizer, X_text


def prepare_features(df, X_text, vectorizer):
    """Combine text features with categorical features."""
    print("\n🔀 Combining text and categorical features...")
    
    # Convert sparse matrix to dense
    X_text_dense = X_text.toarray()
    
    # Add categorical features
    X_combined = np.hstack([
        X_text_dense,
        df[['platform_encoded', 'region_encoded']].values
    ])
    
    print(f"✅ Combined feature matrix shape: {X_combined.shape}")
    return X_combined


def train_engagement_model(X, y_engagement):
    """Train Random Forest Regressor for engagement score."""
    print("\n🏋️ Training Engagement Score Model (Random Forest Regressor)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_engagement, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"✅ Engagement Model trained:")
    print(f"   Train R² Score: {train_r2:.4f}")
    print(f"   Test R² Score: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.4f}")
    
    return model


def train_sentiment_model(X, y_sentiment):
    """Train Logistic Regression for sentiment classification."""
    print("\n🏋️ Training Sentiment Classification Model (Logistic Regression)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sentiment, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"✅ Sentiment Model trained:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    return model


def save_models(engagement_model, sentiment_model, vectorizer, 
                platform_encoder, region_encoder, sentiment_encoder):
    """Save trained models and encoders to pickle files."""
    print("\n💾 Saving models...")
    
    models_dict = {
        'engagement_model': engagement_model,
        'sentiment_model': sentiment_model,
        'vectorizer': vectorizer,
        'platform_encoder': platform_encoder,
        'region_encoder': region_encoder,
        'sentiment_encoder': sentiment_encoder
    }
    
    with open('models.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    
    print("✅ All models saved to 'models.pkl'")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🚀 2026 GLOBAL TREND FORECASTER - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Preprocess data
    df_processed, platform_enc, region_enc, sentiment_enc = preprocess_data(df)
    
    # Create vectorizer
    vectorizer, X_text = create_vectorizer(df['headline'])
    
    # Prepare combined features
    X = prepare_features(df_processed, X_text, vectorizer)
    
    # Train models
    engagement_model = train_engagement_model(X, df['engagement_score'])
    sentiment_model = train_sentiment_model(X, df_processed['sentiment_encoded'])
    
    # Save models
    save_models(engagement_model, sentiment_model, vectorizer,
                platform_enc, region_enc, sentiment_enc)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE! Ready to run Streamlit app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
