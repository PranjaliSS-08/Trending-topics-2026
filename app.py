"""
🚀 2026 GLOBAL TREND FORECASTER
A comprehensive Streamlit application for predicting engagement scores and sentiment
of trending topics in 2026 using NLP and machine learning models.

Version: 1.1 - Fixed deployment issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Optional wordcloud import with fallback
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="2026 Global Trend Forecaster",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        font-size: 3em;
        font-weight: bold;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    .neutral-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def train_and_save_models(df):
    """Train models if they don't exist."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    
    with st.spinner("🤖 Training ML models on first run (this takes ~30 seconds)..."):
        # Preprocess data
        data = df.copy()
        platform_encoder = LabelEncoder()
        region_encoder = LabelEncoder()
        sentiment_encoder = LabelEncoder()
        
        data['platform_encoded'] = platform_encoder.fit_transform(data['platform'])
        data['region_encoded'] = region_encoder.fit_transform(data['region'])
        data['sentiment_encoded'] = sentiment_encoder.fit_transform(data['sentiment'])
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_text = vectorizer.fit_transform(data['headline'])
        X_text_dense = X_text.toarray()
        
        # Combine features
        X = np.hstack([X_text_dense, data[['platform_encoded', 'region_encoded']].values])
        
        # Train engagement model
        X_train, X_test, y_train, y_test = train_test_split(
            X, data['engagement_score'], test_size=0.2, random_state=42
        )
        
        engagement_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        engagement_model.fit(X_train, y_train)
        
        # Train sentiment model with balanced class weights
        # ===================================================================
        # WHY class_weight='balanced'?
        # 
        # Without it: Model predicts only "Positive" (the majority class)
        # because it achieves high accuracy by ignoring minority classes.
        #
        # With it: class_weight='balanced' automatically adjusts weights
        # inversely proportional to class frequencies:
        # - Minority classes (Negative, Neutral) get HIGHER penalties
        # - Majority class (Positive) gets LOWER penalty
        # - Forces model to learn all classes equally well
        #
        # Formula: weight = total_samples / (n_classes * class_samples)
        # ===================================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, data['sentiment_encoded'], test_size=0.2, random_state=42
        )
        
        # Apply SMOTE to training data only (prevents data leakage)
        # SMOTE creates synthetic samples to balance minority classes
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print(f"   Training data balanced via SMOTE: {len(y_train)} → {len(y_train_smote)} samples")
        
        # LogisticRegression with balanced class weights ensures
        # the model predicts Positive, Negative, AND Neutral
        sentiment_model = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight='balanced'  # ← CRITICAL for handling imbalance!
        )
        sentiment_model.fit(X_train_smote, y_train_smote)
        
        # Save models
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
        
        st.success("✅ Models trained and cached!")
    
    return models_dict


@st.cache_resource
def load_models():
    """Load pre-trained models from pickle file or train if missing."""
    if os.path.exists('models.pkl'):
        with open('models.pkl', 'rb') as f:
            models_dict = pickle.load(f)
        return models_dict
    else:
        # Auto-train models on first run
        df = load_dataset()
        return train_and_save_models(df)


@st.cache_data
def load_dataset():
    """Load the trending topics dataset."""
    try:
        df = pd.read_csv('trending_topics_2026.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure trending_topics_2026.csv is in the directory.")
        st.stop()


def vectorize_text(text, vectorizer):
    """Convert headline text into TF-IDF features."""
    return vectorizer.transform([text]).toarray()


def encode_categorical(platform, region, platform_enc, region_enc):
    """Encode categorical variables."""
    try:
        platform_encoded = platform_enc.transform([platform])[0]
        region_encoded = region_enc.transform([region])[0]
        return platform_encoded, region_encoded
    except:
        st.warning("⚠️ Unknown platform or region selected.")
        return 0, 0


def prepare_prediction_features(headline, platform, region, vectorizer, 
                                platform_enc, region_enc):
    """Prepare features for prediction."""
    # Vectorize text
    X_text = vectorize_text(headline, vectorizer)
    
    # Encode categorical
    platform_encoded, region_encoded = encode_categorical(
        platform, region, platform_enc, region_enc
    )
    
    # Combine features
    X_features = np.hstack([
        X_text,
        [[platform_encoded, region_encoded]]
    ])
    
    return X_features


def predict_trend(headline, platform, region, models_dict):
    """
    Predict engagement score and sentiment for a given headline.
    
    Args:
        headline: User input headline
        platform: Selected platform (X, News, Google)
        region: Selected region
        models_dict: Dictionary containing all trained models
    
    Returns:
        Dictionary with engagement_score and sentiment predictions
    """
    # Extract models
    engagement_model = models_dict['engagement_model']
    sentiment_model = models_dict['sentiment_model']
    vectorizer = models_dict['vectorizer']
    platform_enc = models_dict['platform_encoder']
    region_enc = models_dict['region_encoder']
    sentiment_enc = models_dict['sentiment_encoder']
    
    # Prepare features
    X_features = prepare_prediction_features(
        headline, platform, region, vectorizer, platform_enc, region_enc
    )
    
    # Make predictions
    engagement_score = engagement_model.predict(X_features)[0]
    sentiment_pred = sentiment_model.predict(X_features)[0]
    
    # Ensure engagement score is within 0-100
    engagement_score = max(0, min(100, engagement_score))
    
    # Decode sentiment
    sentiment = sentiment_enc.inverse_transform([sentiment_pred])[0]
    
    return {
        'engagement_score': round(engagement_score, 2),
        'sentiment': sentiment
    }


def get_sentiment_emoji(sentiment):
    """Get emoji based on sentiment."""
    emoji_map = {
        'Positive': '😊',
        'Neutral': '😐',
        'Negative': '😢'
    }
    return emoji_map.get(sentiment, '❓')


# ============================================================================
# PAGE: HOME
# ============================================================================

def page_home():
    """Home page with project overview."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-title">🚀 2026 Global Trend Forecaster</div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 What is This App?")
        st.write("""
        The **2026 Global Trend Forecaster** is an advanced NLP-powered application 
        that predicts the viral potential and sentiment of trending topics across 
        global platforms.
        
        Using machine learning models trained on synthetic 2026 data, this app helps:
        - **Predict Engagement**: How likely a topic will go viral (0-100 scale)
        - **Analyze Sentiment**: Determine if a topic is Positive, Neutral, or Negative
        - **Track Trends**: Visualize trending patterns across platforms and regions
        """)
    
    with col2:
        st.subheader("📈 Key Features")
        st.write("""
        ✨ **Real-time Predictions**: Instant engagement and sentiment analysis
        
        🌍 **Multi-Platform Support**: X, News, Google Trends
        
        🗺️ **Regional Analysis**: Global, India, USA perspectives
        
        📊 **Rich Visualizations**: Word clouds, sentiment distribution, platform analytics
        
        🤖 **AI-Powered**: Using Random Forest & Logistic Regression models
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Dataset Overview")
    df = load_dataset()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Platforms", df['platform'].nunique())
    col3.metric("Regions", df['region'].nunique())
    col4.metric("Features", df.shape[1])
    
    st.write("**Dataset Features:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Headline**: News topic/trending story")
        st.write("**Platform**: X, News, Google Trends")
    
    with col2:
        st.write("**Region**: Global, India, USA")
        st.write("**Engagement Score**: 0-100 (higher = more viral)")
    
    with col3:
        st.write("**Sentiment**: Positive, Neutral, Negative")
    
    st.markdown("---")
    
    st.subheader("🎯 How to Use")
    st.write("""
    1. **Navigate** using the sidebar menu
    2. **Enter Headlines** on the Real-time Forecasting page
    3. **Get Predictions** for engagement score and sentiment
    4. **Understand Results** using the provided explanations
    """)
    
    st.markdown("---")
    
    st.info("""
    💡 **Tip**: High engagement scores suggest topics likely to go viral across platforms.
    Use this app to understand what makes news trending and predict future trends!
    """)


# ============================================================================
# PAGE: REAL-TIME FORECASTING
# ============================================================================

def page_forecasting():
    """Real-time forecasting page for user predictions."""
    st.title("🔮 Real-time Trend Forecasting")
    st.markdown("---")
    
    st.write("""
    Enter a headline and select platform/region to get instant predictions on:
    - **Engagement Score**: How viral your topic might become (0-100)
    - **Sentiment**: Whether the topic is Positive, Neutral, or Negative
    """)
    
    st.markdown("---")
    
    # Load models
    models_dict = load_models()
    
    # Input Section
    st.subheader("📝 Enter Your Headline")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        headline = st.text_input(
            "Headline Text",
            placeholder="e.g., 'AI breakthroughs revolutionize healthcare in 2026'",
            label_visibility="collapsed"
        )
    
    with col2:
        platform = st.selectbox(
            "Platform",
            options=['X', 'News', 'Google'],
            label_visibility="collapsed"
        )
    
    with col3:
        region = st.selectbox(
            "Region",
            options=['Global', 'India', 'USA'],
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # Prediction Button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button(
            "🔥 Analyze Trend Power",
            use_container_width=True,
            key="predict_button",
            type="primary"
        )
    
    # Process Prediction
    if predict_button:
        if not headline.strip():
            st.error("⚠️ Please enter a headline!")
        else:
            with st.spinner("⏳ Analyzing 2026 Trends..."):
                # Make prediction
                result = predict_trend(headline, platform, region, models_dict)
                
                engagement = result['engagement_score']
                sentiment = result['sentiment']
                emoji = get_sentiment_emoji(sentiment)
                
                # Display Results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.metric(
                        "🔥 Predicted Engagement Score",
                        f"{engagement}/100",
                        delta="Higher = More Viral" if engagement > 70 else "Moderate" if engagement > 50 else "Lower"
                    )
                    
                    # Engagement Meter
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh([0], [engagement], color='#3498db', height=0.3)
                    ax.barh([0], [100], color='#ecf0f1', height=0.3, alpha=0.3)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    
                    # Add value text
                    ax.text(engagement/2, 0, f'{engagement}%', va='center', ha='center', 
                           fontsize=20, fontweight='bold', color='white')
                    
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    st.metric("🎭 Predicted Sentiment", f"{emoji} {sentiment}")
                    
                    # Sentiment Box
                    if sentiment == 'Positive':
                        st.markdown(f"""
                        <div class="success-box">
                        <h3>✅ POSITIVE SENTIMENT</h3>
                        <p>This topic has a <b>positive tone</b> and is likely to resonate well with audiences. 
                        Great for uplifting content!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment == 'Negative':
                        st.markdown(f"""
                        <div class="warning-box">
                        <h3>⚠️ NEGATIVE SENTIMENT</h3>
                        <p>This topic has a <b>negative tone</b>. While it may still trend, handle with care 
                        and consider the impact.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="neutral-box">
                        <h3>➖ NEUTRAL SENTIMENT</h3>
                        <p>This topic is <b>neutral in tone</b>. It's factual and informative without strong 
                        emotional bias.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed Analysis
                st.subheader("📊 Detailed Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Platform:** " + platform)
                    st.write("**Region:** " + region)
                
                with col2:
                    virality_level = "🔥 VIRAL" if engagement >= 80 else "📈 HIGH" if engagement >= 60 else "➡️ MEDIUM" if engagement >= 40 else "📉 LOW"
                    st.write("**Virality Level:** " + virality_level)
                
                with col3:
                    st.write("**Prediction Time:** " + datetime.now().strftime("%H:%M:%S"))
                
                st.markdown("---")
                
                # Explanation
                st.subheader("💡 What This Means")
                
                explanation = f"""
                **Engagement Score ({engagement}/100):**
                
                Your headline has an engagement score of **{engagement}/100**. This score represents the predicted 
                likelihood of the topic going viral across {platform} in the {region} region.
                """
                
                if engagement >= 80:
                    explanation += """
                    - 🔥 **HIGHLY VIRAL**: This topic is likely to generate massive engagement and widespread sharing
                    - Perfect for trending hashtags and viral campaigns
                    - Expect significant reach and interaction
                    """
                elif engagement >= 60:
                    explanation += """
                    - 📈 **HIGH ENGAGEMENT**: Strong potential for virality
                    - Likely to gain good traction across the platform
                    - Good content for audience reach
                    """
                elif engagement >= 40:
                    explanation += """
                    - ➡️ **MODERATE ENGAGEMENT**: Average engagement potential
                    - Will resonate with some audiences but not explosive growth
                    - Solid content for consistent reach
                    """
                else:
                    explanation += """
                    - 📉 **LOW ENGAGEMENT**: Limited viral potential
                    - May need refinement or different positioning
                    - Consider adjusting the angle or adding more hooks
                    """
                
                explanation += f"""
                
                **Sentiment Analysis ({sentiment}):**
                
                The predicted sentiment is **{sentiment}**. This indicates the emotional tone of your headline.
                """
                
                if sentiment == 'Positive':
                    explanation += """
                    - ✅ Audiences will likely respond favorably
                    - Good for brand building and audience loyalty
                    - Encourages shares and positive discussions
                    """
                elif sentiment == 'Negative':
                    explanation += """
                    - ⚠️ May trigger strong reactions (both for and against)
                    - Can drive engagement but needs careful handling
                    - Consider the broader impact before posting
                    """
                else:
                    explanation += """
                    - ➖ Neutral, informative tone
                    - Professional and balanced perspective
                    - Good for news and factual reporting
                    """
                
                st.info(explanation)
                
                st.markdown("---")
                
                # Recommendations
                st.subheader("🎯 Recommendations")
                
                recommendations = []
                
                if engagement < 50:
                    recommendations.append("• Consider using more compelling language or adding emotional hooks")
                    recommendations.append("• Try adding relevant hashtags or trending keywords")
                    recommendations.append("• Consider the target audience's interests")
                
                if sentiment == 'Negative' and engagement > 70:
                    recommendations.append("• Be prepared for heated discussions and strong reactions")
                    recommendations.append("• Have moderation and response strategies ready")
                
                if len(recommendations) == 0:
                    recommendations.append("• Great headline! High engagement and good sentiment combination")
                    recommendations.append("• Consider this as a template for future content")
                    recommendations.append("• Test similar styles for consistent viral potential")
                
                for rec in recommendations:
                    st.write(rec)


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def main():
    """Main app with sidebar navigation."""
    
    # Sidebar Navigation
    st.sidebar.markdown("## 🚀 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔮 Real-time Forecasting"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Sidebar Info
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.write("""
    **2026 Global Trend Forecaster**
    
    An AI-powered app for predicting viral trends and sentiment analysis.
    
    **Models Used:**
    - Random Forest (Engagement)
    - Logistic Regression (Sentiment)
    
    **Data:** Synthetic 2026 Trends
    
    **Built with:** Streamlit, Scikit-learn, NLP
    """)
    
    st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.caption("© 2026 Global Trend Forecaster | Data Science Assignment 11")
    
    # Route to Pages
    if "🏠 Home" in page:
        page_home()
    elif "🔮 Real-time Forecasting" in page:
        page_forecasting()


if __name__ == "__main__":
    main()
