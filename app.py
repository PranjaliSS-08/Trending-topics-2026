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
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        engagement_model.fit(X_train, y_train)
        
        # Train sentiment model
        X_train, X_test, y_train, y_test = train_test_split(
            X, data['sentiment_encoded'], test_size=0.2, random_state=42
        )
        
        sentiment_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='auto'
        )
        sentiment_model.fit(X_train, y_train)
        
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
    2. **Explore Trends** on the Trend Analytics page
    3. **Enter Headlines** on the Real-time Forecasting page
    4. **Get Predictions** for engagement score and sentiment
    5. **Understand Results** using the provided explanations
    """)
    
    st.markdown("---")
    
    st.info("""
    💡 **Tip**: High engagement scores suggest topics likely to go viral across platforms.
    Use this app to understand what makes news trending and predict future trends!
    """)


# ============================================================================
# PAGE: TREND ANALYTICS (EDA)
# ============================================================================

def page_trend_analytics():
    """Trend Analytics page with visualizations."""
    st.title("📊 Trend Analytics - 2026 Global Insights")
    st.markdown("---")
    
    df = load_dataset()
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Avg Engagement", f"{df['engagement_score'].mean():.1f}/100")
    col2.metric("Max Engagement", f"{df['engagement_score'].max():.0f}/100")
    col3.metric("Min Engagement", f"{df['engagement_score'].min():.0f}/100")
    col4.metric("Total Headlines", len(df))
    
    st.markdown("---")
    
    # Row 1: Sentiment Distribution & Platform Analytics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎭 Sentiment Distribution (Global Feb 2026)")
        sentiment_counts = df['sentiment'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        # Make percentage text white
        for autotext in autotexts:
            autotext.set_color('white')
        
        ax.set_title('Global Sentiment Mood', fontsize=14, weight='bold', pad=20)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("📱 Average Engagement Score per Platform")
        platform_engagement = df.groupby('platform')['engagement_score'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(platform_engagement.index, platform_engagement.values, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('Average Engagement Score', fontsize=11, weight='bold')
        ax.set_xlabel('Platform', fontsize=11, weight='bold')
        ax.set_title('Engagement Score by Platform', fontsize=14, weight='bold', pad=20)
        ax.set_ylim([0, 100])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Region Analytics & Word Cloud
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌍 Engagement Score Distribution by Region")
        region_data = df.groupby('region')['engagement_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(region_data.index, region_data['mean'], color=['#9b59b6', '#f39c12', '#1abc9c'])
        ax.set_xlabel('Average Engagement Score', fontsize=11, weight='bold')
        ax.set_title('Average Engagement by Region', fontsize=14, weight='bold', pad=20)
        ax.set_xlim([0, 100])
        
        # Add value labels
        for i, (idx, row) in enumerate(region_data.iterrows()):
            ax.text(row['mean'], i, f"{row['mean']:.1f}", va='center', ha='left', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("☁️ Word Cloud of Top Trending Terms")
        # Combine all headlines for word cloud
        all_text = ' '.join(df['headline'].astype(str))
        
        if WORDCLOUD_AVAILABLE:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
        else:
            # Fallback: Top words frequency bar chart (no NLTK required)
            # Simple stopwords list
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'is', 'are', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'of', 'from', 'by', 'with', 'as', 'that', 'this', 'it', 'its', 'about',
                'which', 'who', 'whom', 'where', 'when', 'why', 'how', 'all', 'each',
                'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can'
            }
            
            # Extract words using regex (simple tokenization)
            words = re.findall(r'\b[a-z]+\b', all_text.lower())
            # Filter: remove stopwords and short words
            words = [w for w in words if len(w) > 3 and w not in stop_words]
            
            # Get top 15 words
            word_freq = Counter(words).most_common(15)
            
            if word_freq:
                fig, ax = plt.subplots(figsize=(8, 6))
                words_list, freqs = zip(*word_freq)
                colors = plt.cm.viridis(np.linspace(0, 1, len(words_list)))
                ax.barh(words_list, freqs, color=colors)
                ax.set_xlabel('Frequency', fontsize=11, weight='bold')
                ax.set_title('Top 15 Trending Words', fontsize=14, weight='bold', pad=20)
                ax.invert_yaxis()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Not enough words to display word frequency chart.")
    
    st.markdown("---")
    
    # Row 3: Sentiment by Platform & Region Heatmap
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎭 Sentiment Breakdown by Platform")
        sentiment_platform = pd.crosstab(df['platform'], df['sentiment'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_platform.plot(kind='bar', ax=ax, color=['#e74c3c', '#95a5a6', '#2ecc71'])
        ax.set_title('Sentiment Distribution by Platform', fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Platform', fontsize=11, weight='bold')
        ax.set_ylabel('Count', fontsize=11, weight='bold')
        ax.legend(title='Sentiment', fontsize=10)
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Engagement Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['engagement_score'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(df['engagement_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["engagement_score"].mean():.1f}')
        ax.set_xlabel('Engagement Score', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title('Distribution of Engagement Scores', fontsize=14, weight='bold', pad=20)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data Table
    st.subheader("📋 Dataset Preview")
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Engagement Score Statistics**")
        st.write(df['engagement_score'].describe().round(2))
    
    with col2:
        st.write("**Platform & Region Breakdown**")
        st.write(f"Platforms: {', '.join(df['platform'].unique())}")
        st.write(f"Regions: {', '.join(df['region'].unique())}")
        st.write(f"Sentiments: {', '.join(df['sentiment'].unique())}")


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
        ["🏠 Home", "📊 Trend Analytics", "🔮 Real-time Forecasting"],
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
    elif "📊 Trend Analytics" in page:
        page_trend_analytics()
    elif "🔮 Real-time Forecasting" in page:
        page_forecasting()


if __name__ == "__main__":
    main()
