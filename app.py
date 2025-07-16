import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
from wordcloud import WordCloud
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Review Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the review data"""
    try:
        df = pd.read_csv('review_analysis_results.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload 'review_analysis_results.csv'")
        return None


def create_sentiment_distribution(df):
    """Create sentiment distribution chart"""
    sentiment_counts = df['sentiment_label'].value_counts()

    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_rating_distribution(df):
    """Create rating distribution chart"""
    rating_counts = df['rating_score'].value_counts().sort_index()

    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Rating Distribution",
        labels={'x': 'Rating Score', 'y': 'Count'},
        color=rating_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_topic_analysis(df):
    """Create topic analysis visualization"""
    topic_counts = df['dominant_topic'].value_counts().sort_index()

    fig = px.bar(
        x=[f"Topic {i}" for i in topic_counts.index],
        y=topic_counts.values,
        title="Topic Distribution",
        labels={'x': 'Topic', 'y': 'Count'},
        color=topic_counts.values,
        color_continuous_scale='plasma'
    )
    return fig


def create_confidence_analysis(df):
    """Create confidence level analysis"""
    fig = px.histogram(
        df,
        x='confidence_level',
        nbins=30,
        title="Confidence Level Distribution",
        labels={'confidence_level': 'Confidence Level', 'count': 'Frequency'}
    )
    fig.add_vline(x=df['confidence_level'].mean(), line_dash="dash",
                  annotation_text=f"Mean: {df['confidence_level'].mean():.3f}")
    return fig


def create_sentiment_rating_heatmap(df):
    """Create sentiment vs rating heatmap"""
    cross_tab = pd.crosstab(df['sentiment_label'], df['rating_score'])

    fig = px.imshow(
        cross_tab,
        title="Sentiment vs Rating Score Heatmap",
        labels=dict(x="Rating Score", y="Sentiment", color="Count"),
        color_continuous_scale='Blues'
    )
    return fig


def create_topic_sentiment_analysis(df):
    """Create topic-sentiment analysis"""
    topic_sentiment = df.groupby(['dominant_topic', 'sentiment_label']).size().reset_index(name='count')
    topic_sentiment['topic_label'] = topic_sentiment['dominant_topic'].apply(lambda x: f"Topic {x}")

    fig = px.bar(
        topic_sentiment,
        x='topic_label',
        y='count',
        color='sentiment_label',
        title="Sentiment Distribution by Topic",
        labels={'topic_label': 'Topic', 'count': 'Count'},
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        }
    )
    return fig


def extract_top_keywords(df, n=20):
    """Extract top keywords from all reviews"""
    all_keywords = []
    for keywords in df['keyword_tags'].dropna():
        all_keywords.extend([kw.strip() for kw in keywords.split(',')])

    keyword_counts = Counter(all_keywords)
    return keyword_counts.most_common(n)


def create_keyword_chart(df):
    """Create keyword frequency chart"""
    top_keywords = extract_top_keywords(df, 15)
    keywords, counts = zip(*top_keywords)

    fig = px.bar(
        x=list(counts),
        y=list(keywords),
        orientation='h',
        title="Top Keywords",
        labels={'x': 'Frequency', 'y': 'Keywords'},
        color=list(counts),
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def create_model_performance_metrics(df):
    """Create model performance overview"""
    metrics = {
        'Total Reviews': len(df),
        'Avg Confidence': df['confidence_level'].mean(),
        'High Confidence (>0.8)': len(df[df['confidence_level'] > 0.8]),
        'Low Confidence (<0.6)': len(df[df['confidence_level'] < 0.6])
    }
    return metrics


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Review Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Sidebar
    st.sidebar.header("🔧 Dashboard Controls")

    # Filters
    st.sidebar.subheader("Filters")

    # Sentiment filter
    sentiment_filter = st.sidebar.multiselect(
        "Select Sentiment",
        options=df['sentiment_label'].unique(),
        default=df['sentiment_label'].unique()
    )

    # Rating filter
    rating_filter = st.sidebar.slider(
        "Rating Range",
        min_value=int(df['rating_score'].min()),
        max_value=int(df['rating_score'].max()),
        value=(int(df['rating_score'].min()), int(df['rating_score'].max()))
    )

    # Confidence filter
    confidence_filter = st.sidebar.slider(
        "Confidence Level Range",
        min_value=float(df['confidence_level'].min()),
        max_value=float(df['confidence_level'].max()),
        value=(float(df['confidence_level'].min()), float(df['confidence_level'].max())),
        step=0.01
    )

    # Apply filters
    filtered_df = df[
        (df['sentiment_label'].isin(sentiment_filter)) &
        (df['rating_score'] >= rating_filter[0]) &
        (df['rating_score'] <= rating_filter[1]) &
        (df['confidence_level'] >= confidence_filter[0]) &
        (df['confidence_level'] <= confidence_filter[1])
        ]

    # Display filtered data info
    st.sidebar.write(f"📊 **Filtered Data**: {len(filtered_df):,} reviews")

    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Detailed Analysis", "🏷️ Topic Analysis", "📊 Model Performance"])

    with tab1:
        st.header("📈 Overview Dashboard")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Reviews", f"{len(filtered_df):,}")

        with col2:
            avg_rating = filtered_df['rating_score'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")

        with col3:
            avg_confidence = filtered_df['confidence_level'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")

        with col4:
            positive_pct = (filtered_df['sentiment_label'] == 'Positive').mean() * 100
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_sentiment_distribution(filtered_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_rating_distribution(filtered_df), use_container_width=True)

        # Additional charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_confidence_analysis(filtered_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_sentiment_rating_heatmap(filtered_df), use_container_width=True)

    with tab2:
        st.header("🔍 Detailed Analysis")

        # Keyword analysis
        st.subheader("Top Keywords")
        st.plotly_chart(create_keyword_chart(filtered_df), use_container_width=True)

        # Data table
        st.subheader("Raw Data")
        st.dataframe(
            filtered_df.head(100),
            use_container_width=True,
            hide_index=True
        )

        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name='filtered_reviews.csv',
            mime='text/csv'
        )

    with tab3:
        st.header("🏷️ Topic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_topic_analysis(filtered_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_topic_sentiment_analysis(filtered_df), use_container_width=True)

        # Topic details
        st.subheader("Topic Details")
        for topic in sorted(filtered_df['dominant_topic'].unique()):
            topic_data = filtered_df[filtered_df['dominant_topic'] == topic]

            with st.expander(f"📂 Topic {topic} ({len(topic_data)} reviews)"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Average Rating**: {topic_data['rating_score'].mean():.2f}")

                with col2:
                    st.write(f"**Average Confidence**: {topic_data['confidence_level'].mean():.3f}")

                with col3:
                    most_common_sentiment = topic_data['sentiment_label'].mode().iloc[0]
                    st.write(f"**Most Common Sentiment**: {most_common_sentiment}")

                # Top keywords for this topic
                topic_keywords = extract_top_keywords(topic_data, 10)
                st.write("**Top Keywords**:")
                keywords_text = ", ".join([f"{kw} ({count})" for kw, count in topic_keywords])
                st.write(keywords_text)

    with tab4:
        st.header("📊 Model Performance")

        # Performance metrics
        metrics = create_model_performance_metrics(filtered_df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Reviews", f"{metrics['Total Reviews']:,}")

        with col2:
            st.metric("Avg Confidence", f"{metrics['Avg Confidence']:.3f}")

        with col3:
            high_conf_pct = (metrics['High Confidence (>0.8)'] / metrics['Total Reviews']) * 100
            st.metric("High Confidence", f"{high_conf_pct:.1f}%")

        with col4:
            low_conf_pct = (metrics['Low Confidence (<0.6)'] / metrics['Total Reviews']) * 100
            st.metric("Low Confidence", f"{low_conf_pct:.1f}%")

        # Confidence distribution by sentiment
        st.subheader("Confidence Distribution by Sentiment")

        fig = px.box(
            filtered_df,
            x='sentiment_label',
            y='confidence_level',
            title="Model Confidence by Sentiment",
            color='sentiment_label',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#f39c12'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model accuracy analysis
        st.subheader("Model Accuracy Analysis")

        # Correlation between rating and sentiment
        sentiment_mapping = {'Negative': 1, 'Neutral': 3, 'Positive': 5}
        filtered_df['sentiment_numeric'] = filtered_df['sentiment_label'].map(sentiment_mapping)

        correlation = filtered_df['rating_score'].corr(filtered_df['sentiment_numeric'])
        st.write(f"**Correlation between Rating and Sentiment**: {correlation:.3f}")

        # Accuracy by confidence level
        st.subheader("Performance by Confidence Level")

        confidence_bins = pd.cut(filtered_df['confidence_level'], bins=5,
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        accuracy_by_conf = filtered_df.groupby(confidence_bins).agg({
            'rating_score': 'mean',
            'sentiment_label': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        }).round(3)

        st.dataframe(accuracy_by_conf, use_container_width=True)


if __name__ == "__main__":
    main()