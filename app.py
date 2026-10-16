import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Netflix Dataset EDA Dashboard",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #141414; }
    h1, h2, h3 { color: #E50914 !important; }
    stMetric { background-color: #222; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Generate Sample Dataset (If local CSV is missing)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("netflix_titles.csv")
    except FileNotFoundError:
        # Fallback synthetic dataset matching Netflix parameters
        np.random.seed(42)
        n = 8807
        types = np.random.choice(['Movie', 'TV Show'], size=n, p=[0.696, 0.304])
        years = np.array([2018, 2017, 2019, 2020, 2016, 2021, 2015, 2014, 2013, 2012, 2010])
        p_years = np.array([0.15, 0.14, 0.14, 0.13, 0.11, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04])
        p_years /= p_years.sum()
        release_years = np.random.choice(years, size=n, p=p_years)
        
        movie_genres = ['Dramas', 'Comedies', 'Documentaries', 'Action & Adventure', 'Thrillers', 'Romantic Movies']
        tv_genres = ['International TV Shows', 'TV Dramas', 'TV Comedies', 'Crime TV Shows', 'Docuseries']

        genres, durations = [], []
        for t in types:
            if t == 'Movie':
                genres.append(", ".join(np.random.choice(movie_genres, size=np.random.choice([1, 2]), replace=False)))
                durations.append(int(np.random.normal(100, 20)))
            else:
                genres.append(", ".join(np.random.choice(tv_genres, size=np.random.choice([1, 2]), replace=False)))
                durations.append(np.random.choice([1, 2, 3, 4], p=[0.7, 0.18, 0.08, 0.04]))

        df = pd.DataFrame({
            'show_id': [f's{i+1}' for i in range(n)],
            'type': types,
            'title': [f'Title {i+1}' for i in range(n)],
            'release_year': release_years,
            'listed_in': genres,
            'duration_val': durations
        })
    return df

df = load_data()

# Sidebar Filters
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
st.sidebar.title("Filter Content")
type_filter = st.sidebar.multiselect("Select Content Type:", options=df['type'].unique(), default=df['type'].unique())
year_range = st.sidebar.slider("Select Release Year Range:", 
                               int(df['release_year'].min()), 
                               int(df['release_year'].max()), 
                               (2010, 2021))

# Filter Data
filtered_df = df[(df['type'].isin(type_filter)) & 
                 (df['release_year'] >= year_range[0]) & 
                 (df['release_year'] <= year_range[1])]

# Main Dashboard Title
st.title("🎬 Netflix Dataset Exploratory Data Analysis")
st.markdown("Interactive EDA Dashboard analyzing content mix, release year trends, top genres, and duration metrics.")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies Count", len(filtered_df[filtered_df['type'] == 'Movie']))
col3.metric("TV Shows Count", len(filtered_df[filtered_df['type'] == 'TV Show']))
col4.metric("Unique Release Years", filtered_df['release_year'].nunique())

st.markdown("---")

# Row 1: Content Type & Release Year Trend
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Content Distribution (Movie vs TV Show)")
    type_counts = filtered_df['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    fig_type = px.pie(type_counts, values='Count', names='Type', 
                      color_discrete_sequence=['#E50914', '#221F1F'], 
                      hole=0.4)
    st.plotly_chart(fig_type, use_container_width=True)

with col_right:
    st.subheader("2. Release Trends Over Time")
    trend_df = filtered_df.groupby(['release_year', 'type']).size().reset_index(name='Count')
    fig_trend = px.line(trend_df, x='release_year', y='Count', color='type',
                        color_discrete_map={'Movie': '#E50914', 'TV Show': '#B81D24'},
                        markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# Row 2: Top Genres & Top Release Years
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("3. Top 10 Common Genres")
    genres_series = filtered_df['listed_in'].str.split(', ').explode()
    top_genres = genres_series.value_counts().head(10).reset_index()
    top_genres.columns = ['Genre', 'Count']
    fig_genre = px.bar(top_genres, x='Count', y='Genre', orientation='h',
                       color='Count', color_continuous_scale='Reds')
    fig_genre.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_genre, use_container_width=True)

with col_g2:
    st.subheader("4. Top 10 Release Years")
    top_years = filtered_df['release_year'].value_counts().head(10).reset_index()
    top_years.columns = ['Year', 'Count']
    fig_years = px.bar(top_years, x='Year', y='Count',
                       color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig_years, use_container_width=True)

st.markdown("---")

# Row 3: Duration Distribution
st.subheader("5. Runtime & Season Distributions")
movies_df = filtered_df[filtered_df['type'] == 'Movie']
tv_df = filtered_df[filtered_df['type'] == 'TV Show']

d_col1, d_col2 = st.columns(2)

with d_col1:
    if not movies_df.empty:
        fig_movie_dur = px.histogram(movies_df, x='duration_val', nbins=30,
                                     title="Movie Duration Distribution (Minutes)",
                                     color_discrete_sequence=['#E50914'])
        st.plotly_chart(fig_movie_dur, use_container_width=True)
    else:
        st.info("No Movie data available in selected range.")

with d_col2:
    if not tv_df.empty:
        tv_seasons = tv_df['duration_val'].value_counts().reset_index()
        tv_seasons.columns = ['Seasons', 'Count']
        fig_tv_dur = px.bar(tv_seasons, x='Seasons', y='Count',
                            title="TV Show Season Distribution",
                            color_discrete_sequence=['#221F1F'])
        st.plotly_chart(fig_tv_dur, use_container_width=True)
    else:
        st.info("No TV Show data available in selected range.")