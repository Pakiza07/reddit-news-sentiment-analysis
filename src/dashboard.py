import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Reddit/News Sentiment Dashboard", layout="wide")

st.title("Reddit & News Sentiment Analysis Dashboard")

# --------------------------
# Load the predictions CSV
# --------------------------
df = pd.read_csv("outputs/sentiment_results.csv")

# --------------------------
# Sidebar filters
# --------------------------
st.sidebar.header("Filters")

# Filter by sentiment
sentiments = df["predicted_sentiment"].unique().tolist()
selected_sentiments = st.sidebar.multiselect("Select Sentiment", sentiments, default=sentiments)
filtered_df = df[df["predicted_sentiment"].isin(selected_sentiments)]

# --------------------------
# Sentiment distribution chart
# --------------------------
st.subheader("Sentiment Distribution")
fig_dist = px.histogram(
    filtered_df,
    x="predicted_sentiment",
    color="predicted_sentiment",
    title="Sentiment Counts",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_dist, use_container_width=True)

# --------------------------
# Sample posts table
# --------------------------
st.subheader("Sample Headlines")
st.dataframe(filtered_df[["title", "title_clean", "predicted_sentiment"]].head(10))

# --------------------------
# Dashboard footer
# --------------------------
st.write("Dashboard ready! Use the sidebar to filter by sentiment and explore the data.")