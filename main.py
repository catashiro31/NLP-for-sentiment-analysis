import streamlit as st
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
import datetime
import pandas as pd
import plotly.express as px
# Load the diary text
diary_path = Path('diary')
list_of_files = list(diary_path.glob('*.txt'))
dates = []
pos = []
neg = []

for file in list_of_files:
    with open(file, 'r') as f:
        content = f.read()
    
    dates.append(datetime.datetime.strptime(file.stem, '%Y-%m-%d').date())
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(content)

    pos.append(scores['pos'])
    neg.append(scores['neg'])

# Create a DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Positive': pos,
    'Negative': neg
})
# Streamlit app to visualize sentiment analysis
st.title('Analyzing Sentiment in Diary')
fig1 = px.line(df, x='Date', y='Positive', title='Positive Sentiment Over Time')
st.plotly_chart(fig1)
fig2 = px.line(df, x='Date', y='Negative', title='Negative Sentiment Over Time')
st.plotly_chart(fig2)