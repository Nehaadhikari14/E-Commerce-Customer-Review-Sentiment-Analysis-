import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

data = {
    'Review': [
        'I love this product! It is amazing.',
        'Worst purchase I have made, very bad quality.',
        'Not bad, but could be better.',
        'Absolutely fantastic! I will buy again.',
        'Very disappointed, not as expected.',
        'I am happy with this purchase.',
        'Terrible experience, will never buy again.',
        'The product is okay, not great but not bad.',
        'Five stars! Exceeded my expectations!',
        'Not worth the price, really poor quality.'
    ]
}

df = pd.DataFrame(data)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(review):
    sentiment = analyzer.polarity_scores(review)
    return sentiment['compound']

df['Sentiment_Score'] = df['Review'].apply(get_sentiment_score)

def classify_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Sentiment_Score'].apply(classify_sentiment)
print(df)

sentiment_counts = df['Sentiment'].value_counts()

sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('E-commerce Customer Reviews Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


