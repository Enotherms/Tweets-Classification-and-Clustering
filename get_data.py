import tweepy
import csv
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Authenticate to Twitter
auth = tweepy.OAuthHandler("consumer_key", "consumer_secret")
auth.set_access_token("access_token", "access_secret")
# change consumer_key, consumer_secret, access_token, access_secret according to yours


# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True)

# Define search term and number of tweets
search_term = ["pemilu","banjir"]
number_of_tweets = 100

# Create stemmer and stopword remover
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Search tweets
tweets = tweepy.Cursor(api.search_tweets, search_term).items(number_of_tweets)

# Create a CSV file
with open('tweets.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Username", "Preprocessed_Tweet_Text", "Timestamp", "classification"])
    
    # Iterate over tweets
    for tweet in tweets:
        # Remove links from the text
        preprocessed_text = re.sub(r'http\S+', '', tweet.text)
        # Preprocess tweet text using Sastrawi
        preprocessed_text = stemmer.stem(stopword_remover.remove(preprocessed_text))
        csv_writer.writerow([tweet.user.screen_name, preprocessed_text, tweet.created_at, search_term])

print("CSV file saved!")
