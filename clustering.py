import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Read data from CSV file
data = pd.read_csv("tweets-combine3topics.csv")

# Remove blank data
data = data[data['Preprocessed_Tweet_Text'].notna()]

# change text to vector with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Preprocessed_Tweet_Text'])

# do k means clustering, 3 cluster
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
prediction=kmeans.predict(X)
prediction

# Grouping each tweet into the appropriate cluster
clusters = kmeans.predict(X)
data['cluster'] = clusters

#print clustering result
print(data.groupby('cluster').size())
print("\n")
for i in range(0,prediction.size):
    print(prediction[i], data['Preprocessed_Tweet_Text'][i])
