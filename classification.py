import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read data from CSV file
data = pd.read_csv("tweets-combine3topics.csv")

# Remove any rows with missing data
data = data.dropna()

# Define the feature and target columns
X = data['Preprocessed_Tweet_Text']
y = data['classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the accuracy, metrics, report of the classifier
print("============================================================")
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("============================================================")
print("metrics  :\n")
print(confusion_matrix(y_test, y_pred))
print("============================================================")
print("report   :\n")
print(classification_report(y_test, y_pred))
print("============================================================")
