import pandas as pd
import numpy as np
import nltk
import tnkeeh as tn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from joblib import dump, load

# Read in dataset
df = pd.read_csv('data.csv')

# Get stopwords from nltk and remove them from the corpus
stop = set(nltk.corpus.stopwords.words("arabic"))
df["cleaned"] = df["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# Clean the dataset using tnkeeh
df = tn.clean_data_frame(df, "cleaned", segment = False, remove_special_chars = True, 
        remove_english = True, normalize = True, remove_diacritics = True,
        excluded_chars = [], remove_tatweel = True, remove_html_elements = True,
        remove_links = True, remove_twitter_meta = True, remove_long_words = True,
        remove_repeated_chars = True)


#  Split the training and testing sets
X = df["cleaned"]
Y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

# Create tf-idf vectorization of the training data
feature_extractor = TfidfVectorizer()
X_train = feature_extractor.fit_transform(X_train)

# Fit a SVM model on the tf-idf vectors and the labels
clf = SVC()
clf.fit(X_train, y_train)

# Test the model on never seen before data
test = feature_extractor.transform(X_test)
predicted = clf.predict(test)

# Show the performance of the model
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

# Save the SVM model as well as the tf-idf transformer for future predictions
dump(clf, "SVM_text_Classification.joblib")
dump(feature_extractor, "SVM_vectorizer.joblib")