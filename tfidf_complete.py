import sys
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

FP_BASE = 'pickles/reviews_'

# Load the data based off of the number of train reviews
# And the number of test reviews
num_reviews = sys.argv[1]
num_test_reviews = sys.argv[2]

train_path = FP_BASE + num_reviews + '_processed.p'
test_path = FP_BASE + num_test_reviews + '_test_processed.p'

vectors_out_path = FP_BASE + num_reviews + '_tfidf_bow.csv'

train_df = pickle.load(open(train_path, 'rb'))

test_df = pickle.load(open(test_path, 'rb'))

# Use the Sklearn vectorizer
vectorizer = TfidfVectorizer().fit(train_df['text'])
train_vectors = vectorizer.transform(train_df['text']).todense().tolist()
test_vectors = vectorizer.transform(test_df['text']).todense().tolist()
feats = vectorizer.get_feature_names()

train_df_vectors = pd.DataFrame(train_vectors, columns=feats)
train_df_vectors['stars_CATEGORY'] = train_df['stars']

train_df_vectors.to_csv(vectors_out_path)

test_df_vectors = pd.DataFrame(test_vectors, columns=feats)
test_df_vectors['stars_CATEGORY'] = test_df['stars']
test_df_fp = FP_BASE + num_reviews + '_train_' + num_test_reviews + '_test_tfidf.csv'
test_df_vectors.to_csv(test_df_fp)

train_vectors = train_df_vectors.drop('stars_CATEGORY', axis=1)
train_stars = train_df_vectors['stars_CATEGORY']

# Original params: {'alpha': [.001, .01, .1, 1, 5, 10, 25, 50]}
# Found that alpha should be between .01 and 1, so refined the experiment
params = {'alpha': np.arange(.00, 1.05, .05)[1:]}

mnb = MultinomialNB()
clf = GridSearchCV(mnb, params, scoring='precision_macro', n_jobs=-1, cv=5, verbose=2)
clf.fit(train_vectors, train_stars)

print(clf.best_params_)

model_save = FP_BASE + num_reviews + '_tfidf_bow_cv.p'

pickle.dump(clf, open(model_save, 'wb'))

X_test = test_df_vectors.drop('stars_CATEGORY', axis=1)
y = test_df_vectors['stars_CATEGORY']

preds = clf.predict(X_test)
preds_proba = clf.predict_proba(X_test)

preds_out = FP_BASE + num_reviews + '_train_' + num_test_reviews + '_test_tfidf_class.p'
pickle.dump((preds, y), open(preds_out, 'wb'))
preds_proba_out = FP_BASE + num_reviews + '_train_' + num_test_reviews + '_test_tfidf_proba.p'
pickle.dump((preds_proba, y), open(preds_proba_out, 'wb'))


test_report = classification_report(y, preds, output_dict=True)
train_report = classification_report(train_stars, clf.predict(train_vectors), output_dict=True)
report_fp = 'reports/tfidf_report_' + num_reviews + '_train_' + num_test_reviews + '_test.json'
# These do not save in proper JSON
# You need to manually change single quotes to double quotes in order to properly parse
open(report_fp, 'w').close()
print(train_report, file=open(report_fp, 'a'))
print(test_report, file=open(report_fp, 'a'))