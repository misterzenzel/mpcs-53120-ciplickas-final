import sys
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

FP_BASE = 'pickles/reviews_'

num_reviews = sys.argv[1]
num_test_reviews = sys.argv[2]

train_path = FP_BASE + num_reviews + '_processed.p'
test_path = FP_BASE + num_test_reviews + '_test_processed.p'
vocab_path = FP_BASE + num_reviews + '_vocab.p'

vectors_out_path = FP_BASE + num_reviews + '_count_bow.csv'

train_df = pickle.load(open(train_path, 'rb'))

test_df = pickle.load(open(test_path, 'rb'))

vocab = pickle.load(open(vocab_path, 'rb'))

try:
	vocab.remove('')
except:
	pass

# Something to experiment on?
# this just marks word or not word
def get_bag_of_words(review):
	global vocab
	review = review.split()
	features = {word: 0 for word in vocab}
	for word in review:
		if word in vocab:	
			features[word] = 1
	return features

train_reviews = [get_bag_of_words(review) for review in train_df['text']]
test_reviews = [get_bag_of_words(review) for review in test_df['text']]

train_reviews = pd.DataFrame(train_reviews)
train_reviews['stars_CATEGORY'] = train_df['stars']
test_reviews = pd.DataFrame(test_reviews)
test_reviews['stars_CATEGORY'] = test_df['stars']
train_reviews.to_csv(vectors_out_path)
test_df_fp = FP_BASE + num_reviews + '_train_' + num_test_reviews + '_test_count_bow.csv'
test_reviews.to_csv(test_df_fp)

train_vectors = train_reviews.drop('stars_CATEGORY', axis=1)
train_stars = train_reviews['stars_CATEGORY']

# Original params: {'alpha': [.001, .01, .1, 1, 5, 10, 25, 50]}
# Found that alpha should be between .01 and 1, so refined the experiment
params = {'alpha': np.arange(.00, 1.05, .05)[1:]}

mnb = MultinomialNB()
# Using precision bc more interesting metric
# Lots of small differences between 4 v 5, 3 v 4, etc. 
# So, see how well the model is able to capture those nuances
clf = GridSearchCV(mnb, params, scoring='precision_macro', n_jobs=-1, cv=5, verbose=2)
clf.fit(train_vectors, train_stars)

print(clf.best_params_)

model_save = FP_BASE + num_reviews + '_count_bow_cv.p'

pickle.dump(clf, open(model_save, 'wb'))

test_vectors = test_reviews.drop('stars_CATEGORY', axis=1)
test_stars = test_reviews['stars_CATEGORY']

preds = clf.predict(test_vectors)
preds_proba = clf.predict_proba(test_vectors)

preds_out = FP_BASE + num_reviews + '_train_' + num_test_reviews + '_test_bin_class.p'
pickle.dump((preds, test_stars), open(preds_out, 'wb'))
preds_proba_out = FP_BASE + num_reviews + '_train_' + num_test_reviews + '_test_bin_proba.p'
pickle.dump((preds_proba, test_stars), open(preds_proba_out, 'wb'))

test_report = classification_report(test_stars, preds, output_dict=True)
train_report = classification_report(train_stars, clf.predict(train_vectors), output_dict=True)

report_fp = 'reports/bin_report_' + num_reviews + '_train_' + num_test_reviews + '_test.json'
open(report_fp, 'w').close()
# These do not save in proper JSON
# You need to manually change single quotes to double quotes in order to properly parse
print(train_report, file=open(report_fp, 'a'))
print(test_report, file=open(report_fp, 'a'))