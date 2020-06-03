import sys
import pandas as pd
import numpy as np
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize

# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

#https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
# Standard function to use with the NLTK pos tagger: 
# we need to convert pos_tag codes to wordnet codes
def get_wordnet_pos(pos_tag):
  if pos_tag.startswith('J'):
    return wordnet.ADJ
  elif pos_tag.startswith('V'):
    return wordnet.VERB
  elif pos_tag.startswith('N'):
    return wordnet.NOUN
  elif pos_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN

# with help from
# https://towardsdatascience.com/detecting-bad-customer-reviews-with-nlp-d8b36134dc7e
# for ideas / basic form
def prepare_review(review):
  global word_counts
  global review_counter
  review_counter += 1
  print(review_counter)
  review = review.lower()
  review = word_tokenize(review)
  s_a = SentimentAnalyzer()
  #this marks words between a negation phrase (ex, 'not'') and the next punctuation with a 'NEG' tag
  #review = s_a.all_words([mark_negation(review)]) 
  review = [word for word in review if word] # remove empty words
  # For some reason, word lemmatizers have a difficult time handling the word 'hate' and think
  # that it stems from the word 'hat'. So, you have to manually 
  review = [word if word[:4] != 'hate' or word[:5] != 'hatin' else 'hate' for word in review]
  pos_tags = pos_tag(review)
  review = [WordNetLemmatizer().lemmatize(word[0], get_wordnet_pos(word[1])) for word in pos_tags]
  review = s_a.all_words([mark_negation(review)])
  review  = [word for word in review if word not in stopwords.words('english')]
  review = [re.sub(r'[^(a-zA-Z\s]','',word) for word in review]
  review = [word for word in review if len(word) > 1]
  for word in review:
    if word in word_counts:
      word_counts[word] += 1
    else:
      word_counts[word] = 1
  review = ' '.join(review)
  # https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
  return review

def create_vocab(word_counts, min_freq=None, top_perc=None, top_num=None):
  if min_freq:
    return set([word[0] for word in word_counts.items() if word[1] > min_freq])
  elif top_perc:
    words = sorted(list(word_counts.items()), key=lambda x: x[1], reverse=True)
    p = ceil(top * len(word))
    return set([word[0] for word in words][:p])
  elif top_num:
    words = sorted(list(word_counts.items()), key=lambda x: x[1], reverse=True)
    return set([word[0] for word in words][:top_num])

def filter_review(review, vocab):
  review = review.split()
  review = [word for word in review if word in vocab]
  review = ' '.join(review)
  return review

word_counts = dict()
review_counter = 0

# Load the proper data
num_reviews = sys.argv[1]
FP_BASE = 'pickles/reviews_'
df_path = FP_BASE + num_reviews + '.p'
processed_save = FP_BASE + num_reviews + '_processed.p'
vocab_save = FP_BASE + num_reviews + '_vocab.p'
if num_reviews[-4:] == 'test':
  vocab_path = sys.argv[2]

df = pickle.load(open(df_path, 'rb'))
reviews = df[['text', 'stars']]
reviews['text'] = reviews['text'].apply(lambda x: prepare_review(x))
if vocab_save:  
  vocab = create_vocab(word_counts, min_freq=50) # this is something to experiment on
else:
  vocab = pickle.load(open(vocab_path, 'rb'))
reviews['text'] = reviews['text'].apply(lambda x: filter_review(x, vocab))
pickle.dump(reviews, open(processed_save, 'wb'))
if vocab_save != 'test':  
  pickle.dump(vocab, open(vocab_save, 'wb'))
print(reviews.head)