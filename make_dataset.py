import sys
import pandas as pd
import numpy as np
import pickle

# https://towardsdatascience.com/converting-yelp-dataset-to-csv-using-pandas-2a4c8f03bd88
start = int(sys.argv[1])
num_to_keep = int(sys.argv[2])
data_path = sys.argv[3]

df_save = 'pickles/reviews_' + str(num_to_keep) + '.p'

chunk_size = 1000

review = pd.read_json(data_path, lines=True, 
          dtype={'review_id':str,'user_id':str,
                   'business_id':str,'stars':int,
                   'date':str,'text':str,'useful':int,
                   'funny':int,'cool':int},
                    chunksize=chunk_size)

chunk_list = []
processed = 0
curr_idx = 0
for chunk_review in review:
  if curr_idx < start:
    pass
  else:
    chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
    chunk_list.append(chunk_review)
    processed += len(chunk_review)
    if processed > num_to_keep:
      break
  curr_idx += chunk_size

df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)

pickle.dump(df, open(df_save, 'wb'))