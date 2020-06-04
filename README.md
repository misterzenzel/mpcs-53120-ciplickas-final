# Sentiment Analysis of Yelp Reviews

## Directory Structure

First, run ./setup.sh to create the two repos that generated files will save to. 

Next, download review.json from this link:
https://www.kaggle.com/yelp-dataset/yelp-dataset

This project uses the Yelp Academic Dataset, a collection of over 5 million Yelp user reviews. Each
entry has the text of the review, its star value, and other metadata (review / user IDs, helpfulness, etc.)
The file is quite large (c. 5gb); once it is downloaded, move it to this directory

Then, we have the following structure:
1. Top level
	- make_dataset.py
	- make_dataset_test.py
	- yelp_data_processng.py
	- tfidf_complete.py
	- bin_complete.py
	- count_complete.py
	- cnn_10000.ipynb
	- cnn_20000.ipynb
	- cnn_50000.ipynb
	- pickles/
		* This is where any data outputs will be saved
	- reports
		* This is where reports from <x>\_complete.py files will be saved

In order to run the experiments, execute the following:
1. Make the training dataset using make\_dataset.py
	- This takes three inputs: 
		* What index in the data you want to start pulling from. I'd recommend 0 for non-test data
		* How many records to save
		* The path to the data. In this case, it is just 'review.json'
	- example call: python3 make_dataset.py 0 10000 review.json
		* This grabs the first 10,000 reviews from the dataset
2. Process the data using yelp_data_processing.py
		 