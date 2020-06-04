# Sentiment Analysis of Yelp Reviews

## Directory Structure

First, run ./setup.sh to create the two repos that generated files will save to. 

Next, download review.json from this link:
https://www.kaggle.com/yelp-dataset/yelp-dataset

This project uses the Yelp Academic Dataset, a collection of over 5 million Yelp user reviews. Each
entry has the text of the review, its star value, and other metadata (review / user IDs, helpfulness, etc.)
The file is quite large (c. 5gb); once it is downloaded, move it to this directory. I can also provide my pickled
dataset files by request! The pickles/ and reports/ directories have some sample data in them. 

Then, we have the following structure:
1. Top level
	- make_dataset.py - c. 40 lines
	- make_dataset_test.py - c. 40 lines
	- yelp_data_processng.py - c. 110 lines
	- tfidf_complete.py - c. 80 lines
	- bin_complete.py - c. 80 lines 
	- count_complete.py - c. 80 lines
	- cnn_10000.ipynb - c. 130 lines
	- cnn_20000.ipynb - c. 130 lines
	- cnn_50000.ipynb - c. 130 lines
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
	- This requires you to input the number of reviews
3. Make the training dataset:
	- I would recommend running the following:
	- python3 make_dataset_training 100000 10000 reviews.json
	- This will take reviews 100,000 - 110,000 frorm the complete dataset
3. Create the Naive Bayes Classifier models
	- Each of the files tfidf_complete.py, bin_complete.py, and count_complete.py takes
	the same inputs:
		- number of training reviews
		- number of testing reviews
	- The datasets it creates will be stored in the pickles/ directory, and the report it generates
	will be in the reports/ directory
4. Create the Convolutional Neural Network models
	- I would recommend running these on a Google Colab GPU
	- For each of cnn_<num_training_reviews>.ipynb, you need to upload the training dataset with the 
	proper number of reviews to the notebook. Then, you can run each cell, the model will train itself,
	and it will produce the predictions and reports. Make sure to download them!
	- See cnn_500000.ipynb for commented code. I took lots of inspiration from the articles found
	here: https://machinelearningmastery.com/best-practices-document-classification-deep-learning/
	and here: https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

Once you've run the above steps, see the reports/ and pickles/ directories for the outputs!
		 