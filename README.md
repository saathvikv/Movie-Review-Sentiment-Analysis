# Movie-Review-Sentiment-Analysis


Sentiment Analysis with LSTM

Overview

This project implements a sentiment analysis model using a Long Short-Term Memory (LSTM) neural network. The model is trained on a dataset of text reviews, classifying them as either positive or negative. The dataset undergoes preprocessing, tokenization, and vectorization before being fed into the LSTM model for training.

Features

Text Preprocessing: Cleans input text by removing special characters, punctuation, and unnecessary spaces.

Tokenization & Vectorization: Converts words into numerical representations using a custom vocabulary.

Deep Learning Model: Utilizes an LSTM-based neural network for sentiment classification.

Training & Optimization: Implements dropout for regularization and uses cross-entropy loss for training.

Installation

To run this project, install the required dependencies using:

pip install numpy pandas torch nltk sklearn

Dataset

The dataset consists of text reviews labeled as positive or negative. It is preprocessed by:

Converting text to lowercase

Removing punctuation and special characters

Tokenizing sentences into words

Creating a word-to-index dictionary

Model Architecture

The LSTM model consists of:

An embedding layer for word vector representation

An LSTM layer to capture sequential dependencies

A dropout layer to prevent overfitting

A fully connected layer with a sigmoid activation function

Training

The model is trained using the Adam optimizer with a learning rate of 0.005. The loss function used is binary cross-entropy, and training runs for multiple epochs.

Usage

To train the model, run:

python train.py

To test the model on new data, run:

python test.py --input "Your text here"

Improvements & Future Work

Improve preprocessing: Enhance tokenization and handle out-of-vocabulary words more effectively.

Tune hyperparameters: Experiment with different dropout rates, batch sizes, and learning rates.

Implement evaluation metrics: Use precision, recall, and F1-score for better performance evaluation.

Deploy as an API: Convert the trained model into a REST API for easy integration.
