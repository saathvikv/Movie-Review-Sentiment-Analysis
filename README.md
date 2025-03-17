# Movie-Review-Sentiment-Analysis


Sentiment Analysis with LSTM

Overview

This project implements a sentiment analysis model using a Long Short-Term Memory (LSTM) neural network. The model is trained on a dataset of movie/IMDB reviews, classifying them as either positive or negative. The dataset undergoes preprocessing, tokenization, and vectorization before being fed into the LSTM model for training.

Installation

To run this project, install the required dependencies using:

pip install numpy pandas torch nltk sklearn

The LSTM model architecture consists of:

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
