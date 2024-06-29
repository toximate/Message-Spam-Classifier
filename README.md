# Message Classifier Model

This repository implements a message classifier using a Long Short-Term Memory (LSTM) neural network for distinguishing between spam and non-spam messages.

## Overview

The message classifier utilizes natural language processing techniques and LSTM networks to classify messages into two categories: "ham" (non-spam) and "spam". It is built with TensorFlow/Keras and trained on GloVe word embeddings for text representation.

## Features

- **Text Preprocessing:** Tokenization, padding sequences.
- **Model Architecture:** LSTM layers with dropout for sequence classification.
- **Data Handling:** Loading and preprocessing of text data.
- **Evaluation:** Training and validation accuracy/loss monitoring.
- **Prediction:** Real-time prediction of message classification.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/toximate/Message-Spam-Classifier.git
   ```
   ```bash
   cd Message-Spam-Classifier
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the GloVe word embeddings (I also gave you the link in the IMPORTANT.txt if you want to do it manually):

   ```bash
   wget http://nlp.stanford.edu/data/glove.6B.zip
   ```
   ```bash
   unzip glove.6B.zip -d MessageClassifier/glove.6B
   ```

