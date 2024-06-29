import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_dir = os.path.join(current_dir, 'MessageClassifier')
sys.path.append(classifier_dir)

from classifier.utils import load_data, generate_Embedding
from classifier.model import build_model

# Load and prepare data
X, Y = load_data('C:\\Users\\1mahe\\Desktop\\MessageClassifier\\data\\data.txt') #Replace with your own directories beehi?
word2int = {'ham': 0, 'spam': 1}
int2word = {0: 'ham', 1: 'spam'}
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X_digit = tokenizer.texts_to_sequences(X)
X_digit = np.array(X_digit, dtype=object)
Y = np.array([word2int[label] for label in Y])

sen_length = 100
X_digit = pad_sequences(X_digit, maxlen=sen_length)
Y = to_categorical(Y, num_classes=2)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_digit, Y, test_size=0.25, random_state=7)

# Generate embedding matrix
embedding_matrix = generate_Embedding(tokenizer, 100)

# Build and train model
model = build_model(embedding_matrix, 128, sen_length)
history = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), batch_size=64, epochs=10, verbose=1)

# Save the model
model.save('C:\\Users\\1mahe\\Desktop\\MessageClassifier\\msg_classifier_model.keras')

# Plot training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, 11)

plt.plot(epochs, acc, 'r', label="Training Acc")
plt.plot(epochs, val_acc, 'b', label="Validation Acc")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
results = model.evaluate(Xtest, Ytest)
print(f'Accuracy: {results[1]*100:.2f}%')

def prediction(model, sms):
    seq = tokenizer.texts_to_sequences([sms])
    seq = pad_sequences(seq, maxlen=sen_length)
    predict = model.predict(seq)[0]
    return int2word[np.argmax(predict)]

model.save('msg_classifier_model.h5')