import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tqdm

def load_data(path):
    text, label = [], []
    with open(path) as file:
        for line in file:
            tokens = line.split()
            label.append(tokens[0])
            text.append(' '.join(tokens[1:]))
    return text, label

def generate_Embedding(tokenizer, dim):
    embedding = {}
    with open(f"C:\\Users\\1mahe\\Desktop\\MessageClassifier\\glove.6B\\glove.6B.{dim}d.txt", encoding='utf8') as file:
        for line in tqdm.tqdm(file, "Reading the GloVe file"):
            tokens = line.split()
            word = tokens[0]
            vector = np.array(tokens[1:], dtype='float32')
            embedding[word] = vector

    wordI = tokenizer.word_index
    embedding_matrix = np.zeros((len(wordI) + 1, dim))

    for word, indx in wordI.items():
        vector = embedding.get(word)
        if vector is not None:
            embedding_matrix[indx] = vector
            
    return embedding_matrix
