import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentences):
    return nltk.word_tokenize(sentences)

def stemmingg(words):
    return stemmer.stem(words.lower())

def words_container(tokenized_sentence, all_words):
    sentence = [stemmingg(words) for words in tokenized_sentence]
    bags = np.zeros(len(all_words), dtype=np.float32)
    for index, wrd in enumerate(all_words):
        if wrd in sentence:
            bags[index] = 1.0

    return bags




