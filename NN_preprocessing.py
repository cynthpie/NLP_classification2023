import torch
import pickle
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# preprocessing of documents for train set (remove stop words, and puncutation and make lowercase)
def get_tokenized_corpus(corpus):
    # args: 
    #   corpus[list]: list of string
    # return:
    #   tokenized_corpus[list]: list of list containing word tokens
    tokenized_corpus = []
    for sent in  corpus: 
        sent = sent.lower()
        words = word_tokenize(sent) # words[list]: list of tokenized word
        words = [word for word in words if word not in stopwords.words("english") and word.isalpha()] # filter stopwords & non-eng word
        tokenized_corpus.append(words)
    return tokenized_corpus

def get_word2idx(tokenized_corpus):
    # args: 
    #   tokenized_corpus[list]: list of list containing word tokens
    # return:
    #   word2idx[dict]: dict{word:index}
    vocab = []
    for sent in tokenized_corpus:
        for word in sent:
            if word not in vocab:
                vocab.append(word)
    word2idx = {w:idx+1 for (idx, w) in enumerate(vocab)}
    word2idx['<pad>'] = 0
    return word2idx

def get_model_inputs(tokenized_corpus, word2idx, labels):
    # get fixed length index representation of the tokenized sentence
    # args:
    #   labels[np.array]: N x 1 numpy array of sentence label

    # we index our sentences
    vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in tokenized_corpus]

    # Sentence lengths
    sent_lengths = [len(sent) for sent in vectorized_sents]

    # Get maximum length
    max_len = max(sent_lengths)
  
    # we create a tensor of a fixed size filled with zeroes for padding
    sent_tensor = torch.zeros((len(vectorized_sents), max_len)).long()

    # we fill it with our vectorized sentences 
    for idx, (sent, sentlen) in enumerate(zip(vectorized_sents, sent_lengths)):
        sent_tensor[idx, :sentlen] = torch.LongTensor(sent)

    # Label tensor
    label_tensor = torch.FloatTensor(labels)
  
    return sent_tensor, label_tensor

# main
if __name__=="__main__":
    # load data
    with open("train_set.pk1", "rb") as target:
        train_data = pickle.load(target)
    with open("official_dev_set.pk1", "rb") as target:
        dev_data = pickle.load(target)
    # remove rows with null data
    dev_data = dev_data[~dev_data["text"].isnull()]

    train_labels = train_data['label'].to_numpy() #convert pandas series to numpy
    tokenized_corpus = get_tokenized_corpus(train_data['text'])
    word2idx = get_word2idx(tokenized_corpus)
    train_sent_tensor, train_label_tensor = get_model_inputs(tokenized_corpus, word2idx, train_labels)
    print(f'Vocabulary size: {len(word2idx)}')
    print('Training set tensor:')
    print(train_sent_tensor)