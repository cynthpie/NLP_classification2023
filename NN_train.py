from NN import FFNN
from NN_preprocessing import get_tokenized_corpus, get_word2idx, get_model_inputs
from torchmetrics.classification import F1Score
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def fix_seed(seed=234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def accuracy(output, target):
    output = torch.round(torch.sigmoid(output))
    correct = (output==target).float()
    acc = correct.mean()
    return acc

def train(model, nb_epoch, learning_rate, train_sent_tensor, train_label_tensor, valid_sent_tensor, valid_label_tensor):
    model = model
    optimizer  = optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    f1 = F1Score(task="binary", num_classes=1)
    for epoch in range(1, nb_epoch+1):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_sent_tensor).squeeze(1)
        loss = loss_fn(predictions, train_label_tensor)
        train_loss = loss.item()
        train_acc = accuracy(predictions, train_label_tensor)
        pred_train_label = torch.round(torch.sigmoid(predictions))
        train_f1 = f1(pred_train_label, train_label_tensor)
        loss.backward()
        optimizer.step()

        # validation data
        model.eval()
        with torch.no_grad():
            predictions_valid = model(valid_sent_tensor).squeeze(1) # numeric output, NOT binary label
            valid_loss = loss_fn(predictions_valid, valid_label_tensor).item()
            valid_acc = accuracy(predictions_valid, valid_label_tensor)
            pred_val_label = torch.round(torch.sigmoid(predictions_valid))
            valid_f1 = f1(pred_val_label, valid_label_tensor)
        print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:6.2f}% | Train F1: {train_f1*100:6.2f}% | \
                Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:6.2f}% | Val. F1: {valid_f1*100:6.2f}% |')
    torch.save(model, 'NN.pt')  # uncommend to save new model
    return model

if __name__ == "__main__":
    # load train data
    with open("train_set.pk1", "rb") as target:
        train_data = pickle.load(target)

    # split train data into train and valid set, and preprocess both.

    train_data = train_data.sample(frac=1, random_state=1) #shuffle
    val_data = train_data.iloc[0:int(len(train_data)*0.2)]
    train_data = train_data.iloc[int(len(train_data)*0.2):]

    train_labels = train_data['label'].to_numpy() #convert pandas series to numpy
    valid_labels = val_data['label'].to_numpy()
    train_tokenized_corpus = get_tokenized_corpus(train_data['text']) 
    valid_tokenized_corpus = get_tokenized_corpus(val_data['text']) 

    # create word to index with training data and save it for test data preprocessing
    word2idx = get_word2idx(train_tokenized_corpus) # only use words from training data not validation data
    with open("word2idx_train.pk", "wb") as target:
        pickle.dump(word2idx, target)
    train_sent_tensor, train_label_tensor = get_model_inputs(train_tokenized_corpus, word2idx, train_labels)
    valid_sent_tensor, valid_label_tensor = get_model_inputs(valid_tokenized_corpus, word2idx, valid_labels)

    # create model and train
    vocab_size = len(word2idx)
    model = FFNN(embedding_dim=50, hidden_dim=30, vocab_size=vocab_size, num_classes=1)
    model = train(model=model, nb_epoch=100, learning_rate=0.1, train_sent_tensor=train_sent_tensor, 
                  train_label_tensor=train_label_tensor, valid_sent_tensor=valid_sent_tensor, valid_label_tensor=valid_label_tensor)
    
    # evaluation on test set
    with open("official_dev_set.pk1", "rb") as target:
        dev_data = pickle.load(target)

    # remove rows with null data
    dev_data = dev_data[~dev_data["text"].isnull()]
    dev_labels = dev_data['label'].to_numpy()
    dev_tokenized_corpus = get_tokenized_corpus(dev_data['text']) 
    dev_sent_tensor, dev_label_tensor = get_model_inputs(dev_tokenized_corpus, word2idx, dev_labels)
    f1 = F1Score(task="binary", num_classes=1)
    with torch.no_grad():
        predictions_dev = model(dev_sent_tensor).squeeze(1) # numeric output, NOT binary label
        dev_acc = accuracy(predictions_dev, dev_label_tensor)
        pred_dev_label = torch.round(torch.sigmoid(predictions_dev))
        dev_f1 = f1(pred_dev_label, dev_label_tensor)
    print("dev_acc", dev_acc.item(), "f1:", dev_f1.item())