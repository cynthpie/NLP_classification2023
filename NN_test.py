from NN import FFNN
from NN_preprocessing import get_tokenized_corpus, get_model_inputs
import pickle
import torch
from torchmetrics.classification import F1Score

# load data
with open("official_dev_set.pk1", "rb") as target:
    dev_data = pickle.load(target)
dev_data = dev_data[~dev_data["text"].isnull()] # remove rows with null data

# load word to index obtained from training data for preprocessing
with open("word2idx_train.pk", "rb") as target:
    word2idx = pickle.load(target)

# preprocess dev data
dev_label = dev_data["label"].to_numpy()
tokenized_corpus = get_tokenized_corpus(dev_data['text'])
dev_sent_tensor, dev_label_tensor = get_model_inputs(tokenized_corpus, word2idx, dev_label)

# load model and evalute on dev set
model = torch.load("NN.pt")
model.eval()
predictions_dev = model(dev_sent_tensor).squeeze(1)
pred_dev_labels = torch.round(torch.sigmoid(predictions_dev))
f1 = F1Score(task="binary", num_classes=1)
dev_f1 = f1(pred_dev_labels, dev_label_tensor)
dev_acc = (pred_dev_labels == dev_label_tensor).float().mean()
print(f'Dev Acc: {dev_acc*100:6.2f}% | Dev F1: {dev_f1*100:6.2f}% |')

# extract and save misclassified samples
misclassified_idx = (pred_dev_labels.detach().numpy()!=dev_label)
misclassified_dev = dev_data[misclassified_idx]
with open('NN_wrong_predictions.pk', 'wb') as target: # a panda dataframe
    pickle.dump(misclassified_dev, target)