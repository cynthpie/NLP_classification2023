import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

nltk.download('punkt')
nltk.download('stopwords')

with open("train_set.pk1", "rb") as target:
    train_data = pickle.load(target)

with open("official_dev_set.pk1", "rb") as target:
    dev_data = pickle.load(target)

# remove rows with null data
dev_data = dev_data[~dev_data["text"].isnull()]

# preprocessing of documents for train set (remove stop words, and puncutation and make lowercase)
documents_train = []
for document in train_data["text"]:
    document = document.lower()
    words = word_tokenize(document)
    words = [word for word in words if word not in stopwords.words("english") and word.isalpha()]
    document = " ".join(words)
    documents_train.append(document)

# preprocessing of documents for dev set (remove stop words, and puncutation and make lowercase)
documents_dev = []

for document in dev_data["text"]:
    document = document.lower()
    words = word_tokenize(document)
    words = [word for word in words if word not in stopwords.words("english") and word.isalpha()]
    document = " ".join(words)
    documents_dev.append(document)

# put into BOW model for both train and dev sets
combined_sets = documents_train + documents_dev
vectorizer = CountVectorizer().fit(documents_train)

bow_train = vectorizer.transform(documents_train)
X_train = bow_train.toarray()
y_train = train_data["label"]

bow_dev = vectorizer.transform(documents_dev)
X_dev = bow_dev.toarray()
y_dev = dev_data["label"]

# using logistic regression for binary classification
clf = LogisticRegression(random_state=1).fit(X_train,y_train)
dev_score = clf.score(X_dev,y_dev)
dev_predictions = clf.predict(X_dev)

print(f"dev score accuracy: {dev_score}")

#f1 score calculation:
dev_f1score = f1_score(y_dev, dev_predictions)

print(f'dev f1 score: {dev_f1score}')

# dev score accuracy: 0.9025322503583373
# dev f1 score: 0.2714285714285714

# extracting out the wrongly classified examples
filter = dev_predictions!=y_dev
filter = filter.to_numpy()

wrong_prd = dev_data[dev_data["label"]!=dev_predictions]
print(dev_predictions)
dev_data["log_reg_prd"] = dev_predictions
documents_dev = np.array(documents_dev)
wrong_predictions = documents_dev[filter]


with open("dev_pred_lr_.pk","wb") as file:
        pickle.dump(dev_data, file)







