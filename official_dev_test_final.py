import pickle
import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw

# convert data to the form used by the pretrained model and Trainer
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# define compute_metrics function
def compute_metrics(input):
    y_pred = np.argmax(input.predictions, axis=1)
    y_true = input.label_ids
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)

    return {'accuracy': accuracy, 'f1 score':f1score}

# gpu set up
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# load train data
with open("train_set.pk1", "rb") as target:
    train_data = pickle.load(target)

# load official dev data
with open("official_dev_set.pk1", "rb") as target:
    offdev_data = pickle.load(target)

# load test data
with open("test_set.pk1", "rb") as target:
    test_data = pickle.load(target)

# remove rows with null data
offdev_data = offdev_data[~offdev_data["text"].isnull()] 

# apply data augmentation to positive examples only
print("Starting data augmentation")

augmenter = naw.ContextualWordEmbsAug(
    model_path = 'distilbert-base-uncased', 
    device = device,
    action = "substitute",
    top_k = 20
)

train_data_positive = train_data.loc[train_data["label"]]
augmented_data_all = []

for i in range(len(train_data_positive)):
    original_sentence = train_data_positive.iloc[i]["text"]
    for j in range(3):
        augmented_sentence = augmenter.augment(original_sentence)
        augmented_data = {"keyword": train_data_positive.iloc[i]["keyword"], "text": augmented_sentence[0], "label": True}
        augmented_data_all.append(augmented_data)

augmented_data_df = pd.DataFrame.from_records(augmented_data_all)

train_data = pd.concat([train_data, augmented_data_df])
train_data = train_data.sample(frac=1, random_state=1) #shuffle

print("Data augmentation finished")


# initiate tokenizer and get tokenised input - using full training data set & offdev data!
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer(list(train_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
val_encodings = tokenizer(list(offdev_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
test_encodings = tokenizer(list(test_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
train_labels = list(train_data["label"])
val_labels = list(offdev_data["label"])
test_labels = list(np.zeros(len(test_data))) # Dummy labels to be able to create dataset

# convert to Dataset structure
train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)

############

num_labels = 2
lr = 1e-5
wd = 0.05

# Initializing a model (with random weights) from the microsoft/deberta-base style configuration
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=num_labels)

# Accessing the model configuration
print(model.config) # hyperparameter info
model.to(device)

# Final Arguments from hyperparameter tuning
training_args = TrainingArguments(
    output_dir="Deberta/",
    learning_rate=lr,
    weight_decay=wd,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "f1 score",
    greater_is_better = True
)

trainer = Trainer(
    model,
    args=training_args,
    tokenizer = tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# train the model
trainer.train()

# save the model
trainer.save_model(f"./models/final_model/")

# final dev set trainer evaluation
dev_output = trainer.predict(val_dataset)

# print to txt file the output
with open(f"offdev_eval_final.txt", "w") as file:
    for metric in dev_output.metrics.items():
        file.write(f"{metric}")

# output dev set predictions
with open("dev_predictions.txt", "w") as file:
    for prediction in dev_output.predictions:
        file.write(f"{np.argmax(prediction)}\n")

# output test set predictions
test_output = trainer.predict(test_dataset)

with open("test_predictions.txt", "w") as file:
    for prediction in test_output.predictions:
        file.write(f"{np.argmax(prediction)}\n")










