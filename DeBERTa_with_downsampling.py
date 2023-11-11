import pickle
import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

# gpu set up
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# load train data
with open("train_set.pk1", "rb") as target:
    train_data = pickle.load(target)

# shuffle and split data into val and train
train_data = train_data.sample(frac=1, random_state=1) #shuffle
val_data = train_data.iloc[0:int(len(train_data)*0.2)]
train_data = train_data.iloc[int(len(train_data)*0.2):]

# check positive sample percentages in validation and train data
# print(sum(val_data["label"])/len(val_data)) # 0.087164
# print(sum(train_data["label"])/len(train_data)) # 0.09671

# downsample negative examples in training data to match the number of positive examples
train_data_positive = train_data.loc[train_data["label"]]
train_data_negative = train_data.loc[train_data["label"]==0]
train_data_negative = train_data_negative.sample(frac=len(train_data_positive)/len(train_data_negative), random_state=1)
train_data = pd.concat([train_data_positive, train_data_negative])
train_data = train_data.sample(frac=1, random_state=1) #shuffle

# initiate tokenizer and get tokenised input
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer(list(train_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
val_encodings = tokenizer(list(val_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
train_labels = list(train_data["label"])
val_labels = list(val_data["label"])

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

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

############

num_labels = 2
lr = 1e-5
wd = 0.05

# Initializing a model (with random weights) from the microsoft/deberta-base style configuration
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=num_labels)
# Accessing the model configuration
print(model.config) # hyperparameter info
model.to(device)
# training argument: define hyperparameters here
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
trainer.save_model(f"./models/saved_model_downsampling/")

############

# load the model
model = DebertaForSequenceClassification.from_pretrained(f"./models/saved_model_downsampling/")
model.to(device)

# trainer evaluation
val_output = trainer.evaluate(val_dataset)
print(val_output)

# print to txt file the output
with open(f"val_eval_with_downsampling.txt", "w") as file:
    for metric in val_output.items():
        file.write(f"{metric}")