import pickle
import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
# from DeBERTa_test3 import Dataset, compute_metrics

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

# remove rows with null data
offdev_data = offdev_data[~offdev_data["text"].isnull()]  
print(offdev_data)


# initiate tokenizer and get tokenised input - using full training data set & offdev data!
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer(list(train_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
val_encodings = tokenizer(list(offdev_data["text"]), return_tensors="pt", truncation=True, padding=True).to(device)
train_labels = list(train_data["label"])
val_labels = list(offdev_data["label"])

# convert to Dataset structure
train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

############




# Initializing a model (with random weights) from the microsoft/deberta-base style configuration
num_labels = 2
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=num_labels)

# Accessing the model configuration
print(model.config) # hyperparameter info
model.to(device)

# Final Arguments from Hyperparameter - to be determined!
training_args = TrainingArguments(
    output_dir="Deberta/",
    learning_rate=1e-5,
    weight_decay=0.00,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    dataloader_num_workers=2,
    do_eval=True,
    evaluation_strategy="epoch",
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
dev_output = trainer.evaluate(val_dataset)


# print to txt file the output
with open(f"offdev_eval_bs.txt", "w") as file:
    for metric in dev_output.items():
        file.write(f"{metric}")










