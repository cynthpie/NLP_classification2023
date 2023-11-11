import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# cd into folder with dataset
os.chdir("dontpatronizeme_v1.4")

# Read data into dataframe
column_names = ["par_id", "keyword", "text", "label"]
all_data = pd.read_csv("dontpatronizeme_pcl.tsv", sep="\t", names=column_names, index_col="par_id", usecols=[0, 2, 4, 5], skiprows=4)

# Convert labels into binary values: {0, 1} = False (no PCL), {2, 3, 4} = True (PCL)
all_data["label"] = all_data["label"]>1.5

# cd back to main folder
os.chdir("..")

### Data analysis ###

# Number of examples in each class
n_positive = all_data["label"].sum()
n_negative = (all_data["label"]==0).sum()
print(f"Number of positive examples: {n_positive}")
print(f"Number of negative examples: {n_negative}")

# Number of unique keywords
n_unique_kw = len(np.unique(all_data["keyword"]))
unique_kw = list(np.unique(all_data["keyword"]))
print(f"Number of unique keywords: {n_unique_kw}")
print(unique_kw)

# For each keyword, extract the number of +ve and -ve examples as a percentage of the total
n_positive_per_kw = []
n_negative_per_kw = []
for kw in unique_kw:
    data_kw = all_data.loc[all_data["keyword"]==kw]
    n_positive_kw = round(100 * data_kw["label"].sum() / n_positive, 1)
    n_negative_kw = round(100 * (data_kw["label"]==0).sum() / n_negative, 1)
    n_positive_per_kw.append(n_positive_kw)
    n_negative_per_kw.append(n_negative_kw)

# Plot bar chart of +ve/-ve examples per keyword
x = np.arange(len(unique_kw))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
rects1 = ax.bar(x - width/2, n_positive_per_kw, width, label="Positive class")
rects2 = ax.bar(x + width/2, n_negative_per_kw, width, label="Negative class")

ax.set_ylabel("Percentage of total positive/negative examples")
ax.set_title("Class labels by keyword")
ax.set_xticks(x, unique_kw)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
fig.savefig("analysis_keyword.png")
# plt.show()

# Extract length of sentence in each class
length_positive = []
length_negative = []

for row in all_data.itertuples():
    sentence = row.text
    label = row.label
    try:
        sentence_length = len(sentence.split())
    except AttributeError:
        continue

    if label:
        length_positive.append(sentence_length)
    else:
        length_negative.append(sentence_length)

# Plot histogram of lengths per class
bins = np.linspace(0, 200, 50)

fig2, ax2 = plt.subplots()
ax2.hist(length_positive, bins, density=True, alpha=0.9, label="Positive class")
ax2.hist(length_negative, bins, density=True, alpha=0.5, label="Negative class")

ax2.legend(loc="upper right")
ax2.set_xlabel("Sentence length")
ax2.set_ylabel("Normalised frequency")
ax2.set_title("Sentence length distribution per class")
fig2.savefig("analysis_length.png")
# plt.show()


# Split data into official train and dev sets
train_parids = pd.read_csv("train_parids.csv")
dev_parids = pd.read_csv("dev_parids.csv")

train_data = all_data.loc[train_parids["par_id"]]
dev_data = all_data.loc[dev_parids["par_id"]]

# Save datasets into pickle files
# with open("train_set.pk1", "wb") as target:
#     pickle.dump(train_data, target)

# with open("official_dev_set.pk1", "wb") as target:
#     pickle.dump(dev_data, target)