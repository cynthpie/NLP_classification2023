import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# preprossed dataset
with open("official_dev_set.pk1", "rb") as target:
    dev_data = pickle.load(target)

# predicted output
with open("dev.txt", "r") as f:
    lines = f.readlines()
    pred = [bool(int(prd.strip("\n"))) for prd in lines]
dev_data["pred"] = pred

# extract pcr level from original dataset
os.chdir("dontpatronizeme_v1.4")
column_names = ["par_id", "keyword", "text", "label"]
all_data = pd.read_csv("dontpatronizeme_pcl.tsv", sep="\t", names=column_names, index_col="par_id", usecols=[0, 2, 4, 5], skiprows=4)
dev_data["pcr_level"] = all_data["label"]
print(dev_data)

# level of patronising content
wrong_pred = dev_data[(dev_data["label"] != dev_data["pred"])]
false_neg = dev_data[(dev_data["label"]==True) & (dev_data["pred"]==False)]
false_pos = dev_data[(dev_data["label"]==False) & (dev_data["pred"]==True)]
a = wrong_pred.value_counts("pcr_level", sort=False)
total = dev_data.value_counts("pcr_level", sort=False)
#print(a/total)

fig, ax = plt.subplots()
ax.bar(["0", "1","2","3","4"], list(a))
ax.set_xlabel("PCL level")
ax.set_ylabel("Misclassification count")
ax.set_title("Misclassification count by PCL level")
#fig.tight_layout()
# fig.savefig("wrong_pr_pcl.png")

# length of the input sequence
# dev_data["text_len"] = [len(text.split(" ")) for text in dev_data["text"]]
false_neg["text_len"] = [len(text.split(" ")) for text in false_neg["text"]]
false_pos["text_len"] = [len(text.split(" ")) for text in false_pos["text"]]
fig, ax = plt.subplots()
ax.hist(false_neg["text_len"], alpha=0.9, label="False negative", density=True)
ax.hist(false_pos["text_len"], alpha=0.5, label="False positive", density=True)
ax.set_xlabel("Text length")
ax.set_ylabel("Normalised frequency")
ax.set_title("Sentence length distribution in misclassified instances")
ax.legend()
#fig.tight_layout()
# fig.savefig("wrong_pr_pcl.png")
#plt.show()

# data category
b = false_neg.value_counts("keyword", sort=False)
c = false_pos.value_counts("keyword", sort=False)
c = (c+b-b).replace(np.nan, 0).astype(float)

# Plot bar chart of +ve/-ve examples per keyword
x = np.arange(len(np.unique(false_neg["keyword"])))  # the label locations
unique_kw = list(np.unique(false_neg["keyword"]))
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
rects1 = ax.bar(x - width/2, round(b/len(false_neg),3), width, label="False negative")
rects2 = ax.bar(x + width/2, round(c/len(false_pos),3), width, label="False positive")

ax.set_ylabel("Misclassification percentage")
ax.set_title("Misclassification percentage by keyword")
ax.set_xticks(x, unique_kw)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
# fig.savefig("model_analysis_keyword.png")
#plt.show()
print(dev_data[(dev_data["keyword"]=="homeless") & (dev_data["pred"]!=dev_data["label"])])
##############################