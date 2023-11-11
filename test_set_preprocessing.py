import os
import pandas as pd
import pickle

# cd into folder with dataset
os.chdir("dontpatronizeme_v1.4")

# Read data into dataframe
column_names = ["par_id", "keyword", "text"]
test_data = pd.read_csv("task4_test.tsv", sep="\t", names=column_names, index_col="par_id", usecols=[0, 2, 4])

# cd back to main folder
os.chdir("..")

# Save dataset into pickle file
with open("test_set.pk1", "wb") as target:
    pickle.dump(test_data, target)