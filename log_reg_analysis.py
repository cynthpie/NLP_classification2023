import pandas as pd
import numpy as np
import pickle
with open("dev_pred_lr_.pk", "rb") as target:
    dev_prd_lreg = pickle.load(target)

print(len(dev_prd_lreg[(dev_prd_lreg["keyword"]=="homeless") & (dev_prd_lreg["label"]==True)]))
wrong_homeless = (dev_prd_lreg[(dev_prd_lreg["keyword"]=="homeless") & (dev_prd_lreg["log_reg_prd"]!=dev_prd_lreg["label"])])
correct_true_homeless = dev_prd_lreg[(dev_prd_lreg["keyword"]=="homeless") & (dev_prd_lreg["label"]==False) & (dev_prd_lreg["log_reg_prd"]==False)]
print(wrong_homeless)

# wrong example
print(wrong_homeless["text"].iloc[1])

