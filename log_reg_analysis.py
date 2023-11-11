import pandas as pd
import numpy as np
import pickle
with open("dev_pred_lr_.pk", "rb") as target:
    dev_prd_lreg = pickle.load(target)

with open("wrong_predictions_LR.pk","rb") as file:
    wrong_prd_reg = pickle.load(file)
print(len(dev_prd_lreg[(dev_prd_lreg["keyword"]=="homeless") & (dev_prd_lreg["label"]==True)]))
wrong_disabled = (dev_prd_lreg[(dev_prd_lreg["keyword"]=="homeless") & (dev_prd_lreg["log_reg_prd"]!=dev_prd_lreg["label"])])
correct_true_disabled = dev_prd_lreg[(dev_prd_lreg["keyword"]=="homeless") & (dev_prd_lreg["label"]==False) & (dev_prd_lreg["log_reg_prd"]==False)]
print(wrong_disabled)
print(wrong_disabled["text"].iloc[1])

