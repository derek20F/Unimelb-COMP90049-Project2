# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:32:08 2020
This code is using zero-R as the baseline model
@author: Chen-An Fan
"""

import pandas as pd

train_labels = pd.read_csv(open("train_labels.tsv", "r", encoding="utf-8"), sep='\t')
valid_labels = pd.read_csv(open("valid_labels.tsv", "r", encoding="utf-8"), sep='\t')

# Take out the most frequently happened genre
mostGenre = train_labels["genres"].describe().top

validAcc = 0
for i in range(valid_labels.shape[0]):
    if valid_labels.loc[i, "genres"] == mostGenre:
        validAcc = validAcc + 1

validAcc = validAcc / valid_labels.shape[0]
print("The accuracy of the valid dataset is: " + str(validAcc))

    