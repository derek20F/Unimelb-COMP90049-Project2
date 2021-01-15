# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:53:23 2020
Using Multi layer perceptron with one-hot encoding for text type features
@author: Chen-An Fan
"""
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# %% Some Functions ===========================================================

'''
This function is used to take out all tags in training feature set
'''
def getAllTags(features_dataframe):
    allTags = []
    tagSeries = features_dataframe['tag']
    for i in range(tagSeries.shape[0]):
        tags = tagSeries[i].split(",")
        for tag in tags:
            if tag not in allTags:
                allTags.append(tag)
    return allTags


'''
This function is used to apply One-hot encoding to seperate tag features.
And remove the original tag attribute in the dataframe
'''
def onehot(features_dataframe, allTags):
    startTime = time.time()    
    # Seperate the tags and turn then into binary value
    for tag in allTags:
        temp = []
        for instance in range(features_dataframe.shape[0]):
            if tag in features_dataframe.loc[instance, "tag"]:
                temp.append(1)
            else:
                temp.append(0)
        features_dataframe[tag] = temp
    features_dataframe = features_dataframe.drop(["tag"], axis = 1)
    timeSpent = time.time() - startTime
    print("Time spent in onehot is: " + str(timeSpent) + " sec")
    return features_dataframe


# %% Data Preprocess ==========================================================

# Load Data
train_features = pd.read_csv(open("train_features.tsv", "r", encoding="utf-8"), sep='\t')
train_labels = pd.read_csv(open("train_labels.tsv", "r", encoding="utf-8"), sep='\t')
valid_features = pd.read_csv(open("valid_features.tsv", "r", encoding="utf-8"), sep='\t')
valid_labels = pd.read_csv(open("valid_labels.tsv", "r", encoding="utf-8"), sep='\t')
test_features = pd.read_csv(open("NEW_test_features.tsv", "r", encoding="utf-8"), sep='\t')

# Remove unhelpful features
train_features_clean = train_features.drop(["title", "movieId", "YTId", "year"], axis = 1)
valid_features_clean = valid_features.drop(["title", "movieId", "YTId", "year"], axis = 1)
test_features_clean = test_features.drop(["title", "movieId", "YTId", "year"], axis = 1)

# Do the one-hot one "tag" feature
allTags = getAllTags(train_features_clean) #allTags must from the training set to maintain the order
train_features_clean = onehot(train_features_clean, allTags)
valid_features_clean = onehot(valid_features_clean, allTags)
test_features_clean = onehot(test_features_clean, allTags)


# Turn label into numerical expression
train_labels = train_labels["genres"] #take out as a array
train_labels_numerical, train_label_map = pd.factorize(train_labels) #this only take array, not dataframe
# Turn valid label into numerical expression with the same map as the training set
valid_labels_numerical = valid_labels.copy()
for instance in range(valid_labels.shape[0]):
    for labelIndex in range(train_label_map.shape[0]):
        if valid_labels.loc[instance, "genres"] == train_label_map[labelIndex]:
            valid_labels_numerical.at[instance, "genres"] = labelIndex

valid_labels_numerical = valid_labels_numerical["genres"] #take out as a array
valid_labels_numerical = valid_labels_numerical.values #Turn series into array of object
valid_labels_numerical = valid_labels_numerical.astype(numpy.int64) #Turn array of object into array of int64

# %% Train the MLP model ======================================================

trainStartTime = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(100,50,25), activation = 'logistic' ,solver='adam', learning_rate_init=0.0005, learning_rate = 'constant')

y_class = [] #This is used to store the all possible classes of y
for i in range(18):
    y_class.append(i)

epochs = 1000
train_score_curve = []
valid_score_curve = []

for i in range(epochs):
    mlp.partial_fit(train_features_clean, train_labels_numerical, y_class)
    print("========== Step " + str(i) + " ==========")
    
    train_score = mlp.score(train_features_clean, train_labels_numerical) 
    train_score_curve.append(train_score)
    print("Train score = " + str(train_score))
    # See the score on unseen valid dataset
    valid_score = mlp.score(valid_features_clean, valid_labels_numerical)
    valid_score_curve.append(valid_score)
    print("Valid score = " + str(valid_score))
    
    valid_predict_numerical = mlp.predict(valid_features_clean)
    print("predict = " + str(valid_predict_numerical))

# Save this for manual inspection
valid_predict_label = train_label_map[valid_predict_numerical]

trainFinishTime = time.time()
print("Time spent on training is: " + str(trainFinishTime - trainStartTime) + " sec")

# Test on the unlabled testing dataset
test_predict_numerical = mlp.predict(test_features_clean)
test_predict_label = train_label_map[test_predict_numerical]

# Find the max accuracy of validation set and where it happened
maxValidAcc = max(valid_score_curve)
maxValidAccP = round(maxValidAcc * 100, 2)
maxValidAccStep = valid_score_curve.index(max(valid_score_curve))
print("The max accuracy of valid set is: " + str(maxValidAcc))
print("at step: " + str(maxValidAccStep))

print(classification_report(valid_labels["genres"], valid_predict_label))

# %% Plot training and valid curve

plt.close('all')
x_axis = []
for i in range(epochs):
    x_axis.append(i+1)

plt.figure()
plt.plot(x_axis, train_score_curve, "r", x_axis, valid_score_curve, "b")
plt.legend(labels=['train','valid'],loc='best')
plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.plot(maxValidAccStep, maxValidAcc, marker='x', markersize=5, color="Green")
plt.annotate(str(maxValidAccP)+"%",xy=(maxValidAccStep, maxValidAcc))

plt.savefig('train and valid curve.png', dpi=600)

# %%Output csv ================================================================
test_result = pd.DataFrame()
test_result['movieId'] = test_features["movieId"]
test_result['genres'] = test_predict_label
test_result.to_csv('test_result.csv', index=False, encoding='utf-8')

                    