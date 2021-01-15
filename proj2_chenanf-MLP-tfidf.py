# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:53:23 2020
Using Multi layer perceptron with TF-IDF encoding for text type features
@author: Chen-An Fan
"""
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
import numpy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# %% Some Functions ===========================================================
'''
This function take features DataFrame as input.
And take out all tags in training feature set.
Then remove the original tag attribute in the dataframe.
In this function, a tag with only one character would not be ignored.
''' 
def getAllTags(features_dataframe):
    #token_pattern="(?u)\\b\\w+\\b" >> This means do not ignore words of only one character.
    vectorizer = TfidfVectorizer(sublinear_tf=False, stop_words=None, token_pattern="(?u)\\b\\w+\\b", smooth_idf=True, norm='l2')
    tfidf = vectorizer.fit_transform(features_dataframe["tag"])
    allTags = vectorizer.get_feature_names()
    return allTags

'''
This function take features DataFrame as input.
And take out all titles in training feature set.
Then remove the original title attribute in the dataframe
'''
def getAllTitles(features_dataframe):
    vectorizer = TfidfVectorizer(sublinear_tf=False, stop_words=None, smooth_idf=True, norm='l2')
    tfidf = vectorizer.fit_transform(features_dataframe["title"].values.astype('U'))
    allTitles = vectorizer.get_feature_names()
    return allTitles

'''
This function does the normalized TF-IDF on the "title" attribute.
This function will ignore the word shorter than 2 characters.
'''
def normalized_tfidf_title(dataframe, allTitles):
    vectorizer = TfidfVectorizer(sublinear_tf=False, stop_words=None, smooth_idf=True, norm='l2')
    tfidf = vectorizer.fit_transform(dataframe["title"].values.astype('U')) #convert the dtype object to unicode string
    df_tfidf = pd.DataFrame(tfidf.toarray(),columns=vectorizer.get_feature_names())
    dataframe = dataframe.drop(["title"],axis = 1)
    for title in allTitles:    
        
        if title in df_tfidf.columns:
            dataframe[title] = df_tfidf[title]
        else:
            dataframe[title] = [0 for i in range(dataframe.shape[0])]
            
    return dataframe

'''
This function does the normalized TF-IDF on the "tag" attribute.
'''
def normalized_tfidf_tag(dataframe, allTags):
    vectorizer = TfidfVectorizer(sublinear_tf=False, stop_words=None, token_pattern="(?u)\\b\\w+\\b", smooth_idf=True, norm='l2')
    
    tfidf = vectorizer.fit_transform(dataframe["tag"])
    df_tfidf = pd.DataFrame(tfidf.toarray(),columns=vectorizer.get_feature_names())
    #df_tfidf = pd.DataFrame(tfidf.toarray())
    #allTags = vectorizer.get_feature_names()
    dataframe = dataframe.drop(["tag"],axis = 1)
    for tag in allTags:
        
        if tag in df_tfidf.columns:
            dataframe[tag] = df_tfidf[tag]
        else:
            dataframe[tag] = [0 for i in range(dataframe.shape[0])]
            
    return dataframe
    
# %% Data Preprocess ==========================================================

# Load Data
train_features = pd.read_csv(open("train_features.tsv", "r", encoding="utf-8"), sep='\t')
train_labels = pd.read_csv(open("train_labels.tsv", "r", encoding="utf-8"), sep='\t')
valid_features = pd.read_csv(open("valid_features.tsv", "r", encoding="utf-8"), sep='\t')
valid_labels = pd.read_csv(open("valid_labels.tsv", "r", encoding="utf-8"), sep='\t')
test_features = pd.read_csv(open("NEW_test_features.tsv", "r", encoding="utf-8"), sep='\t')
# Remove unhelpful features
train_features_clean = train_features.drop(["movieId", "YTId", "year"], axis = 1)
valid_features_clean = valid_features.drop(["movieId", "YTId", "year"], axis = 1)
test_features_clean = test_features.drop(["movieId", "YTId", "year"], axis = 1)


# Generate the attribute names of visual data
avfName = []
for i in range(107):
    tempstr = "avf"+str(i+1)
    avfName.append(tempstr)

# Generate the attribute names of audio data
ivecName = []
for i in range(20):
    tempstr = "ivec"+str(i+1)
    ivecName.append(tempstr)

'''
Please comment out this part, if you want to keep the audio and visual data for training!
/=============================================================================\
'''
# Drop all the audio and visual data. Only keep the tag attributes.
###train_features_clean = train_features_clean.drop(avfName, axis = 1)
###valid_features_clean = valid_features_clean.drop(avfName, axis = 1)
###test_features_clean = test_features_clean.drop(avfName, axis = 1)
###train_features_clean = train_features_clean.drop(ivecName, axis = 1)
###valid_features_clean = valid_features_clean.drop(ivecName, axis = 1)
###test_features_clean = test_features_clean.drop(ivecName, axis = 1)
'''
\=============================================================================/
'''


'''
Please comment out this part, if you want to drop tag and title attributes
/=============================================================================\
'''
# Do the normalized tfidf on "title" feature
allTitles = getAllTitles(train_features_clean)
train_features_clean = normalized_tfidf_title(train_features_clean, allTitles)
valid_features_clean = normalized_tfidf_title(valid_features_clean, allTitles)
test_features_clean = normalized_tfidf_title(test_features_clean, allTitles)

# Do the normalized tfidf on "tag" feature
allTags = getAllTags(train_features_clean) #allTags must from the training set to maintain order
train_features_clean = normalized_tfidf_tag(train_features_clean, allTags)
valid_features_clean = normalized_tfidf_tag(valid_features_clean, allTags)
test_features_clean = normalized_tfidf_tag(test_features_clean, allTags)
'''
\=============================================================================/
'''


'''
Please comment out this part, if you want to keep tag and title attributes
/=============================================================================\
'''
###train_features_clean = train_features_clean.drop(['tag'], axis = 1)
###valid_features_clean = valid_features_clean.drop(['tag'], axis = 1)
###test_features_clean = test_features_clean.drop(['tag'], axis = 1)
###train_features_clean = train_features_clean.drop(['title'], axis = 1)
###valid_features_clean = valid_features_clean.drop(['title'], axis = 1)
###test_features_clean = test_features_clean.drop(['title'], axis = 1)
'''
\=============================================================================/
'''

# Turn label into numerical expression
train_labels = train_labels["genres"] #take out as a array
train_labels_numerical, train_label_map = pd.factorize(train_labels) #this only take array not dataframe

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
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation = 'logistic' ,solver='adam', learning_rate_init=0.0005, learning_rate = 'constant')

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


# %% Plot training and valid curve ============================================
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
  
                    