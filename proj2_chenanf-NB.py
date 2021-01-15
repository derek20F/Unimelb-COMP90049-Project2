# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:46:55 2020
This code is to use Naive Bayes to predict the movie genre
Using one-hot feature to deal with the text type features

@author: Chen-An Fan
"""
import time
import pandas as pd
from sklearn import preprocessing #LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

'''
This function is used to categorize the audio and visual data into 4 classes 
'''
def categorizeData(features_data_frame, avfName, ivecName, avf_ivec_for_train):
    for instance in range(features_data_frame.shape[0]):
        for avf in avfName:
            if features_data_frame.loc[instance, avf] <= avf_ivec_for_train.loc["25%", avf]:
                features_data_frame.loc[instance, avf] = 1
            elif features_data_frame.loc[instance, avf] <= avf_ivec_for_train.loc["50%", avf]:
                features_data_frame.loc[instance, avf] = 2
            elif features_data_frame.loc[instance, avf] <= avf_ivec_for_train.loc["75%", avf]:
                features_data_frame.loc[instance, avf] = 3
            else:
                features_data_frame.loc[instance, avf] = 4
        for ivec in ivecName:
            if features_data_frame.loc[instance, ivec] <= avf_ivec_for_train.loc["25%", ivec]:
                features_data_frame.loc[instance, ivec] = 1
            elif features_data_frame.loc[instance, ivec] <= avf_ivec_for_train.loc["50%", ivec]:
                features_data_frame.loc[instance, ivec] = 2
            elif features_data_frame.loc[instance, ivec] <= avf_ivec_for_train.loc["75%", ivec]:
                features_data_frame.loc[instance, ivec] = 3
            else:
                features_data_frame.loc[instance, ivec] = 4
    return features_data_frame

# %% Functions for One-hot Encoding
'''
This function is used to get all tag in training data set for One-hot encoding
'''
def getAllTags_Onehot(features_dataframe):
    allTags = []
    tagSeries = features_dataframe['tag']
    for i in range(tagSeries.shape[0]):
        tags = tagSeries[i].split(",")
        for tag in tags:
            if tag not in allTags:
                allTags.append(tag)
    return allTags


'''
This function is used to apply One-hot encoding to seperate tag attribute
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
    # Drop the original tag attribute
    features_dataframe = features_dataframe.drop(["tag"], axis = 1)
    timeSpent = time.time() - startTime
    print("Time spent in onehot is: " + str(timeSpent) + " sec")
    return features_dataframe

# %% Functions for TF-IDF =====================================================
'''
This function take features DataFrame as input.
And take out all tags in training feature set.
Then remove the original tag attribute in the dataframe.
In this function, a tag with only one character would not be ignored.
''' 
def getAllTags_tfidf(features_dataframe):
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
def getAllTitles_tfidf(features_dataframe):
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
    #df_tfidf = pd.DataFrame(tfidf.toarray())
    #allTags = vectorizer.get_feature_names()
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





# %% Data Processing ==========================================================
t = time.time()

# Load data
train_features = pd.read_csv(open("train_features.tsv", "r", encoding="utf-8"), sep='\t')
train_labels = pd.read_csv(open("train_labels.tsv", "r", encoding="utf-8"), sep='\t')
valid_features = pd.read_csv(open("valid_features.tsv", "r", encoding="utf-8"), sep='\t')
valid_labels = pd.read_csv(open("valid_labels.tsv", "r", encoding="utf-8"), sep='\t')

# Drop unnecessary Attributes
train_features_clean = train_features.drop(["title", "movieId", "YTId", "year"], axis = 1)
valid_features_clean = valid_features.drop(["title", "movieId", "YTId", "year"], axis = 1)

#Using the statistic of training set for categorizing
avf_ivec_analysis = train_features_clean.describe() 

# Get all the tag in training set
allTags=getAllTags_Onehot(train_features_clean)

# Take out the name of all features
features_titles = []
for feature in train_features_clean:
    features_titles.append(feature)

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
Please comment out this part, if you only want to use tag attribute to train!
/=============================================================================\
'''
# Categorize the audio and vusial data into 4 classes by their relative value in training set
train_features_clean = categorizeData(train_features_clean, avfName, ivecName, avf_ivec_analysis)
valid_features_clean = categorizeData(valid_features_clean, avfName, ivecName, avf_ivec_analysis)
'''
\=============================================================================/
'''

'''
Please comment out this part, if you want to use the audio and visual data for training!
/=============================================================================\
'''
# Drop all the audio and visual data. Only keep the tag attributes.
###train_features_clean = train_features_clean.drop(avfName, axis = 1)
###valid_features_clean = valid_features_clean.drop(avfName, axis = 1)
###train_features_clean = train_features_clean.drop(ivecName, axis = 1)
###valid_features_clean = valid_features_clean.drop(ivecName, axis = 1)
'''
\=============================================================================/
'''


'''
Please comment out this part, if you want to keep tag attributes
/=============================================================================\
'''
###train_features_clean = train_features_clean.drop(['tag'], axis = 1)
###valid_features_clean = valid_features_clean.drop(['tag'], axis = 1)
'''
\=============================================================================/
'''


# Apply one hot encoding on tag attribute
train_features_clean = onehot(train_features_clean, allTags)
valid_features_clean = onehot(valid_features_clean, allTags)

elapsed = time.time() - t
print("Time spent on pre-processing data is: " + str(elapsed))

# %% Train the model ==========================================================

train_labels_array = train_labels["genres"] #take out as a array
train_labels_numerical, train_label_map = pd.factorize(train_labels_array)

'''
Indicate which NB model to use here!
'''
#model = BernoulliNB()
model = ComplementNB()
#model = GaussianNB()
#model = CategoricalNB()
#model = MultinomialNB()

t_train = time.time()

model.fit(train_features_clean, train_labels_numerical)
valid_labels_predicted_numerical = model.predict(valid_features_clean)
result = train_label_map[valid_labels_predicted_numerical]
# Compare result with valid_labels
valid_labels = valid_labels["genres"]
acc = 0

for i in range(valid_features_clean.shape[0]):
    if result[i] == valid_labels[i]:
        acc = acc + 1
        
acc = acc / valid_features_clean.shape[0]

print("acc = " + str(acc))


elapsed_train = time.time() - t_train
print("Time spent on training is: " + str(elapsed_train))





  
    