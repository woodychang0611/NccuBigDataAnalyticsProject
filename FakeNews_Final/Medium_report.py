#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

#Read Data from label.txt and source_tweets.txt to a panda dataframe
label_dataframe = pd.read_csv("label.txt",sep=':',index_col=1,header=None,names=['label','index'])
one_hot_label_dataframe = pd.get_dummies(label_dataframe,prefix=['label'])
label_dataframe = pd.concat([label_dataframe,one_hot_label_dataframe], axis=1) 
tweets_dataframe = pd.read_csv("source_tweets.txt",sep='\t',index_col=0,header=None,names=['index','text'])
tweets_dataframe = pd.concat([tweets_dataframe,label_dataframe], axis=1)  

#Vectorize the text 
#Can use TfidfVectorizer or CountVectorizer
#vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(decode_error='ignore')
X = vectorizer.fit_transform(tweets_dataframe['text'])
Y = tweets_dataframe['label']

#Model available, can add new sklearn learn classification model here
models = {
  'Naive Bayes':MultinomialNB(),
  'Ada Boost':AdaBoostClassifier(),
  'Random Forest':RandomForestClassifier(n_estimators=100),
  'SVM':svm.SVC(gamma='scale')
}

#Hold-out validation 
#Seperate the data to train data and test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2)
for key  in models:
  print('='*80)
  print(f'Model: {key}')
  model = models[key]
  model.fit (Xtrain,Ytrain)
  print(f'Traing Score: {model.score(Xtrain,Ytrain)}')
  print(f'Test Score: {model.score(Xtest,Ytest)}')
  predict_string = 'This is a book'
  predict_result = model.predict(vectorizer.transform([predict_string]))
  print (f'Predict result for "{predict_string}" is "{predict_result[0]}"')
  print('='*80)

#Cross Validation
for key  in models:
  print('='*80)
  print(f'Model: {key}')
  model = models[key]
  scores = cross_val_score(model, X, Y, cv=5)
  mean_score = np.mean(scores)
  print(f'Scores: {scores}')
  print(f'Mean Score: {mean_score}')
  predict_string = 'This is a book'
  predict_result = model.predict(vectorizer.transform([predict_string]))
  print (f'Predict result for "{predict_string}" is "{predict_result[0]}"')
  print('='*80)


# In[ ]:





# 
