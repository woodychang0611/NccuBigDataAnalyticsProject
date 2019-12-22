#!/usr/bin/env python
# coding: utf-8

# # 測試各種model在預設條件的成績

# In[1]:


import os
import numpy as np
from gen1_preprocessor import Preprocessor
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime


# In[2]:


classifiers = {
  'Naive Bayes':MultinomialNB(),
  #'Ada Boost 50 (default)':AdaBoostClassifier(n_estimators=50),     
  #'Ada Boost 100':AdaBoostClassifier(n_estimators=100),   
  #'Ada Boost 150':AdaBoostClassifier(n_estimators=150),
  #'Ada Boost 200':AdaBoostClassifier(n_estimators=200),
  #'Ada Boost 250':AdaBoostClassifier(n_estimators=250),
  'Random Forest':RandomForestClassifier(n_estimators=100),
  'SVM':svm.SVC(gamma='scale')
}

preprocessors ={
  "No PCA": Preprocessor(),      
  "PCA 2000": Preprocessor(pca_n_components=2000), 
  "PCA 1000": Preprocessor(pca_n_components=1000),
  "PCA 100": Preprocessor(pca_n_components=100),
  
} 


# In[3]:



current_path = os.path.abspath('')
traning_set_path = os.path.join(current_path,"training")
print (f'Traning Set:{traning_set_path}')


# In[ ]:


#Cross Validation without PCA
for preprocessor_name in preprocessors:
    print(f'Preprocessor: {preprocessor_name}')
    x_traning,y_traning = preprocessors[preprocessor_name].load_training_data(traning_set_path,limit=None)
    for classifier_name in classifiers:
      print("Time:", datetime.now())
      loop_count =20
      print('='*80)
      print(f'Classifier: {classifier_name}')
      model = classifiers[classifier_name]
      scores = []
      try:
          for _ in range(0,loop_count):
              scores+= list(cross_val_score(model, x_traning, y_traning, scoring='accuracy',cv=5,error_score='raise'))
              print("+", end = '')
          print("")
      except:
          scores=[0]
      min_score = np.min(scores)
      max_score = np.max(scores)
      mean_score = np.mean(scores)
      std_score = np.std(scores)
      print(f'Mean Score: {mean_score},Max Score: {max_score},Min Score: {min_score},Std: {std_score}')
      print('='*80)


# In[ ]:





# In[ ]:




