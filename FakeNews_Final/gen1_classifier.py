import pandas as pd
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
class Classifier:
    def __init__ (self,max_features ='auto'):
        self.model = MultinomialNB()
        #self.model = RandomForestClassifier(n_estimators=100,max_features =max_features )
        #self.model = AdaBoostClassifier()
        #self.model = svm.SVC(gamma='scale',C=1,probability=True)
        pass
    def fit(self,x,y):
        self.model.fit(x,y)
        pass
    def predict(self,data):
        return self.model.predict(data)
    def predict_proba(self,data):
        return self.model.predict_proba(data)
