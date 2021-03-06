import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
class Classifier:
    def __init__ (self):
        self.model = RandomForestClassifier(n_estimators=300)
        pass
    def fit(self,x,y):
        self.model.fit(x,y)
        pass
    def predict(self,data):
        return self.model.predict(data)
    def predict_proba(self,data):
        return self.model.predict_proba(data)
