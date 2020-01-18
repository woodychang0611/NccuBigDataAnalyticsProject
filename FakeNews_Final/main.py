import os
import numpy as np
from gen1_preprocessor import Preprocessor as Preprocessor1
from gen2_preprocessor import Preprocessor as Preprocessor2
#from simple_classifier import Classifier
from gen1_classifier import Classifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import sparse
import time

preprocessor1 = Preprocessor1(pca_n_components=2)
preprocessor2 = Preprocessor2(pca_n_components=2)
text_classifier = Classifier()
traning_set_path = os.path.join((os.path.dirname(__file__)),"training")
testing_set_path = os.path.join((os.path.dirname(__file__)),"training")
print (f'Traning Set:{traning_set_path}')
print (f'Testing Set:{testing_set_path}')

preprocessors ={
  "No PCA": Preprocessor1(),
  "PCA 1300": Preprocessor1(pca_n_components=1300),    
  "Preprocessor2 special No PCA": Preprocessor2(),  
#  "Preprocessor2 special PCA 2000": Preprocessor2(pca_n_components=2000), 
  "Preprocessor2 special PCA 1500": Preprocessor2(pca_n_components=1500),
  "Preprocessor2 special PCA 1300": Preprocessor2(pca_n_components=1300),
#  "PCA 1100": Preprocessor(pca_n_components=1100),    
#  "PCA 1000": Preprocessor(pca_n_components=1000),
#  "PCA 500": Preprocessor(pca_n_components=500),    
#  "PCA 300": Preprocessor(pca_n_components=300),
#  "PCA 100": Preprocessor(pca_n_components=100),
} 

for preprocessor_name in preprocessors:
    print(preprocessor_name)
    x,y = preprocessors[preprocessor_name].load_training_data(traning_set_path,limit=None)
    print(f'x type: {type(x)} y type: {type(y)}')
    model =svm.SVC(gamma='scale',C=5)
    start = time.time()
    s = np.mean(cross_val_score(model, x, y, scoring='accuracy',cv=5,error_score='raise'))
    end = time.time()
    print(f'time: {end - start}')
    print(s)
 
exit(0)
#x_test,y_test = preprocessor1.load_testing_data(testing_set_path,limit=100,hasLabel=True)
#x_test2,y_test2 = preprocessor2.load_testing_data(testing_set_path,limit=100,hasLabel=True)
print(x1.shape)
#print(x2.shape)
model = RandomForestClassifier(n_estimators=100)
model =svm.SVC(gamma='scale',C=5)
s1 = np.mean(cross_val_score(model, x1, y1, scoring='accuracy',cv=5,error_score='raise'))
#s2 = np.mean(cross_val_score(model, x2, y2, scoring='accuracy',cv=5,error_score='raise'))
print (f' {s1}')
#print (f' {s2}')
exit(0)
text_classifier.fit(x,y)
y_test_predictions = text_classifier.predict(x_test)
y_test_predict_probabilities = text_classifier.predict_proba(x_test)

count=0
print ('Accuracy:',(y_test_predictions == y_test).sum().astype(float)/(y_test.shape[0]))
print ('Classification report:')
print (classification_report(y_test, y_test_predictions))

y_score = y_test_predict_probabilities[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(false_positive_rate, true_positive_rate)
# Plotting
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, c='navy', label=('AUC-'+'= %0.2f'%roc_auc))
plt.legend(loc='lower right', prop={'size':8})
plt.plot([0,1],[0,1], color='lightgrey', linestyle='--')
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()