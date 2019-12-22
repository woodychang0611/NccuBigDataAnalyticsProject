import os
import numpy as np
from gen1_preprocessor import Preprocessor
#from simple_classifier import Classifier
from gen1_classifier import Classifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

preprocessor = Preprocessor(pca_n_components=1000)
text_classifier = Classifier()
traning_set_path = os.path.join((os.path.dirname(__file__)),"training")
testing_set_path = os.path.join((os.path.dirname(__file__)),"training")
print (f'Traning Set:{traning_set_path}')
print (f'Testing Set:{testing_set_path}')


x,y = preprocessor.load_training_data(traning_set_path,limit=None)
x_test,y_test = preprocessor.load_testing_data(testing_set_path,limit=100,hasLabel=True)


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