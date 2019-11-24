import os
from simple_preprocessor import Preprocessor
from simple_classifier import Classifier

preprocessor = Preprocessor()
classifier = Classifier()
traning_set_path = os.path.join((os.path.dirname(__file__)),"training")
testing_set_path = os.path.join((os.path.dirname(__file__)),"test")

print (f'Traning Set:{traning_set_path}')
print (f'Testing Set:{testing_set_path}')


x,y = preprocessor.load_training_data(traning_set_path,limit=5)
print (x)
print (y)
classifier.fit(None)
