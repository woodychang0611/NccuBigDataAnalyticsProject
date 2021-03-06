import pandas as pd
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import glob
import json
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate
import numpy as np
import scipy

#Read testing data 
#store all traning data in a hash table
raw_traning_data= {}


#process tweet folder
#e.g. .\training\charliehebdo-all-rnr-threads\rumours\552783238415265792
def process_tweet(folder):
    tweets_id = os.path.basename(folder)
    source_tweets = os.path.join(folder,f'./source-tweets/{tweets_id}.json')
    with open(source_tweets) as json_file:
        data = json.load(json_file)
        text = data['text']
    hash = {'tweets_id':tweets_id,'text':text}
    annotation = os.path.join(folder,'annotation.json') 
    with open(annotation) as json_file:
        data = json.load(json_file)
        label_map ={'is_rumour':1,'rumour':1,'nonrumour':0,'unclear':0}
        hash['is_rumour'] = label_map[data['is_rumour']]
    return hash

#process source folder 
#e.g. .\training\charliehebdo-all-rnr-threads

def process_source_folder(folder):
    tweets__source_folder_pattern = f'{folder}/*/*'
    tweets_folders = glob.glob(tweets__source_folder_pattern)
    return pd.DataFrame(map(process_tweet,tweets_folders[:]))

def get_intersection_vocabulary(raw_data):
    vocabulary_sets = {}
    for theme in raw_traning_data: 
        vectorizer = CountVectorizer(decode_error='ignore',stop_words='english')
        vectorizer.fit(raw_traning_data[theme]['text'])
        vocabulary_set = set(vectorizer.vocabulary_.keys())
        vocabulary_sets[theme] = vocabulary_set
    for theme in vocabulary_sets:
        print (f'count of vocabulary in {theme}: {len(vocabulary_sets[theme])}')
    #intersection
    intersection_vocabulary = set.intersection(*map(lambda key:vocabulary_sets[key],vocabulary_sets))
    return intersection_vocabulary
 

current_path = os.path.abspath('')
traning_set_path = os.path.join(current_path,"training")
print (f'read training data from {traning_set_path}')
source_folder_pattern = f'{traning_set_path}/*'
source_folders = glob.glob(source_folder_pattern)
for source_folder in source_folders:
    print (source_folder)
    theme = os.path.basename(source_folder)
    print (theme)
    raw_traning_data[theme] = process_source_folder(source_folder)

for theme in raw_traning_data:
    print (f'{theme} data count: {len (raw_traning_data[theme])}')
    

intersection_vocabulary = get_intersection_vocabulary(raw_traning_data)
print(f'len of intersection_vocabulary :{len(intersection_vocabulary)}')

vectorizer = CountVectorizer(decode_error='ignore',vocabulary=intersection_vocabulary)

x_traning = None
y_traning = None
for theme in raw_traning_data:
    data = raw_traning_data[theme]
    x_data =  vectorizer.transform(data['text']).toarray()
    y_data = list(data['is_rumour'])
    if (x_traning is None):
        x_traning=x_data
    else:
        x_traning =np.concatenate([x_traning,x_data])
        pass
    if (y_traning is None):
        y_traning=y_data
    else:
        y_traning+=y_data

#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_traning = scipy.sparse.csr_matrix(x_traning)

#balance the traning set 
x_traning_true = []
y_traning_true =  []
x_traning_false =  []
y_traning_false = []
for x,y in zip(x_traning,y_traning):
    if(y==1):
        x_traning_true.append(x.toarray())
        y_traning_true.append(y)
    else:
        x_traning_false.append(x.toarray())
        y_traning_false.append(y)
min_len = min(int(len(x_traning_true)),len(x_traning_false))
x_traning_balanced = np.concatenate(x_traning_true[:int(min_len)] +  x_traning_false[:int(min_len)])
y_traning_balanced = np.array(y_traning_true[:int(min_len)] +  y_traning_false[:int(min_len)])
#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_traning_balanced = scipy.sparse.csr_matrix(x_traning_balanced)
print(x_traning_balanced.shape)
print(y_traning_balanced.shape)

#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_traning_balanced = scipy.sparse.csr_matrix(x_traning_balanced)

#Train model
model = RandomForestClassifier(n_estimators=100)
score = cross_validate(model,x_traning_balanced,y_traning_balanced,cv=5,scoring="accuracy")
model.fit(x_traning_balanced,y_traning_balanced)
print(f'Cross validation score: {np.mean(score["test_score"])}')
#Read testing data
testing_path = os.path.join(current_path,"source_tweets_final.txt")
testing_dataframe = pd.read_csv(testing_path,sep='\t',header=None,names=['index','text'])
x_testing = vectorizer.transform(testing_dataframe['text'])
#Test Model
y_testing = model.predict(x_testing)
print(f'rummor rate: {np.sum(y_testing)/len(y_testing)}')

#Output Result
result_path = os.path.join(current_path,"result_real_final.txt")
print (f'Output result to {result_path}')
with open(result_path, 'w',encoding='utf-8') as result_file:
    testing_index = list(testing_dataframe['index'])
    for index,label in zip(testing_index,y_testing):
        rummor_label = 'rumours' if label == 1 else 'non-rumours'
        text = f"{str(index)}\t{str(rummor_label)}\n"
        result_file.writelines(text)