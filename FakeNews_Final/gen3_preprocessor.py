import glob
import json
import random
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA 
import numpy as np
import re
from nltk.corpus import stopwords
from scipy import sparse

#Get data, e.g. text, label from the folder

def cleanup_text(text):
    #remove punctuation
    #text = re.sub('[^A-Za-z0-9]+', ' ', text)
    #remove stopwords
    #text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
    return text


def process_folder(folder,has_label=False):
    try:
        tweets_id = os.path.basename(folder)
        source_tweets = os.path.join(folder,f'./source-tweets/{tweets_id}.json')
        with open(source_tweets) as json_file:
            data = json.load(json_file)
            text = cleanup_text(data['text'])
            #text = json_file.read()
        ret_hash = {'tweets_id':tweets_id,'text':text}
        if(has_label):
            annotation = os.path.join(folder,'annotation.json') 
            #map the label           
            with open(annotation) as json_file:
                data = json.load(json_file)
                label_map ={'is_rumour':1,'rumour':1,'nonrumour':0,'unclear':0}
                ret_hash['is_rumour'] = label_map[data['is_rumour']]

        #Process reactions        
        reaction_tweets_pattern = os.path.join(folder,f'./reactions/*')
        reaction_tweets = glob.glob(reaction_tweets_pattern)
        ret_hash['reaction_count'] = len(reaction_tweets)
        ret_hash['reaction_texts']=''
        for reaction_tweet in reaction_tweets:
            with open(reaction_tweet) as json_file:
                data = json.load(json_file)
                ret_hash['reaction_texts'] += cleanup_text(data['text'])
        return ret_hash

    except Exception as e:
        print (f'Process {folder} failed')
        print (f'exception {e}')
        return None

class Preprocessor:
    def __init__ (self,pca_n_components = None):
        self.vectorizer=None
        self.pca = None
        self.pca_n_components = pca_n_components
        pass
    def load_training_data(self,src,testing,limit: int=None):
        tweets_data=[]
        testing_data=[]
        print(f'load data src:{src}, limit:{limit}')
        source_folder_pattern = f'{src}/*/*/*'
        source_folders = glob.glob(source_folder_pattern)
        random.shuffle(source_folders)
        for folder in source_folders[:limit]:
            result=process_folder(folder,has_label=True)
            if (result!=None):
                tweets_data.append(result)
        testing_folder_pattern = f'{testing}/*/*/*'
        testing_folders = glob.glob(testing_folder_pattern)
        for folder in testing_folders[:limit]:
            result=process_folder(folder,has_label=False)
            if (result!=None):
                testing_data.append(result)                
        pattern_regex = '(?u)\b\w\w+\b'
        print (f'{len(tweets_data)} data read')
        tweets_dataframe = pd.DataFrame(tweets_data)
        testing_data = pd.DataFrame(testing_data)
        vectorizer = CountVectorizer(stop_words='english')
        vectorizer.fit(tweets_dataframe['text'])
        vocabulary_traning = set(vectorizer.vocabulary_.keys())

        vectorizer = CountVectorizer(stop_words='english')
        vectorizer.fit(testing_data['text'])
        vocabulary_testing = set(vectorizer.vocabulary_.keys())
        count=0
        vocabulary_table={}
        for text in vocabulary_traning.intersection(vocabulary_testing):
            vocabulary_table[text]=count
            count+=1
        print(f'vocabulary_traning len {len(vocabulary_traning)}')
        print(f'vocabulary_testing len {len(vocabulary_testing)}') 
        print(f'vocabulary_table len {len(vocabulary_table)}')  
        print(vocabulary_table)
        self.vectorizer = CountVectorizer(decode_error='ignore',vocabulary=vocabulary_table)
         #fit with traning and testing data
        self.vectorizer.fit(pd.concat([tweets_dataframe['text'],testing_data['text']]))
        vectorized_text = self.vectorizer.transform(tweets_dataframe['text']).toarray()
        reaction_count_dataframe = np.reshape (tweets_dataframe['reaction_count'].values,(-1,1))
        input_data = np.concatenate([vectorized_text],axis=1) 
        label_dataframe = tweets_dataframe['is_rumour']
        if(self.pca_n_components!=None):
            self.pca = PCA(n_components=self.pca_n_components)
            input_data= self.pca.fit_transform(input_data)
        #transform to sparse matrix to incrase speed, somehow numpy array is slower
        input_data = sparse.csr_matrix(input_data)
        return input_data,label_dataframe

    def load_testing_data(self,src,limit: int=None,has_label=False):
        tweets_data=[]
        print(f'load data src:{src}, limit:{limit}, has_label:{has_label}')
        source_folder_pattern = f'{src}/*/*/*'
        source_folders = glob.glob(source_folder_pattern)
        for folder in source_folders[:limit]:
            result=process_folder(folder,has_label=has_label)
            if (result!=None):
                tweets_data.append(result)
        tweets_dataframe = pd.DataFrame(tweets_data)
        vectorized_text = self.vectorizer.transform(tweets_dataframe['text']).toarray()
        reaction_count_dataframe = np.reshape (tweets_dataframe['reaction_count'].values,(-1,1))
        input_data = np.concatenate([vectorized_text],axis=1) 
        if(self.pca_n_components!=None):
             input_data= self.pca.transform(input_data)
        #transform to sparse matrix to incrase speed, somehow numpy array is slower
        input_data = sparse.csr_matrix(input_data)
        indexes =  tweets_dataframe['tweets_id']
        if(has_label):
            label_dataframe = tweets_dataframe['is_rumour']    
            return input_data,label_dataframe,indexes
        else:
            return input_data,None,indexes