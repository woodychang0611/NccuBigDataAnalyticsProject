import glob
import json
import random
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
#Get data, e.g. text, label from the folder
def process_folder(folder,has_label=False):
    try:
        tweets_id = os.path.basename(folder)
        source_tweets = os.path.join(folder,f'./source-tweets/{tweets_id}.json')
        with open(source_tweets) as json_file:
            data = json.load(json_file)
            text = data['text']
        hash = {'tweets_id':tweets_id,'text':text}
        if(has_label):
            annotation = os.path.join(folder,'annotation.json') 
            #map the label           
            with open(annotation) as json_file:
                data = json.load(json_file)
                label_map ={'is_rumour':1,'rumour':1,'nonrumour':0,'unclear':0}
                hash['is_rumour'] = label_map[data['is_rumour']]
        return hash

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
    def load_training_data(self,src,limit: int=None):
        tweets_data=[]
        print(f'load data src:{src}, limit:{limit}')
        source_folder_pattern = f'{src}/*/*/*'
        source_folders = glob.glob(source_folder_pattern)
        random.shuffle(source_folders)
        for folder in source_folders[:limit]:
            result=process_folder(folder,has_label=True)
            if (result!=None):
                tweets_data.append(result)
        tweets_dataframe = pd.DataFrame(tweets_data)
        self.vectorizer = CountVectorizer(decode_error='ignore')
        vectorized_dataframe = self.vectorizer.fit_transform(tweets_dataframe['text'])
        label_dataframe = tweets_dataframe['is_rumour']
        if(self.pca_n_components!=None):
            self.pca = PCA(n_components=self.pca_n_components)
            vectorized_dataframe= self.pca.fit_transform(vectorized_dataframe.toarray())
        return vectorized_dataframe,label_dataframe
    def load_testing_data(self,src,limit: int=None,hasLabel=False):
        tweets_data=[]
        print(f'load data src:{src}, limit:{limit}')
        source_folder_pattern = f'{src}/*/*/*'
        source_folders = glob.glob(source_folder_pattern)
        random.shuffle(source_folders)
        for folder in source_folders[:limit]:
            result=process_folder(folder,has_label=hasLabel)
            if (result!=None):
                tweets_data.append(result)
        tweets_dataframe = pd.DataFrame(tweets_data)
        if(self.use_pca):
            vectorized_dataframe = self.vectorizer.transform(tweets_dataframe['text'])
            vectorized_dataframe= self.pca.transform(vectorized_dataframe.toarray())
        if (hasLabel):
            label_dataframe = tweets_dataframe['is_rumour']
            return vectorized_dataframe,label_dataframe
        return vectorized_dataframe