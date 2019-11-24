import glob
import json
import random
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#Get data, e.g. text, label from the folder
def process_folder(folder):
    try:
        tweets_id = os.path.basename(folder)
        source_tweets = os.path.join(folder,f'.\\source-tweets\\{tweets_id}.json')
        annotation = os.path.join(folder,'annotation.json')            
        with open(source_tweets) as json_file:
            data = json.load(json_file)
            text = data['text']
        with open(annotation) as json_file:
            data = json.load(json_file)
            is_rumour = data['is_rumour']
        return {'tweets_id':tweets_id,'text':text,'is_rumour':is_rumour}

    except:
        print (f'Process {folder} failed')
        return None

class Preprocessor:
    def __init__ (self):
        self.vectorizer=None
        self.label_map ={'is_rumour':1,'rumour':1,'nonrumour':0,'unclear':0}
        pass
    def load_training_data(self,src,limit: int=None):
        tweets_data=[]
        print(f'load data src:{src}, limit:{limit}')
        source_folder_pattern = f'{src}\\*\\*\\*'
        source_folders = glob.glob(source_folder_pattern,recursive=True)
        random.shuffle(source_folders)
        for folder in source_folders[:limit]:
            result=process_folder(folder)
            if (result!=None):
                tweets_data.append(result)
        tweets_dataframe = pd.DataFrame(tweets_data)
        self.vectorizer = CountVectorizer(decode_error='ignore')
        vectorized_dataframe = self.vectorizer.fit_transform(tweets_dataframe['text'])
        label_dataframe = pd.DataFrame(list(map(lambda a: self.label_map[a],tweets_dataframe['is_rumour'])))
        return vectorized_dataframe,label_dataframe
    def load_testing_data(self,src,limit: int=None):
        pass