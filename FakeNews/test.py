from numpy import loadtxt
import pandas as pd
from enum import Enum

class FakeNewsLabel(Enum):
  non_rumor = 0
  false = 1
  true = 2
  unverified = 3

def process_text(text):
  text_length = len(text)
  return [text_length]

label = pd.read_csv("label.txt",sep=':',index_col=1,header=None,names=['label','index'])
tweets = pd.read_csv("source_tweets.txt",sep='\t',index_col=0,header=None,names=['index','text'])
tweets = pd.concat([tweets,label], axis=1)  
tweets['text_length'] = tweets.text.apply(lambda s: process_text(s))
print (tweets)
for row in tweets.iterrows():
  index = row[0]
  text = row[1].text


