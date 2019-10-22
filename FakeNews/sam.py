import pandas as pd
import random
import numpy as np
#import nltk as nltk
#from sklearn.ensemble import RandomForestClassifier


#nltk.download()
def process_text(text):
  text_length = len(text)
  return [text_length]
#讀結果檔
label_dataframe = pd.read_csv("D://label.txt",sep=':',index_col=1,header=None,names=['label','index'])
one_hot_label_dataframe = pd.get_dummies(label_dataframe,prefix=['label'])
label_dataframe = pd.concat([label_dataframe,one_hot_label_dataframe], axis=1)
#讀分析檔
tweets_dataframe = pd.read_csv("D://source_tweets.txt",sep='\t',index_col=0,header=None,names=['index','text'])
tweets_dataframe['text_length'] = tweets_dataframe.text.apply(lambda s: process_text(s))
tweets_dataframe = pd.concat([tweets_dataframe,label_dataframe], axis=1)

#print(tweets_dataframe.index.size)
#print (tweets_dataframe)
#print (tweets_dataframe.columns)

##########隨機抽樣##########
#
# 使用pandas
# tweets_dataframe.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
# n是要抽取的行數。（例如n=20000時，抽取其中的2W行）
# frac是抽取的比列。（有一些時候，我們並對具體抽取的行數不關係，我們想抽取其中的百分比，這個時候就可以選擇使用frac，例如frac=0.8，就是抽取其中80%）
# replace：是否為有放回抽樣，取replace=True時為有放回抽樣。
# weights這個是每個樣本的權重，具體可以看官方文件說明。
# random_state這個在之前的文章已經介紹過了。
# axis是選擇抽取資料的行還是列。axis=0的時是抽取行，axis=1時是抽取列（也就是說axis=1時，在列中隨機抽取n列，在axis=0時，在行中隨機抽取n行）

# 資料準備
tweetdata = tweets_dataframe.values
# 使用random(抽100筆)
data_sample = random.sample(list(tweetdata), 100)
len(data_sample)
print(data_sample)

