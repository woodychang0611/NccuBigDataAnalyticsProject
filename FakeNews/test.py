import pandas as pd

def process_text(text):
  text_length = len(text)
  return [text_length]

label_dataframe = pd.read_csv("label.txt",sep=':',index_col=1,header=None,names=['label','index'])
one_hot_label_dataframe = pd.get_dummies(label_dataframe,prefix=['label'])
label_dataframe = pd.concat([label_dataframe,one_hot_label_dataframe], axis=1) 
tweets_dataframe = pd.read_csv("source_tweets.txt",sep='\t',index_col=0,header=None,names=['index','text'])
tweets_dataframe['text_length'] = tweets_dataframe.text.apply(lambda s: process_text(s))
tweets_dataframe = pd.concat([tweets_dataframe,label_dataframe], axis=1)  


print (tweets_dataframe)
print (tweets_dataframe.columns)


