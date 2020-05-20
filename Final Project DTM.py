#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import datetime
from time import time
import string
import pandas as pd
import numpy as np
import re

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer # Was not as accurate as WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim
from gensim.models.wrappers import DtmModel
#from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

from bokeh.palettes import Category10_10
from bokeh.models import HoverTool, Legend, ColumnDataSource
from bokeh.plotting import figure, output_file, show, output_notebook


# In[5]:


#Overall reviews
hr = pd.read_csv('output_reviews_updated.csv')


# In[10]:


#Assigning negative reviews where ratings are 1 or 2, positive reviews where ratings are 3, 4, or 5.
negrev, posrev = hr.loc[hr['Rating'] < 3], hr.loc[hr['Rating'] >= 3]


# In[6]:


#reviews['DateofStay'] = reviews['DateofStay'].map(lambda x: x.lstrip('Date of stay: '))


# In[14]:


reviews = negrev[['reviewID','Review','Title','Rating','DateofStay']]  #Use hr, regrev, or posrev, depending on situation
#Strip MoYe to the Form to convert to DateTime
reviewst = reviews.assign(MoYe = reviews['DateofStay'].str.replace(r'Date of stay: ', ''))
#Convert MoYe to DateTime
for x in ['MoYe']:
    reviewst[x] = pd.to_datetime(reviewst[x])


# In[15]:


#Function to get the root word of all the words in the reviews. We used both PorterStemmer and WordNetLemmatizer,
#but WordNetLemmatizer gave a higher accuracy.
def generate_root(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word.lower(), pos = 'n') for word in words]
    return lemmatized
#Function to prprocess the hotel reviews
def preprocess(review):
    #Remove Punctuations
    no_punc = ''.join([character for character in review if character not in string.punctuation])  
    #Tokenize the Review
    tokenized = nltk.word_tokenize(no_punc.lower())
    #Remove Stop Words
    stop_words = set(stopwords.words('english'))
    no_stop = [word.lower() for word in tokenized if word.lower() not in stop_words]
    no_numeric = [word for word in no_stop if not any(num.isdigit() for num in word)]
    #Generating the root words of the tokens
    root_word = generate_root(no_numeric)        
    return root_word


# In[16]:


#Only run once to Pre-process
reviews['Review'] = reviews['Review'].apply(lambda review: preprocess(review) if not pd.isnull(review) else review)
reviews.head()


# In[17]:


revs = reviews.Review
revs.head()


# In[19]:


#Creating the parameters for the DTM Model and a mapping of the unique characters to an id number.
id2word = gensim.corpora.Dictionary(revs)
#Crating a corpus with id2word in BOW format
corpus = [id2word.doc2bow(rev) for rev in revs]


# In[24]:


reviewst['Assigned_Month'] = reviewst.MoYe.dt.to_period('m').apply(str)
sorted_reviewt = reviewst.copy().sort_values(by = 'Assigned_Month').reset_index(drop = True)
sorted_reviewt.head()


# In[27]:


timeDF = pd.DataFrame(sorted_reviewt.groupby(by = 'Assigned_Month').apply(len), columns = ['total_reviews'])
revst = sorted_reviewt.Review
time_slice = list(timeDF.total_reviews)


# In[28]:


#dtm_path = r"C:\Users\leech\Documents\Fall2019\MSA\MSA8040\Final Project\dtm-win64.exe"
print("Dynamic Topic Modeling started.")
start = datetime.datetime.now()
dtm_model = DtmModel('dtm-win64.exe', corpus=corpus, time_slices=time_slice, num_topics=5, id2word=id2word, initialize_lda=True)
finish = datetime.datetime.now()
print(f"\nComplete! Elapsed Time: {(finish-start).total_seconds()} seconds\n")


# In[81]:


#dtm_model.save("full_model5tneg.gensim")


# In[ ]:


#dtm_model.save("full_model5tpos.gensim")


# In[22]:


#dtm_model.save("full_model5t.gensim")


# In[66]:


#dtm_model.save("full_model10t.gensim")


# In[100]:


model1 = DtmModel.load("full_model.gensim")
model1.show_topic(topicid = 1, time = 5, topn = 15)


# In[34]:


model = DtmModel.load("full_model5tneg.gensim")
model.show_topic(topicid = 0, time = 5, topn = 15)


# In[42]:


topicMapping = {0: 'Restaurant/Bar',
                1: 'Hotel Quality',
                2: 'Location',
                3: 'Aesthetic',
                4: 'Staff Service',}


# In[76]:


model.dtm_coherence(time=5, num_words = 5)


# In[43]:


df_doc_topics = sorted_reviewt[['reviewID','Review','Assigned_Month']].copy()
df_doc_topics['DominantTopic'] = None
df_doc_topics['DominantTopic_Prop'] = None
for r in range(df_doc_topics.shape[0]):
    maxID = 0
    maxProp = 0   
    for i in range(model.num_topics):
        proportion = model.gamma_[r, i]
        if proportion > maxProp:
            maxProp = proportion
            maxID = i
    df_doc_topics.loc[r, 'DominantTopic'] = maxID
    df_doc_topics.loc[r, 'DominantTopic_Prop'] = maxProp
df_doc_topics.head()


# In[33]:


topicData = pd.DataFrame(df_doc_topics.groupby(by=['AssignedMonth','DominantTopic']).apply(len)).reset_index()
topicData.columns = ['AssignedMonth','DominantTopic', 'Totals']
topicData['MonthPercentage'] = topicData.copy().apply(lambda x: x['Totals']/timeDF.loc[x['AssignedMonth']], axis=1)
topicData.head()


# In[35]:


def get_things(keys, lines):
    listed = []
    for key in keys:
         listed.append((topicMapping[key], [lines[key]]))   
    return listed


# In[43]:


from bokeh.palettes import Category10_10
output_notebook()

df_list=[]
lines = []

TOOLS = "crosshair,pan,zoom_in,zoom_out,undo,redo,reset,save,box_select"
p = figure(plot_width=800, plot_height=500, x_axis_type="datetime", tools=TOOLS, toolbar_location="below")

for topicID, color in zip(np.unique(topicData['DominantTopic']), Category10_10):
    df = topicData[topicData.DominantTopic==topicID].reset_index(drop=True)
    lines.append(p.line(x=pd.to_datetime(df['AssignedMonth'][5:]), y=df['MonthPercentage'][5:], color=color, line_width=2, alpha=0.8))
    
legend = Legend(items=get_things(np.unique(topicData['DominantTopic']), lines), location="center", click_policy="hide")  
p.add_layout(legend, 'right') 

show(p)

