# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 02:06:13 2022

@author: Bhargav
"""

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text) #seprating each word
    l = []
    for i in text:
        if i.isalnum():
            l.append(i) #considering only alphabets and numbers
    
    text = l[:]
    l.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)
            
    text = l[:]
    l.clear()        
    for i in text:
        l.append(ps.stem(i)) #applying stemming
    
    return " ".join(l)


tfidf = pickle.load(open('D:/nlp/vectorizer.pkl','rb'))
model = pickle.load(open('D:/nlp/model.pkl','rb'))

st.title("Text Spam or HAm Classifier")

input_sms = st.text_area('Enter the message')

if st.button('Predict'):


    transformed_sms= text_transform(input_sms)
    
    vector_input = tfidf.transform([transformed_sms])
    
    result = model.predict(vector_input)[0]
    
    
    if result == 1:
        st.header('Spam Detected')
        
    else:
        st.header('No Spam Detected')
        