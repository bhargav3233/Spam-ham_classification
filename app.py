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
ps= PorterStemmer()

def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    #removing special chars
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)



tfidf = pickle.load(open('D:/spamham/vectorizer.pkl','rb'))
model  = pickle.load(open('D:/spamham/model.pkl','rb'))

st.title('Text Spam / ham Classifier')

input_sms = st.text_area('Enter the text')

if st.button('Predict'):
    transformed_sms = text_transform(input_sms)
    
    vector_input = tfidf.transform([transformed_sms])
    
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header('Spam Detected')
    else:
        st.header('No Spam Detected')


