import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import re
from nltk.corpus import stopwords

from keras.models import load_model
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from tensorflow.keras.preprocessing.text import one_hot
voc_size =40000
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = load_model('my_model.h5')


def predict_fun(test_input):
    corpus_test = []
    sent_length=100
    review = re.sub('[^a-zA-Z]', ' ',test_input)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)
    onehot_repr_test= [one_hot(words,voc_size)for words in corpus_test]
    #print(onehot_repr_test)
    embedded_docs_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length)
    #print(embedded_docs_test)
    x_to_test = np.array(embedded_docs_test)
    y_to_pred = model.predict(x_to_test)
    y_to_pred = [int(i > .5) for i in y_to_pred]
    return y_to_pred


def fake_or_true(y_to_pred):
    if y_to_pred == [0]:
        return 'fake'
    elif y_to_pred == [1]:
        return "True"  
     
 


def main():
    st.title("Fake News Detection ")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit FAKE NEWS DETECTOR ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Textinput = st.text_area("INPUT","Type Here")

    result_new=""
    if st.button("Predict"):
        result =predict_fun(Textinput)
        result_new = fake_or_true(result)
    
        #result= predict(Textinput)
    st.success('The content is {}'.format(result_new))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with love with Streamlit")

if __name__=='__main__':
    main()
    