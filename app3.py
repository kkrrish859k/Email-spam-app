import streamlit as st
import pickle
import string
import re
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

STOPWORDS = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
    "it","its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of","at","by","for",
    "with","about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over","under",
    "again","further","then","once","here","there","when","where","why","how","all",
    "any","both","each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","s","t","can","will","just","don",
    "should","now"
])

def transform_text(text):
    text = text.lower()
    
    # regex tokenizer
    text = re.findall(r'\b\w+\b', text)

    # remove stopwords & punctuation + stemming
    y = [ps.stem(i) for i in text if i.isalnum() and i not in STOPWORDS]
    
    return " ".join(y)


# Load your pickled vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
