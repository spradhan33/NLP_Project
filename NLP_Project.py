import streamlit as st
import nltk
import spacy
import string
import re
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# DOWNLOAD NLTK DATA
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NLP Preprocessing",
    layout="wide"
)

# APP TITLE
st.title("NLP Preprocessing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization, BoW, TF-IDF and Word Embedding")

# USER INPUT
text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Aman is the HOD of HIT and loves NLP"
)

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embedding"
    ]
)

# PROCESS BUTTON
if st.button("Process Text"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    # TOKENIZATION
    elif option == "Tokenization":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(sent_tokenize(text))

        with col2:
            st.write(word_tokenize(text))

        with col3:
            st.write(list(text))

    # TEXT CLEANING
    elif option == "Text Cleaning":
        text_lower = text.lower()

        cleaned_text = "".join(
            ch for ch in text_lower
            if ch not in string.punctuation and not ch.isdigit()
        )

        doc = nlp(cleaned_text)

        final_words = [
            token.text
            for token in doc
            if not token.is_stop and token.text.strip() != ""
        ]

        st.write(" ".join(final_words))

    # STEMMING
    elif option == "Stemming":
        words = word_tokenize(text)

        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        porter_stem = [porter.stem(word) for word in words]
        lancaster_stem = [lancaster.stem(word) for word in words]

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": porter_stem,
            "Lancaster Stemmer": lancaster_stem
        })

        st.dataframe(df, use_container_width=True)

    # LEMMATIZATION
    elif option == "Lemmatization":
        doc = nlp(text)

        df = pd.DataFrame(
            [(token.text, token.pos_, token.lemma_) for token in doc],
            columns=["Word", "POS", "Lemma"]
        )

        st.dataframe(df, use_container_width=True)

    # BAG OF WORDS
    elif option == "Bag of Words":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])

        vocab = vectorizer.get_feature_names_out()
        freq = X.toarray()[0]

        df = pd.DataFrame({
            "Word": vocab,
            "Frequency": freq
        }).sort_values(by="Frequency", ascending=False)

        st.dataframe(df, use_container_width=True)

    # TF-IDF (REGEX USED ONLY HERE)
    elif option == "TF-IDF":
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b[a-z]{2,}\b'
        )

        X = vectorizer.fit_transform([cleaned])

        words = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]

        df = pd.DataFrame({
            "Word": words,
            "TF-IDF Score": scores
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

    # WORD EMBEDDING (REGEX USED ONLY HERE)
    elif option == "Word Embedding":
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        vectorizer = CountVectorizer(
            token_pattern=r'(?u)\b[a-z]{2,}\b',
            binary=True
        )

        X = vectorizer.fit_transform([cleaned])

        words = vectorizer.get_feature_names_out()
        vectors = X.toarray()[0]

        df = pd.DataFrame({
            "Word": words,
            "Embedding Value": vectors
        })

        st.dataframe(df, use_container_width=True)