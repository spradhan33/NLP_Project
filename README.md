**🧠 NLP Preprocessing using Streamlit**

This project implements an interactive NLP Preprocessing system that demonstrates core Natural Language Processing (NLP) techniques. The application allows users to input raw text and visually observe how different preprocessing methods transform the text step by step. An interactive Streamlit web application is provided for real-time experimentation.

**📌 Project Overview**

Domain: Natural Language Processing (NLP)
Application Type: Text Preprocessing & Feature Extraction
Techniques: Tokenization, Cleaning, Stemming, Lemmatization, BoW, TF-IDF, Word Embedding
Interface: Streamlit Web App

Text preprocessing is a fundamental step in NLP pipelines such as sentiment analysis, text classification, and information retrieval. This project focuses on concept clarity and visualization rather than model training.

## ✨ Features

- Sentence-level and word-level tokenization  
- Text cleaning:
  - Lowercasing  
  - Punctuation removal  
  - Digit removal  
  - Stopword removal  
- Stemming using:
  - Porter Stemmer  
  - Lancaster Stemmer  
- Lemmatization with Part-of-Speech (POS) tagging  
- Bag of Words (BoW) representation  
- TF-IDF scoring  
- Simple binary word embedding representation  
- Interactive and easy-to-use Streamlit interface


## 🛠️ Technologies Used

- Python 3.11  
- Streamlit  
- NLTK  
- spaCy  
- scikit-learn  
- Pandas  
- Regular Expressions (re)
```
## 📂 Project Structure
NLP/
├── NLP_Project.py # Streamlit application
├── README.md # Project documentation
├── .gitignore # Ignored files and folders
├── runtime.txt # Python runtime configuration
└── nlpenv/ # Virtual environment (ignored)
```
## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install streamlit nltk spacy pandas scikit-learn
python -m spacy download en_core_web_sm
```

- simple_text: "Aman is the HOD of HIT and loves NLP."
- noisy_text: "NLP!!! is 100% useful in 2025 & beyond :)"
- lemmatization_example: "The children are playing games and running fast."

```
## 📊 NLP Technique Logic
| Technique        | Description |
|------------------|-------------|
| Tokenization     | Splits text into sentences and individual words |
| Text Cleaning    | Removes noise such as punctuation, digits, and stopwords |
| Stemming         | Reduces words to their root or base forms |
| Lemmatization    | Converts words into meaningful dictionary  base forms |
| Bag of Words     | Represents text using word frequency counts |
| TF-IDF           | Measures the importance of words in a document |
| Word Embedding   | Represents words as binary numerical vectors |
``` 

### ⚠️ Limitations 
No deep learning or ML model is used
Word embeddings are binary and not semantic
Designed for learning and demonstration purposes
Advanced embeddings like Word2Vec, GloVe, or BERT are not included intentionally to keep the project beginner-friendly.

## 🎓 Academic Note 
This project is suitable for:
NLP coursework
Python & Streamlit practice
Mini projects
Lab experiments and viva
It emphasizes understanding NLP preprocessing concepts visually.

```## 👤 Author 
Satyabrata Pradhan
B.Tech – Computer Science & Engineering
```

## 📌 Future Improvements 
Add Word2Vec / GloVe embeddings
Integrate sentiment analysis model
Deploy on Streamlit Cloud
Add data export (CSV) feature
Improve UI with charts and graphs

**⭐ If you find this project helpful, feel free to star the repository!**