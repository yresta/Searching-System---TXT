import pandas as pd
import string
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Data
data = pd.read_csv('stories_history.csv')  # Pastikan dataset sudah ada

# Clean dataset by removing rows containing the word "tidak"
columns_to_check = ['title', 'storylink', 'article']
data_cleaned = data[~data[columns_to_check].apply(lambda x: x.str.contains(r'\btidak\b', case=False, na=False)).any(axis=1)]

# Pre-processing functions
def remove_number(text):
    return re.sub(r'\d+', '', text)

def remove_punctuation(text):
    punc = string.punctuation
    return text.translate(str.maketrans('', '', punc))

def remove_stopwords(text):
    stopw = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in stopw])

def remove_single_quotes(text):
    return re.sub(r"[‘’']", "", text)

# Apply text cleaning
def clean_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_number(text)
    text = remove_stopwords(text)
    return remove_single_quotes(text)

data_cleaned["cleaned_title"] = data_cleaned["title"].apply(clean_text)

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

data_cleaned['tokenized_title'] = data_cleaned['cleaned_title'].apply(word_tokenize)
data_cleaned['lemmatized_title'] = data_cleaned['tokenized_title'].apply(lemmatize_tokens)
data_cleaned['lemmatized_text'] = data_cleaned['lemmatized_title'].apply(lambda x: ' '.join(x))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data_cleaned['lemmatized_text'])

# Streamlit setup
st.set_page_config(page_title="Sistem Pencarian Artikel - History.com", layout="wide")

# Function to search for articles based on keyword
def search_articles(keyword):
    lemmatized_keyword = lemmatizer.lemmatize(keyword.lower())
    
    # Transform keyword to vector
    keyword_vector = vectorizer.transform([lemmatized_keyword])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(keyword_vector, tfidf_matrix)
    
    # Threshold for similarity (e.g., >0 untuk ambil semua yang relevan)
    relevant_indices = [i for i, similarity in enumerate(similarities[0]) if similarity > 0]
    
    # Display search results
    st.subheader(f"Hasil pencarian untuk kata kunci '{keyword}':")
    
    if relevant_indices:
        for idx in relevant_indices:
            title = data_cleaned.iloc[idx]['title']
            article = data_cleaned.iloc[idx]['article']  # Get full article
            
            # Create a button for each article title
            if st.button(f"{title}", key=f"article_{idx}"):
                st.subheader(f"{title}")
                st.write(article)  # Show the full article when clicked
                st.write("-" * 50)
    else:
        st.write("Tidak ditemukan artikel yang cocok.")

# Streamlit UI for input
st.title("Sistem Pencarian Artikel")
keyword = st.text_input("Masukkan kata kunci untuk pencarian:", placeholder="Contoh: American")

# If keyword is entered, search for articles
if keyword:
    search_articles(keyword)