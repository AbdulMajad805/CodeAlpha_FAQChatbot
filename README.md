# 🤖 FAQ Chatbot — CodeAlpha AI Internship

An intelligent FAQ chatbot for an e-commerce store, built using Natural Language Processing (NLP). The chatbot matches user questions to the most relevant FAQ using TF-IDF vectorization and Cosine Similarity.

---

## 🔗 Live Demo
👉 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-codealpha.streamlit.app/)

---

## 🧠 How It Works

1. User types a question in the chat
2. The question is preprocessed using **NLTK** (tokenization, lemmatization, stopword removal)
3. A **synonym expansion** map converts casual words to FAQ vocabulary
4. **TF-IDF vectorization** converts text into numerical vectors
5. **Cosine Similarity** finds the closest matching FAQ question
6. The best matching answer is returned with a confidence score

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| NLTK | Text preprocessing (tokenization, lemmatization, stopwords) |
| Scikit-learn | TF-IDF Vectorizer + Cosine Similarity |
| HTML/CSS | Custom UI styling inside Streamlit |

