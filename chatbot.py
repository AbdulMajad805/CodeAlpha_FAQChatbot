import nltk
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download NLTK data (only runs once) ───────────────────────────────
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# ── Setup ──────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ── 1. Load FAQs from CSV ──────────────────────────────────────────────
def load_faqs(filepath="faqs.csv"):
    faqs = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            faqs.append({
                "question": row["question"],
                "answer": row["answer"]
            })
    return faqs

# ── 2. Preprocess text using NLTK ─────────────────────────────────────
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha()            # remove punctuation & numbers
        and t not in stop_words   # remove stopwords (the, is, a ...)
    ]
    return " ".join(tokens)

# ── 3. Build TF-IDF matrix from FAQ questions ──────────────────────────
def build_vectorizer(faqs):
    questions = [faq["question"] for faq in faqs]
    preprocessed = [preprocess(q) for q in questions]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed)

    return vectorizer, tfidf_matrix

# ── 4. Find best match using cosine similarity ─────────────────────────
def get_best_match(user_query, faqs, vectorizer, tfidf_matrix, threshold=0.1):
    cleaned_query = preprocess(user_query)
    query_vec = vectorizer.transform([cleaned_query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < threshold:
        return {
            "answer": "Sorry, I couldn't find a good match. Please try rephrasing your question.",
            "matched_question": None,
            "score": round(float(best_score), 4)
        }

    return {
        "answer": faqs[best_index]["answer"],
        "matched_question": faqs[best_index]["question"],
        "score": round(float(best_score), 4)
    }