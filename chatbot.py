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

SYNONYMS = {
    "send back"     : "return refund",
    "sending back"  : "return refund",
    "give back"     : "return refund",
    "sent back"     : "return refund",
    "send it back"  : "return refund",
    "refund"        : "return refund",
    "money back"    : "return refund",
    "ship back"     : "return refund",
    "return"        : "return refund",
    "arrive"        : "delivery shipping",
    "arriving"      : "delivery shipping",
    "get my order"  : "delivery shipping",
    "how long"      : "delivery time",
    "card"          : "payment method",
    "cards"         : "payment method",
    "pay"           : "payment method",
    "paying"        : "payment method",
    "cancel"        : "cancel order",
    "cancellation"  : "cancel order",
    "modify"        : "change order",
    "broken"        : "damaged defective",
    "damaged"       : "damaged defective",
    "defective"     : "damaged defective",
    "lost"          : "missing package lost",
    "missing"       : "missing package lost",
    "safe"          : "secure privacy",
    "secure"        : "secure privacy",
    "swap"          : "exchange size color",
    "different size": "exchange size color",
    "coupon"        : "discount promo code",
    "promo"         : "discount promo code",
    "sold out"      : "restock available",
    "back in stock" : "restock available",
    "warranty"      : "warranty guarantee",
    "guarantee"     : "warranty guarantee",
    "gift"          : "gift wrapping",
    "bulk"          : "wholesale discount",
    "wholesale"     : "wholesale discount",
    "sign up"       : "create account register",
    "register"      : "create account register",
    "forgot"        : "reset password forgot",
    "password"      : "reset password forgot",
    "track"         : "track order",
    "tracking"      : "track order",
    "where is"      : "track order",
    "international" : "international shipping worldwide",
    "overseas"      : "international shipping worldwide",
    "abroad"        : "international shipping worldwide",
}

# ── 1. Load FAQs from CSV ──────────────────────────────────────────────
def load_faqs(filepath="faqs.csv"):
    faqs = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            faqs.append({
                "question": row["question"],
                "answer"  : row["answer"]
            })
    return faqs

# ── 2. Expand synonyms in text ─────────────────────────────────────────
def expand_synonyms(text):
    text_lower = text.lower().strip()
    # Sort by length so longer phrases match first
    for phrase in sorted(SYNONYMS.keys(), key=len, reverse=True):
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, SYNONYMS[phrase])
    return text_lower

# ── 3. Preprocess text using NLTK ─────────────────────────────────────
def preprocess(text):
    text = expand_synonyms(text)
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha()
        and t not in stop_words
    ]
    return " ".join(tokens)

# ── 4. Build TF-IDF matrix ─────────────────────────────────────────────
def build_vectorizer(faqs):
    # Combine question + answer for richer matching
    documents = [
        faq["question"] + " " + faq["answer"]
        for faq in faqs
    ]
    preprocessed = [preprocess(doc) for doc in documents]

    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        sublinear_tf=True     # dampens effect of very frequent words
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed)

    return vectorizer, tfidf_matrix

# ── 5. Find best match using cosine similarity ─────────────────────────
def get_best_match(user_query, faqs, vectorizer, tfidf_matrix, threshold=0.05):
    cleaned_query = preprocess(user_query)
    query_vec = vectorizer.transform([cleaned_query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < threshold:
        return {
            "answer"          : "Sorry, I couldn't find a good match. Try rephrasing or ask about orders, shipping, returns, or payments.",
            "matched_question": None,
            "score"           : round(float(best_score), 4)
        }

    return {
        "answer"          : faqs[best_index]["answer"],
        "matched_question": faqs[best_index]["question"],
        "score"           : round(float(best_score), 4)
    }