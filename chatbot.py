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

# ── Synonym map — expands user words to FAQ vocabulary ────────────────
SYNONYMS = {
    "send back"   : "return",
    "sending back": "return",
    "refund"      : "return",
    "give back"   : "return",
    "ship"        : "shipping delivery",
    "shipping"    : "shipping delivery",
    "arrive"      : "delivery",
    "when"        : "delivery time",
    "card"        : "payment",
    "cards"       : "payment",
    "pay"         : "payment",
    "paying"      : "payment",
    "cancel"      : "cancel order",
    "cancellation": "cancel order",
    "modify"      : "change order",
    "change"      : "change order",
    "broken"      : "damaged defective",
    "damaged"     : "damaged defective",
    "defective"   : "damaged defective",
    "lost"        : "missing package",
    "missing"     : "missing package",
    "safe"        : "secure privacy",
    "secure"      : "secure privacy",
    "privacy"     : "secure privacy",
    "swap"        : "exchange",
    "exchange"    : "exchange",
    "different size": "exchange",
    "coupon"      : "discount promo code",
    "promo"       : "discount promo code",
    "discount"    : "discount promo code",
    "back in stock": "restock",
    "sold out"    : "restock",
    "warranty"    : "warranty guarantee",
    "guarantee"   : "warranty guarantee",
    "gift"        : "gift wrapping",
    "wrap"        : "gift wrapping",
    "bulk"        : "wholesale discount",
    "wholesale"   : "wholesale discount",
    "store"       : "physical store location",
    "location"    : "physical store location",
    "account"     : "create account login",
    "login"       : "create account login",
    "sign up"     : "create account",
    "password"    : "reset password forgot",
    "forgot"      : "reset password forgot",
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
    text_lower = text.lower()
    for phrase, replacement in SYNONYMS.items():
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, replacement)
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