from flask import Flask, render_template, request, jsonify
from chatbot import load_faqs, build_vectorizer, get_best_match

app = Flask(__name__)

# ── Load FAQs and build TF-IDF matrix once at startup ─────────────────
faqs = load_faqs("faqs.csv")
vectorizer, tfidf_matrix = build_vectorizer(faqs)

# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "").strip()

    if not user_question:
        return jsonify({"answer": "Please enter a question.", "matched": None, "score": 0})

    result = get_best_match(user_question, faqs, vectorizer, tfidf_matrix)

    return jsonify({
        "answer": result["answer"],
        "matched": result["matched_question"],
        "score": result["score"]
    })


if __name__ == "__main__":
    app.run(debug=True)