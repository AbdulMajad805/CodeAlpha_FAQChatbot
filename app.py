import streamlit as st
from chatbot import load_faqs, build_vectorizer, get_best_match

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="🤖",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Syne:wght@700&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* ── Hide Streamlit default UI ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Header ── */
.chat-header {
    background: #0F6E56;
    padding: 20px 24px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.chat-header::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.chat-header h1 {
    font-family: 'Syne', sans-serif;
    color: white;
    font-size: 20px;
    margin: 0;
}
.chat-header p {
    color: rgba(255,255,255,0.72);
    font-size: 12px;
    margin: 4px 0 0 0;
}
.avatar {
    width: 46px; height: 46px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
    border: 1.5px solid rgba(255,255,255,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: #0F6E56;
    color: white;
    padding: 11px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 4px 0 4px auto;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.65;
    width: fit-content;
    margin-left: auto;
}
.bot-bubble {
    background: white;
    color: #1a1a1a;
    padding: 11px 16px;
    border-radius: 18px 18px 18px 4px;
    border: 1px solid #e4eae8;
    margin: 4px auto 4px 0;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.65;
    width: fit-content;
}
.meta-info {
    font-size: 11px;
    color: #8fa89e;
    margin: 2px 0 12px 4px;
}

/* ── Suggestion buttons ── */
.stButton > button {
    background: #f0faf6 !important;
    color: #085041 !important;
    border: 1px solid #d4e8e0 !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 5px 14px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    background: #1D9E75 !important;
    color: white !important;
    border-color: #1D9E75 !important;
}

/* ── Chat input ── */
.stChatInput > div {
    border-radius: 24px !important;
    border: 1.5px solid #e0eae6 !important;
}
.stChatInput > div:focus-within {
    border-color: #1D9E75 !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model once ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    faqs = load_faqs("faqs.csv")
    vectorizer, tfidf_matrix = build_vectorizer(faqs)
    return faqs, vectorizer, tfidf_matrix

faqs, vectorizer, tfidf_matrix = load_model()

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="avatar">🤖</div>
    <div>
        <h1>FAQ Assistant</h1>
        <p>E-commerce Support · Powered by NLTK + TF-IDF + Cosine Similarity</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Initialize session state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role"   : "bot",
        "text"   : "Hi there! 👋 I'm your FAQ Assistant. Tap a suggestion below or ask me anything about orders, shipping, returns, or payments!",
        "matched": None,
        "score"  : None
    }]
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# ── Suggestion buttons ─────────────────────────────────────────────────
if st.session_state.show_suggestions:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📦 Track order"):
            st.session_state.pending = "How do I track my order?"
    with col2:
        if st.button("↩️ Returns"):
            st.session_state.pending = "What is your return policy?"
    with col3:
        if st.button("🌍 Shipping"):
            st.session_state.pending = "Do you offer international shipping?"
    with col4:
        if st.button("💬 Support"):
            st.session_state.pending = "How do I contact support?"

# ── Display chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["text"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">{msg["text"]}</div>',
            unsafe_allow_html=True
        )
        if msg["matched"]:
            pct = round(msg["score"] * 100)
            st.markdown(
                f'<div class="meta-info">Matched: "{msg["matched"]}" · {pct}% confidence</div>',
                unsafe_allow_html=True
            )

# ── Handle a question ──────────────────────────────────────────────────
def handle_question(question):
    st.session_state.show_suggestions = False
    st.session_state.messages.append({
        "role": "user",
        "text": question
    })
    result = get_best_match(question, faqs, vectorizer, tfidf_matrix)
    st.session_state.messages.append({
        "role"   : "bot",
        "text"   : result["answer"],
        "matched": result["matched_question"],
        "score"  : result["score"]
    })

# ── Handle suggestion click ────────────────────────────────────────────
if "pending" in st.session_state:
    handle_question(st.session_state.pending)
    del st.session_state.pending
    st.rerun()

# ── Chat input at the bottom ───────────────────────────────────────────
question = st.chat_input("Ask a question about your order...")
if question:
    handle_question(question)
    st.rerun()