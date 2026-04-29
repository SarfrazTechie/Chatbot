import streamlit as st
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.matcher import FAQMatcher
from utils.logger import QueryLogger
from utils.ai_engine import get_ai_answer

st.set_page_config(page_title="FAQ Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #212121 !important;
    font-family: 'Inter', sans-serif !important;
    color: #ececec !important;
}
[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
[data-testid="stSidebar"] { background: #171717 !important; border-right: 1px solid #2f2f2f !important; }
[data-testid="stSidebar"] * { color: #ececec !important; }
[data-testid="stMainBlockContainer"] { max-width: 800px !important; margin: 0 auto !important; padding-bottom: 100px !important; }
.stButton > button { background: transparent !important; border: none !important; border-radius: 8px !important; color: #ececec !important; font-size: 13px !important; text-align: left !important; padding: 8px 12px !important; }
.stButton > button:hover { background: #2f2f2f !important; }
[data-testid="stChatInput"] { background: #2f2f2f !important; border-radius: 14px !important; border: 1px solid #3f3f3f !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; border: none !important; color: #ececec !important; font-size: 15px !important; caret-color: #ececec !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #555 !important; }
[data-testid="stChatInputSubmitButton"] button { background: #ececec !important; border-radius: 8px !important; color: #212121 !important; }

/* USER bubble — RIGHT */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 8px 0;
}
.bubble-user {
    background: #2f2f2f;
    color: #ececec;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 60%;
    font-size: 15px;
    line-height: 1.7;
    word-wrap: break-word;
}

/* BOT bubble — LEFT */
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 8px 0;
}
.bubble-bot {
    background: transparent;
    color: #ececec;
    padding: 12px 0;
    max-width: 70%;
    font-size: 15px;
    line-height: 1.75;
    word-wrap: break-word;
}

.src-row { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
.src-pill { font-size:11px; padding:2px 9px; border-radius:20px; background:#2a2a2a; border:1px solid #3a3a3a; color:#777; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #3f3f3f; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_matcher():
    return FAQMatcher(faq_path="data/faqs.json", confidence_threshold=0.05)

@st.cache_resource
def load_logger():
    return QueryLogger(log_dir="logs")

matcher      = load_matcher()
query_logger = load_logger()

if "messages"     not in st.session_state: st.session_state.messages     = []
if "api_messages" not in st.session_state: st.session_state.api_messages = []
if "prefill"      not in st.session_state: st.session_state.prefill      = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 FAQ Assistant")
    st.caption("Powered by Llama AI")
    st.divider()

    stats = query_logger.get_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total",   stats.get('total_queries', 0))
    col2.metric("Matched", stats.get('successful', 0))
    col3.metric("Failed",  stats.get('fallback_count', 0))

    st.divider()
    st.markdown("**Suggested**")
    samples = [
        "How do I reset my password?",
        "What payment methods accepted?",
        "Do you offer refunds?",
        "How is my data protected?",
        "Is there a mobile app?",
        "What is in the free plan?",
        "How to cancel subscription?",
        "Is there a free trial?",
    ]
    for s in samples:
        if st.button(s, key=f"sq_{s}", use_container_width=True):
            st.session_state.prefill = s

    st.divider()
    if st.button("🗑  New conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.api_messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#ececec;font-size:30px;font-weight:600;margin-bottom:8px'>How can I help you today?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;font-size:15px;margin-bottom:40px'>Ask me anything about accounts, billing, features or support</p>", unsafe_allow_html=True)

# ── Render messages manually so alignment works 100% ─────────────────────────
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        text = msg['content'].replace('<','&lt;').replace('>','&gt;')
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble-user">{text}</div>
        </div>""", unsafe_allow_html=True)
    else:
        text = msg['content'].replace('<','&lt;').replace('>','&gt;').replace('\n','<br>')
        sources_html = ""
        if msg.get('sources'):
            pills = "".join(f"<span class='src-pill'>↗ {s[:50]}{'…' if len(s)>50 else ''}</span>" for s in msg['sources'])
            sources_html = f"<div class='src-row'>{pills}</div>"
        st.markdown(f"""
        <div class="msg-bot">
            <div class="bubble-bot">{text}{sources_html}</div>
        </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input("Message FAQ Assistant…")
if prefill and not user_input:
    user_input = prefill

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    with st.spinner(""):
        candidates = matcher.get_top_matches(user_input, top_n=5)
        ai_result  = get_ai_answer(
            user_query=user_input,
            faq_candidates=candidates,
            chat_history=st.session_state.api_messages
        )

    st.session_state.api_messages.append({"role": "user",      "content": user_input})
    st.session_state.api_messages.append({"role": "assistant",  "content": ai_result['answer']})

    query_logger.log(user_input, {
        'answer':           ai_result['answer'],
        'matched_question': ai_result['sources'][0] if ai_result['sources'] else None,
        'confidence':       candidates[0]['confidence'] if candidates else 0,
        'category':         candidates[0]['category']   if candidates else 'Unknown',
        'is_fallback':      ai_result['used_fallback'],
    })

    st.session_state.messages.append({
        'role':         'bot',
        'content':      ai_result['answer'],
        'sources':      ai_result['sources'],
        'used_fallback': ai_result['used_fallback'],
    })
    st.rerun()
