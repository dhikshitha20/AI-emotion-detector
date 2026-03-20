import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from transformers import pipeline
from fpdf import FPDF
import datetime
import tempfile
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

MONGO_URI = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "admin123"))

@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client
    except ConnectionFailure:
        return None

def log_to_mongo(text, top_emotion, confidence, all_scores):
    client = get_mongo_client()
    if client is None:
        return False
    try:
        db = client["emotion_detector"]
        collection = db["logs"]
        collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "input_text": text,
            "dominant_emotion": top_emotion,
            "confidence": confidence,
            "all_scores": all_scores,
            "word_count": len(text.split())
        })
        return True
    except Exception:
        return False

def fetch_logs(limit=50):
    client = get_mongo_client()
    if client is None:
        return []
    try:
        db = client["emotion_detector"]
        collection = db["logs"]
        logs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        return logs
    except Exception:
        return []

def delete_all_logs():
    client = get_mongo_client()
    if client is None:
        return False
    try:
        db = client["emotion_detector"]
        collection = db["logs"]
        collection.delete_many({})
        return True
    except Exception:
        return False

st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #FDF6F0 !important; }
    [data-testid="stAppViewContainer"] { background-color: #FDF6F0 !important; }
    [data-testid="stHeader"] { background-color: #FDF6F0 !important; }
    [data-testid="stSidebar"] { background-color: #FFF0E8 !important; }
    html, body, p, label, h1, h2, h3, h4, h5 {
        color: #1A1A1A !important;
        font-family: 'Georgia', serif !important;
    }
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1.5px solid #E8C9B0 !important;
        border-radius: 10px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFF0E8 !important;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1A1A1A !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background-color: #C96A8A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    .stButton > button:hover { background-color: #B05575 !important; }
    .stDownloadButton > button {
        background-color: #2E7D5E !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    .emotion-card {
        background: #FFF8F4 !important;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E8C9B0;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: #FFF8F4 !important;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #E8C9B0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .log-card {
        background: #F5F0FF !important;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #D0BEF8;
        margin: 8px 0;
        font-size: 13px;
    }
    .stSpinner > div { border-top-color: #C96A8A !important; }
    [data-testid="stFileUploader"] {
        background-color: #FFF8F4 !important;
        border: 1.5px dashed #E8C9B0 !important;
        border-radius: 10px !important;
    }
    .stAlert { background-color: #FFF0E8 !important; color: #1A1A1A !important; }
    hr { border-color: #E8C9B0 !important; }
    .mongo-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .mongo-ok { background: #D4EDDA; color: #155724; }
    .mongo-fail { background: #F8D7DA; color: #721C24; }
</style>
""", unsafe_allow_html=True)

EMOTION_CONFIG = {
    "joy":      {"emoji": "😊", "color": "#2E7D5E", "label": "Joy"},
    "sadness":  {"emoji": "😢", "color": "#3B6CB7", "label": "Sadness"},
    "anger":    {"emoji": "😠", "color": "#C0392B", "label": "Anger"},
    "fear":     {"emoji": "😨", "color": "#7B5EA7", "label": "Fear"},
    "surprise": {"emoji": "😲", "color": "#D4860A", "label": "Surprise"},
    "disgust":  {"emoji": "🤢", "color": "#2A7A6F", "label": "Disgust"},
    "neutral":  {"emoji": "😐", "color": "#6B7280", "label": "Neutral"},
}

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("<h1 style='text-align:center;margin-bottom:4px;color:#1A1A1A !important;'>🧠 AI Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555555;margin-bottom:8px;font-size:16px;'>Detect emotions from text using advanced AI</p>", unsafe_allow_html=True)

mongo_client = get_mongo_client()
if mongo_client:
    st.markdown("<div style='text-align:center'><span class='mongo-badge mongo-ok'>🍃 MongoDB Connected</span></div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:center'><span class='mongo-badge mongo-fail'>⚠️ MongoDB Not Connected — logs won't be saved</span></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("Loading AI model..."):
    classifier = load_model()

st.markdown("### Analyze Text")
tab1, tab2 = st.tabs(["✏️ Type Text", "📄 Upload File"])

input_text = ""

with tab1:
    input_text = st.text_area(
        "Enter your text here",
        placeholder="Type something like: I am so excited about my internship interview tomorrow!",
        height=150,
        label_visibility="collapsed"
    )

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area("File content", input_text, height=150, disabled=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Emotion", use_container_width=True)

if analyze_btn and input_text.strip():
    with st.spinner("Analyzing emotions..."):
        results = classifier(input_text[:512])[0]
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        top = results_sorted[0]
        top_config = EMOTION_CONFIG.get(top["label"], {"emoji": "🤔", "color": "#555", "label": top["label"]})
        all_scores = {r["label"]: round(r["score"] * 100, 1) for r in results_sorted}

        logged = log_to_mongo(
            text=input_text,
            top_emotion=top_config["label"],
            confidence=round(top["score"] * 100, 1),
            all_scores=all_scores
        )
        if logged:
            st.toast("✅ Logged to MongoDB", icon="🍃")
        else:
            st.toast("⚠️ Could not log to MongoDB", icon="⚠️")

        st.session_state.history.append({
            "text": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "emotion": top_config["label"],
            "emoji": top_config["emoji"],
            "confidence": round(top["score"] * 100, 1),
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "all_scores": all_scores
        })

        st.markdown("---")
        st.markdown("### Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:48px'>{top_config['emoji']}</div>
                <div style='font-size:22px;font-weight:600;color:{top_config['color']}'>{top_config['label']}</div>
                <div style='color:#555555;font-size:13px'>Dominant Emotion</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:36px;font-weight:700;color:{top_config['color']}'>{round(top['score']*100,1)}%</div>
                <div style='color:#555555;font-size:13px'>Confidence Score</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:36px;font-weight:700;color:#1A1A1A'>{len(input_text.split())}</div>
                <div style='color:#555555;font-size:13px'>Words Analyzed</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Emotion Breakdown")
        labels = [EMOTION_CONFIG.get(r["label"], {"label": r["label"]})["label"] for r in results_sorted]
        scores = [round(r["score"] * 100, 1) for r in results_sorted]
        colors = [EMOTION_CONFIG.get(r["label"], {"color": "#999"})["color"] for r in results_sorted]

        fig = go.Figure(go.Bar(
            x=scores, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{s}%" for s in scores],
            textposition="outside"
        ))
        fig.update_layout(
            plot_bgcolor="#FDF6F0", paper_bgcolor="#FDF6F0",
            xaxis=dict(range=[0, 110], showgrid=False, title="Confidence (%)", color="#1A1A1A"),
            yaxis=dict(autorange="reversed", color="#1A1A1A"),
            height=320, margin=dict(l=20, r=40, t=20, b=20),
            font=dict(family="Georgia, serif", size=13, color="#1A1A1A")
        )
        st.plotly_chart(fig, use_container_width=True)

        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 12, "AI Emotion Detector - Analysis Report", ln=True, align="C")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True, align="C")
            pdf.ln(8)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Analyzed Text:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, input_text[:500])
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, f"Dominant Emotion: {top_config['label']} ({round(top['score']*100,1)}%)", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "All Emotion Scores:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            for r in results_sorted:
                label = EMOTION_CONFIG.get(r["label"], {"label": r["label"]})["label"]
                pdf.cell(0, 8, f"  {label}: {round(r['score']*100,1)}%", ln=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                pdf.output(f.name)
                return f.name

        pdf_path = generate_pdf()
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name=f"emotion_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        os.unlink(pdf_path)

elif analyze_btn:
    st.warning("Please enter some text to analyze!")

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Session History")

    if len(st.session_state.history) > 1:
        st.markdown("#### Emotion Trend")
        df = pd.DataFrame(st.session_state.history)
        fig2 = go.Figure(go.Scatter(
            x=list(range(1, len(df)+1)),
            y=df["confidence"],
            mode="lines+markers+text",
            text=df["emoji"],
            textposition="top center",
            line=dict(color="#C96A8A", width=2),
            marker=dict(size=10, color="#C96A8A")
        ))
        fig2.update_layout(
            plot_bgcolor="#FDF6F0", paper_bgcolor="#FDF6F0",
            xaxis=dict(title="Analysis #", showgrid=False, color="#1A1A1A"),
            yaxis=dict(title="Confidence (%)", range=[0, 110], color="#1A1A1A"),
            height=250, margin=dict(l=20, r=20, t=20, b=40),
            font=dict(color="#1A1A1A")
        )
        st.plotly_chart(fig2, use_container_width=True)

    for h in reversed(st.session_state.history):
        cfg = EMOTION_CONFIG.get(h["emotion"].lower(), {"emoji": "🤔", "color": "#555"})
        st.markdown(f"""
        <div class='emotion-card'>
            <span style='font-size:20px'>{h['emoji']}</span>
            <span style='font-weight:600;color:{cfg.get('color','#555')};margin-left:8px'>{h['emotion']}</span>
            <span style='color:#555555;font-size:13px;margin-left:8px'>({h['confidence']}% confidence)</span>
            <span style='float:right;color:#555555;font-size:12px'>{h['timestamp']}</span>
            <br><span style='color:#1A1A1A;font-size:13px'>{h['text']}</span>
        </div>""", unsafe_allow_html=True)

    if st.button("🗑️ Clear Session History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
with st.expander("🔐 Admin — View MongoDB Logs"):
    admin_pass = st.text_input("Enter admin password", type="password", key="admin_pass")


    if admin_pass == ADMIN_PASSWORD:
        st.success("Access granted!")

        col_a, col_b = st.columns([3, 1])
        with col_a:
            limit = st.slider("Number of recent logs to show", 5, 100, 20)
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Logs"):
                st.rerun()

        logs = fetch_logs(limit=limit)

        if logs:
            st.markdown(f"**{len(logs)} log(s) found**")

            emotions = [l["dominant_emotion"] for l in logs]
            emotion_counts = pd.Series(emotions).value_counts()

            fig3 = go.Figure(go.Pie(
                labels=emotion_counts.index,
                values=emotion_counts.values,
                hole=0.4,
                marker_colors=[EMOTION_CONFIG.get(e.lower(), {"color": "#999"})["color"] for e in emotion_counts.index]
            ))
            fig3.update_layout(
                title="Emotion Distribution (All Logs)",
                paper_bgcolor="#FDF6F0",
                font=dict(color="#1A1A1A"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig3, use_container_width=True)

            for log in logs:
                ts = log.get("timestamp", "")
                if isinstance(ts, datetime.datetime):
                    ts = ts.strftime("%d %b %Y, %H:%M:%S UTC")
                emotion = log.get("dominant_emotion", "Unknown")
                conf = log.get("confidence", 0)
                text = log.get("input_text", "")[:120]
                cfg = EMOTION_CONFIG.get(emotion.lower(), {"emoji": "🤔", "color": "#555"})
                st.markdown(f"""
                <div class='log-card'>
                    <span style='font-weight:600;color:{cfg.get('color','#555')}'>{cfg.get('emoji','🤔')} {emotion}</span>
                    <span style='color:#555;margin-left:8px'>({conf}% confidence)</span>
                    <span style='float:right;color:#888;font-size:11px'>{ts}</span>
                    <br><span style='color:#333'>{text}{'...' if len(log.get('input_text','')) > 120 else ''}</span>
                </div>""", unsafe_allow_html=True)

            df_logs = pd.DataFrame(logs)
            csv = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Logs as CSV",
                data=csv,
                file_name=f"emotion_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            st.markdown("---")
            if st.button("🗑️ Delete ALL Logs from MongoDB", type="primary"):
                if delete_all_logs():
                    st.success("All logs deleted!")
                    st.rerun()
                else:
                    st.error("Failed to delete logs.")
        else:
            st.info("No logs found in MongoDB yet.")

    elif admin_pass:
        st.error("Wrong password!")
