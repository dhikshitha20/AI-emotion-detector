import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from transformers import pipeline
from fpdf import FPDF
import datetime
import tempfile
import os

# Page config
st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS - Force beige light theme
st.markdown("""
<style>
    /* Force entire app beige */
    .stApp {
        background-color: #FDF6F0 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #FDF6F0 !important;
    }
    [data-testid="stHeader"] {
        background-color: #FDF6F0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFF0E8 !important;
    }
    /* All text black */
    html, body, p, span, div, label, h1, h2, h3, h4, h5 {
        color: #1A1A1A !important;
        font-family: 'Georgia', serif !important;
    }
    /* Text area and inputs */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1.5px solid #E8C9B0 !important;
        border-radius: 10px !important;
    
    }
    /* Tabs */
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
    /* Button */
    .stButton > button {
        background-color: #C96A8A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    .stButton > button:hover {
        background-color: #B05575 !important;
    }
    /* Download button */
    .stDownloadButton > button {
        background-color: #2E7D5E !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    /* Cards */
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
    /* Spinner */
    .stSpinner > div {
        border-top-color: #C96A8A !important;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFF8F4 !important;
        border: 1.5px dashed #E8C9B0 !important;
        border-radius: 10px !important;
    }
    /* Warning */
    .stAlert {
        background-color: #FFF0E8 !important;
        color: #1A1A1A !important;
    }
    /* Divider */
    hr {
        border-color: #E8C9B0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Emotion config
EMOTION_CONFIG = {
    "joy":      {"emoji": "😊", "color": "#2E7D5E", "label": "Joy"},
    "sadness":  {"emoji": "😢", "color": "#3B6CB7", "label": "Sadness"},
    "anger":    {"emoji": "😠", "color": "#C0392B", "label": "Anger"},
    "fear":     {"emoji": "😨", "color": "#7B5EA7", "label": "Fear"},
    "surprise": {"emoji": "😲", "color": "#D4860A", "label": "Surprise"},
    "disgust":  {"emoji": "🤢", "color": "#2A7A6F", "label": "Disgust"},
    "neutral":  {"emoji": "😐", "color": "#6B7280", "label": "Neutral"},
}

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("<h1 style='text-align:center;margin-bottom:4px;color:#1A1A1A !important;'>🧠 AI Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555555;margin-bottom:32px;font-size:16px;'>Detect emotions from text using advanced AI</p>", unsafe_allow_html=True)

# Load model
with st.spinner("Loading AI model..."):
    classifier = load_model()

# Input section
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

# Analyze button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Emotion", use_container_width=True)

if analyze_btn and input_text.strip():
    with st.spinner("Analyzing emotions..."):
        results = classifier(input_text[:512])[0]
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        top = results_sorted[0]
        top_config = EMOTION_CONFIG.get(top["label"], {"emoji": "🤔", "color": "#555", "label": top["label"]})

        # Save to history
        st.session_state.history.append({
            "text": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "emotion": top_config["label"],
            "emoji": top_config["emoji"],
            "confidence": round(top["score"] * 100, 1),
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "all_scores": {r["label"]: round(r["score"] * 100, 1) for r in results_sorted}
        })

        # Result display
        st.markdown("---")
        st.markdown("### Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:48px'>{top_config['emoji']}</div>
                <div style='font-size:22px;font-weight:600;color:{top_config['color']}'>{top_config['label']}</div>
                <div style='color:#555555;font-size:13px'>Dominant Emotion</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:36px;font-weight:700;color:{top_config['color']}'>{round(top['score']*100,1)}%</div>
                <div style='color:#555555;font-size:13px'>Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:36px;font-weight:700;color:#1A1A1A'>{len(input_text.split())}</div>
                <div style='color:#555555;font-size:13px'>Words Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        # Bar chart
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

        # PDF Download
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

# History section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Analysis History")

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

    for i, h in enumerate(reversed(st.session_state.history)):
        cfg = EMOTION_CONFIG.get(h["emotion"].lower(), {"emoji": "🤔", "color": "#555"})
        st.markdown(f"""
        <div class='emotion-card'>
            <span style='font-size:20px'>{h['emoji']}</span>
            <span style='font-weight:600;color:{cfg.get('color','#555')};margin-left:8px'>{h['emotion']}</span>
            <span style='color:#555555;font-size:13px;margin-left:8px'>({h['confidence']}% confidence)</span>
            <span style='float:right;color:#555555;font-size:12px'>{h['timestamp']}</span>
            <br><span style='color:#1A1A1A;font-size:13px'>{h['text']}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
