import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Ensure root folder is accessible for preprocessing
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import preprocess_text

# Page Configuration
st.set_page_config(
    page_title="Gen Z Sentiment & Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Minimalist Styling with Purple Accents
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #05050A;
        color: #E2E8F0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Glassmorphism Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(88, 28, 135, 0.4) 0%, rgba(15, 23, 42, 0.8) 50%, rgba(5, 5, 10, 0.95) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 20px;
        padding: 35px 30px;
        margin-bottom: 30px;
        box-shadow: 0 12px 40px -10px rgba(147, 51, 234, 0.3);
        text-align: center;
    }

    .hero-title {
        background: linear-gradient(90deg, #D8B4FE 0%, #C084FC 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
    }

    .hero-sub {
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
    }

    /* Glass Cards Container */
    .glass-card {
        background: #0D0D18;
        border: 1px solid #1E1B4B;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
    }

    /* Custom Text Input & Textarea Styling */
    .stTextInput input, .stTextArea textarea {
        background-color: #080811 !important;
        color: #F8FAFC !important;
        border: 1px solid #4C1D95 !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        padding: 12px 15px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #A855F7 !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.4) !important;
    }

    /* Glowing Primary Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7C3AED 0%, #A855F7 50%, #EC4899 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.5);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.7);
    }

    /* Result Glow Box */
    .output-glow-box {
        background: linear-gradient(180deg, #120D26 0%, #090614 100%);
        border: 1px solid #6D28D9;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 0 30px rgba(109, 40, 217, 0.25);
    }

    /* Bottom Info Card */
    .info-card {
        background: linear-gradient(135deg, #0F0C20 0%, #080612 100%);
        border: 1px solid #3B0764;
        border-radius: 16px;
        padding: 25px;
        margin-top: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Cache Model & Data Loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_dataset_stats():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'reviews.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        valid = ['Positive', 'Negative', 'Neutral', 'Mixed']
        return df[df['sentiment'].isin(valid)].copy()
    return None

model = load_model()
df_dataset = load_dataset_stats()

# Header Section
st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⚡ Context & Confidence AI Analytics</div>
        <div class="hero-sub">Single-Line Slang Decoder • Multi-Line Review Aggregation • Uncertainty Alerts</div>
    </div>
""", unsafe_allow_html=True)

# Helper Function to Render Output Box
def render_analysis_output(text_input):
    cleaned_text = preprocess_text(text_input)
    probabilities = model.predict_proba([cleaned_text])[0]
    classes = model.classes_
    max_idx = int(np.argmax(probabilities))
    
    sentiment = str(classes[max_idx])
    confidence = float(probabilities[max_idx])
    
    st.markdown('<div class="output-glow-box">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction Results")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Predicted Sentiment", sentiment)
    with m2:
        st.metric("Confidence Score", f"{round(confidence * 100, 2)}%")
    
    st.markdown("**Confidence Probability Gauge:**")
    st.progress(confidence)
    st.caption(f"**Preprocessed Tokens:** `{cleaned_text}`")
    
    st.markdown("---")
    
    if confidence < 0.65:
        st.warning("⚠️ **Low Confidence Alert**: Detected ambiguity, sarcasm, or context-dependent slang.")
    else:
        st.success("✅ **High Confidence Prediction**: Calibrated above safety threshold.")
    
    st.markdown("**Probability Class Breakdown:**")
    prob_df = pd.DataFrame({
        "Sentiment Class": classes,
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    }).sort_values(by="Probability (%)", ascending=False)
    
    st.dataframe(prob_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Analyzer", "📊 Dataset Analytics", "💡 How It Works"])

# TAB 1: REAL-TIME ANALYZER
with tab1:
    # SECTION 1: SINGLE-LINE COMMENT ANALYZER
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚡ 1. Single-Line Comment / Slang Analyzer")
    st.write("Ideal for quick 1-line feedback, slang expressions, or emoji comments.")
    
    col_s1, col_s2 = st.columns([1.1, 0.9], gap="large")
    with col_s1:
        single_input = st.text_input(
            "Enter a 1-line comment:",
            placeholder="e.g., 'I\\'m dead 💀', 'you ate', 'that\\'s sick 🔥', 'this app is mid'..."
        )
        single_btn = st.button("Analyze Single Comment ✨", key="btn_single")
    
    with col_s2:
        if single_btn and single_input.strip() != "":
            render_analysis_output(single_input)
        elif single_btn:
            st.error("Please enter a single-line comment first.")
        else:
            st.info("Enter a 1-line comment on the left to see instant predictions.")
    st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 2: MULTI-LINE REVIEW ANALYZER
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📝 2. Multi-Line Review / Paragraph Analyzer")
    st.write("Ideal for long student reviews containing multiple sentences or paragraphs.")
    
    col_m1, col_m2 = st.columns([1.1, 0.9], gap="large")
    with col_m1:
        multi_input = st.text_area(
            "Paste multi-line feedback:",
            placeholder=(
                "Example multi-line review:\n"
                "This phone update is absolutely lit 💀\n"
                "The UI design slays and feels super smooth.\n"
                "However the battery drain looks a bit sus fr."
            ),
            height=160
        )
        multi_btn = st.button("Analyze Multi-Line Review ✨", key="btn_multi")
    
    with col_m2:
        if multi_btn and multi_input.strip() != "":
            render_analysis_output(multi_input)
        elif multi_btn:
            st.error("Please paste a multi-line review first.")
        else:
            st.info("Paste a multi-line review on the left to see aggregated predictions.")
    st.markdown('</div>', unsafe_allow_html=True)

    # BOTTOM INFO SECTION
    st.markdown("""
        <div class="info-card">
            <h3>📌 System Capabilities & Processing Workflow</h3>
            <p>This sentiment analysis engine evaluates both single-line slang comments and multi-line paragraph reviews:</p>
            <ul>
                <li><b>Single-Line Decoding:</b> Instantly maps informal Gen Z expressions (e.g., <i>"lit"</i>, <i>"sus"</i>, <i>"you ate"</i>) to conversational sentiment.</li>
                <li><b>Multi-Sentence Context Aggregation:</b> Analyzes multi-line feedback blocks, balancing mixed or contrasting sentiment shifts across paragraphs.</li>
                <li><b>Emoji Demojization:</b> Converts non-verbal visual emojis like 💀, 🔥, and 😭 into explicit text tokens before classification[cite: 1].</li>
                <li><b>Confidence Calibration & Warning Alerts:</b> Computes probability distributions and flags low-confidence predictions when confidence falls below $65\%$[cite: 5].</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# TAB 2: DATASET ANALYTICS DASHBOARD
with tab2:
    st.subheader("📈 Dataset Overview & Distribution")
    if df_dataset is not None:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Corpus Rows", f"{len(df_dataset):,}")
        kpi2.metric("Positive Reviews", f"{len(df_dataset[df_dataset['sentiment']=='Positive']):,}")
        kpi3.metric("Negative Reviews", f"{len(df_dataset[df_dataset['sentiment']=='Negative']):,}")
        kpi4.metric("Neutral / Mixed", f"{len(df_dataset[df_dataset['sentiment'].isin(['Neutral', 'Mixed'])]):,}")
        
        st.markdown("---")
        
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.subheader("Sentiment Class Distribution")
            sentiment_counts = df_dataset['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
        with c2:
            st.subheader("Top Gen Z Slang Terms in Corpus")
            slang_counts = df_dataset['word'].value_counts().head(10)
            st.bar_chart(slang_counts)
    else:
        st.warning("Dataset file (`data/reviews.csv`) not found for analytics display.")

# TAB 3: HOW IT WORKS
with tab3:
    st.subheader("💡 Technical Architecture & Capabilities")
    st.markdown("""
    - **1. Context-Aware Gen Z Interpretation**: Decodes informal expressions whose literal dictionary meaning differs from conversational sentiment (e.g., distinguishing slang like *"you ate"* or *"that's sick 🔥"* from literal complaints)[cite: 1].
    - **2. Emoji Token Demojization**: Converts visual non-verbal cues like `💀` and `😭` into standard text tokens (`emoji_skull`, `emoji_loudly_crying_face`) prior to feature vectorization[cite: 1].
    - **3. Calibrated Confidence Scoring**: Uses multi-class probability estimation ($P(y\vert{}x)$) rather than raw binary flags to quantify classification certainty[cite: 5].
    - **4. Uncertainty Detection**: Automatically triggers low-confidence warning alerts whenever maximum probability falls below $65\%$, flagging ambiguous, sarcastic, or mixed feedback for human review[cite: 5].
    """)