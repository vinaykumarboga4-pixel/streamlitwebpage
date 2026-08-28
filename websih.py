"""
VoiceShield — Real-time voice clone detection demo
SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import time
from datetime import datetime

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

st.set_page_config(page_title="VoiceShield", page_icon="🛡️", layout="centered")

MODEL_NAME = "facebook/wav2vec2-base-960h"   # swap for an anti-spoofing checkpoint if you find one
CHUNK_SECONDS = 2.5
SAMPLE_RATE = 16000
RISK_THRESHOLD = 55  # % above which we treat audio as a likely clone

# --------------------------------------------------------------------------
# Model loading (cached so it only downloads/loads once per session)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading detection model (first run downloads ~360MB)...")
def load_model():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    return extractor, model


@st.cache_resource(show_spinner=False)
def load_classifier_head(embedding_dim=768):
    """
    Lightweight classifier head on top of wav2vec2 embeddings.
    NOTE: this is randomly initialized. For a real result you must fine-tune
    it on a labeled dataset (e.g. ASVspoof) before the demo — see the
    train_classifier.py script referenced in the README. Until then this
    acts as a stand-in so the full pipeline runs end-to-end.
    """
    torch.manual_seed(42)
    head = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid(),
    )
    head.eval()
    return head


# --------------------------------------------------------------------------
# Audio pipeline
# --------------------------------------------------------------------------

def load_audio(file) -> np.ndarray:
    audio, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)
    return audio


def chunk_audio(audio: np.ndarray, sr: int = SAMPLE_RATE, chunk_seconds: float = CHUNK_SECONDS):
    chunk_len = int(chunk_seconds * sr)
    chunks = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < sr * 0.5:  # skip trailing sliver shorter than 0.5s
            continue
        chunks.append(chunk)
    return chunks if chunks else [audio]


def score_chunk(chunk, extractor, model, head) -> float:
    inputs = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # mean-pool over time
        risk = head(embedding).item()
    return round(risk * 100, 1)


def analyze_audio(audio, extractor, model, head, progress_callback=None):
    chunks = chunk_audio(audio)
    scores = []
    for i, chunk in enumerate(chunks):
        score = score_chunk(chunk, extractor, model, head)
        scores.append(score)
        if progress_callback:
            progress_callback(i + 1, len(chunks), score)
    return scores


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_verification" not in st.session_state:
    st.session_state.pending_verification = None

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .risk-badge {padding: 4px 10px; border-radius: 999px; font-family: monospace; font-size: 0.85rem;}
    .safe {background: rgba(45,212,191,0.15); color: #2DD4BF;}
    .warn {background: rgba(245,166,35,0.15); color: #F5A623;}
    .danger {background: rgba(239,68,68,0.15); color: #EF4444;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ VoiceShield")
st.caption("Real-time voice clone detection for phone calls — SIH26104")

extractor, model = load_model()
head = load_classifier_head()

st.warning(
    "The classifier head is currently untrained (random weights) — wire in a "
    "fine-tuned checkpoint before relying on real scores. See README.",
    icon="⚠️",
)

tab_live, tab_history = st.tabs(["Live analysis", "Call history"])

# ---- Live analysis tab ----------------------------------------------------
with tab_live:
    st.subheader("Analyze a call clip")
    source = st.radio("Audio source", ["Upload a file", "Record with mic"], horizontal=True)

    audio_file = None
    if source == "Upload a file":
        audio_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
    else:
        audio_file = st.audio_input("Record audio")

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Run detection", type="primary"):
            audio = load_audio(audio_file)

            progress_bar = st.progress(0, text="Analyzing chunks...")
            chart_placeholder = st.empty()
            scores = []

            def update_progress(done, total, score):
                scores.append(score)
                progress_bar.progress(done / total, text=f"Chunk {done}/{total} — risk {score}%")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=scores, mode="lines+markers",
                    line=dict(color="#2DD4BF" if max(scores) < RISK_THRESHOLD else "#EF4444"),
                ))
                fig.update_layout(
                    height=250, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(title="Clone probability (%)", range=[0, 100]),
                    xaxis=dict(title="Chunk"),
                    template="plotly_dark",
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.15)  # small delay so the live update is visible in-demo

            scores = analyze_audio(audio, extractor, model, head, update_progress)
            progress_bar.empty()

            overall_risk = round(max(scores), 1) if scores else 0
            verdict = "Likely cloned" if overall_risk >= RISK_THRESHOLD else "Likely genuine"
            badge_class = "danger" if overall_risk >= RISK_THRESHOLD else "safe"

            st.markdown(
                f'<span class="risk-badge {badge_class}">{verdict} · {overall_risk}% peak risk</span>',
                unsafe_allow_html=True,
            )

            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "filename": getattr(audio_file, "name", "mic recording"),
                "risk": overall_risk,
                "verdict": verdict,
                "outcome": "pending",
            }

            if overall_risk >= RISK_THRESHOLD:
                st.error("⚠️ Possible voice clone detected — verify caller identity.", icon="🚨")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Identity confirmed"):
                        entry["outcome"] = "verified"
                        st.session_state.history.insert(0, entry)
                        st.rerun()
                with col2:
                    if st.button("🚫 Flag as fraud"):
                        entry["outcome"] = "blocked"
                        st.session_state.history.insert(0, entry)
                        st.rerun()
            else:
                st.success("✅ Voice verified — no action needed.")
                entry["outcome"] = "verified"
                st.session_state.history.insert(0, entry)

# ---- History tab -----------------------------------------------------------
with tab_history:
    st.subheader("Call history")
    if not st.session_state.history:
        st.caption("No calls analyzed yet.")
    else:
        for entry in st.session_state.history:
            badge_class = "danger" if entry["outcome"] == "blocked" else "safe"
            st.markdown(
                f"**{entry['filename']}** — {entry['timestamp']}  "
                f'<span class="risk-badge {badge_class}">{entry["risk"]}% · {entry["outcome"]}</span>',
                unsafe_allow_html=True,
            )
            st.divider()
