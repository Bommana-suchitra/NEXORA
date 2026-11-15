"""
MediAssist AI - Next-Gen Ambient AI Platform
============================================
Features:
1. Ultra-Premium Light-Themed Glassmorphism UI (Soft white, ice blue, lavender, cyan).
2. Stunning 3D Image Cards for Mode Selection:
   - Medical Assistant: Photorealistic 3D healthcare AI visual with floating animation, glowing blue border, and hover lift.
   - General AI Chat: Photorealistic 3D AI chatbot visual with floating animation, soft purple glow, and hover lift.
3. Smooth Animations:
   - Floating visual animations.
   - Entrance fade-in and slide-up animations.
   - Hover scale, elevation, and glowing borders.
   - Smooth mode transition animations.
4. Preserved Core Functionality:
   - Medical Mode: Whisper STT, CLIP ViT-B/32, Edge-TTS (JennyNeural), dynamic 1-at-a-time follow-up questions, direct image analysis, healthy state detection, specialist department cards.
   - General Chat: Streaming responses, text & voice input, all-purpose knowledge, zero medical triage logic.
   - Navigation: Home, Mode Selection, and Back navigation.
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import os
import re
import tempfile
import time
from typing import Generator, List, Optional, Tuple

import clip
from dotenv import find_dotenv, load_dotenv
import edge_tts
from gtts import gTTS
from PIL import Image
from pydantic import BaseModel, Field
import streamlit as st
import torch
import torchvision.transforms as transforms
import whisper

from triagesim.llm import Request, _json_span, load_backend

# Load environment configuration
load_dotenv(find_dotenv(usecwd=True))

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="MediAssist AI • Next-Gen Ambient Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Handle Query Parameters for direct URL routing (?page=landing | mode_select | medical | general)
query_params = st.query_params
if "page" in query_params:
    st.session_state.current_page = query_params["page"]


# ==============================================================================
# ASSET CACHE (BASE64 3D IMAGES FOR SUB-SECOND RENDERING)
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_image_base64(path: str) -> str:
    """Loads and caches base64 representation of local images."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# Preload the 3D visual assets
med_img_b64 = get_image_base64("assets/medical_ai_3d.jpg")
gen_img_b64 = get_image_base64("assets/general_ai_3d.jpg")


# ==============================================================================
# ADVANCED LIGHT-THEMED GLASSMORPHISM CSS & ANIMATIONS
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');
    
    :root {
        --primary-blue: #0284c7;
        --primary-cyan: #06b6d4;
        --primary-indigo: #6366f1;
        --primary-purple: #8b5cf6;
        --primary-teal: #0d9488;
        --glass-surface: rgba(255, 255, 255, 0.78);
        --glass-border: rgba(255, 255, 255, 0.95);
        --glass-shadow: 0 16px 40px -6px rgba(15, 23, 42, 0.05), 0 4px 16px -2px rgba(15, 23, 42, 0.02);
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    /* Background Canvas: Ice blue, lavender, subtle cyan with neural dot grid */
    .stApp {
        background: linear-gradient(135deg, #f8faff 0%, #eff6ff 25%, #f5f3ff 50%, #ecfdf5 75%, #f8fafc 100%);
        background-attachment: fixed;
        color: #0f172a;
    }
    
    /* Neural Dot Pattern Overlay */
    .neural-grid-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: radial-gradient(rgba(99, 102, 241, 0.07) 1.2px, transparent 1.2px);
        background-size: 34px 34px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Ambient Animated Floating Gradient Light Orbs */
    .ambient-orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(65px);
        z-index: 0;
        pointer-events: none;
        opacity: 0.65;
    }
    
    .orb-cyan {
        top: -120px;
        left: 10%;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.32) 0%, rgba(255, 255, 255, 0) 70%);
        animation: floatOrb1 18s ease-in-out infinite alternate;
    }
    
    .orb-purple {
        top: 35%;
        right: 5%;
        width: 580px;
        height: 580px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.28) 0%, rgba(255, 255, 255, 0) 70%);
        animation: floatOrb2 22s ease-in-out infinite alternate;
    }
    
    .orb-blue {
        bottom: -80px;
        left: 20%;
        width: 480px;
        height: 480px;
        background: radial-gradient(circle, rgba(2, 132, 199, 0.28) 0%, rgba(255, 255, 255, 0) 70%);
        animation: floatOrb3 16s ease-in-out infinite alternate;
    }
    
    @keyframes floatOrb1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(40px, 30px) scale(1.08); }
        100% { transform: translate(-30px, -20px) scale(0.95); }
    }
    
    @keyframes floatOrb2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-45px, -35px) scale(1.12); }
        100% { transform: translate(30px, 25px) scale(0.92); }
    }
    
    @keyframes floatOrb3 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, -30px) scale(1.05); }
        100% { transform: translate(-25px, 35px) scale(0.96); }
    }
    
    /* Entrance Animation (Fade-In & Slide-Up) */
    .animate-entrance {
        animation: fadeInUp 0.75s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    .animate-delay-1 { animation-delay: 0.1s; }
    .animate-delay-2 { animation-delay: 0.2s; }
    .animate-delay-3 { animation-delay: 0.3s; }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(28px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Transparent Floating Glass Navbar */
    .glass-nav {
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 22px;
        padding: 14px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.9);
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        z-index: 10;
    }
    
    .glass-nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
    }
    
    .glass-nav-brand h2 {
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .glass-nav-links {
        display: flex;
        align-items: center;
        gap: 28px;
    }
    
    .glass-nav-link {
        color: #475569;
        font-size: 0.92rem;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    
    .glass-nav-link:hover {
        color: #0284c7;
        transform: translateY(-1px);
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 55px 20px 40px 20px;
        position: relative;
        z-index: 5;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(186, 230, 253, 0.9);
        color: #0284c7;
        padding: 7px 20px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.08);
        margin-bottom: 24px;
        animation: pulseBadge 3s ease-in-out infinite;
    }
    
    @keyframes pulseBadge {
        0%, 100% { box-shadow: 0 4px 18px rgba(2, 132, 199, 0.08); transform: translateY(0); }
        50% { box-shadow: 0 4px 26px rgba(99, 102, 241, 0.22); transform: translateY(-2px); }
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.1rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.035em;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 35%, #4f46e5 75%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #475569;
        max-width: 720px;
        margin: 0 auto 36px auto;
        line-height: 1.62;
        font-weight: 400;
    }
    
    /* Scroll Down Indicator */
    .scroll-indicator {
        margin: 18px auto 0 auto;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .mouse-pill {
        width: 22px;
        height: 36px;
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        position: relative;
    }
    
    .mouse-wheel {
        width: 4px;
        height: 8px;
        background: #0284c7;
        border-radius: 2px;
        position: absolute;
        top: 6px;
        left: 50%;
        transform: translateX(-50%);
        animation: scrollWheel 2s ease-in-out infinite;
    }
    
    @keyframes scrollWheel {
        0% { transform: translate(-50%, 0); opacity: 1; }
        100% { transform: translate(-50%, 14px); opacity: 0; }
    }
    
    /* =========================================================================
       PREMIUM 3D MODE SELECTION IMAGE CARDS
       ========================================================================= */
    .mode-3d-card {
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(28px) saturate(180%);
        -webkit-backdrop-filter: blur(28px) saturate(180%);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        border-radius: 28px;
        padding: 34px 30px;
        box-shadow: 0 20px 50px -10px rgba(15, 23, 42, 0.07), inset 0 1px 2px 0 rgba(255, 255, 255, 0.9);
        transition: all 0.38s cubic-bezier(0.16, 1, 0.3, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    
    /* Specular Light Reflection Sweep */
    .mode-3d-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -150%;
        width: 100%;
        height: 100%;
        background: linear-gradient(115deg, transparent 20%, rgba(255, 255, 255, 0.6) 50%, transparent 80%);
        transition: all 0.75s ease;
        pointer-events: none;
    }
    
    .mode-3d-card:hover::before {
        left: 150%;
    }
    
    .mode-3d-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 30px 65px -10px rgba(2, 132, 199, 0.22), 0 0 28px rgba(56, 189, 248, 0.25);
        border-color: rgba(147, 197, 253, 0.95);
    }
    
    .mode-3d-card-general:hover {
        box-shadow: 0 30px 65px -10px rgba(147, 51, 234, 0.22), 0 0 28px rgba(196, 181, 253, 0.25);
        border-color: rgba(196, 181, 253, 0.95);
    }
    
    /* 3D Image Display & Floating Animation */
    .image-3d-container {
        width: 100%;
        height: 250px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(240, 249, 255, 0.6) 0%, rgba(245, 243, 255, 0.6) 100%);
        margin-bottom: 24px;
        position: relative;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    .image-3d-visual {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 18px;
        transition: transform 0.45s cubic-bezier(0.16, 1, 0.3, 1), filter 0.45s ease;
        animation: floatVisual 6s ease-in-out infinite alternate;
    }
    
    .mode-3d-card:hover .image-3d-visual {
        transform: scale(1.07) translateY(-6px);
        filter: brightness(1.03) contrast(1.02);
    }
    
    @keyframes floatVisual {
        0% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-8px) scale(1.02); }
        100% { transform: translateY(4px) scale(0.99); }
    }
    
    /* Feature Pills */
    .glass-tag {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(226, 232, 240, 0.9);
        color: #334155;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 10px;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }
    
    /* General Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.76);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.92);
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: var(--glass-shadow);
        transition: all 0.25s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 45px -8px rgba(15, 23, 42, 0.08);
    }
    
    /* Department Cards */
    .dept-glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(240, 249, 255, 0.9) 100%);
        backdrop-filter: blur(16px);
        border: 1.5px solid rgba(186, 230, 253, 0.85);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.06);
    }
    
    /* Healthy / Normal card */
    .healthy-glass-card {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.94) 0%, rgba(236, 253, 245, 0.94) 100%);
        backdrop-filter: blur(16px);
        border: 1.5px solid #86efac;
        border-radius: 20px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.08);
    }
    
    /* Spoken Follow-up Box */
    .voice-glass-box {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.94) 0%, rgba(240, 253, 244, 0.94) 100%);
        backdrop-filter: blur(18px);
        border: 1.5px solid rgba(147, 197, 253, 0.9);
        border-radius: 20px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 8px 26px rgba(37, 99, 235, 0.09);
    }
    
    /* Red Flag Warning */
    .warning-glass-box {
        background: rgba(254, 242, 242, 0.94);
        backdrop-filter: blur(14px);
        border: 1.5px solid rgba(239, 68, 68, 0.7);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 16px;
        color: #991b1b;
    }
    </style>
    
    <!-- Background Canvas & Ambient Orbs -->
    <div class="neural-grid-bg"></div>
    <div class="ambient-orb orb-cyan"></div>
    <div class="ambient-orb orb-purple"></div>
    <div class="ambient-orb orb-blue"></div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# MODEL LOADERS (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_whisper_model():
    """Loads lightweight local Whisper model for real-time STT."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("tiny", device=device)


@st.cache_resource(show_spinner=False)
def get_clip_vision():
    """Loads OpenAI CLIP model for medical image understanding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


@st.cache_resource(show_spinner=False)
def get_llm_backend(model_spec: str):
    """Loads active LLM backend with local GPU TF32 acceleration."""
    return load_backend(model_spec)


# ==============================================================================
# NATURAL AI VOICE GENERATOR (TTS)
# ==============================================================================
@st.cache_data(show_spinner=False)
def generate_voice_output_mp3(text: str) -> bytes:
    """
    Synthesizes natural, human-like speech using Microsoft Edge Neural TTS
    (en-US-JennyNeural) with in-memory fallbacks to gTTS and pyttsx3.
    """
    clean_text = re.sub(r"[\*\#\_\`\(\)\[\]]", "", text).strip()
    if not clean_text:
        return b""

    # 1. Primary: Natural Microsoft Neural Voice
    try:
        async def _run_edge():
            comm = edge_tts.Communicate(clean_text, voice="en-US-JennyNeural", rate="+0%", pitch="+0Hz")
            chunks = []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(_run_edge()))
            data = future.result(timeout=4.0)
            if data and len(data) > 500:
                return data
    except Exception:
        pass

    # 2. Fallback: In-memory gTTS
    try:
        fp = io.BytesIO()
        tts = gTTS(text=clean_text, lang="en", slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        pass

    # 3. Fallback: Local pyttsx3 SAPI5
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "female" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tpath = f.name
        engine.save_to_file(clean_text, tpath)
        engine.runAndWait()
        with open(tpath, "rb") as f:
            data = f.read()
        try:
            os.remove(tpath)
        except Exception:
            pass
        return data
    except Exception:
        return b""


# ==============================================================================
# AUDIO TRANSCRIPTION (WHISPER STT)
# ==============================================================================
def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Converts recorded or uploaded audio into text using Whisper."""
    try:
        whisper_model = get_whisper_model()
        ext = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        result = whisper_model.transcribe(tmp_path, fp16=torch.cuda.is_available())
        text = result.get("text", "").strip()

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return text
    except Exception as e:
        return f"[Audio transcription error: {e}]"


# ==============================================================================
# MEDICAL IMAGE UNDERSTANDING & DIRECT DIAGNOSTICS (CLIP VISION)
# ==============================================================================
def analyze_medical_image(pil_img: Image.Image) -> Tuple[str, str, float]:
    """Recognizes medical image modality and key visual features via CLIP."""
    try:
        model, preprocess, device = get_clip_vision()
        img_tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)

        modalities = [
            ("Chest X-ray (Radiograph)", "a chest X-ray thoracic radiograph showing lungs, cardiac silhouette, and ribcage"),
            ("Dermatology / Skin Lesion", "a clinical dermatological photograph of skin showing erythema, rash, or pigmented lesion"),
            ("Dental / Oral Radiograph", "a dental panoramic or bitewing radiograph showing teeth, roots, and mandibular bone"),
            ("Neuroimaging (Brain MRI/CT)", "a cross-sectional brain CT or MRI neuroimaging slice"),
            ("Orthopedic Bone / Joint X-ray", "a musculoskeletal radiograph of bones and joints showing skeletal alignment"),
            ("Electrocardiogram (ECG/EKG)", "a medical 12-lead electrocardiogram ECG rhythm strip tracing with P-QRS-T waveforms"),
            ("Ophthalmic / Eye Examination", "an ocular eye photograph showing sclera, conjunctiva, cornea or pupil"),
            ("Surgical Wound / Laceration", "a clinical photograph of a traumatic laceration, surgical incision, or wound"),
            ("Normal / Unremarkable Clinical Scan", "a completely normal clear medical radiograph or healthy skin surface with no acute pathology"),
        ]

        tokens = clip.tokenize([m[1] for m in modalities]).to(device)

        with torch.no_grad():
            img_feats = model.encode_image(img_tensor)
            text_feats = model.encode_text(tokens)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            similarity = (img_feats @ text_feats.T).softmax(dim=-1)[0]

        top_idx = similarity.argmax().item()
        modality_name = modalities[top_idx][0]
        confidence = float(similarity[top_idx])

        if "Normal" in modality_name:
            features = "No acute radiological or structural abnormality visualized. Anatomy appears intact."
        elif "Dental" in modality_name:
            features = "Dentition, alveolar bone crest, and periapical regions evaluated."
        elif "Chest" in modality_name:
            features = "Thoracic cavity with bilateral lung fields and cardiac silhouette evaluated."
        elif "Skin" in modality_name or "Dermatology" in modality_name:
            features = "Cutaneous region evaluated for erythema, scaling, and inflammatory changes."
        elif "Neuroimaging" in modality_name:
            features = "Intracranial compartments and ventricular symmetry visualized."
        elif "Orthopedic" in modality_name:
            features = "Cortical bone continuity and joint spacing evaluated."
        elif "Electrocardiogram" in modality_name:
            features = "Cardiac rhythm, ST-segments, and repolarization waves evaluated."
        else:
            features = "Diagnostic image reviewed for pathological clinical signs."

        return modality_name, features, confidence
    except Exception as e:
        return "Clinical Medical Image", f"Clinical image provided ({e}).", 0.5


def analyze_medical_image_diagnostics(pil_img: Image.Image) -> dict:
    """
    Directly analyzes uploaded medical image and provides:
    1. Findings
    2. Possible diagnosis
    3. Recommended department in simple cards
    Without requiring unnecessary follow-up questions.
    """
    modality, features, conf = analyze_medical_image(pil_img)

    if "Normal" in modality:
        is_healthy = True
        findings = "Normal scan. Clear anatomical borders without consolidation, focal lesion, or effusion."
        possible_diagnosis = "No obvious health concern detected based on available information. The person appears to be doing well."
        department = "General Physician"
        dept_reason = "No apparent abnormality identified. Routine preventative health checkup advised."
    elif "Dental" in modality:
        is_healthy = False
        findings = "Dental radiograph evaluating crowns, roots, and alveolar bone architecture."
        possible_diagnosis = "Possible Dental Caries / Periapical Periodontitis"
        department = "Dentist"
        dept_reason = "Comprehensive dental examination, cleaning, or tooth restoration evaluation."
    elif "Chest" in modality:
        is_healthy = False
        findings = "Thoracic radiograph demonstrating lung fields and cardiac shadow with localized bronchovascular markings."
        possible_diagnosis = "Possible Community-Acquired Pneumonia / Lung Opacity"
        department = "Pulmonologist"
        dept_reason = "Respiratory auscultation, pulse oximetry, and follow-up chest imaging recommended."
    elif "Skin" in modality or "Dermatology" in modality:
        is_healthy = False
        findings = "Cutaneous evaluation shows localized erythema and superficial epidermal irritation."
        possible_diagnosis = "Possible Contact Dermatitis / Inflammatory Eruption"
        department = "Dermatologist"
        dept_reason = "Dermoscopy and targeted topical anti-inflammatory therapy."
    elif "Neuroimaging" in modality or "Brain" in modality:
        is_healthy = False
        findings = "Cross-sectional intracranial parenchyma displaying ventricular contours."
        possible_diagnosis = "Focal Intracranial Change / Non-acute Asymmetry"
        department = "Neurologist"
        dept_reason = "Formal radiological interpretation advised to exclude space-occupying lesion or ischemia."
    elif "Orthopedic" in modality or "Bone" in modality:
        is_healthy = False
        findings = "Musculoskeletal radiograph demonstrating osseous contours and articular alignment."
        possible_diagnosis = "Possible Cortical Bone Disruption / Traumatic Joint Sprain"
        department = "Orthopedic Specialist"
        dept_reason = "Physical examination, palpation, and splinting or immobilization evaluation."
    elif "Electrocardiogram" in modality or "ECG" in modality:
        is_healthy = False
        findings = "12-lead ECG tracing indicating rate, rhythm, and ventricular repolarization."
        possible_diagnosis = "Possible Cardiac Rhythm / ST-Segment Deviation"
        department = "Cardiologist"
        dept_reason = "Serum cardiac biomarker testing and continuous telemetry monitoring."
    elif "Wound" in modality or "Laceration" in modality:
        is_healthy = False
        findings = "Soft tissue discontinuity with localized tissue trauma."
        possible_diagnosis = "Traumatic Soft Tissue Laceration"
        department = "General Physician"
        dept_reason = "Wound cleansing, closure assessment, and tetanus immunization status review."
    else:
        is_healthy = False
        findings = "Diagnostic imaging evaluated for clinical signs."
        possible_diagnosis = "Clinical Diagnostic Finding under Review"
        department = "General Physician"
        dept_reason = "Routine clinical correlation and outpatient consultation advised."

    return {
        "modality": modality,
        "features": features,
        "confidence": conf,
        "is_healthy": is_healthy,
        "findings": findings,
        "possible_diagnosis": possible_diagnosis,
        "department": department,
        "dept_reason": dept_reason,
    }


# ==============================================================================
# CLINICAL DATA STRUCTURES & LOGIC
# ==============================================================================
class DiseaseDiagnosis(BaseModel):
    condition: str
    confidence: float
    evidence: str


class ClinicalAssessment(BaseModel):
    is_healthy: bool = False
    healthy_message: Optional[str] = None
    top_3_diseases: List[DiseaseDiagnosis]
    recommended_department: str
    department_reason: str
    clinical_summary: str
    recommended_next_steps: List[str]
    emergency_warning_signs: List[str]
    esi_level: int
    triage_priority: str


def check_vital_thresholds(vitals: dict, vitals_taken: bool) -> List[str]:
    """Identifies critical physiological red flags."""
    if not vitals_taken or not vitals:
        return []

    flags = []
    hr = vitals.get("heartrate")
    if hr and (hr > 120 or hr < 45):
        flags.append(f"Heart Rate: {int(hr)} bpm ({'Severe Tachycardia' if hr > 120 else 'Severe Bradycardia'})")

    sbp = vitals.get("sbp")
    if sbp and (sbp < 90 or sbp > 185):
        flags.append(f"Blood Pressure: SBP {int(sbp)} mmHg ({'Hypotension Risk' if sbp < 90 else 'Hypertensive Crisis'})")

    spo2 = vitals.get("o2sat")
    if spo2 and spo2 < 92:
        flags.append(f"Oxygen Saturation: SpO2 {int(spo2)}% (Hypoxemia Risk)")

    rr = vitals.get("resprate")
    if rr and (rr > 28 or rr < 8):
        flags.append(f"Respiratory Rate: {int(rr)} /min ({'Severe Tachypnea' if rr > 28 else 'Respiratory Depression'})")

    temp = vitals.get("temperature")
    if temp and temp > 103.5:
        flags.append(f"Temperature: {temp}°F (Severe Fever)")

    return flags


def generate_medical_followup_question(
    backend,
    question_idx: int,
    total_questions: int,
    symptoms: str,
    vitals: dict,
    vitals_taken: bool,
    image_info: Optional[dict],
    chat_history: list[dict],
) -> str:
    """
    Generates a short, simple, plain-language clinical follow-up question (under 15 words)
    asking one question at a time.
    """
    s_full = " ".join(m["content"] for m in chat_history if m["role"] == "user")

    if question_idx == 1:
        focus = "how many days they have had this symptom and if the pain is continuous or comes and goes"
    elif question_idx == 2:
        focus = "if the discomfort spreads anywhere else, or what makes it feel better or worse"
    elif question_idx == 3:
        focus = "if they have any other symptoms like fever, shortness of breath, dizziness, or nausea"
    else:
        focus = "if they have any preexisting medical conditions or take regular medications"

    system_prompt = f"""You are Dr. MediAssist, a caring and friendly clinician speaking with a patient.
The patient said: "{symptoms}"
Recent answers: "{s_full}"

YOUR TASK:
Ask Follow-up Question {question_idx} of {total_questions}.
Focus: {focus}.

CRITICAL RULES:
1. Ask only ONE single, short, simple question in plain everyday English.
2. Keep it strictly under 15 words.
3. Speak directly to the patient in a warm, gentle, conversational tone.
4. Do NOT use medical jargon, do NOT introduce yourself, and do NOT give any explanations.
5. Output ONLY the single question sentence ending with a question mark."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is your next single follow-up question for me?"},
    ]

    try:
        res = backend.complete([Request(messages=messages, max_tokens=35)])
        if isinstance(res[0], str):
            clean_q = res[0].strip().replace('"', '').replace("Question:", "").strip()
            clean_q = re.split(r"(?<=[?])\s+", clean_q)[0]
            if "?" not in clean_q:
                clean_q += "?"
            return clean_q
    except Exception:
        pass

    fallbacks = {
        1: "How many days have you been experiencing these symptoms?",
        2: "Does the pain spread to your arm, back, or neck?",
        3: "Are you having any shortness of breath, dizziness, or nausea?",
    }
    return fallbacks.get(question_idx, "Do you have any existing medical conditions?")


def generate_final_medical_assessment(
    backend,
    symptoms: str,
    vitals: dict,
    vitals_taken: bool,
    image_info: Optional[dict],
    chat_history: list[dict],
) -> ClinicalAssessment:
    """
    Generates the top 3 possible diseases, simple department recommendations
    (Cardiologist, Dentist, General Physician, Dermatologist, Neurologist, etc.),
    and concise assessment. Communicates absence of obvious concern when healthy.
    """
    flags = check_vital_thresholds(vitals, vitals_taken)
    s_full = " ".join(m["content"] for m in chat_history if m["role"] == "user")
    s_lower = (symptoms + " " + s_full).lower()

    # Check for healthy / normal / no concern presentation
    healthy_keywords = ["fine", "good", "healthy", "normal", "no symptoms", "no issue", "routine check", "feeling great", "well", "none"]
    if any(k in s_lower for k in healthy_keywords) and not any(k in s_lower for k in ["pain", "hurt", "ache", "fever", "cough", "shortness", "breath", "blood", "dizzy", "vomit", "chest", "tooth"]):
        return ClinicalAssessment(
            is_healthy=True,
            healthy_message="No obvious health concern detected based on available information. The person appears to be doing well.",
            top_3_diseases=[
                DiseaseDiagnosis(condition="General Healthy State / Normal Baseline", confidence=95, evidence="Patient reports absence of acute symptoms, normal wellness indicators"),
                DiseaseDiagnosis(condition="Routine Wellness / Preventative Follow-up", confidence=85, evidence="No acute physiological distress or functional impairment reported"),
                DiseaseDiagnosis(condition="Mild Transient Fatigue (Non-pathological)", confidence=15, evidence="Standard physiological baseline without clinical red flags"),
            ],
            recommended_department="General Physician",
            department_reason="Routine periodic wellness maintenance and preventive primary care.",
            clinical_summary="No obvious health concern detected based on available information. The person appears to be doing well with normal vital signs and no acute complaints.",
            recommended_next_steps=[
                "Maintain balanced nutrition, adequate sleep, and daily hydration",
                "Continue routine physical activity and wellness habits",
                "Schedule periodic routine preventative health checks",
            ],
            emergency_warning_signs=[],
            esi_level=5,
            triage_priority="Non-Urgent / Healthy",
        )

    # Department and condition mapping
    if any(k in s_lower for k in ["tooth", "teeth", "gum", "jaw", "mouth", "dental", "cavity"]):
        dept = "Dentist"
        reason = "Oral or dental pain presentation concerning for dental caries, gingivitis, or periapical inflammation."
        top_3 = [
            DiseaseDiagnosis(condition="Dental Caries / Pulpitis", confidence=65, evidence="Localized oral discomfort with sensitivity or toothache"),
            DiseaseDiagnosis(condition="Gingivitis / Periodontal Abscess", confidence=25, evidence="Gingival tenderness or localized swelling"),
            DiseaseDiagnosis(condition="Temporomandibular Joint (TMJ) Strain", confidence=10, evidence="Masticatory tenderness without tooth defect"),
        ]
        esi, priority = 4, "Less Urgent"
        steps = ["Dental clinical examination and intraoral radiograph", "Gentle warm salt-water rinses", "Avoid excessively hot, cold, or sugary foods"]
    elif any(k in s_lower for k in ["chest", "heart", "arm", "pressure", "angina", "tightness"]):
        dept = "Cardiologist"
        reason = "Substernal thoracic symptoms concerning for acute coronary syndrome or myocardial ischemia."
        top_3 = [
            DiseaseDiagnosis(condition="Acute Coronary Syndrome (ACS / Angina)", confidence=65, evidence="Substernal chest pressure with cardiovascular symptom pattern"),
            DiseaseDiagnosis(condition="Pulmonary Embolism (PE)", confidence=20, evidence="Acute onset thoracic discomfort and dyspnea risk"),
            DiseaseDiagnosis(condition="Acute Pericarditis / Costochondritis", confidence=15, evidence="Chest wall / pericardial irritation requiring ECG differentiation"),
        ]
        esi, priority = 2, "Emergent"
        steps = ["Immediate 12-lead ECG", "Continuous cardiac telemetry monitoring", "Stat cardiac enzymes (Troponin I/T)"]
    elif any(k in s_lower for k in ["breath", "wheez", "cough", "asthma", "dyspnea", "lungs", "pneumonia"]) or (image_info and "Chest" in image_info.get("modality", "")):
        dept = "Pulmonologist"
        reason = "Respiratory symptoms and lower airway involvement requiring targeted bronchodilator or oxygen therapy."
        top_3 = [
            DiseaseDiagnosis(condition="Acute Asthma Exacerbation / Bronchitis", confidence=55, evidence="Dyspnea, airway reactivity, and persistent cough symptoms"),
            DiseaseDiagnosis(condition="Community-Acquired Pneumonia", confidence=30, evidence="Lower respiratory tract symptoms and localized opacity signs"),
            DiseaseDiagnosis(condition="Pleural Effusion / Pneumothorax", confidence=15, evidence="Thoracic radiological evaluation indicated"),
        ]
        esi, priority = 2, "Emergent"
        steps = ["Maintain oxygen saturation >= 94%", "Stat Chest Radiograph (CXR)", "Inhaled bronchodilator therapy"]
    elif any(k in s_lower for k in ["abdomen", "belly", "stomach", "nausea", "vomit", "appendix", "cramp"]):
        dept = "Gastroenterologist"
        reason = "Abdominal discomfort presentation concerning for acute gastrointestinal or inflammatory process."
        top_3 = [
            DiseaseDiagnosis(condition="Acute Appendicitis", confidence=60, evidence="Localized abdominal discomfort with gastrointestinal symptoms"),
            DiseaseDiagnosis(condition="Acute Cholecystitis / Biliary Colic", confidence=25, evidence="Right upper abdominal tenderness and digestive distress"),
            DiseaseDiagnosis(condition="Acute Gastroenteritis / Peptic Disease", confidence=15, evidence="Epigastric irritation and nausea"),
        ]
        esi, priority = 3, "Urgent"
        steps = ["Abdominal ultrasound or CT scan", "NPO status (avoid solid food)", "Complete blood count and metabolic panel"]
    elif any(k in s_lower for k in ["headache", "dizziness", "faint", "syncope", "numb", "stroke", "weakness"]):
        dept = "Neurologist"
        reason = "Neurological symptoms with risk of transient cerebral hypoperfusion or focal deficit."
        top_3 = [
            DiseaseDiagnosis(condition="Transient Ischemic Attack (TIA) / Acute Migraine", confidence=50, evidence="Neurological symptom onset, cephalalgia, or transient dizziness"),
            DiseaseDiagnosis(condition="Vasovagal Syncope / Orthostatic Hypotension", confidence=30, evidence="Transient cerebral hypoperfusion without persistent focal deficit"),
            DiseaseDiagnosis(condition="Vestibular Neuritis / Benign Positional Vertigo", confidence=20, evidence="Episodic vertigo and balance disturbance"),
        ]
        esi, priority = 2 if "weakness" in s_lower or "numb" in s_lower else 3, "Emergent" if "weakness" in s_lower else "Urgent"
        steps = ["Immediate neurological assessment (BE-FAST check)", "Non-contrast Brain CT/MRI scan", "Orthostatic blood pressure check"]
    elif any(k in s_lower for k in ["skin", "rash", "itch", "lesion", "hives", "burn"]) or (image_info and "Skin" in image_info.get("modality", "")):
        dept = "Dermatologist"
        reason = "Cutaneous eruption and epidermal inflammatory changes requiring targeted dermatological treatment."
        top_3 = [
            DiseaseDiagnosis(condition="Contact Dermatitis / Urticaria", confidence=60, evidence="Cutaneous erythema, pruritus, and localized inflammatory skin changes"),
            DiseaseDiagnosis(condition="Cellulitis / Bacterial Skin Infection", confidence=25, evidence="Localized erythema and soft tissue inflammation risk"),
            DiseaseDiagnosis(condition="Viral Exanthem / Psoriasis Flare", confidence=15, evidence="Epidermal scaling and widespread cutaneous pattern"),
        ]
        esi, priority = 4, "Less Urgent"
        steps = ["Dermatological examination and lesion dermoscopy", "Topical soothing therapy or antihistamines", "Monitor for spreading borders or systemic fever"]
    elif any(k in s_lower for k in ["bone", "joint", "fracture", "sprain", "knee", "ankle", "wrist", "back"]):
        dept = "Orthopedic Specialist"
        reason = "Musculoskeletal injury or articular inflammation requiring orthopedic assessment."
        top_3 = [
            DiseaseDiagnosis(condition="Musculoskeletal Sprain / Tendinopathy", confidence=55, evidence="Localized joint or tendon tenderness without deformity"),
            DiseaseDiagnosis(condition="Acute Ligamentous Tear / Strain", confidence=30, evidence="Joint pain exacerbated by weight-bearing"),
            DiseaseDiagnosis(condition="Occult Bone Fracture", confidence=15, evidence="Localized osseous tenderness requiring radiograph"),
        ]
        esi, priority = 4, "Less Urgent"
        steps = ["X-ray radiograph of affected joint/limb", "RICE protocol (Rest, Ice, Compression, Elevation)", "Avoid heavy weight-bearing"]
    else:
        dept = "General Physician"
        reason = "Non-specific systemic presentation requiring comprehensive primary care evaluation."
        top_3 = [
            DiseaseDiagnosis(condition="Acute Viral Syndrome / Upper Respiratory Infection", confidence=50, evidence="Systemic constitutional complaints and malaise"),
            DiseaseDiagnosis(condition="General Metabolic Fatigue / Dehydration", confidence=30, evidence="Fluctuating energy and non-focal discomfort"),
            DiseaseDiagnosis(condition="Musculoskeletal Strain", confidence=20, evidence="Localized physical tenderness without red flags"),
        ]
        esi, priority = 4, "Less Urgent"
        steps = ["Outpatient medical evaluation", "Rest and adequate fluid hydration", "Return immediately if red-flag symptoms develop"]

    if flags:
        esi = 1 if any("Shock" in f or "Hypoxemia" in f for f in flags) else 2
        priority = "Immediate Resuscitation" if esi == 1 else "Emergent"

    summary = f"Patient presents with {symptoms[:100].strip().lower()}. Intake completed with structured follow-ups. Primary consideration is {top_3[0].condition} ({top_3[0].confidence}% relevance). Consultation with {dept} is recommended."

    return ClinicalAssessment(
        is_healthy=False,
        top_3_diseases=top_3,
        recommended_department=dept,
        department_reason=reason,
        clinical_summary=summary,
        recommended_next_steps=steps,
        emergency_warning_signs=flags,
        esi_level=esi,
        triage_priority=priority,
    )


# ==============================================================================
# ROUTING & SESSION STATE INITIALIZATION
# ==============================================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"

# Medical Mode States
if "med_messages" not in st.session_state:
    st.session_state.med_messages = []
if "med_question_idx" not in st.session_state:
    st.session_state.med_question_idx = 0
if "med_total_questions" not in st.session_state:
    st.session_state.med_total_questions = 3
if "med_current_voice_question" not in st.session_state:
    st.session_state.med_current_voice_question = None
if "med_initial_symptoms" not in st.session_state:
    st.session_state.med_initial_symptoms = ""
if "clinical_assessment" not in st.session_state:
    st.session_state.clinical_assessment = None
if "image_diagnostic_result" not in st.session_state:
    st.session_state.image_diagnostic_result = None
if "uploaded_image_info" not in st.session_state:
    st.session_state.uploaded_image_info = None
if "vitals_taken" not in st.session_state:
    st.session_state.vitals_taken = False
if "current_vitals" not in st.session_state:
    st.session_state.current_vitals = {
        "heartrate": 82.0,
        "sbp": 120.0,
        "resprate": 16.0,
        "o2sat": 98.0,
        "temperature": 98.6,
    }

# General Chat States
if "gen_messages" not in st.session_state:
    st.session_state.gen_messages = []

active_model = os.environ.get("MODEL", "hf-local:Qwen/Qwen2.5-1.5B-Instruct")


# ==============================================================================
# PAGE 1: ANIMATED LANDING PAGE WITH STUNNING 3D MODE CARDS
# ==============================================================================
if st.session_state.current_page == "landing":
    # 1. Floating Glass Navbar
    st.markdown(
        """
        <div class="glass-nav animate-entrance">
            <div class="glass-nav-brand">
                <span style="font-size: 1.85rem;">✨</span>
                <h2>MediAssist AI</h2>
                <span style="background: rgba(224, 242, 254, 0.9); color: #0284c7; font-size: 0.78rem; font-weight: 800; padding: 4px 12px; border-radius: 999px;">v2.5 SaaS</span>
            </div>
            <div class="glass-nav-links">
                <a href="#modes" class="glass-nav-link">Mode Selection</a>
                <a href="#features" class="glass-nav-link">Features</a>
                <a href="#technology" class="glass-nav-link">Technology</a>
                <a href="#about" class="glass-nav-link">About</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Hero Section
    st.markdown(
        """
        <div class="hero-container animate-entrance animate-delay-1">
            <div class="hero-badge">
                <span>✦</span>
                <span>Next-Generation Ambient AI Platform</span>
            </div>
            <h1 class="hero-title">Intelligence That<br>Understands You.</h1>
            <p class="hero-subtitle">
                One intelligent platform for clinical healthcare assistance and everyday conversational intelligence.
                Engineered with multi-modal medical vision, human-like voice synthesis, and local GPU acceleration.
            </p>
            <div class="scroll-indicator">
                <div class="mouse-pill"><div class="mouse-wheel"></div></div>
                <span>Scroll to Explore Modes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Quick Jump Button
    cta_c1, cta_c2, cta_c3 = st.columns([1.5, 2, 1.5])
    with cta_c2:
        if st.button("🚀 Explore Platform & Choose Mode ↓", type="primary", use_container_width=True):
            st.session_state.current_page = "mode_select"
            st.rerun()

    st.markdown("<div style='height: 35px;' id='modes'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. HIGHLIGHT: INTERACTIVE 3D MODE SELECTION SHOWCASE CARDS (WITH 3D IMAGES)
    # =========================================================================
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 28px;" class="animate-entrance animate-delay-2">
            <span style="color: #0284c7; font-weight: 800; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.08em;">Choose Your Experience</span>
            <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.7rem; font-weight: 800; color: #0f172a; margin-top: 4px;">Two Dedicated AI Experiences</h2>
            <p style="color: #64748b; font-size: 1.05rem; max-width: 600px; margin: 0 auto;">Select between clinical triage decision support or all-purpose conversational AI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card1, card2 = st.columns(2, gap="large")

    # CARD 1: MEDICAL ASSISTANT (High-Resolution 3D Visual & Glass)
    with card1:
        st.markdown(
            f"""
            <div class="mode-3d-card animate-entrance animate-delay-3">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="background: rgba(224, 242, 254, 0.95); color: #0284c7; font-size: 0.76rem; font-weight: 800; padding: 6px 14px; border-radius: 999px; letter-spacing: 0.06em;">✦ CLINICAL ENGINE</span>
                        <span style="font-size: 0.8rem; color: #16a34a; font-weight: 700;">● Online & Accelerated</span>
                    </div>
                    <div class="image-3d-container">
                        <img src="data:image/jpeg;base64,{med_img_b64}" alt="3D Medical AI Visual" class="image-3d-visual" />
                    </div>
                    <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800; color: #0f172a; margin: 0 0 10px 0;">Medical Assistant</h2>
                    <p style="color: #475569; font-size: 1.02rem; line-height: 1.6; margin-bottom: 20px;">
                        Intelligent health insights through voice, vision and AI. Experience context-aware dynamic follow-up questions, direct medical image diagnostics, and specialist department mapping.
                    </p>
                    <div style="margin-bottom: 16px;">
                        <span class="glass-tag">🎙️ Spoken Follow-Ups</span>
                        <span class="glass-tag">🖼️ Direct Vision Diagnostics</span>
                        <span class="glass-tag">🏥 Department Routing</span>
                        <span class="glass-tag">❤️ Real-Time Vitals</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🩺 Enter Medical Mode →", type="primary", use_container_width=True, key="landing_btn_med"):
            st.session_state.current_page = "medical"
            st.rerun()

    # CARD 2: GENERAL AI CHAT (High-Resolution 3D Visual & Glass)
    with card2:
        st.markdown(
            f"""
            <div class="mode-3d-card mode-3d-card-general animate-entrance animate-delay-3">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="background: rgba(243, 232, 255, 0.95); color: #7c3aed; font-size: 0.76rem; font-weight: 800; padding: 6px 14px; border-radius: 999px; letter-spacing: 0.06em;">✦ ALL-PURPOSE AI</span>
                        <span style="font-size: 0.8rem; color: #7c3aed; font-weight: 700;">● Token Streaming</span>
                    </div>
                    <div class="image-3d-container">
                        <img src="data:image/jpeg;base64,{gen_img_b64}" alt="3D AI Chatbot Visual" class="image-3d-visual" />
                    </div>
                    <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800; color: #0f172a; margin: 0 0 10px 0;">General Chat</h2>
                    <p style="color: #475569; font-size: 1.02rem; line-height: 1.6; margin-bottom: 20px;">
                        Explore ideas, ask questions and interact with AI. Solve programming challenges, analyze scientific concepts, brainstorm creative writing, and chat freely without clinical logic.
                    </p>
                    <div style="margin-bottom: 16px;">
                        <span class="glass-tag">⚡ Low Latency Streaming</span>
                        <span class="glass-tag">💻 Code & Technology</span>
                        <span class="glass-tag">📚 Science & Education</span>
                        <span class="glass-tag">🚫 Zero Medical Triage</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("💬 Enter General Chat →", use_container_width=True, key="landing_btn_gen"):
            st.session_state.current_page = "general"
            st.rerun()

    st.markdown("<div style='height: 60px;' id='features'></div>", unsafe_allow_html=True)

    # 4. Features Grid
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 32px;">
            <span style="color: #0284c7; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;">Architectural Highlights</span>
            <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 800; color: #0f172a; margin-top: 4px;">Precision Healthcare & Open Reasoning</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feat1, feat2, feat3 = st.columns(3)
    with feat1:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">🎙️</div>
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Neural Spoken Follow-Ups</h3>
                <p style="font-size: 0.9rem; color: #64748b; line-height: 1.6;">
                    Medical follow-up questions are delivered through natural, warm human-like speech output (TTS) one question at a time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">🖼️</div>
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Direct Vision Diagnostics</h3>
                <p style="font-size: 0.9rem; color: #64748b; line-height: 1.6;">
                    Upload medical imaging (X-rays, ECGs, MRIs, dermatology) for immediate findings and department mapping without unnecessary follow-ups.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat3:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">🛡️</div>
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Private Local GPU Compute</h3>
                <p style="font-size: 0.9rem; color: #64748b; line-height: 1.6;">
                    Powered locally on RTX GPU using Ada Lovelace TensorFloat-32 acceleration for instantaneous responses with strict data privacy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Technology & About (Anchor: #technology)
    st.markdown("<div style='height: 40px;' id='technology'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.88) 0%, rgba(240, 249, 255, 0.88) 100%); padding: 32px 36px;" id="about">
            <span style="color: #0284c7; font-weight: 800; font-size: 0.82rem; text-transform: uppercase;">Technical Specifications</span>
            <h3 style="font-size: 1.7rem; font-weight: 800; color: #0f172a; margin: 6px 0 12px 0;">Built on Cutting-Edge Open Foundations</h3>
            <p style="color: #475569; font-size: 0.98rem; line-height: 1.65; margin-bottom: 16px;">
                MediAssist AI harnesses Qwen2.5-1.5B-Instruct running locally in float16/TF32, OpenAI Whisper for low-latency acoustic transcription,
                CLIP ViT-B/32 zero-shot vision classification, and Edge Neural voice synthesis. All health decision algorithms require clinical licensed clinician oversight.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span class="glass-tag" style="background:#e0f2fe; color:#0369a1;">NVIDIA RTX 4050 GPU</span>
                <span class="glass-tag" style="background:#dcfce7; color:#15803d;">Whisper-Tiny STT</span>
                <span class="glass-tag" style="background:#f3e8ff; color:#7e22ce;">CLIP ViT-B/32</span>
                <span class="glass-tag" style="background:#fef3c7; color:#b45309;">Edge-TTS JennyNeural</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; padding: 30px 10px; border-top: 1px solid rgba(226, 232, 240, 0.7); color: #64748b; font-size: 0.85rem;">
            <p style="margin: 0 0 6px 0;"><strong>MediAssist AI</strong> • Designed for Healthcare Professionals & Everyday Intelligence</p>
            <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Decision support only. Clinical professional verification required for all medical treatments.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 2: INTERACTIVE DEDICATED MODE SELECTION
# ==============================================================================
elif st.session_state.current_page == "mode_select":
    # Top Navigation Bar
    s_col1, s_col2, s_col3 = st.columns([2, 6, 2])
    with s_col1:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.current_page = "landing"
            st.rerun()
    with s_col3:
        st.markdown(
            "<div style='text-align: right;'><span class='glass-tag' style='background:#dbeafe; color:#0369a1;'>Step 2 of 2</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="text-align: center; margin: 24px 0 36px 0;" class="animate-entrance">
            <div class="hero-badge">✦ Interactive Mode Selection</div>
            <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 3.0rem; font-weight: 800; color: #0f172a; margin: 8px 0;">
                Choose Your Experience
            </h1>
            <p style="color: #64748b; font-size: 1.15rem; max-width: 600px; margin: 0 auto;">
                Select between our specialized clinical intelligence engine or all-purpose conversational AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sel_col1, sel_col2 = st.columns(2, gap="large")

    with sel_col1:
        st.markdown(
            f"""
            <div class="mode-3d-card animate-entrance animate-delay-1">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="background: rgba(224, 242, 254, 0.95); color: #0284c7; font-size: 0.76rem; font-weight: 800; padding: 6px 14px; border-radius: 999px;">✦ CLINICAL ENGINE</span>
                        <span style="font-size: 0.8rem; color: #16a34a; font-weight: 700;">● Online</span>
                    </div>
                    <div class="image-3d-container">
                        <img src="data:image/jpeg;base64,{med_img_b64}" alt="3D Medical AI Visual" class="image-3d-visual" />
                    </div>
                    <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800; color: #0f172a; margin: 0 0 10px 0;">Medical Assistant</h2>
                    <p style="color: #475569; font-size: 1.02rem; line-height: 1.6; margin-bottom: 20px;">
                        Intelligent health insights through voice, vision and AI. Experience context-aware dynamic follow-up questions, direct medical image diagnostics, and specialist department mapping.
                    </p>
                    <div style="margin-bottom: 16px;">
                        <span class="glass-tag">🎙️ Spoken Follow-Ups</span>
                        <span class="glass-tag">🖼️ Direct Vision Diagnostics</span>
                        <span class="glass-tag">🏥 Department Routing</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Enter Medical Mode →", type="primary", use_container_width=True, key="sel_btn_med"):
            st.session_state.current_page = "medical"
            st.rerun()

    with sel_col2:
        st.markdown(
            f"""
            <div class="mode-3d-card mode-3d-card-general animate-entrance animate-delay-1">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="background: rgba(243, 232, 255, 0.95); color: #7c3aed; font-size: 0.76rem; font-weight: 800; padding: 6px 14px; border-radius: 999px;">✦ ALL-PURPOSE AI</span>
                        <span style="font-size: 0.8rem; color: #7c3aed; font-weight: 700;">● Real-Time</span>
                    </div>
                    <div class="image-3d-container">
                        <img src="data:image/jpeg;base64,{gen_img_b64}" alt="3D AI Chatbot Visual" class="image-3d-visual" />
                    </div>
                    <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800; color: #0f172a; margin: 0 0 10px 0;">General Chat</h2>
                    <p style="color: #475569; font-size: 1.02rem; line-height: 1.6; margin-bottom: 20px;">
                        Explore ideas, ask questions and interact with AI. Solve programming challenges, analyze scientific concepts, brainstorm creative writing, and chat freely without clinical logic.
                    </p>
                    <div style="margin-bottom: 16px;">
                        <span class="glass-tag">⚡ Low Latency Streaming</span>
                        <span class="glass-tag">💻 Code & Technology</span>
                        <span class="glass-tag">📚 Science & Education</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Enter General Chat →", use_container_width=True, key="sel_btn_gen"):
            st.session_state.current_page = "general"
            st.rerun()


# ==============================================================================
# PAGE 3: GENERAL AI CHAT (ANY TOPIC, NO MEDICAL LOGIC)
# ==============================================================================
elif st.session_state.current_page == "general":
    # Top Glass Navigation Bar
    n_col1, n_col2, n_col3 = st.columns([2, 6, 2])
    with n_col1:
        if st.button("← Mode Selection", use_container_width=True):
            st.session_state.current_page = "mode_select"
            st.rerun()
    with n_col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <h3 style="margin: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 800; color: #6d28d9; font-size: 1.4rem;">
                    💬 General Chat
                </h3>
                <span style="font-size: 0.82rem; color: #7c3aed; font-weight: 600;">● Versatile AI Assistant • No Medical Triage Logic</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n_col3:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "landing"
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # General Chat Controls: Voice Input & Clear
    g_ctrl1, g_ctrl2 = st.columns([3, 1])
    with g_ctrl1:
        audio_gen = st.audio_input("🎙️ Speak your question:", key="gen_voice_active")
        if audio_gen is not None:
            a_bytes = audio_gen.read()
            with st.spinner("Transcribing your speech with Whisper..."):
                t_text = transcribe_audio_bytes(a_bytes)
            if t_text and not any(m.get("voice_id") == id(audio_gen) for m in st.session_state.gen_messages):
                st.session_state.gen_messages.append({"role": "user", "content": t_text, "voice_id": id(audio_gen)})
                backend = get_llm_backend(active_model)
                with st.chat_message("user", avatar="👤"):
                    st.write(t_text)
                with st.chat_message("assistant", avatar="🤖"):
                    gen_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "You are a friendly, intelligent, and versatile AI assistant. Answer the user's questions "
                                "clearly, concisely, and helpfully across any topic including technology, education, coding, "
                                "science, and general knowledge. Do NOT ask medical triage questions or give medical diagnoses."
                            ),
                        },
                        *st.session_state.gen_messages,
                    ]
                    full_gen = st.write_stream(backend.stream_chat(gen_prompt, max_new_tokens=300))
                st.session_state.gen_messages.append({"role": "assistant", "content": full_gen})
                st.rerun()

    with g_ctrl2:
        st.write("")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.gen_messages = []
            st.rerun()

    # Chat Messages Container
    gen_box = st.container(height=480)
    with gen_box:
        if not st.session_state.gen_messages:
            st.markdown(
                """
                <div class="glass-card" style="text-align: center; padding: 42px 20px;">
                    <span style="font-size: 2.6rem;">💡</span>
                    <h3 style="color: #6d28d9; margin: 12px 0 6px 0; font-weight: 800; font-family: 'Space Grotesk', sans-serif;">How can I assist you today?</h3>
                    <p style="color: #64748b; font-size: 0.96rem; max-width: 520px; margin: 0 auto; line-height: 1.6;">
                        Ask me about programming, software engineering, science, literature, history, or everyday curiosities.
                        This mode operates as a normal AI assistant with zero medical triage logic.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.gen_messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["content"])

    # Chat Text Input
    if user_gen_text := st.chat_input("Ask any topic (e.g., Python code, explain quantum computing, brainstorm ideas)..."):
        st.session_state.gen_messages.append({"role": "user", "content": user_gen_text})
        with st.chat_message("user", avatar="👤"):
            st.write(user_gen_text)

        backend = get_llm_backend(active_model)
        with st.chat_message("assistant", avatar="🤖"):
            gen_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly, intelligent, and versatile AI assistant. Answer the user's questions "
                        "clearly, concisely, and helpfully across any topic including technology, education, coding, "
                        "science, and general knowledge. Do NOT ask medical triage questions or give medical diagnoses."
                    ),
                },
                *st.session_state.gen_messages,
            ]
            full_gen = st.write_stream(backend.stream_chat(gen_prompt, max_new_tokens=300))

        st.session_state.gen_messages.append({"role": "assistant", "content": full_gen})
        st.rerun()


# ==============================================================================
# PAGE 4: MEDICAL MODE (INTELLIGENT MEDICAL ASSISTANT)
# ==============================================================================
elif st.session_state.current_page == "medical":
    # Top Glass Navigation Bar
    m_col1, m_col2, m_col3 = st.columns([2, 6, 2])
    with m_col1:
        if st.button("← Mode Selection", use_container_width=True):
            st.session_state.current_page = "mode_select"
            st.rerun()
    with m_col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <h3 style="margin: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 800; color: #0284c7; font-size: 1.4rem;">
                    🩺 MediAssist • Medical Mode
                </h3>
                <span style="font-size: 0.82rem; color: #16a34a; font-weight: 700;">● Clinical Decision Support Active • Voice Output Enabled</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "landing"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Vitals Glass Bar
    with st.container():
        vcols = st.columns([1, 1, 1, 1, 1, 1.3])
        v = st.session_state.current_vitals
        v_taken = st.session_state.vitals_taken

        with vcols[0]:
            hr = v.get("heartrate")
            st.metric("❤️ Heart Rate", f"{int(hr)} bpm" if (v_taken and hr) else "--")
        with vcols[1]:
            sbp = v.get("sbp")
            st.metric("🩺 BP Systolic", f"{int(sbp)} mmHg" if (v_taken and sbp) else "--")
        with vcols[2]:
            rr = v.get("resprate")
            st.metric("🫁 Respiration", f"{int(rr)} /min" if (v_taken and rr) else "--")
        with vcols[3]:
            spo2 = v.get("o2sat")
            st.metric("💨 SpO2 Oxygen", f"{int(spo2)}%" if (v_taken and spo2) else "--")
        with vcols[4]:
            temp = v.get("temperature")
            st.metric("🌡️ Temperature", f"{temp}°F" if (v_taken and temp) else "--")

        with vcols[5]:
            st.write("")
            if st.button("🩺 Record Vitals", use_container_width=True):
                st.session_state.vitals_taken = True
                v_msg = f"🩺 **Vitals Logged:** HR {int(v.get('heartrate', 80))} bpm | SBP {int(v.get('sbp', 120))} mmHg | RR {int(v.get('resprate', 16))} /min | SpO2 {int(v.get('o2sat', 98))}% | Temp {v.get('temperature', 98.6)}°F"
                st.session_state.med_messages.append({"role": "system", "content": v_msg})
                if st.session_state.clinical_assessment:
                    st.session_state.clinical_assessment = generate_final_medical_assessment(
                        backend=get_llm_backend(active_model),
                        symptoms=st.session_state.med_initial_symptoms,
                        vitals=st.session_state.current_vitals,
                        vitals_taken=True,
                        image_info=st.session_state.uploaded_image_info,
                        chat_history=st.session_state.med_messages,
                    )
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2-Column Split: Left = Patient Consultation; Right = Structured Assessment & Department Cards
    med_left, med_right = st.columns([1.18, 1.0], gap="large")

    # --------------------------------------------------------------------------
    # LEFT COLUMN: CONSULTATION & MULTI-INPUT OPTIONS
    # --------------------------------------------------------------------------
    with med_left:
        st.subheader("💬 Patient Consultation")

        # MULTIPLE INPUT OPTIONS (Tabs for Live Mic, Audio Upload, Medical Image)
        in_tab1, in_tab2, in_tab3 = st.tabs([
            "🎙️ Live Microphone",
            "📁 Upload Audio File",
            "🖼️ Medical Image Upload",
        ])

        # 1. Live Microphone Input
        with in_tab1:
            audio_med = st.audio_input("Record voice (Microphone):", key="med_mic_active")

        # 2. Audio File Upload (.mp3, .wav, .m4a)
        with in_tab2:
            uploaded_audio_file = st.file_uploader(
                "Upload voice note (.wav, .mp3, .m4a, .ogg):",
                type=["wav", "mp3", "m4a", "ogg"],
                key="med_audio_active",
            )

        # 3. Medical Image Upload (Direct analysis without unnecessary follow-ups)
        with in_tab3:
            uploaded_img_file = st.file_uploader(
                "Upload medical image (X-ray, ECG, MRI, Skin):",
                type=["png", "jpg", "jpeg", "dcm"],
                key="med_img_active",
            )
            if uploaded_img_file is not None and (
                not st.session_state.uploaded_image_info
                or st.session_state.uploaded_image_info.get("filename") != uploaded_img_file.name
            ):
                try:
                    pil_img = Image.open(uploaded_img_file)
                    st.image(pil_img, caption="Uploaded Medical Image", width=160)
                    with st.spinner("Analyzing image directly with CLIP Vision..."):
                        diag_res = analyze_medical_image_diagnostics(pil_img)
                        st.session_state.image_diagnostic_result = diag_res
                        st.session_state.uploaded_image_info = {
                            "modality": diag_res["modality"],
                            "features": diag_res["features"],
                            "confidence": diag_res["confidence"],
                            "filename": uploaded_img_file.name,
                        }
                except Exception as e:
                    st.error(f"Image analysis error: {e}")

        # --- DIRECT MEDICAL IMAGE DIAGNOSTIC RESULT ---
        if st.session_state.image_diagnostic_result:
            img_res = st.session_state.image_diagnostic_result
            if img_res.get("is_healthy"):
                st.markdown(
                    f"""
                    <div class="healthy-glass-card">
                        <div style="font-weight: 800; font-size: 1.1rem; margin-bottom: 4px; color: #166534;">✅ Image Findings: Unremarkable</div>
                        <div style="font-size: 1.02rem; font-weight: 600; color: #15803d; line-height: 1.5;">
                            "No obvious health concern detected based on available information. The person appears to be doing well."
                        </div>
                        <div style="margin-top: 8px; font-size: 0.88rem; color: #334155;"><strong>Findings:</strong> {img_res['findings']}</div>
                        <div style="margin-top: 4px; font-size: 0.88rem; color: #334155;"><strong>Recommended Department:</strong> 🏥 {img_res['department']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="dept-glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 800; color: #0284c7; font-size: 1.05rem;">🖼️ Direct Image Analysis</span>
                            <span class="glass-tag" style="background:#e0f2fe; color:#0369a1;">{img_res['modality']} ({img_res['confidence']*100:.0f}% conf.)</span>
                        </div>
                        <div style="margin-bottom: 6px;">
                            <strong style="color: #0f172a;">Key Findings:</strong>
                            <div style="color: #334155; font-size: 0.88rem; margin-top: 2px;">{img_res['findings']}</div>
                        </div>
                        <div style="margin-bottom: 6px;">
                            <strong style="color: #0f172a;">Possible Diagnosis:</strong>
                            <span style="color: #0369a1; font-weight: 700; font-size: 0.98rem;"> {img_res['possible_diagnosis']}</span>
                        </div>
                        <div>
                            <strong style="color: #0f172a;">Recommended Department:</strong>
                            <span style="background: rgba(224, 242, 254, 0.95); color: #0284c7; padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.9rem;">🏥 {img_res['department']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Intake Status Progress
        q_idx = st.session_state.med_question_idx
        total_q = st.session_state.med_total_questions

        if q_idx == 0:
            st.info("👋 **Describe your symptoms** via microphone, audio upload, or the text chat below to begin.")
        elif 1 <= q_idx <= total_q:
            st.markdown(
                f"""
                <div style="background: rgba(240, 253, 244, 0.88); border: 1px solid #86efac; border-radius: 12px; padding: 8px 16px; margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #166534; font-size: 0.9rem;">🩺 Follow-up Question {q_idx} of {total_q}</span>
                    <span style="font-size: 0.82rem; color: #15803d; font-weight: 600;">Spoken Voice Output below</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success("✅ **Consultation Intake Complete:** Assessment and department cards ready on right.")

        # --- NATURAL AI VOICE OUTPUT (TTS) FOR FOLLOW-UP QUESTIONS ---
        if st.session_state.med_current_voice_question:
            q_text = st.session_state.med_current_voice_question
            mp3_bytes = generate_voice_output_mp3(q_text)

            st.markdown(
                f"""
                <div class="voice-glass-box">
                    <span style="background: #0284c7; color: #ffffff; padding: 3px 10px; border-radius: 8px; font-size: 0.78rem; font-weight: 700;">🔊 Doctor's Spoken Question ({min(q_idx, total_q)} of {total_q})</span>
                    <h4 style="margin: 8px 0 6px 0; color: #0c4a6e; font-size: 1.18rem; font-weight: 800;">"{q_text}"</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if mp3_bytes:
                st.audio(mp3_bytes, format="audio/mp3", autoplay=True)

        # Medical Chat History Container
        chat_box_med = st.container(height=340)
        with chat_box_med:
            for msg in st.session_state.med_messages:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.write(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant", avatar="👩‍⚕️"):
                        st.write(msg["content"])
                elif msg["role"] == "system":
                    st.info(msg["content"])

        # Patient Input Processing Function (Voice or Text)
        def handle_patient_input(patient_text: str, voice_token: Optional[int] = None):
            """Advances follow-up questions 1 to 3 and triggers final assessment."""
            st.session_state.med_messages.append({"role": "user", "content": patient_text, "voice_id": voice_token})
            backend = get_llm_backend(active_model)

            # Turn 0: Initial symptom presentation -> Generate Question 1
            if st.session_state.med_question_idx == 0:
                st.session_state.med_initial_symptoms = patient_text
                st.session_state.med_question_idx = 1
                q1 = generate_medical_followup_question(
                    backend=backend,
                    question_idx=1,
                    total_questions=st.session_state.med_total_questions,
                    symptoms=st.session_state.med_initial_symptoms,
                    vitals=st.session_state.current_vitals,
                    vitals_taken=st.session_state.vitals_taken,
                    image_info=st.session_state.uploaded_image_info,
                    chat_history=st.session_state.med_messages,
                )
                st.session_state.med_current_voice_question = q1
                st.session_state.med_messages.append({
                    "role": "assistant",
                    "content": f"🔊 *[Spoken Question 1 of 3]*: {q1}",
                })

            # Turn 1: Patient answered Q1 -> Generate Question 2
            elif st.session_state.med_question_idx == 1:
                st.session_state.med_question_idx = 2
                q2 = generate_medical_followup_question(
                    backend=backend,
                    question_idx=2,
                    total_questions=st.session_state.med_total_questions,
                    symptoms=st.session_state.med_initial_symptoms,
                    vitals=st.session_state.current_vitals,
                    vitals_taken=st.session_state.vitals_taken,
                    image_info=st.session_state.uploaded_image_info,
                    chat_history=st.session_state.med_messages,
                )
                st.session_state.med_current_voice_question = q2
                st.session_state.med_messages.append({
                    "role": "assistant",
                    "content": f"🔊 *[Spoken Question 2 of 3]*: {q2}",
                })

            # Turn 2: Patient answered Q2 -> Generate Question 3
            elif st.session_state.med_question_idx == 2:
                st.session_state.med_question_idx = 3
                q3 = generate_medical_followup_question(
                    backend=backend,
                    question_idx=3,
                    total_questions=st.session_state.med_total_questions,
                    symptoms=st.session_state.med_initial_symptoms,
                    vitals=st.session_state.current_vitals,
                    vitals_taken=st.session_state.vitals_taken,
                    image_info=st.session_state.uploaded_image_info,
                    chat_history=st.session_state.med_messages,
                )
                st.session_state.med_current_voice_question = q3
                st.session_state.med_messages.append({
                    "role": "assistant",
                    "content": f"🔊 *[Spoken Question 3 of 3]*: {q3}",
                })

            # Turn 3: Patient answered Q3 -> Intake Complete! Generate Assessment
            elif st.session_state.med_question_idx >= 3:
                st.session_state.med_question_idx = 4
                st.session_state.med_current_voice_question = None

                conclusion_msg = "Thank you. I have collected all the necessary clinical information. Your assessment and department recommendations are ready on the right."
                st.session_state.med_messages.append({"role": "assistant", "content": conclusion_msg})

                assessment = generate_final_medical_assessment(
                    backend=backend,
                    symptoms=st.session_state.med_initial_symptoms,
                    vitals=st.session_state.current_vitals,
                    vitals_taken=st.session_state.vitals_taken,
                    image_info=st.session_state.uploaded_image_info,
                    chat_history=st.session_state.med_messages,
                )
                st.session_state.clinical_assessment = assessment

            st.rerun()

        # Handle Live Microphone input
        if audio_med is not None:
            a_bytes = audio_med.read()
            with st.spinner("Transcribing your spoken symptoms with Whisper..."):
                transcribed_med = transcribe_audio_bytes(a_bytes)
            if transcribed_med and not any(m.get("voice_id") == id(audio_med) for m in st.session_state.med_messages):
                handle_patient_input(transcribed_med, voice_token=id(audio_med))

        # Handle Audio File Upload input
        if uploaded_audio_file is not None:
            if not st.session_state.get("last_uploaded_audio_name") == uploaded_audio_file.name:
                st.session_state["last_uploaded_audio_name"] = uploaded_audio_file.name
                f_bytes = uploaded_audio_file.read()
                with st.spinner("Transcribing audio file with Whisper..."):
                    file_text = transcribe_audio_bytes(f_bytes, filename=uploaded_audio_file.name)
                if file_text:
                    handle_patient_input(file_text, voice_token=id(uploaded_audio_file))

        # Handle Text Chat input
        if user_med_text := st.chat_input("Describe your symptoms or answer the doctor's question..."):
            handle_patient_input(user_med_text)

    # --------------------------------------------------------------------------
    # RIGHT COLUMN: CLINICAL ASSESSMENT & SIMPLE DEPARTMENT CARDS
    # --------------------------------------------------------------------------
    with med_right:
        st.subheader("📋 Clinical Assessment & Department")

        assessment = st.session_state.clinical_assessment

        if assessment:
            # 1. Healthy / No Obvious Abnormality State
            if assessment.is_healthy:
                st.markdown(
                    """
                    <div class="healthy-glass-card">
                        <div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 6px; color: #166534;">
                            🌿 Health Status: Normal / No Apparent Concern
                        </div>
                        <div style="font-size: 1.02rem; font-weight: 600; line-height: 1.5; color: #15803d;">
                            "No obvious health concern detected based on available information. The person appears to be doing well."
                        </div>
                        <p style="margin-top: 8px; font-size: 0.85rem; color: #4b5563;">
                            Note: This assessment is based on the provided information and does not guarantee complete health. Continue regular checkups with your General Physician.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 2. Emergency Warning Signs Banner (if present)
            elif assessment.emergency_warning_signs:
                st.markdown(
                    f"""
                    <div class="warning-glass-box">
                        <strong style="font-size: 0.95rem;">🚨 Immediate Warning Signs Detected:</strong>
                        <ul style="margin: 4px 0 0 0; padding-left: 20px;">
                            {''.join(f'<li>{flag}</li>' for flag in assessment.emergency_warning_signs)}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3. Department Recommendation in Simple Glass Card
            st.markdown(
                f"""
                <div class="dept-glass-card">
                    <div style="font-size: 0.78rem; color: #0284c7; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">Recommended Department</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #0369a1; margin: 3px 0;">🏥 {assessment.recommended_department}</div>
                    <div style="font-size: 0.88rem; color: #334155; line-height: 1.45;">{assessment.department_reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 4. Possible Conditions
            st.markdown("##### 🔬 Possible Conditions")
            for rank, diag in enumerate(assessment.top_3_diseases[:3], 1):
                conf = min(max(int(diag.confidence), 1), 100)
                color = "#dc2626" if conf >= 60 else "#ea580c" if conf >= 30 else "#16a34a"

                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 14px 18px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                            <strong style="color: #0f172a; font-size: 0.96rem;">#{rank}. {diag.condition}</strong>
                            <span style="font-weight: 700; color: {color}; font-size: 0.88rem;">{conf}% Match</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #475569;">
                            <strong>Evidence:</strong> {diag.evidence}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(conf / 100)

            # 5. Summary & Next Steps
            st.markdown("##### 📝 Clinical Summary & Recommended Steps")
            st.markdown(
                f"""
                <div class="glass-card" style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    {assessment.clinical_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )

            for step in assessment.recommended_next_steps:
                st.markdown(f"- **{step}**")

            # Reset Button
            st.write("")
            if st.button("🔄 Start New Patient Intake", use_container_width=True):
                st.session_state.med_messages = []
                st.session_state.med_question_idx = 0
                st.session_state.med_current_voice_question = None
                st.session_state.med_initial_symptoms = ""
                st.session_state.clinical_assessment = None
                st.session_state.image_diagnostic_result = None
                st.session_state.uploaded_image_info = None
                st.session_state.vitals_taken = False
                st.rerun()

        else:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; padding: 46px 20px;">
                    <span style="font-size: 2.8rem;">🩺</span>
                    <h4 style="color: #0284c7; margin: 12px 0 6px 0; font-weight: 800; font-family: 'Space Grotesk', sans-serif;">Clinical Assessment Pending</h4>
                    <p style="color: #64748b; font-size: 0.92rem; max-width: 340px; margin: 0 auto; line-height: 1.55;">
                        Step {min(q_idx, total_q)} of {total_q}. Describe your symptoms and answer the short spoken questions.
                        The possible conditions and recommended department card will appear here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
