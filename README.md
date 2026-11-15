# MediAssist AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Powered by: PyTorch](https://img.shields.io/badge/ML-PyTorch%20%7C%20Transformers-orange.svg)](https://pytorch.org/)

**MediAssist AI** is a production-grade, multimodal AI platform that combines an intelligent clinical healthcare assistant with a versatile conversational AI chatbot. Featuring a modern light-themed glassmorphism interface, natural neural voice synthesis, computer vision diagnostics, and real-time streaming, MediAssist delivers an intuitive and responsive clinical triage experience.

---

## Key Features

### 🩺 Medical Assistant Mode
- **Multimodal Patient Input**: Supports live microphone speech, audio file uploads, medical image scans, and text input.
- **Computer Vision Diagnostics**: Instant medical image evaluation using zero-shot visual representation (CLIP) to identify conditions and route to the correct medical department without unnecessary questions.
- **Spoken Follow-Up Protocol**: Generates concise, dynamic medical follow-up questions delivered via natural AI voice output (Edge-TTS) one question at a time.
- **Differential Assessment & Triage**: Delivers top-3 probable conditions, concise clinical assessments, and recommended specialist departments (*Cardiology, Dermatology, Neurology, Pulmonology, Orthopedics, Dentistry, General Physician*).
- **Healthy Case Recognition**: Recognizes when vital indicators appear normal, offering reassurance and preventive health guidance.

### 💬 General AI Chat Mode
- **Unrestricted Conversational AI**: ChatGPT-style interface for general knowledge, coding, writing, research, and technical questions.
- **Real-Time Token Streaming**: Low-latency token-by-token generation for an immediate, responsive experience.
- **Strict Isolation**: Completely isolated from medical triage logic and clinical restrictions.

### 🚑 TriageSim Engine (CLI & Evaluation)
- **Synthetic Dialogue Simulation**: Generates multi-speaker clinical dialogues grounded in real emergency department (ED) vignettes (ESI and ATS protocols).
- **Interactive Terminal Triage**: Live doctor/nurse-to-patient interactive interview mode (`live_conversation.py`).

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA GPU recommended for local inference (CPU fallback supported)
- [Git](https://git-scm.com/)

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/dipankarsrirag/triage-sim.git
cd triage-sim
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory (refer to `.env.example`):

```bash
cp .env.example .env
```

Set your configuration parameters in `.env`:

```env
HF_TOKEN=your_huggingface_token_here
MODEL=hf-local:Qwen/Qwen2.5-1.5B-Instruct
```

*Note: You can also use OpenRouter or hosted API backends by specifying `MODEL=openrouter:<model_id>` and adding `OPENROUTER_API_KEY`.*

### 4. Run the Web Application

Launch the Streamlit platform:

```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`.

---

## Command-Line Usage

### Interactive Live Triage Simulation
Interview an AI-simulated patient in real time from your terminal:

```bash
python live_conversation.py
```

### Batch Triage Simulation
Run automated clinical evaluation across patient cases:

```bash
triagesim --cases examples/cases.jsonl \
    --nurse-personas examples/personas/nurse.yaml \
    --patient-personas examples/personas/patient.yaml \
    --out outputs/simulation_results.jsonl
```

---

## Architecture & Technology Stack

- **Frontend & UI**: Streamlit with custom CSS3 glassmorphism, responsive animations, and HTML5 audio player.
- **Speech Recognition (STT)**: OpenAI Whisper for robust acoustic transcription.
- **Voice Synthesis (TTS)**: Microsoft Edge Neural TTS (`en-US-JennyNeural`) for high-fidelity spoken follow-ups with gTTS and pyttsx3 fallbacks.
- **Computer Vision**: Pretrained OpenAI CLIP for zero-shot clinical image classification.
- **Inference & Language**: Hugging Face Transformers (`Qwen2.5-1.5B-Instruct` or hosted models) with Pydantic schema validation.

---

## Disclaimer

MediAssist AI and TriageSim are developed for educational, simulation, and research assistance. They are not a replacement for professional clinical judgment, emergency medical services, or formal medical diagnosis. Always consult certified healthcare professionals for medical emergencies and clinical care.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
