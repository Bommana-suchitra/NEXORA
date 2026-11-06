import os
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from triagesim.cases import load_cases
from triagesim.llm import Request, generate, load_backend
from triagesim.personas import PatientPersona, load_personas
from triagesim.prompts import patient_messages
from triagesim.schemas import PatientOutput

# Load environment configuration (.env)
load_dotenv(find_dotenv(usecwd=True))

st.set_page_config(
    page_title="TriageSim - Interactive Emergency Department",
    page_icon="🚑",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .vital-alert {
        color: #d90429;
        font-weight: bold;
    }
    .vital-normal {
        color: #2b9348;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading AI model into GPU memory...")
def get_backend(model_spec: str):
    return load_backend(model_spec)


@st.cache_data
def get_cases_and_personas():
    cases = load_cases("examples/cases.jsonl")
    personas = load_personas("examples/personas/patient.yaml", PatientPersona)
    return cases, personas


cases, personas = get_cases_and_personas()

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🚑 TriageSim Controls")

    default_model = os.environ.get("MODEL", "hf-local:Qwen/Qwen2.5-1.5B-Instruct")
    model_choice = st.text_input(
        "Active Model Backend",
        value=default_model,
        help="Model string: e.g. hf-local:Qwen/Qwen2.5-1.5B-Instruct, hf:<repo_id>, or openrouter:<model_id>",
    )

    st.subheader("Select Clinical Vignette")
    case_options = {f"{c.case_id} ({c.chief_complaint})": c for c in cases}
    selected_case_name = st.selectbox("Emergency Case", list(case_options.keys()))
    selected_case = case_options[selected_case_name]

    st.subheader("Patient Persona Traits")
    persona_labels = {
        f"Persona {i+1} ({p.distress.capitalize()} / {p.disclosure.capitalize()})": p
        for i, p in enumerate(personas)
    }
    selected_persona_name = st.selectbox("Persona", list(persona_labels.keys()))
    selected_persona = persona_labels[selected_persona_name]

    if st.button("🔄 Reset / Start New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.vitals_taken = False
        st.session_state.triage_result = None
        st.rerun()

    st.divider()
    st.markdown("### Persona Profile")
    st.write(f"**Distress:** {selected_persona.distress}")
    st.write(f"**Disclosure:** {selected_persona.disclosure}")
    st.write(f"**Health Literacy:** {selected_persona.health_literacy}")
    st.write(f"**Verbosity:** {selected_persona.verbosity}")
    st.write(f"**Demeanor:** {selected_persona.demeanour}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "vitals_taken" not in st.session_state:
    st.session_state.vitals_taken = False
if "triage_result" not in st.session_state:
    st.session_state.triage_result = None

# --- Main App Layout ---
st.title("🏥 Emergency Department Triage Simulation")
st.caption(
    "Conduct a live clinical interview as the triage nurse. Ask diagnostic questions, assess vital signs, and assign the appropriate Emergency Severity Index (ESI) level."
)

# Patient Arrival Banner
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.info(f"**Patient Presentation:** {selected_case.chief_complaint}")
with col_info2:
    pain_level = selected_case.pain if selected_case.pain else "Not reported"
    st.info(f"**Reported Pain Score:** {pain_level} / 10")
with col_info3:
    st.info(f"**Acuity Scale:** Emergency Severity Index (ESI)")

# --- Vital Signs Section ---
st.subheader("🩺 Patient Vital Signs")
vcol1, vcol2, vcol3, vcol4, vcol5, vcol_btn = st.columns([1, 1, 1, 1, 1, 1.5])

v = selected_case.vitals

with vcol1:
    st.metric(
        "Heart Rate",
        f"{int(v['heartrate'])} bpm" if st.session_state.vitals_taken and "heartrate" in v else "--",
    )
with vcol2:
    st.metric(
        "Systolic BP",
        f"{int(v['sbp'])} mmHg" if st.session_state.vitals_taken and "sbp" in v else "--",
    )
with vcol3:
    st.metric(
        "Respirations",
        f"{int(v['resprate'])} /min" if st.session_state.vitals_taken and "resprate" in v else "--",
    )
with vcol4:
    st.metric(
        "SpO2",
        f"{int(v['o2sat'])}%" if st.session_state.vitals_taken and "o2sat" in v else "--",
    )
with vcol5:
    st.metric(
        "Temperature",
        f"{v['temperature']} °F" if st.session_state.vitals_taken and "temperature" in v else "--",
    )

with vcol_btn:
    st.write("")
    st.write("")
    if st.button("🌡️ Take Vital Signs", disabled=st.session_state.vitals_taken, use_container_width=True):
        st.session_state.vitals_taken = True
        current_turn = len(st.session_state.messages) // 2
        for vital_name, val in v.items():
            st.session_state.history.append({
                "turn": current_turn,
                "actor": "system",
                "event": "vital",
                "name": vital_name,
                "value": val,
            })
        st.session_state.messages.append({
            "role": "system",
            "content": f"🩺 **Vitals Taken:** HR: {v.get('heartrate')} bpm | SBP: {v.get('sbp')} mmHg | RR: {v.get('resprate')} | SpO2: {v.get('o2sat')}% | Temp: {v.get('temperature')}°F",
        })
        st.rerun()

st.divider()

# --- Chat Interface ---
st.subheader("💬 Live Nurse ↔ Patient Interview")

# Display previous conversation messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👩‍⚕️"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🧑"):
            st.write(msg["content"])
    elif msg["role"] == "system":
        st.info(msg["content"])

# User Input
if prompt := st.chat_input("Ask the patient a question (e.g. 'Can you describe your pain? When did this start?')..."):
    # Append nurse message
    st.session_state.messages.append({"role": "user", "content": prompt})
    current_turn = len(st.session_state.messages) // 2
    st.session_state.history.append({
        "turn": current_turn,
        "actor": "nurse",
        "action": "utterance",
        "utterance": prompt,
    })

    with st.chat_message("user", avatar="👩‍⚕️"):
        st.write(prompt)

    # Generate patient response
    with st.chat_message("assistant", avatar="🧑"):
        with st.spinner("Patient is replying..."):
            backend = get_backend(model_choice)
            messages = patient_messages(
                selected_persona, selected_case, script=None, history=st.session_state.history
            )
            req = Request(messages=messages, output_type=PatientOutput)
            [out] = generate(backend, [req])
            patient_reply = out.utterance if hasattr(out, "utterance") else str(out)
            st.write(patient_reply)

    st.session_state.messages.append({"role": "assistant", "content": patient_reply})
    st.session_state.history.append({
        "turn": current_turn,
        "actor": "patient",
        "utterance": patient_reply,
        "disclosed": getattr(out, "disclosed", patient_reply),
    })

st.divider()

# --- Triage Assessment & Submission ---
st.subheader("📋 Final Triage Determination")

col_triage1, col_triage2 = st.columns([2, 1])

with col_triage1:
    esi_level = st.radio(
        "Select Emergency Severity Index (ESI) Level:",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "Level 1: Resuscitation (Immediate life-saving intervention needed)",
            2: "Level 2: Emergent (High risk, acute chest pain/syncope/altered mental status)",
            3: "Level 3: Urgent (Multiple diagnostic resources needed, stable vitals)",
            4: "Level 4: Less Urgent (One resource needed, e.g. single X-ray/sutures)",
            5: "Level 5: Non-Urgent (No resources needed, prescription refill/exam only)",
        }[x],
        horizontal=False,
    )

with col_triage2:
    st.write("")
    st.write("")
    if st.button("✅ Submit Triage Decision", use_container_width=True, type="primary"):
        ground_truth = selected_case.acuity
        st.session_state.triage_result = {
            "assigned": esi_level,
            "ground_truth": ground_truth,
        }

if st.session_state.triage_result:
    res = st.session_state.triage_result
    assigned = res["assigned"]
    gt = res["ground_truth"]

    st.markdown("### 🎯 Assessment Feedback")
    if assigned == gt:
        st.success(
            f"🎉 **Correct Assessment!** You assigned **ESI Level {assigned}**, which matches the clinical ground truth."
        )
    elif assigned < gt:
        st.warning(
            f"⚠️ **Over-triage:** You assigned **ESI Level {assigned}**, but the clinical ground truth is **ESI Level {gt}** (patient was less acute)."
        )
    else:
        st.error(
            f"❌ **Under-triage Risk:** You assigned **ESI Level {assigned}**, but the clinical ground truth is **ESI Level {gt}** (patient required more urgent care!)."
        )

