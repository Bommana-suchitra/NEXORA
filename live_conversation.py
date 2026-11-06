"""
live_conversation.py: Interactive Live ED Triage Conversation.
Allows a human to interview an AI Patient in real-time, order vital signs,
and assign an emergency triage acuity level.
"""

import os
import sys
from dotenv import find_dotenv, load_dotenv

from triagesim.cases import load_cases
from triagesim.llm import Request, generate, load_backend
from triagesim.personas import PatientPersona, load_personas
from triagesim.prompts import patient_messages
from triagesim.schemas import PatientOutput

load_dotenv(find_dotenv(usecwd=True))


def main():
    print("=" * 70)
    print("🚑 TRIAGESIM: LIVE INTERACTIVE EMERGENCY TRIAGE")
    print("=" * 70)
    print("You are the Triage Nurse in the Emergency Department.")
    print("A patient has just arrived. Conduct the triage interview!")
    print("\nCommands during interview:")
    print("  /vitals        - Measure patient vital signs (BP, HR, SpO2, Temp)")
    print("  /triage <1-5>  - Conclude interview and assign Emergency Severity Level")
    print("  /quit          - Exit the session")
    print("=" * 70 + "\n")

    # Load cases and personas
    cases = load_cases("examples/cases.jsonl")
    patient_personas = load_personas("examples/personas/patient.yaml", PatientPersona)

    # Pick first case (or let user choose)
    case = cases[0]
    persona = patient_personas[0]

    model_spec = os.environ.get("MODEL", "hf-local:Qwen/Qwen2.5-1.5B-Instruct")
    print(f"Loading Patient AI with model: {model_spec} ...")
    backend = load_backend(model_spec)
    print("Patient AI is ready!\n")

    print(f"[Patient enters the triage bay: {case.age}-year-old {case.gender.lower()}]")
    print(f"[Persona Traits: {', '.join(f'{k}={v}' for k, v in persona.traits.items() if v)}]\n")

    history = []
    turn = 0

    while True:
        try:
            nurse_input = input("\n[YOU (Nurse)]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting triage session.")
            break

        if not nurse_input:
            continue

        if nurse_input.lower() in ("/quit", "exit", "quit"):
            print("Session ended.")
            break

        if nurse_input.lower() == "/vitals":
            print("\n--- MEASURED VITAL SIGNS ---")
            print(f"  Heart Rate:         {case.vitals.get('heartrate', 'N/A')} bpm")
            print(f"  Systolic BP:        {case.vitals.get('sbp', 'N/A')} mmHg")
            print(f"  Respiratory Rate:   {case.vitals.get('resprate', 'N/A')} bpm")
            print(f"  Oxygen Saturation:  {case.vitals.get('o2sat', 'N/A')}%")
            print(f"  Temperature:        {case.vitals.get('temperature', 'N/A')} °F")
            print("----------------------------\n")
            # Record in history
            for vital_name, val in case.vitals.items():
                history.append({
                    "turn": turn,
                    "actor": "system",
                    "event": "vital",
                    "name": vital_name,
                    "value": val,
                })
            continue

        if nurse_input.lower().startswith("/triage"):
            parts = nurse_input.split()
            if len(parts) < 2 or not parts[1].isdigit():
                print("Usage: /triage <level 1-5> (e.g. /triage 2)")
                continue
            chosen_level = int(parts[1])
            ground_truth = case.acuity
            print("\n" + "=" * 70)
            print("📋 TRIAGE RESULTS & FEEDBACK")
            print("=" * 70)
            print(f"Your Triage Assignment:  ESI Level {chosen_level}")
            print(f"Ground-Truth Urgency:    ESI Level {ground_truth}")
            if chosen_level == ground_truth:
                print("✅ Exact Match! Excellent clinical assessment.")
            elif chosen_level < ground_truth:
                print("⚠️ Over-triage (you assigned a higher urgency than ground truth).")
            else:
                print("❌ Under-triage (patient acuity was more severe than assessed!).")

            print(f"\nChief Complaint: {case.chief_complaint}")
            print(f"Final ED Disposition: {case.metadata.get('disposition', 'Admitted/Observed')}")
            print("=" * 70)
            break

        # Normal dialogue turn
        history.append({
            "turn": turn,
            "actor": "nurse",
            "action": "utterance",
            "utterance": nurse_input,
        })

        # Ask patient model to reply
        messages = patient_messages(persona, case, script=None, history=history)
        req = Request(messages=messages, output_type=PatientOutput)
        
        print("\n[PATIENT is replying...]")
        [out] = generate(backend, [req])
        
        patient_text = out.utterance if hasattr(out, "utterance") else str(out)
        print(f"[PATIENT]: {patient_text}")

        history.append({
            "turn": turn,
            "actor": "patient",
            "utterance": patient_text,
            "disclosed": getattr(out, "disclosed", patient_text),
        })
        turn += 1


if __name__ == "__main__":
    main()

