"""
run_demo.py: Runs a demonstration ED triage simulation offline using
example cases and personas without requiring an active external API key.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from triagesim.cases import load_cases
from triagesim.personas import NursePersona, PatientPersona, load_personas
from triagesim.metrics import summarize
from triagesim.schemas import NurseOutput, PatientOutput, RECORD_FIELDS, RECORD_UNKNOWN
from triagesim.simulation import simulate


class DemoLLM:
    """Simulated LLM backend for local offline demonstrations."""

    batched = True

    def __init__(self, name="demo-agent"):
        self.model = name
        self.usage = Counter()

    def complete(self, requests):
        responses = []
        for r in requests:
            out_type = r.output_type
            dialogue = r.messages[1]["content"] if len(r.messages) > 1 else ""

            if out_type is PatientOutput or "disclosed" in out_type.model_fields:
                resp = {
                    "utterance": "I've had severe chest tightness and dizziness since this morning.",
                    "disclosed": "Severe chest pain and dizziness",
                }
            else:
                # Nurse agent logic
                schema = out_type.model_json_schema()
                actions_prop = schema["properties"].get("action", {})
                actions = actions_prop.get("enum", [])
                
                vitals_prop = schema["properties"].get("vital", {})
                vitals = vitals_prop.get("enum", [])

                needs_understood = schema["properties"].get("understood", {}).get("type") == "string"
                understood = "Patient reports acute chest tightness and dizziness." if needs_understood else None

                patient_turns = dialogue.count("Patient:")
                
                if "check_vital" in actions and "heartrate" in vitals and "[Vital] heartrate" not in dialogue:
                    resp = {
                        "action": "check_vital",
                        "vital": "heartrate",
                        "utterance": "I'm going to check your heart rate and pulse first.",
                        "red_flags": [],
                        "triage": 2,
                        "confidence": "medium",
                        "explanation": "Assessing cardiac stability.",
                        "understood": understood,
                    }
                elif "log_red_flag" in actions and "[Vital] heartrate" in dialogue and "Tachycardia" not in dialogue:
                    resp = {
                        "action": "log_red_flag",
                        "vital": None,
                        "utterance": None,
                        "red_flags": ["Potential Acute Coronary Syndrome", "Tachycardia"],
                        "triage": 2,
                        "confidence": "high",
                        "explanation": "Vitals and reported symptoms suggest high acuity.",
                        "understood": understood,
                    }
                elif patient_turns >= 2 and "end" in actions:
                    resp = {
                        "action": "end",
                        "vital": None,
                        "utterance": "We need to get you seen by an emergency doctor right away. Please follow me.",
                        "red_flags": ["Acute Chest Pain"],
                        "triage": 2,
                        "confidence": "high",
                        "explanation": "Immediate medical attention needed (ESI 2).",
                        "understood": understood,
                    }
                else:
                    resp = {
                        "action": "utterance",
                        "vital": None,
                        "utterance": "Can you describe the pain? Does it radiate to your arm or jaw?",
                        "red_flags": [],
                        "triage": 2,
                        "confidence": "medium",
                        "explanation": "Exploring chest pain characteristics.",
                        "understood": understood,
                    }

            responses.append(json.dumps(resp))
        return responses


def main():
    print("=" * 70)
    print("TriageSim Offline Demonstration Run")
    print("=" * 70)

    cases = load_cases("examples/cases.jsonl")[:2]
    nurse_personas = load_personas("examples/personas/nurse.yaml", NursePersona)[:1]
    patient_personas = load_personas("examples/personas/patient.yaml", PatientPersona)[:1]

    print(f"Loaded {len(cases)} cases, {len(nurse_personas)} nurse persona, {len(patient_personas)} patient persona.\n")

    demo_llm = DemoLLM("demo-local-agent")

    episodes = list(
        simulate(
            cases,
            nurse_llm=demo_llm,
            patient_llm=demo_llm,
            nurse_personas=nurse_personas,
            patient_personas=patient_personas,
            max_turns=6,
            min_questions=1,
        )
    )

    print(f"Generated {len(episodes)} simulated triage dialogues:\n")
    for i, ep in enumerate(episodes, 1):
        print(f"--- Episode {i}: Episode ID {ep['episode_id']} ---")
        print(f"Final Triage Level: ESI {ep['final_triage']} (End Reason: {ep['end_reason']})")
        print(f"Red Flags Logged: {ep.get('red_flags', [])}")
        print("\nDialogue Transcript:")
        for turn in ep["transcript"]:
            print(f"  [{turn['speaker'].upper()}]: {turn['text']}")
        print()

    summary = summarize(episodes)
    print("=" * 70)
    print("Summary Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()

