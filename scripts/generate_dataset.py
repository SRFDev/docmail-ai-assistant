import json
import random
import logging
import time
from faker import Faker
from config.loader import AppConfig
from core.aws_service import AwsService
from llama_index.core import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fake = Faker()

# --- Generator Configuration ---
SPECIALTIES = ["Cardiology", "Dermatology", "Primary Care", "Endocrinology"]

TONES = {
    "Cardiology": ["reassuring", "data-driven", "formal", "urgent"],
    "Dermatology": ["empathetic", "descriptive", "concise"],
    "Primary Care": ["friendly", "holistic", "firm"],
    "Endocrinology": ["supportive", "analytical", "encouraging"]
}

SCENARIOS = {
    "Cardiology": [
        "Patient concerned about palpitations after coffee",
        "Follow-up question about statin side effects",
        "Blood pressure monitor readings seem high",
    ],
    "Dermatology": [
        "Question about a new mole changing color",
        "Rash worsening after using prescribed cream",
        "Request for sunscreen recommendations",
    ],
    "Primary Care": [
        "Request for antibiotics for a cold",
        "Back pain from working from home",
        "Scheduling annual physical",
    ],
    "Endocrinology": [
        "Blood sugar levels fluctuating in the morning",
        "Forgot to take thyroid medication for two days",
        "Weight gain despite diet changes"
    ]
}

GENERATOR_PROMPT = """
You are a synthetic data generator for a medical AI training set.
Generate a realistic email exchange between a patient and a physician.

PARAMETERS:
- Physician Name: {physician_name}
- Specialty: {specialty}
- Physician Tone: {physician_tone}
- Patient Name: {patient_name}
- Scenario: {scenario}

INSTRUCTIONS:
1. Write a **Patient Email** that reflects the scenario. The patient should sound natural (maybe anxious, maybe casual).
2. Write a **Physician Reply** that strictly adheres to the assigned tone. The reply must be medically responsible (no definitive diagnosis over email) and professional.
3. Output MUST be valid JSON only. Do not add markdown blocks.

JSON FORMAT:
{{
  "physician_persona": {{
    "name": "{physician_name}",
    "specialty": "{specialty}",
    "tone": "{physician_tone}"
  }},
  "medical_scenario": {{
    "topic": "{scenario}"
  }},
  "patient_email": "...",
  "physician_reply": "..."
}}
"""

def generate_record(svc: AwsService, specialty: str):
    # Randomize parameters
    p_name = fake.name()
    doc_name = f"Dr. {fake.last_name()}"
    tone = random.choice(TONES[specialty])
    scenario = random.choice(SCENARIOS[specialty])

    prompt = GENERATOR_PROMPT.format(
        physician_name=doc_name,
        specialty=specialty,
        physician_tone=tone,
        patient_name=p_name,
        scenario=scenario
    )

    try:
        # Call Bedrock
        response = svc.llm.complete(prompt)
        text = response.text.strip()
        
        # Clean potential markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        record = json.loads(text)
        # Add a unique ID
        record['id'] = fake.uuid4()
        return record
        
    except Exception as e:
        logger.error(f"Failed to generate record: {e}")
        return None

def main():
    config = AppConfig()
    svc = AwsService.get_instance(config)
    
    output_file = config.physician_style_path # Defined in config.toml
    target_count = 20 # Start small for the sprint (increase to 50 if time permits)
    
    logger.info(f"Starting generation of {target_count} records to {output_file}...")
    
    records = []
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for i in range(target_count):
        spec = random.choice(SPECIALTIES)
        logger.info(f"Generating record {i+1}/{target_count} ({spec})...")
        
        record = generate_record(svc, spec)
        if record:
            records.append(record)
        
        # Sleep briefly to avoid hitting Bedrock rate limits (though Sonnet is fast)
        time.sleep(0.5)

    # Write to JSONL
    with open(output_file, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    logger.info(f"✅ Generation Complete. Saved {len(records)} records.")

if __name__ == "__main__":
    main()