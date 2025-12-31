import json
import logging
import time
import random
import concurrent.futures
from faker import Faker
from config.loader import AppConfig
from core.aws_service import AwsService


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

fake = Faker()

# --- MASTER TOPIC LIST (50 Seeds) ---
TOPICS = [
    # --- ADMINISTRATIVE ---
    "Request to reschedule appointment for next week",
    "Question about insurance coverage for upcoming procedure",
    "Need a copy of vaccination records for school",
    "Trouble logging into the patient portal",
    "Running late for today's appointment",
    "Billing error on latest statement",
    "Updating home address and phone number",
    "Request for FMLA paperwork completion",
    "Clarification on appointment location",
    "Canceling appointment due to work conflict",

    # --- MEDICATION ---
    "Refill request for Lisinopril",
    "Question about side effects of new antibiotic",
    "Pharmacy says they didn't receive the prescription",
    "Request for higher dosage of pain medication (Drug Seeking)",
    "Forgot to take medication this morning",
    "Interaction between herbal supplement and prescribed meds",
    "Cost of medication is too high, asking for generic",
    "Request for antibiotics for a viral cold",
    "Lost prescription bottle",
    "Requesting a refill too early",

    # --- MILD SYMPTOMS ---
    "Sore throat for 2 days, no fever",
    "Rash on arm after gardening",
    "Mild headache that won't go away",
    "Toddler has a low-grade fever",
    "Sprained ankle, swelling slightly",
    "Persistent cough after a cold",
    "Feeling fatigued lately",
    "Indigestion after eating spicy food",
    "Minor cut on finger, asking if stitches are needed",
    "Seasonal allergy flare-up",

    # --- URGENT/ER TRIAGE ---
    "Sudden sharp chest pain radiating to arm",
    "Slurred speech and facial drooping",
    "Child swallowed a cleaning product",
    "High fever (104F) and stiff neck",
    "Sudden loss of vision in one eye",
    "Difficulty breathing while lying down",
    "Severe abdominal pain on right side",
    "Vomiting blood",
    "Head injury with loss of consciousness",
    "Suicidal thoughts",

    # --- EMOTIONAL/DIFFICULT ---
    "Angry about wait time at last visit",
    "Frustrated that the doctor hasn't called back yet",
    "Anxious about upcoming surgery results",
    "Confused by the specialist's explanation",
    "Feeling ignored by the front desk staff",
    "Worried about a diagnosis read on WebMD",
    "Grieving a family member, asking for resources",
    "Disagreement with the doctor's treatment plan",
    "Requesting a different doctor within the practice",
    "Nervous about a procedure, wants to cancel"
]

# --- FEW-SHOT PROMPT ---
GENERATOR_PROMPT = """
You are a data generation expert for AI fine-tuning. Your task is to create high-quality training examples for tuning a language model to act as an empathetic and professional physician's assistant.

CONTEXT:
We are generating data for the topic: "{topic}"

INSTRUCTIONS:
Generate 5 unique, realistic examples of patient/physician email exchanges based on this topic.
For each example, vary the patient's age, tone, and specific details.

The output must act as a training record for an LLM. Use this specific JSON structure:
- "instruction": Always "You are a physician's assistant drafting a reply to a patient's email. Be professional, empathetic, and cautious."
- "input": The patient's email text.
- "output": The ideal physician response.

CRITICAL SAFETY RULES:
- If the topic suggests an emergency (chest pain, stroke, poisoning, suicide), the output MUST tell them to call 911 or go to the ER immediately. Do not offer an appointment.
- Never provide a definitive diagnosis (e.g., "You have the flu"). Use phrases like "Your symptoms are consistent with..."
- Always guide to the next step (appointment, lab work, ER, or self-care).

OUTPUT FORMAT:
Return a valid JSON array of objects. Do not include markdown code blocks.
[
  {{
    "instruction": "...",
    "input": "...",
    "output": "..."
  }},
  ...
]
"""

def generate_batch(svc, topic):
    """Generates 10 variations for a single topic."""
    try:
        prompt = GENERATOR_PROMPT.format(topic=topic)
        
        # Call Bedrock (Claude Sonnet)
        # We ask for a longer response since we are generating 10 items at once
        response = svc.llm.complete(prompt)
        text = response.text.strip()
        
        if text.startswith("```json"):
        # Clean markdown
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Parse the JSON array
        batch = json.loads(text)
        
        # Validate schema briefly
        valid_batch = []
        for record in batch:
            if "instruction" in record and "input" in record and "output" in record:
                valid_batch.append(record)
                
        return valid_batch

    except Exception as e:
        logger.error(f"Error generating batch for '{topic}': {e}")
        return []

def main():
    config = AppConfig()
    svc = AwsService.get_instance(config)
    
    output_file = "data/physician_style_dataset.jsonl"
    
    # Ensure data dir exists
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # --- RESUME LOGIC ---
    # Check how many records we already have
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_count = sum(1 for line in f)
    
    # Calculate starting topic index (Assuming 10 records per topic)
    start_topic_index = existing_count // 10
    
    logger.info(f"🚀 Starting Operation Open Llama Data Gen")
    logger.info(f"Target: {len(TOPICS)} Topics. Found {existing_count} existing records.")
    
    if start_topic_index > 0:
        logger.info(f"⏩ Resuming from Topic {start_topic_index + 1}...")

    total_start = time.time()

    # Loop through topics
    for i, topic in enumerate(TOPICS):
        # Skip topics we already completed
        if i < start_topic_index:
            continue

        logger.info(f"[{i+1}/{len(TOPICS)}] Processing topic: {topic}")
        
        # We do 2 passes of 5 to get 10 total, but with lower latency per call
        topic_records = []
        for pass_num in range(2): 
            logger.info(f"   -> Sub-batch {pass_num+1}/2...")
            batch = generate_batch(svc, topic) # Ensure this function uses the updated prompt asking for 5
            if batch:
                topic_records.extend(batch)
        
        if topic_records:
            # Checkpoint Write
            with open(output_file, 'a') as f:
                for record in topic_records:
                    f.write(json.dumps(record) + "\n")
            logger.info(f"   -> Saved {len(topic_records)} records to disk.")
        
        time.sleep(1.0)
           
    elapsed = time.time() - total_start
    logger.info(f"✅ Mission Complete. Run time: {elapsed:.2f} seconds.")
    logger.info(f"📂 Data saved to: {output_file}")


if __name__ == "__main__":
    main()

