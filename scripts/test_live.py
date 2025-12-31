import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv() # Load from .env

# SECURE LOADING
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")

PROMPT = "You are a physician. Explain 'Tachycardia' to a patient."

# Safety Check
if not API_KEY or not ENDPOINT_ID:
    print("❌ Error: Missing credentials in .env file")
    exit(1)


# --- THE CODE ---
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "messages": [
            {"role": "system", "content": "You are a helpful physician assistant."},
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }
}

print(f"📡 Sending request to {ENDPOINT_ID}...")
print("⏳ Waiting for Cold Start (this may take 2-3 minutes)...")

start_time = time.time()
try:
    # Timeout set to 300s (5 mins) to allow for cold start
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    
    elapsed = time.time() - start_time
    print(f"✅ Response received in {elapsed:.1f} seconds!")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- MODEL OUTPUT ---")
        # Handle vLLM response format
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"💥 Request failed: {e}")