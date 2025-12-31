import os
import requests
import json
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

class RunPodService:
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        
        if not self.api_key or not self.endpoint_id:
            raise ValueError("❌ CRITICAL: Missing RunPod credentials in .env")

    def generate_reply(self, patient_message: str, tone: str = "professional"):
        """
        Sends a prompt to the Fine-Tuned Llama 3 Physician model.
        """
        print(f"📡 Connecting to RunPod Endpoint: {self.endpoint_id}")
        
        # 1. Construct the Prompt (System + User)
        system_prompt = (
            "You are a helpful, professional physician's assistant. "
            "Draft a concise response to the patient inquiry. "
            "Do not diagnose. Refer serious issues to an in-person visit."
        )
        
        # 2. Payload Structure (vLLM / OpenAI Format)
        payload = {
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Patient Query: {patient_message}\n\nDraft a {tone} response."}
                ],
                "max_tokens": 250,
                "temperature": 0.3
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 3. Execution with Timeout
        try:
            # 60s timeout handles Cold Starts gracefully
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status() # Raise error for 4xx/5xx codes
            
            data = response.json()
            
            # 4. Parse vLLM Output (Protective Logic)
            if "output" in data and len(data["output"]) > 0:
                # RunPod Serverless wrapper structure
                choices = data["output"][0].get("choices", [])
                if choices:
                    return choices[0]["tokens"][0] # The generated text
            
            return "⚠️ Error: Model returned empty response."

        except requests.exceptions.Timeout:
            return "⚠️ Error: RunPod timed out (Cold Start). Try again in 30s."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

# Singleton Instance for import
llm_client = RunPodService()
