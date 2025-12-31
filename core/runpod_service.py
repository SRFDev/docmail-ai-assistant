# core/runpod_service.py
import requests
import logging
from core.llm_interface import LLMProvider

logger = logging.getLogger(__name__)

class RunPodService(LLMProvider):
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        # Construct the vLLM / OpenAI-compatible endpoint URL
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        
        if not self.api_key or not endpoint_id:
            logger.error("❌ RunPod Service initialized without credentials. Check .env and config.toml")

    def generate_draft(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        logger.info("Generating draft via RunPod (Fine-Tuned Llama 3)...")
        
        # Prepare Payload (OpenAI Chat Format for vLLM)
        payload = {
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # Nest parameters here so vLLM finds them
                "sampling_params": {
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "temperature": kwargs.get("temperature", 0.6),
                    "stop": ["<|end_of_text|>", "<|eot_id|>"]
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 60s timeout to allow for Serverless Cold Start
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse vLLM Response
            if "output" in data and len(data["output"]) > 0:
                choices = data["output"][0].get("choices", [])
                if choices:
                    first_choice = choices[0]
                    text_content = ""
                    
                    # Handle Text vs Tokens
                    if "text" in first_choice:
                        text_content = first_choice["text"]
                    elif "tokens" in first_choice:
                        raw = first_choice["tokens"]
                        text_content = "".join(raw) if isinstance(raw, list) else raw
                    
                    # FINAL CLEANUP: Strip stop tokens if they leaked into the string
                    return text_content.replace("<|end_of_text|>", "").replace("<|eot_id|>", "").strip()
                
            return "Error: Model returned no content."
            
        except requests.exceptions.Timeout:
            logger.error("RunPod Request Timed Out (Cold Start)")
            raise TimeoutError("The Physician Model is waking up. Please try again in 1 minute.")
        except Exception as e:
            logger.error(f"RunPod Error: {e}")
            raise e
        

        