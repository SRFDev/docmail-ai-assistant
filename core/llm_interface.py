# core/llm_interface.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers (Strategy Pattern).
    Ensures Bedrock and RunPod implementations are interchangeable
    within the main application logic.
    """
    
    @abstractmethod
    def generate_draft(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generates a text response based on system and user prompts.
        
        Args:
            system_prompt (str): The 'persona' or instruction.
            user_prompt (str): The user's input (e.g., patient email).
            **kwargs: Provider-specific params (temperature, max_tokens).
            
        Returns:
            str: The generated text (string only).
        """
        pass