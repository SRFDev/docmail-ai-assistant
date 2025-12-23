# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import tomllib
import logging
from typing import Optional, Any
from core import constants

logger = logging.getLogger(__name__)

# Singleton instance
_instance: Optional["PromptManager"] = None

class PromptManager:
    """A class to load, manage, and format prompts from a TOML file."""
    def __init__(self, prompts_file_path: str):
        logger.info(f"Initializing PromptManager with file: {prompts_file_path}")
        try:
            with open(prompts_file_path, "rb") as f:
                self._prompts = tomllib.load(f)
        except FileNotFoundError:
            logger.error(f"Prompts file not found at: {prompts_file_path}")
            raise

    def get_prompt(self, section: str, name: str) -> str:
        try:
            return self._prompts[section][name]
        except KeyError:
            raise KeyError(f"Prompt '{name}' not found in section '{section}'.")

    def format_prompt(self, section: str, name: str, **kwargs: Any) -> str:
        template = self.get_prompt(section, name)
        return template.format(**kwargs)


def initialize_prompt_manager(prompts_file_path: str = constants.PROMPTS_FILE_PATH): 
    """
    Initializes the singleton instance of the PromptManager.
    """
    global _instance
    if _instance is not None:
        logger.warning("PromptManager is already initialized. Ignoring call.")
        return
    _instance = PromptManager(prompts_file_path)


def get_prompt_manager() -> PromptManager:
    """
    Retrieves the singleton instance.
    """
    if _instance is None:
        raise RuntimeError("PromptManager has not been initialized.")
    return _instance

