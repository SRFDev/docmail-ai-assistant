# Copyright © 2025 SRF Development, Inc. All rights reserved.
#
# This file is part of the "Howie AI Assistant" project.
#
# This project is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by the Open Source
# Initiative.
#
# This project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with this project.
# If not, see <https://opensource.org/licenses/MIT>.
#
# SPDX-License-Identifier: MIT
import tomllib
from typing import Optional, Any
from core import constants

# This "private" module-level variable will hold our single instance.
_instance: Optional["PromptManager"] = None

class PromptManager:
    """A class to load, manage, and format prompts from a TOML file."""
    def __init__(self, prompts_file_path: str):
        print(f"INFO:     Initializing PromptManager with prompts file: {prompts_file_path}")
        try:
            with open(prompts_file_path, "rb") as f:
                self._prompts = tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found at: {prompts_file_path}")

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
    This should be called ONCE at application startup.
    """
    global _instance
    if _instance is not None:
        print("WARN:     PromptManager is already initialized. Ignoring subsequent calls.")
        return
    _instance = PromptManager(prompts_file_path)


def get_prompt_manager() -> PromptManager:
    """
    Retrieves the singleton instance of the PromptManager.
    Will raise an exception if initialize_prompt_manager has not been called.
    """
    if _instance is None:
        raise RuntimeError(
            "PromptManager has not been initialized. "
            "Please call initialize_prompt_manager() at application startup."
        )
    return _instance
