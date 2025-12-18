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
import logging
import sys

def setup_logging(logger_name: str, log_level: str = "INFO"):
    """
    Configures a specific logger for the application, not the root logger.
    """
    # Get the ROOT logger
    logger = logging.getLogger()
    
    log_level_enum = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_enum)
    
    logger.propagate = True  # DON'T stop messages from going to the root

    # This prevents duplicate messages if the function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handler and formatter as before
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Now, tell noisy libraries to be quiet
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google.cloud.aiplatform").setLevel(logging.WARNING)
    logging.getLogger("google.cloud.aiplatform_v1").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)    
    
    # Log a message using the logger we just configured
    logger.info(f"Logger '{logger_name}' configured successfully.")


