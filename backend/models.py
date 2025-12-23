# Copyright © 2025 SRF Development, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pydantic import BaseModel, Field
from typing import List, Optional

class DraftRequest(BaseModel):
    """The input model for a patient email draft request."""
    patient_email: str = Field(..., description="The full text of the patient's email.")

class DraftResponse(BaseModel):
    """The output model for the drafted reply."""
    draft_reply: str = Field(..., description="The AI-generated email draft.")
    source_nodes: List[str] = Field(..., description="List of style examples used as context.")

    