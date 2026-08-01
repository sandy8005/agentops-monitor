import json
import os

ALLOWED_WORK_MODES = ["remote", "hybrid", "onsite"]
ALLOWED_EMPLOYMENT_TYPES = ["full-time", "part-time", "contract", "internship"]


def receive_user_input(config_path="user_input.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    required_fields = ["resume_file", "target_role"]
    for field in required_fields:
        if not data.get(field):
            raise ValueError(f"Missing required field: {field}")

    if not os.path.exists(data["resume_file"]):
        raise FileNotFoundError(f"Resume file not found: {data['resume_file']}")

    if not data["resume_file"].lower().endswith(".pdf"):
        raise ValueError("resume_file must be a .pdf")

    work_mode = data.get("work_mode", "").lower()
    if work_mode and work_mode not in ALLOWED_WORK_MODES:
        raise ValueError(f"work_mode must be one of {ALLOWED_WORK_MODES}")

    employment_type = data.get("employment_type", "").lower()
    if employment_type and employment_type not in ALLOWED_EMPLOYMENT_TYPES:
        raise ValueError(f"employment_type must be one of {ALLOWED_EMPLOYMENT_TYPES}")

    return {
        "resume_file": data["resume_file"],
        "target_role": data["target_role"],
        "location": data.get("location", ""),
        "work_mode": work_mode,
        "employment_type": employment_type
    }