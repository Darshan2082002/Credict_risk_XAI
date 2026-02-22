import json
from datetime import datetime

def register_model(metadata_path):
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata["registered_at"] = str(datetime.now())

    with open("governance/model_registry.json", "w") as f:
        json.dump(metadata, f, indent=4)