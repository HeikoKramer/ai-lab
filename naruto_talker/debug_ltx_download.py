import torch
from huggingface_hub import list_repo_files

print("Checking Lightricks/LTX-Video...")
try:
    files = list_repo_files("Lightricks/LTX-Video")
    print("Lightricks/LTX-Video is accessible. Files found.")
    print(files[:5])
except Exception as e:
    print(f"Lightricks/LTX-Video access failed: {e}")

print("\nChecking Lightricks/LTX-Video-0.9.8-dev again...")
try:
    files = list_repo_files("Lightricks/LTX-Video-0.9.8-dev")
    print("Lightricks/LTX-Video-0.9.8-dev is accessible.")
except Exception as e:
    print(f"Lightricks/LTX-Video-0.9.8-dev access failed: {e}")
