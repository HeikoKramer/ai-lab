from huggingface_hub import list_repo_files

print("Checking Lightricks/ltxv-spatial-upscaler-0.9.8...")
try:
    files = list_repo_files("Lightricks/ltxv-spatial-upscaler-0.9.8")
    print("Upscaler is accessible.")
except Exception as e:
    print(f"Upscaler access failed: {e}")
