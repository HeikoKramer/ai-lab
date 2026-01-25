import os
from huggingface_hub import snapshot_download, hf_hub_download

base_dir = "MuseTalk/models"
os.makedirs(base_dir, exist_ok=True)

def dl(repo, local_dir, allow_patterns=None):
    print(f"Downloading {repo} to {local_dir}...")
    snapshot_download(
        repo_id=repo,
        local_dir=local_dir,
        allow_patterns=allow_patterns
    )

# 1. MuseTalk models
dl("TMElyralab/MuseTalk", base_dir, allow_patterns=["musetalk/*", "musetalkV15/*"])

# 2. Whisper
dl("openai/whisper-tiny", os.path.join(base_dir, "whisper"))

# 3. DW Pose
dl("yzd-v/DWPose", os.path.join(base_dir, "dwpose"))

# 4. SD VAE
dl("stabilityai/sd-vae-ft-mse", os.path.join(base_dir, "sd-vae"))

# 5. Face Parse (already downloaded via gdown in shell script, but let's ensure)
# Since it's google drive, we skip here or assume user has it. 
# But let's check if the shell script succeeded for it.
if not os.path.exists(os.path.join(base_dir, "face-parse-bisent/79999_iter.pth")):
    print("WARNING: face-parse-bisent weights missing. Please ensure gdown script ran or download manually.")
else:
    print("face-parse-bisent weights present.")
    
# 6. ResNet18
resnet_path = os.path.join(base_dir, "face-parse-bisent/resnet18-5c106cde.pth")
if not os.path.exists(resnet_path):
    print("Downloading ResNet18...")
    os.system(f"wget -O {resnet_path} https://download.pytorch.org/models/resnet18-5c106cde.pth")

print("Download complete.")
