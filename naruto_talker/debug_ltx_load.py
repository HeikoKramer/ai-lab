import torch
from diffusers import LTXPipeline, LTXConditionPipeline

model_id = "Lightricks/LTX-Video"
print(f"Loading {model_id}...")
try:
    # Try ConditionPipeline first as we need I2V
    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    print("LTXConditionPipeline loaded successfully.")
except Exception as e:
    print(f"LTXConditionPipeline failed: {e}")
    try:
        pipe = LTXPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        print("LTXPipeline loaded successfully. (Note: this might be T2V only)")
    except Exception as e2:
        print(f"LTXPipeline failed: {e2}")
