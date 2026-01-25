import os
import sys

# Mock Gradio progress
class MockProgress:
    def __call__(self, val, desc=""):
        print(f"[Progress {val:.2f}] {desc}")

def main():
    # Ensure we are in project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    print("Initializing App components...")
    from app import generate_pipeline
    
    gender = "Male"
    pitch = 0.0
    speed = 1.0
    emotions = ["confident"]
    visual_desc = "blonde spiky hair, blue eyes, whisker marks on cheeks"
    speech_text = "I am ready to become the Hokage! Dattebayo!"
    seed = 999 
    
    print("Running Full Pipeline Test (Headless)...")
    try:
        # visual_prompt (DE), negative_prompt, seed, voice_prompt (DE), ref_audio, script_text, output_language, lipsync_offset, restoration_strength
        inputs=[
            "Eine Cyberpunk Hackerin mit lila Haaren, Neonlicht", # Visual (DE to be translated)
            "low quality, blurry, painting", # Negative
            42, # Seed
            "Eine ruhige, tiefe Stimme, sehr ernst", # Voice (DE to be translated)
            None, # Ref Audio
            "System bereit. Zugriff gewährt. Wir starten Phase 1.", # Script
            "de", # Language
            0, # Offset
            1.0, # Restoration Strength
            "Static" # Motion Speed
        ]
        results = generate_pipeline(*inputs, progress=MockProgress())
        portrait, speech, silent, final = results
        
        print("\n=== SUCCESS === ")
        print(f"Portrait: {portrait}")
        print(f"Speech: {speech}")
        print(f"Silent Video: {silent}")
        print(f"Final Video: {final}")
        
        if os.path.exists(final):
            print("Final video file exists.")
        else:
            print("ERROR: Final video file missing.")
            
    except Exception as e:
        print(f"\n=== FAILURE ===\n{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
