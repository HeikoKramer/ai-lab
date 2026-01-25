import os
import sys
import time
import traceback

# Ensure src is in path
sys.path.append(os.path.dirname(__file__))

print("Importing pipelines...", flush=True)
from src.t2i import T2IPipeline
from src.i2v import I2VPipeline
from src.tts import TTSPipeline
from src.lipsync import LipSyncPipeline

OUTPUT_DIR = "outputs/debug_headless"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_run():
    print("Initializing pipelines...", flush=True)
    t2i_pipe = T2IPipeline()
    i2v_pipe = I2VPipeline()
    tts_pipe = TTSPipeline()
    lipsync_pipe = LipSyncPipeline()
    
    print("Pipelines initialized.", flush=True)
    
    # Inputs
    timestamp = int(time.time())
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    visual_desc = "blonde spiky hair, blue eyes, whisker marks on cheeks"
    speech_text = "Dattebayo!"
    seed = 42
    
    try:
        # 1. T2I
        print(f"[{timestamp}] Step 1: T2I Generation...", flush=True)
        portrait_img = t2i_pipe.generate(visual_desc, seed=seed)
        portrait_path = os.path.join(run_dir, "portrait.png")
        portrait_img.save(portrait_path)
        print(f"[{timestamp}] Portrait saved.", flush=True)
        
        print("Unloading T2I...")
        t2i_pipe.unload()

        # 2. I2V
        print(f"[{timestamp}] Step 2: I2V Animation...", flush=True)
        # Using fewer frames for debug speed if needed, but let's stick to default to reproduce crash
        silent_video_frames = i2v_pipe.generate(portrait_img, seed=seed, num_frames=25) 
        
        print("Unloading I2V...")
        i2v_pipe.unload()
        
        # Save silent video
        import imageio
        silent_video_path = os.path.join(run_dir, "silent.mp4")
        imageio.mimsave(silent_video_path, silent_video_frames, fps=25)
        print(f"[{timestamp}] Silent video saved.", flush=True)
        
        # 3. TTS
        print(f"[{timestamp}] Step 3: TTS Synthesis...", flush=True)
        # Mock voice desc
        voice_desc = "A cheerful, energetic young ninja voice, confident, excited"
        sr, wav = tts_pipe.generate(speech_text, voice_desc, speed=1.0)
        
        import soundfile as sf
        speech_path = os.path.join(run_dir, "speech.wav")
        sf.write(speech_path, wav, sr)
        print(f"[{timestamp}] Audio saved.", flush=True)
        
        # 4. Lip-Sync
        print(f"[{timestamp}] Step 4: Lip-Sync...", flush=True)
        final_video_path = os.path.join(run_dir, "final.mp4")
        lipsync_pipe.inference(silent_video_path, speech_path, final_video_path)
        print(f"[{timestamp}] Final video saved at {final_video_path}", flush=True)
        
        print("TEST SUCCESSFUL!", flush=True)
        
    except Exception as e:
        print("TEST FAILED WITH EXCEPTION:", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_run()
