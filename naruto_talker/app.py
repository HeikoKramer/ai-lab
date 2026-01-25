import gradio as gr
import os
import time
import sys
import traceback
import shutil

# Ensure src is in path
sys.path.append(os.path.dirname(__file__))

from src.t2i import T2IPipeline
from src.i2v import I2VPipeline
from src.tts import TTSPipeline
from src.lipsync import LipSyncPipeline
from src.translation import translate_text

# Initialize pipelines lazily
t2i_pipe = T2IPipeline()
i2v_pipe = I2VPipeline()
tts_pipe = TTSPipeline()
lipsync_pipe = LipSyncPipeline()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pipeline(
    visual_prompt,
    negative_prompt,
    seed,
    voice_prompt,
    ref_audio,
    script_text,
    output_language,
    lipsync_offset,
    restoration_strength,
    motion_speed,
    progress=gr.Progress()
):
    MAX_RETRIES = 5
    from src.verification import VerificationPipeline
    
    # Map short codes to Qwen supported names (for TTS Language)
    # Qwen expects "German", "English", etc.
    LANG_MAP_TTS = {
        "auto": "auto",
        "en": "English",
        "de": "German",
        "fr": "French",
        "ja": "Japanese",
        "zh": "Chinese",
        "es": "Spanish",
        "it": "Italian",
    }
    # For Translation Utility (ISO codes)
    LANG_MAP_TRANS = {
        "en": "en",
        "de": "de",
        "fr": "fr",
        "ja": "ja",
        "zh": "zh-CN",
        "es": "es",
        "it": "it",
    }
    
    tts_lang_name = LANG_MAP_TTS.get(output_language, "auto")
    trans_target_code = LANG_MAP_TRANS.get(output_language, "en")

    try:
        timestamp = int(time.time())
        run_id = f"run_{timestamp}"
        run_dir = os.path.join(OUTPUT_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"[{run_id}] Starting generation pipeline...", flush=True)
        print(f"Inputs: Visual='{visual_prompt[:20]}...', Voice='{voice_prompt[:20]}...', RefAudio={bool(ref_audio)}, Lang={output_language}, Motion={motion_speed}", flush=True)

        # --- PRE-PROCESSING & TRANSLATION ---
        progress(0.05, desc="Translating Inputs...")
        
        # 1. Visual Prompt: Always translate to English for SDXL
        # (Assuming SDXL works best with EN)
        visual_prompt_en = translate_text(visual_prompt, "en")
        print(f"[{run_id}] Visual Prompt (EN): {visual_prompt_en}")
        
        # Append "anime" styles if not present (optional, but good for consistency)
        # We rely on the user or the hidden base prompt. 
        # Application requirement says "Anime", but Refactoring says "Direct Control".
        # Let's preserve the "Anime" bias slightly but allow user control.
        # Actually, let's stick to user prompt + some quality boosters.
        full_visual_prompt = f"{visual_prompt_en}, high quality, detailed"
        if "anime" not in full_visual_prompt.lower():
             full_visual_prompt += ", anime style"

        # 2. Voice Prompt: Translate to English (Qwen VoiceDesign works well with EN descriptions)
        voice_prompt_en = translate_text(voice_prompt, "en") if voice_prompt else "A neutral voice"
        print(f"[{run_id}] Voice Prompt (EN): {voice_prompt_en}")
        
        # 3. Script Text: Translate to Target Language
        script_text_translated = translate_text(script_text, trans_target_code)
        print(f"[{run_id}] Script (Translated to {output_language}): {script_text_translated}")

        # --- 1. IMAGE GENERATION ---
        progress(0.1, desc="Generating Character...")
        
        input_seed = int(seed) if seed != -1 else int(time.time())
        portrait_path = None
        verifier = VerificationPipeline()
        valid_portrait = False
        
        # Try up to 3 times
        for p_attempt in range(1, 4):
            current_seed = input_seed + (p_attempt * 100)
            try:
                portrait_img = t2i_pipe.generate(
                    prompt=full_visual_prompt, 
                    negative_prompt=negative_prompt, 
                    seed=current_seed, 
                    width=512, 
                    height=512
                )
                temp_path = os.path.join(run_dir, f"portrait_attempt_{p_attempt}.png")
                portrait_img.save(temp_path)
                t2i_pipe.unload()
                
                if verifier.verify_image(temp_path):
                    portrait_path = temp_path
                    valid_portrait = True
                    break
                else:
                    print(f"[{run_id}] Portrait verification failed.")
            except Exception as e:
                print(f"[{run_id}] T2I Failed: {e}")
                t2i_pipe.unload()
                
        if not valid_portrait:
            raise RuntimeError("Could not generate a valid portrait. Please adjust your visual explanation.")

        # --- 2. VIDEO ANIMATION (Silent) ---
        progress(0.3, desc=f"Animating Portrait ({motion_speed})...")
        silent_video_path = None
        success_anim = False
        
        for v_attempt in range(1, 4):
            vid_seed = input_seed + (v_attempt * 555)
            try:
                from PIL import Image
                clean_img = Image.open(portrait_path)
                # Pass motion_speed here
                frames = i2v_pipe.generate(
                    clean_img, 
                    seed=vid_seed, 
                    num_frames=25, 
                    prompt=full_visual_prompt,
                    motion_speed=motion_speed
                )
                i2v_pipe.unload()
                
                if verifier.verify_video(frames, sample_rate=3, success_threshold=0.8):
                    import imageio
                    silent_video_path = os.path.join(run_dir, "silent.mp4")
                    imageio.mimsave(silent_video_path, frames, fps=25)
                    success_anim = True
                    break
            except Exception as e:
                print(f"[{run_id}] I2V Failed: {e}")
                i2v_pipe.unload()
        
        verifier.close()
        
        if not success_anim:
            raise RuntimeError("Failed to generate stable animation.")

        # --- 3. TTS GENERATION ---
        progress(0.5, desc="Synthesizing Voice...")
        
        # Check Reference Audio
        ref_audio_path = None
        if ref_audio: # ref_audio is a tuple (sr, data) or filepath from Gradio?
            # Gradio 'sources=["upload","microphone"]' returns a filepath (str) usually, OR numpy if type="numpy".
            # By default type="filepath".
            if isinstance(ref_audio, str):
                 ref_audio_path = ref_audio
        
        # Generation
        sr, wav = tts_pipe.generate(
            text=script_text_translated,
            voice_description=voice_prompt_en,
            ref_audio_path=ref_audio_path,
            language=tts_lang_name
        )
        
        import soundfile as sf
        speech_path = os.path.join(run_dir, "speech.wav")
        sf.write(speech_path, wav, sr)
        audio_duration = len(wav) / sr
        tts_pipe.unload()
        
        # --- 4. PREPARE VIDEO LOOP ---
        print(f"[{run_id}] Extending video to {audio_duration:.2f}s...")
        import imageio_ffmpeg
        extended_video_path = os.path.join(run_dir, "silent_extended.mp4")
        ffmpeg_cmd = (
            f"{imageio_ffmpeg.get_ffmpeg_exe()} -y -stream_loop -1 -i {silent_video_path} "
            f"-t {audio_duration + 0.1} " 
            f"-c:v copy {extended_video_path}"
        )
        os.system(ffmpeg_cmd)
        if not os.path.exists(extended_video_path):
             extended_video_path = silent_video_path # Fallback

        # --- 5. LIP SYNC ---
        progress(0.7, desc="Lip-Syncing...")
        final_video_path = os.path.join(run_dir, "final.mp4")
        
        lipsync_pipe.inference(
            video_path=extended_video_path,
            audio_path=speech_path,
            output_path=final_video_path,
            bbox_shift=int(lipsync_offset),
            restoration_strength=float(restoration_strength)
        )
        lipsync_pipe.unload()
        
        # --- 6. FINAL RESTORATION (Global) ---
        # Note: The lipsync pipeline already did mouth restoration.
        # Use existing restoration pipe for full face polish?
        # The SoW didn't explicitly remove Step 4 (Restoration), but lipsync now handles "mouth super res".
        # Let's keep the global restoration for overall quality but make it optional or lighter if needed.
        # Or just return the output of lipsync if we trust the mouth restoration.
        # User requirement: "Restoration Strength (Slider)" -> "Wie stark soll der 'Mouth Super-Resolution' Filter arbeiten?"
        # That slider was passed to lipsync.
        # So manual global restoration might be redundant or destructive (double restoration).
        # Let's skip the global restoration step here to save VRAM and trust the local mouth restoration.
        # Also refactoring plan didn't mention keeping the old "Step 4 Restoration" as a separate block.
        
        final_output = final_video_path

        progress(1.0, desc="Done!")
        return portrait_path, speech_path, silent_video_path, final_output

    except Exception as e:
        print(f"[{run_id if 'run_id' in locals() else 'N/A'}] Job Failed: {e}")
        traceback.print_exc()
        raise gr.Error(str(e))


# --- UI LAYOUT ---
with gr.Blocks(title="Character Studio", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Character Studio (Direct Control)")
    
    with gr.Row():
        # LEFT COLUMN: CONTROLS
        with gr.Column(scale=1):
            
            with gr.Group():
                gr.Markdown("### 1. Visual Identity (The Look)")
                visual_prompt_in = gr.Textbox(
                    label="Visuelle Beschreibung (Visual Prompt)",
                    value="portrait of a cyberpunk hacker, neon lights, purple hair, detailed eyes, 8k, close up",
                    lines=3,
                    info="Beschreiben Sie den Charakter. (DE/EN)"
                )
                negative_prompt_in = gr.Textbox(
                    label="Was nicht enthalten sein soll (Negative)",
                    value="blurry, bad anatomy, drawing, painting, low quality",
                    lines=1
                )
                seed_in = gr.Number(label="Portrait Seed (-1 = Random)", value=-1, precision=0)

            with gr.Group():
                gr.Markdown("### 2. Voice Identity (The Sound)")
                voice_prompt_in = gr.Textbox(
                    label="Stimm-Beschreibung (Voice Prompt)",
                    value="Eine junge, fröhliche Stimme, klar und deutlich.",
                    lines=2,
                    info="Was für eine Stimme ist es? (wird überschrieben wenn Audio hochgeladen wird)"
                )
                ref_audio_in = gr.Audio(
                    label="Stimm-Referenz (Optional - Voice Cloning)",
                    sources=["upload", "microphone"],
                    type="filepath"
                )

            with gr.Group():
                gr.Markdown("### 3. Action (The Script)")
                script_in = gr.Textbox(
                    label="Text (Was wird gesagt?)",
                    value="Hallo! Ich bin dein neuer AI Avatar. Wie gefällt dir meine Stimme?",
                    lines=3
                )
                language_in = gr.Dropdown(
                    ["de", "en", "fr", "ja", "es"], 
                    label="Zielsprache (Language)", 
                    value="de",
                    info="Der Text wird automatisch in diese Sprache übersetzt."
                )

            with gr.Accordion("Technical Tuning", open=True):
                lipsync_offset_in = gr.Slider(-20, 20, value=0, label="Mund-Position (Offset)", step=1)
                restoration_strength_in = gr.Slider(0.0, 1.0, value=1.0, label="Gesichtskorrektur Stärke", step=0.1)
                motion_speed_in = gr.Radio(["Static", "Subtle", "Dynamic"], value="Static", label="Kamera/Hintergrund Bewegung", info="Static = Hintergrund eingefroren (Empfohlen)")

            gen_btn = gr.Button("GENERATE SCENE", variant="primary", size="lg")

        # RIGHT COLUMN: PREVIEWS
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Final Video"):
                    video_out = gr.Video(label="Final Result")
                with gr.Tab("Assets"):
                    portrait_out = gr.Image(label="Portrait", height=300)
                    audio_out = gr.Audio(label="Voice")
                    silent_out = gr.Video(label="Base Animation", height=300)

    # WIRING
    gen_btn.click(
        generate_pipeline,
        inputs=[
            visual_prompt_in, 
            negative_prompt_in, 
            seed_in,
            voice_prompt_in,
            ref_audio_in,
            script_in,
            language_in,
            lipsync_offset_in,
            restoration_strength_in,
            motion_speed_in
        ],
        outputs=[portrait_out, audio_out, silent_out, video_out]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", debug=True)
