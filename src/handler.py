import runpod
import base64
import io
import sys
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

# Add chatterbox to path
sys.path.append('/src')

# Import ChatterboxTTS from tts.py
from chatterbox.tts import ChatterboxTTS

# Initialize the model once (outside handler for efficiency)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ChatterboxTTS on {device}...")
tts_model = ChatterboxTTS.from_pretrained(device)
print("ChatterboxTTS loaded successfully!")

def handler(job):
    """
    RunPod handler for Chatterbox TTS
    """
    try:
        job_input = job['input']
        text = job_input.get('text', 'Hello, this is a test.')
        
        # Optional parameters with defaults
        temperature = job_input.get('temperature', 0.8)
        repetition_penalty = job_input.get('repetition_penalty', 1.2)
        min_p = job_input.get('min_p', 0.05)
        top_p = job_input.get('top_p', 1.0)
        cfg_weight = job_input.get('cfg_weight', 0.5)
        exaggeration = job_input.get('exaggeration', 0.5)
        
        # Generate audio using ChatterboxTTS
        print(f"Generating audio for text: {text[:50]}...")
        audio_tensor = tts_model.generate(
            text=text,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration
        )
        
        # Convert tensor to numpy array
        audio_data = audio_tensor.squeeze().cpu().numpy()
        
        # Get sample rate from model
        sample_rate = tts_model.sr
        
        # Ensure audio is in the right format for WAV
        # Check if already in int16 range or needs conversion
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            # Normalize to [-1, 1] if needed
            audio_max = np.abs(audio_data).max()
            if audio_max > 1.0:
                audio_data = audio_data / audio_max
            # Convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV buffer
        wav_buffer = io.BytesIO()
        write_wav(wav_buffer, sample_rate, audio_data)
        wav_buffer.seek(0)
        
        # Encode to base64
        wav_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        print(f"Audio generated successfully. Sample rate: {sample_rate}, Length: {len(audio_data)} samples")
        
        return {
            "output": {
                "audio_base64": wav_base64,
                "format": "wav",
                "sample_rate": sample_rate,
                "duration_seconds": len(audio_data) / sample_rate
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
