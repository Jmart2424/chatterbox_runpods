import runpod
import base64
import io
import sys
import numpy as np
from scipy.io.wavfile import write as write_wav

# Add chatterbox to path
sys.path.append('/src')

# Import from your existing TTS file
from chatterbox.tts import synthesize  # Or whatever the main function is called in tts.py

def handler(job):
    """
    RunPod handler for Chatterbox TTS
    """
    try:
        job_input = job['input']
        text = job_input.get('text', '')
        
        # Call your existing TTS function from tts.py
        # You need to check what function is available in tts.py
        audio_data = synthesize(text)  # CHECK tts.py for the actual function name!
        
        # Convert to WAV format
        if hasattr(audio_data, 'cpu'):  # If it's a tensor
            audio_data = audio_data.cpu().numpy()
        
        # Ensure proper shape
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()
        
        # Normalize to int16 for WAV
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = np.clip(audio_data, -1, 1)
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV buffer
        wav_buffer = io.BytesIO()
        sample_rate = 24000  # Adjust based on Chatterbox model
        write_wav(wav_buffer, sample_rate, audio_data)
        wav_buffer.seek(0)
        
        # Encode to base64
        wav_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return {
            "output": {
                "audio_base64": wav_base64,
                "format": "wav",
                "sample_rate": sample_rate
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
