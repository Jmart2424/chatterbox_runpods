FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libsndfile1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy requirements and install dependencies
COPY src/requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r /src/requirements.txt

# Copy the entire source code
COPY src/ /src/

# Pre-download the model weights (optional - makes container bigger but faster cold start)
# Uncomment the line below if you want to include model in the image
# RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained('cpu')"

# Run the handler
CMD ["python", "-u", "/src/handler.py"]
