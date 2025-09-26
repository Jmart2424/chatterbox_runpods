try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

# Don't call version(), just hardcode it
__version__ = "0.2.6"

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
