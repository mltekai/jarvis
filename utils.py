import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel
import threading
from queue import Queue
import sounddevice as sd
import time
from rich.console import Console
import whisper


#nltk.download('punkt_tab')
console = Console()
stt = whisper.load_model("base.en")

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

def play_audio(sample_rate: int, audio_array: np.ndarray) -> None:
    try:
        sd.play(audio_array, sample_rate)
        sd.wait()
    except Exception as exc:
        console.print(f"[red]Audio playback error: {exc}")

def record_audio(stop_event: threading.Event, data_queue: Queue) -> None:
    """Record raw audio into a queue until stop_event is set.

    Uses a small sleep in the loop to be responsive to stop events.
    """

    def callback(indata, frames, time_, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.05)


def transcribe(audio_np: np.ndarray, use_fp16: bool = False) -> str:
    """Transcribe audio with Whisper and return text (safe for empty input)."""
    if audio_np is None or audio_np.size == 0:
        return ""
    try:
        result = stt.transcribe(audio_np, fp16=use_fp16)
        return result.get("text", "").strip()
    except Exception as exc:
        console.print(f"[red]Transcription error: {exc}")
        return ""


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class.

        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_6"):
        """
        Synthesizes audio from the given text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_6".

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_6"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)
