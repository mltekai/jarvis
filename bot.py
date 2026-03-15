import threading
from queue import Queue
from typing import Optional

from utils import play_audio, record_audio, transcribe 

import numpy as np
import sounddevice as sd
from rich.console import Console
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from langchain_classic.prompts import PromptTemplate
from common.common_util import create_agent_live, run_agent
from utils import TextToSpeechService
import asyncio
import os

console = Console()
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "20"))
tts = TextToSpeechService()

# Prompt template
template = """
You are a helpful and friendly AI Agent. You are polite, respectful, and aim to provide concise responses of less
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

# Lazily create and cache the agent to avoid import-time complexity
AGENT = None

def get_or_create_agent():
    global AGENT
    if AGENT is not None:
        return AGENT

    agent = create_agent_live(PROMPT.template)
    if asyncio.iscoroutine(agent):
        try:
            AGENT = asyncio.run(agent)
        except RuntimeError:
            console.print("[yellow]Event loop already running; agent coroutine will be created later.")
            return None
    else:
        AGENT = agent
    return AGENT


def get_agent_response(text: str) -> str:
    if not text:
        return ""
    try:
        agent = get_or_create_agent()
        if agent is None:
            try:
                agent = asyncio.run(create_agent_live(PROMPT.template))
            except RuntimeError:
                console.print("[yellow]Event loop running; cannot create agent synchronously.")
                return ""
            except Exception as exc:
                console.print(f"[red]create_agent_live failed: {exc}")
                return ""

        agent_response = run_agent(text, agent, timeout=AGENT_TIMEOUT)
        return agent_response if isinstance(agent_response, str) else str(agent_response)
    except Exception as exc:
        console.print(f"[red]Agent error: {exc}")
        return ""


class VoiceAgentApp:
    """Simple tkinter-based UI for the voice agent.

    - Start/Stop recording
    - Transcript and response panes
    - Status indicator
    """

    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        root.title("Jarvis Voice Agent")
        root.geometry("700x500")

        # Controls frame
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        self.start_btn = tk.Button(ctrl, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = tk.Button(ctrl, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.status_label = tk.Label(ctrl, text="Idle")
        self.status_label.pack(side=tk.LEFT, padx=8)

        # Transcript display
        tk.Label(root, text="Transcript:").pack(anchor=tk.W, padx=8)
        self.transcript_box = ScrolledText(root, height=8)
        self.transcript_box.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        # Response display
        tk.Label(root, text="Agent Response:").pack(anchor=tk.W, padx=8)
        self.response_box = ScrolledText(root, height=8)
        self.response_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Recording control
        self._data_queue: Optional[Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._recording_thread: Optional[threading.Thread] = None

    def start_recording(self) -> None:
        self.transcript_box.delete("1.0", tk.END)
        self.response_box.delete("1.0", tk.END)
        self._data_queue = Queue()
        self._stop_event = threading.Event()
        self._recording_thread = threading.Thread(target=record_audio, args=(self._stop_event, self._data_queue))
        self._recording_thread.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Recording...")

    def stop_recording(self) -> None:
        if not self._stop_event:
            return
        self._stop_event.set()
        if self._recording_thread:
            self._recording_thread.join()

        # collect audio bytes
        audio_data = b"".join(list(self._data_queue.queue)) if self._data_queue else b""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0 if audio_data else np.array([], dtype=np.float32)

        self.status_label.config(text="Processing...")
        self.root.update_idletasks()

        text = transcribe(audio_np)
        if text:
            self.transcript_box.insert(tk.END, text)

        response = get_agent_response(text)
        if response:
            self.response_box.insert(tk.END, response)
            try:
                sample_rate, audio_array = tts.long_form_synthesize(response)
                play_audio(sample_rate, audio_array)
            except Exception as exc:
                console.print(f"[red]TTS error: {exc}")

        self.status_label.config(text="Idle")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


class HeadlessVoiceAgent:
    """Headless agent for CLI use reusing the same recording/transcription/response logic."""

    def __init__(self) -> None:
        self._data_queue: Optional[Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._recording_thread: Optional[threading.Thread] = None

    def start_recording(self) -> None:
        self._data_queue = Queue()
        self._stop_event = threading.Event()
        self._recording_thread = threading.Thread(target=record_audio, args=(self._stop_event, self._data_queue))
        self._recording_thread.start()

    def stop_recording(self) -> tuple[str, str]:
        if not self._stop_event:
            return "", ""
        self._stop_event.set()
        if self._recording_thread:
            self._recording_thread.join()

        audio_data = b"".join(list(self._data_queue.queue)) if self._data_queue else b""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0 if audio_data else np.array([], dtype=np.float32)

        text = transcribe(audio_np)
        response = get_agent_response(text) if text else ""
        return text, response


def run_cli() -> None:
    console.print("[cyan]Agent started! Press Ctrl+C to exit.")
    try:
        headless = HeadlessVoiceAgent()
        while True:
            console.input("Press Enter to start recording, then press Enter again to stop.")
            headless.start_recording()

            input()
            text, response = headless.stop_recording()

            if text:
                console.print(f"[yellow]You: {text}")
                console.print(f"[cyan]Agent: {response}")
                if response:
                    sample_rate, audio_array = tts.long_form_synthesize(response)
                    play_audio(sample_rate, audio_array)
                else:
                    console.print("[red]No Agent response generated.")
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")


if __name__ == "__main__":
    if tk is None:
        console.print(
            "[yellow]tkinter not available. Falling back to CLI. To enable GUI install tkinter/a framework Python."
        )
        run_cli()
    else:
        # On macOS ensure Python is a framework build for GUI to display correctly.
        try:
            root = tk.Tk()
        except Exception as exc:
            console.print(f"[red]Unable to start GUI: {exc}\nFalling back to CLI.")
            run_cli()
        else:
            app = VoiceAgentApp(root)
            root.mainloop()
