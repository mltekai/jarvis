# Jarvis Voice Agent

A voice-based AI agent application with both graphical and command-line interfaces. Jarvis listens to voice input, transcribes it, processes through an AI agent, and responds with synthesized speech.

## Features

- **Voice Recording** - Real-time audio capture from microphone
- **Speech-to-Text** - Automatic transcription using OpenAI Whisper
- **AI Agent** - Intelligent responses powered by local Ollama LLM (llama3.2:3b)
- **Text-to-Speech** - Natural speech synthesis using Bark TTS
- **Dual Interface** - GUI mode (tkinter) or CLI mode for flexibility
- **Configurable Timeout** - Environment variable control over agent response time
- **Cross-platform** - Works on macOS, Linux, and Windows

## Requirements

### System Dependencies
- Python 3.8+
- Microphone and speakers
- CUDA-capable GPU (optional, but recommended for TTS performance)

### Python Dependencies
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library
- `sounddevice` - Audio input/output
- `openai-whisper` - Speech recognition
- `langchain` - LLM/Agent framework
- `langchain-ollama` - Ollama integration
- `langchain-mcp-adapters` - MCP tools integration
- `rich` - Rich terminal output

### External Services
- **Ollama** - Local LLM server (running at `http://localhost:11434/`)
  - Model: `llama3.2:3b`
  - Download: [ollama.ai](https://ollama.ai)
- **MCP Servers** - DuckDuckGo search capability (optional)

## Installation

1. **Clone/download the project**
   ```bash
   cd /Users/namburi/work_poc/jarvis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch transformers sounddevice openai-whisper langchain langchain-ollama langchain-mcp-adapters rich
   ```

4. **Start Ollama service**
   ```bash
   ollama run llama3.2:3b
   ```
   
   Ensure it's running at `http://localhost:11434/`

## Configuration

### Environment Variables

- `AGENT_TIMEOUT` (default: `20`) - Maximum seconds to wait for agent response
  ```bash
  export AGENT_TIMEOUT=30
  ```

## Usage

### GUI Mode
Start the application with the graphical interface:
```bash
python voicebot.py
```

**Controls:**
- **Start Recording** - Begin capturing audio
- **Stop Recording** - End capture and process
- **Transcript pane** - Displays transcribed text
- **Agent Response pane** - Shows AI's response and plays audio

### CLI Mode
Run in command-line mode (automatically used if tkinter unavailable):
```bash
python voicebot.py
```

**Instructions:**
1. Press Enter to start recording
2. Press Enter again to stop and process
3. Listen to the agent's synthesized response
4. Press Ctrl+C to exit

## Project Structure

```
jarvis/
├── voicebot.py          # Main application (GUI + CLI)
├── utils.py             # Audio utilities (recording, TTS, STT)
├── common/
│   └── common_util.py   # Agent creation and management
├── __pycache__/         # Python cache
└── README.md            # This file
```

## Architecture

### Components

1. **VoiceAgentApp** - Tkinter GUI application
   - Handles user interactions
   - Manages recording lifecycle
   - Displays transcript and responses

2. **HeadlessVoiceAgent** - CLI backend
   - Recording control
   - Processing pipeline execution

3. **TextToSpeechService** - Bark TTS
   - Converts text to audio
   - Supports long-form synthesis

4. **Agent System** - LangChain integration
   - Local LLM (Ollama llama3.2:3b)
   - Configurable timeouts
   - Tool integration via MCP

### Data Flow

```
Microphone
    ↓
Record Audio (sounddevice)
    ↓
Transcribe (Whisper)
    ↓
Agent Processing (Ollama + LangChain)
    ↓
Text-to-Speech (Bark)
    ↓
Play Audio (sounddevice)
```

## Troubleshooting

### GUI won't start
- Install tkinter: `brew install python-tk@3.11` (macOS)
- Falls back to CLI automatically

### Agent timeout
- Increase `AGENT_TIMEOUT`: `export AGENT_TIMEOUT=30`
- Check Ollama service is running

### No audio recorded
- Verify microphone permissions
- Check system audio input settings

### Transcription issues
- Whisper "base.en" model will auto-download on first run
- Ensure stable internet connection

## Performance Notes

- First run downloads Whisper model (~140MB)
- Bark model (~2GB) downloads on first TTS usage
- Ollama should run efficiently on modern hardware
- CUDA GPU recommended for faster TTS inference

## License

Project uses open-source components. See dependencies for individual licenses.
