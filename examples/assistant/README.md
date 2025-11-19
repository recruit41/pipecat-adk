# Voice Assistant Example

A self-contained example demonstrating a voice assistant using pipecat-adk with the small-webrtc-prebuilt UI.

## Features

- **Google ADK Integration**: Uses Google's Agent Development Kit for conversation management
- **Automatic Interruption Handling**: Gracefully handles user interruptions mid-response
- **Voice Input/Output**: Speech-to-text and text-to-speech via Google services
- **WebRTC Transport**: Browser-based audio communication via small-webrtc-prebuilt
- **Prebuilt UI**: Ready-to-use web interface for testing

## Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install pipecat-adk** (from parent directory):
   ```bash
   pip install -e ../..
   ```

4. **Configure environment variables**:
   ```bash
   cp env.example .env
   ```

   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Running the Assistant

Start the server:
```bash
python run.py
```

By default, the server runs on `http://localhost:7860`. Open this URL in your browser to interact with the assistant.

### Command-line Options

- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 7860)
- `--verbose` or `-v`: Enable debug logging

Example:
```bash
python run.py --host 0.0.0.0 --port 8000 -v
```

## How It Works

### Architecture

The assistant uses a Pipecat pipeline with the following flow:

1. **Audio Input**: User speaks into browser microphone
2. **Speech-to-Text**: Google STT converts speech to text
3. **User Context Aggregator**: Packages input for ADK agent
4. **ADK Agent**: Google Gemini processes the conversation
5. **Text-to-Speech**: Google TTS converts response to audio
6. **Audio Output**: Browser plays the response

### Interruption Handling

The library uses an "accountant's approach" for interruptions:

1. When user interrupts, the system tracks what was actually spoken
2. A synthetic event is added to the ADK session with the partial response
3. On the next user message, the InterruptionHandlerPlugin filters the conversation history
4. The LLM only sees what the user actually heard, ensuring natural conversation flow

### Session Management

Each WebRTC connection gets its own ADK session identified by the connection's `pc_id`. Sessions are managed in-memory and automatically cleaned up when connections close.

## Customization

### Modifying the Agent

Edit the `SYSTEM_INSTRUCTION` in `bot.py` to change the assistant's personality and behavior:

```python
SYSTEM_INSTRUCTION = """You are a helpful AI assistant. Be concise and friendly.
Your custom instructions here..."""
```

### Changing Voice Settings

Modify the TTS service configuration in `bot.py`:

```python
tts = GoogleTTSService(
    voice_id="en-US-Wavenet-C",  # Change voice here
    params=GoogleTTSService.InputParams(language=Language.EN_US),
)
```

Available voices: See [Google TTS documentation](https://cloud.google.com/text-to-speech/docs/voices)

### Adjusting Pipeline Behavior

Enable/disable interruptions in `bot.py`:

```python
task = PipelineTask(
    pipeline,
    params=PipelineParams(allow_interruptions=True),  # Set to False to disable
)
```

## Troubleshooting

### "GEMINI_API_KEY environment variable not set"

Make sure you've created a `.env` file with your API key (see Setup step 4).

### Connection issues

- Check that port 7860 is not already in use
- Try running with a different port: `python run.py --port 8000`
- Make sure your browser allows microphone access

### Audio not working

- Check browser console for errors
- Ensure microphone permissions are granted
- Try refreshing the page

### Debug mode

Run with verbose logging to see detailed information:
```bash
python run.py -v
```

## Project Structure

```
examples/assistant/
├── bot.py              # Bot logic with ADK agent and Pipecat pipeline
├── run.py              # FastAPI server and WebRTC signaling
├── requirements.txt    # Python dependencies
├── env.example         # Environment variable template
└── README.md          # This file
```

## Next Steps

- Explore other examples in the `examples/` directory
- Read the [pipecat-adk documentation](../../README.md)
- Check out the [Pipecat documentation](https://github.com/pipecat-ai/pipecat)
- Learn more about [Google ADK](https://github.com/google/adk)
