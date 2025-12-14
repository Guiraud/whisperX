# 🎤 WhisperX GUI

A beautiful macOS application for transcribing audio with WhisperX.

![WhisperX GUI](../figures/gui_preview.png)

## Features

- ✨ **Modern Dark Theme** - Beautiful, eye-friendly interface
- 🎯 **Drag & Drop** - Simply drag audio files to transcribe
- 🌍 **Multi-language Support** - Auto-detect or select from 50+ languages
- ⚡ **Fast Transcription** - Powered by WhisperX's batched inference
- 🎯 **Word-level Timestamps** - Accurate alignment with wav2vec2
- 👥 **Speaker Diarization** - Identify who said what
- 📤 **Multiple Export Formats** - TXT, SRT, VTT, JSON

## Installation

### Prerequisites

Make sure you have WhisperX installed:

```bash
cd /path/to/whisperX
pip install -e .
```

### Install GUI Dependencies

```bash
pip install PyQt6>=6.4.0
```

Or using the requirements file:

```bash
pip install -r whisperx_gui/requirements.txt
```

## Usage

### Run the Application

```bash
# From the whisperX directory
python -m whisperx_gui.run

# Or directly
python whisperx_gui/run.py
```

### Quick Start

1. **Select Audio File**: Drag & drop a file or click the drop zone
2. **Configure Settings**: Choose model, language, and options
3. **Transcribe**: Click the "Transcribe" button
4. **Export**: Save results in TXT, SRT, or JSON format

## Settings

### Basic Settings

| Option | Description |
|--------|-------------|
| Model | Whisper model size (tiny to large-v3) |
| Language | Audio language (or auto-detect) |
| Device | CPU, CUDA, or MPS (Apple Silicon) |
| Output Format | Export format preference |

### Advanced Settings

| Option | Description |
|--------|-------------|
| Compute Type | float16, float32, or int8 |
| Batch Size | Number of segments to process at once |
| Word Alignment | Enable precise word timestamps |
| VAD Method | Voice Activity Detection method |

### Speaker Diarization

To enable speaker identification:

1. Get a HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the model agreements:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Enter your token in the "Speakers" tab
4. Enable "Speaker Diarization"

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- WMA
- AAC

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘O | Open file |
| ⌘S | Save transcription |
| ⌘, | Preferences |
| ⌘Q | Quit |

## Troubleshooting

### "No module named whisperx"

Make sure WhisperX is installed in your Python environment:

```bash
pip install -e /path/to/whisperX
```

### "CUDA out of memory"

Try these solutions:
1. Reduce batch size (Advanced settings)
2. Use a smaller model (tiny or base)
3. Use CPU instead of CUDA

### "Model download failed"

Check your internet connection and try again. Models are downloaded from HuggingFace.

## License

Same as WhisperX - BSD-2-Clause
