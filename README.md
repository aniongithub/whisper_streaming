# Enhanced Whisper Streaming with Diarization & Probabilities

**Real-Time Transcription with Speaker Identification and Streaming Confidence Scores**

This repository is a significantly enhanced fork of the original [whisper_streaming](https://github.com/ufal/whisper_streaming) project, featuring:

- **Real-time speaker diarization** using Resemblyzer embeddings and Silero VAD
- **Streaming word-level confidence probabilities** for quality assessment
- **Modular architecture** with pluggable ASR backends
- **Development container environment** with GPU support and VS Code integration
- **Optimized performance** for real-time multi-speaker scenarios

## Key Features

### Speaker Diarization
- **Real-time speaker identification** during streaming transcription
- **Automatic speaker detection** with similarity-based clustering
- **Voice activity detection** using Silero VAD models
- **Speaker persistence** across audio segments

### Streaming Probabilities
- **Word-level confidence scores** for each transcribed word
- **Real-time quality assessment** for streaming applications
- **Probability aggregation** at sentence and utterance levels
- **Quality-based filtering** for improved accuracy

### Enhanced Architecture
- **Modular ASR backends**: faster-whisper, whisper-timestamped, OpenAI API, MLX-whisper
- **Containerized development** with full GPU support
- **Flexible audio processing** with VAD and voice activity control
- **Streaming-optimized buffering** with local agreement policies

## Architecture Overview

This enhanced streaming system is built with a modular architecture centered around these core components:

### ASR Module (`asr/`)
- **`online_processor.py`** - Main streaming processor with probability tracking
- **`streaming_diarizer.py`** - Real-time speaker diarization engine
- **`hypothesis_buffer.py`** - Advanced buffering with probability aggregation
- **`impl/`** - Pluggable ASR backends (faster-whisper, OpenAI API, MLX, etc.)
- **`server.py`** - TCP server for real-time streaming
- **`vac_processor.py`** - Voice Activity Controller integration

### Enhanced Features

#### Streaming Probabilities
The system now provides real-time confidence assessment:
- **Word-level probabilities** from Whisper model outputs
- **Sentence-level aggregation** using probability averaging
- **Quality filtering** based on confidence thresholds
- **Real-time feedback** for streaming applications

#### Real-Time Speaker Diarization
Advanced speaker identification using modern ML models:
- **Resemblyzer embeddings** for speaker characterization
- **Silero VAD** for voice activity detection
- **Similarity clustering** for automatic speaker identification
- **Streaming-optimized** processing with minimal latency

The diarizer automatically:
1. Detects voice activity in real-time
2. Extracts speaker embeddings from speech segments
3. Clusters similar voices into speaker profiles
4. Assigns speaker labels to transcribed text

```python
# Example output structure:
(start_time, end_time, text, avg_probability, [SPEAKER_<id>] [(word, prob), ...])
```

#### Backend Flexibility
Multiple ASR backends supported with consistent interfaces:
- **faster-whisper** (recommended) - GPU-optimized, 4x faster
- **whisper-timestamped** - Enhanced timestamp accuracy
- **OpenAI API** - Cloud-based processing
- **MLX-whisper** - Apple Silicon optimization

## API Usage

### Basic Streaming with Probabilities

```python
from asr.utils import create_asr_engine, add_shared_args
from asr.online_processor import OnlineASRProcessor

# Initialize ASR with probability tracking
asr = create_asr_engine("faster-whisper", language="en", model="large-v3")
processor = OnlineASRProcessor(asr)

# Process audio chunks
while audio_available:
    audio_chunk = get_audio_chunk()
    processor.insert_audio_chunk(audio_chunk)
    
    # Get transcription with probabilities
    result = processor.process_iter()
    if result[0] is not None:  # New confirmed text
        start, end, text, avg_prob, word_probs = result
        print(f"[{start:.1f}-{end:.1f}] {text} (confidence: {avg_prob:.2f})")
```

### Speaker Diarization Integration

```python
from asr.streaming_diarizer import StreamingDiarizer

# Initialize diarization (requires resemblyzer and silero-vad)
diarizer = StreamingDiarizer()

# Process with speaker identification
while audio_available:
    audio_chunk = get_audio_chunk()
    
    # Add audio to both processor and diarizer
    processor.insert_audio_chunk(audio_chunk)
    diarizer.add_audio_chunk(audio_chunk)
    
    # Get transcription
    result = processor.process_iter()
    
    # Get speaker assignments
    speaker_assignments = diarizer.process_chunk()
    
    if result[0] is not None:
        start, end, text, avg_prob, word_probs = result
        # Find speaker for this timestamp
        speaker = diarizer.get_speaker_for_timestamp((start + end) / 2)
        print(f"[{speaker or 'UNKNOWN'}] {text} (confidence: {avg_prob:.2f})")
```

## Development Setup

This project provides a complete containerized development environment with GPU acceleration support for efficient development and testing.

### Prerequisites

- **Docker** with NVIDIA container runtime (for GPU support)
- **VS Code** with Dev Containers extension
- **NVIDIA GPU** (recommended) with drivers ≥470.57.02
- **Git** with LFS support

### Getting Started with Dev Containers

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/whisper_streaming.git
   cd whisper_streaming
   ```

2. **Open in VS Code:**
   ```bash
   code .
   ```

3. **Reopen in Container:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Select "Dev Containers: Reopen in Container"
   - VS Code will build and start the development container

### Container Environment

The development container includes:

- **Base**: Ubuntu 22.04 with CUDA 12.3.2 and cuDNN 9
- **Python**: Latest Python 3 with optimized package installations
- **GPU Support**: Full NVIDIA GPU passthrough for acceleration
- **Extensions**: Pre-configured Python development tools
- **Port Forwarding**: Automatic forwarding for streaming server (port 43007)

#### Container Features

**GPU Access**: The container runs with `--gpus all` to access all available GPUs:
```json
{
  "runArgs": ["--privileged", "--gpus", "all"]
}
```

**Python Environment**: All dependencies pre-installed via `requirements.txt`:
- Core ML libraries: `torch`, `torchaudio`, `faster-whisper`
- Audio processing: `librosa`, `soundfile`
- Diarization models: `resemblyzer`, `silero-vad`
- Text processing: `opus-fast-mosestokenizer`, `wtpsplit`

**VS Code Integration**: Pre-configured extensions for Python development:
- Python language support with IntelliSense
- Task runner for custom commands
- Markdown editing capabilities

### Running the System

#### Server Mode (Recommended for Development)

Start the streaming server in the container:

```bash
# Basic server with default settings
python asr/server.py --port 43007 --model large-v3

# With diarization enabled
python asr/server.py --port 43007 --model large-v3 --diarization

# With custom settings
python asr/server.py \
  --port 43007 \
  --model large-v3-turbo \
  --language en \
  --vad \
  --buffer_trimming_sec 2
```

#### Client Testing

From the host machine or another container:

```bash
# Stream audio file to server
python client/client.py --server localhost:43007 --file assets/jfk.flac

# Real-time microphone streaming (requires audio setup)
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43007
```

### Development Workflow

#### Code Organization

```
asr/                    # Core ASR processing modules
├── base.py            # ASR backend interface
├── online_processor.py # Main streaming processor
├── streaming_diarizer.py # Speaker diarization
├── hypothesis_buffer.py # Streaming buffer management
├── impl/              # ASR backend implementations
│   ├── faster_whisper.py
│   ├── openai_api.py
│   └── mlx_whisper.py
└── server.py          # TCP streaming server

client/                # Client implementations
└── client.py          # Example streaming client

common/                # Shared utilities
└── line_packet.py     # Network packet handling
```

#### Adding New ASR Backends

1. Create new implementation in `asr/impl/your_backend.py`
2. Inherit from `ASRBase` in `asr/base.py`
3. Implement required methods: `load_model()`, `transcribe()`, `ts_words()`
4. Register backend in `asr/utils.py`

#### Testing Changes

```bash
# Run with test audio file
python -c "
from asr.utils import create_asr_engine
from asr.online_processor import OnlineASRProcessor
import numpy as np

# Test basic functionality
asr = create_asr_engine('faster-whisper', language='en', model='base')
processor = OnlineASRProcessor(asr)
print('Setup successful!')
"

# Test with sample audio
python client/client.py --file assets/jfk.flac --simulate-realtime
```

### GPU Memory Management

For development with limited GPU memory:

```bash
# Use smaller models
export WHISPER_MODEL=base  # Instead of large-v3

# Monitor GPU usage
nvidia-smi -l 1

# Or use CPU-only mode for debugging
export CUDA_VISIBLE_DEVICES=""
```

### Debugging and Logging

Enable detailed logging:

```bash
# Set log level for debugging
export LOG_LEVEL=DEBUG

# Or configure in code
import logging
logging.getLogger("whisper_streaming").setLevel(logging.DEBUG)
```

Common debug scenarios:
- **Model loading issues**: Check CUDA version compatibility
- **Audio format problems**: Ensure 16kHz mono input
- **Diarization failures**: Verify resemblyzer/silero-vad installation
- **Network issues**: Check port forwarding and firewall settings

### Performance Optimization

#### GPU Settings

```bash
# Optimize for RTX series GPUs
export WHISPER_COMPUTE_TYPE=float16
export WHISPER_DEVICE=cuda

# For older GPUs or stability issues
export WHISPER_COMPUTE_TYPE=int8_float16
```

#### Memory Tuning

```bash
# Reduce buffer sizes for lower latency
export BUFFER_TRIMMING_SEC=1

# Increase for better accuracy
export BUFFER_TRIMMING_SEC=5
```

## Output Format

This enhanced system provides rich output with probabilities and speaker information:

### Standard Output Format

Each line contains:
```
<emission_time> <start_ms> <end_ms> [<speaker>] <text> (confidence: <probability>)
```

**Example Output:**
```
2691.44 300 1380 [SPEAKER_00] Chairman, thank you. (confidence: 0.94)
6914.55 1940 4940 [SPEAKER_01] If the debate today had a (confidence: 0.87)
9019.03 5160 7160 [SPEAKER_01] the subject the situation in (confidence: 0.91)
10065.13 7180 7480 [SPEAKER_01] Gaza (confidence: 0.95)
11058.36 7480 9460 [SPEAKER_02] Strip, I might (confidence: 0.89)
```

### API Response Format

When using the Python API, responses include detailed probability information:

```python
# process_iter() returns:
(start_time, end_time, text, avg_probability, word_probabilities)

# Example:
(2691.44, 1380, "Chairman, thank you.", 0.94, 
 [("Chairman,", 0.96), ("thank", 0.93), ("you.", 0.93)])
```

### Diarization Output

When speaker diarization is enabled:

```python
from asr.streaming_diarizer import StreamingDiarizer

diarizer = StreamingDiarizer()
# ... process audio ...

# Get speaker assignments
speaker_assignments = diarizer.process_chunk()
# Returns: {(start_time, end_time): "SPEAKER_ID", ...}

# Get speaker for specific timestamp
speaker = diarizer.get_speaker_for_timestamp(timestamp)
# Returns: "SPEAKER_00", "SPEAKER_01", etc., or None

# Get overall statistics
stats = diarizer.get_speaker_stats()
# Returns: {
#   'total_speakers': 3,
#   'speaker_names': ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'],
#   'total_processed_time': 125.3,
#   'models_loaded': True
# }
```

### Probability Interpretation

- **Word-level probabilities** (0.0-1.0): Individual word confidence from Whisper
- **Average probabilities** (0.0-1.0): Sentence/utterance average for quality assessment
- **Thresholds for filtering**:
  - `> 0.9`: High confidence, likely accurate
  - `0.7-0.9`: Good confidence, generally reliable
  - `0.5-0.7`: Moderate confidence, may need review
  - `< 0.5`: Low confidence, likely errors

### JSON Output Mode

For structured data applications:

```python
import json

# Enable JSON output in server mode
python asr/server.py --output-format json

# Client receives:
{
    "timestamp": 2691.44,
    "start_ms": 300,
    "end_ms": 1380,
    "text": "Chairman, thank you.",
    "speaker": "SPEAKER_00",
    "confidence": 0.94,
    "word_probabilities": [
        {"word": "Chairman,", "probability": 0.96},
        {"word": "thank", "probability": 0.93},
        {"word": "you.", "probability": 0.93}
    ]
}
```

## Background & Technical Approach

This enhanced system builds upon the foundational streaming Whisper architecture while adding significant new capabilities for real-world applications.

### Core Streaming Architecture

The base streaming approach addresses Whisper's original 30-second chunk limitation through:

- **Local Agreement Policy**: Consecutive updates must agree on transcript prefixes before confirmation
- **Dynamic Buffer Management**: Smart audio buffer trimming based on sentence/segment boundaries  
- **Init Prompt Handling**: Proper context management for continuous processing
- **Timestamp Synchronization**: Accurate alignment between audio and text outputs

### Enhanced Features in This Fork

#### Streaming Probabilities
- **Word-level confidence tracking** from Whisper model outputs
- **Real-time quality assessment** for streaming applications
- **Probability aggregation** for sentence and utterance-level confidence
- **Quality-based filtering** to improve transcription reliability

#### Real-Time Speaker Diarization
- **Resemblyzer embeddings** for robust speaker characterization
- **Silero VAD integration** for precise voice activity detection
- **Streaming-optimized clustering** with minimal latency impact
- **Speaker persistence** across audio segments and sessions

#### Modular Architecture
- **Pluggable ASR backends** supporting multiple Whisper implementations
- **Containerized development** with full GPU acceleration
- **Enhanced buffering** with probability-aware processing
- **Network-optimized streaming** for real-time applications

### Technical Implementation

The system processes audio through these key stages:

1. **Audio Buffering**: Accumulate chunks with Voice Activity Detection
2. **ASR Processing**: Extract transcriptions with word-level probabilities
3. **Speaker Analysis**: Generate embeddings and assign speaker identities
4. **Hypothesis Management**: Buffer and confirm transcripts using local agreement
5. **Output Generation**: Combine text, timing, speakers, and confidence scores

This approach achieves low-latency streaming while maintaining high accuracy and providing rich metadata for downstream applications.

### Performance Characteristics

Based on the original research and enhancements:
- **Latency**: ~3.3 seconds for high-quality transcription
- **Accuracy**: Maintains Whisper model quality with streaming optimizations
- **Speaker Identification**: Real-time diarization with minimal overhead
- **Scalability**: Efficient GPU utilization for multiple concurrent streams

## Acknowledgments & Credits

This work builds upon significant contributions from the research and open-source communities:

### Original Research
- **Dominik Macháček, Raj Dabre, Ondřej Bojar** for the foundational streaming Whisper research
- **Paper**: ["Turning Whisper into Real-Time Transcription System"](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) (IJCNLP-AACL 2023)

### Technical Foundations  
- **Peter Polák** for the [original streaming demo concept](https://github.com/pe-trik/transformers/blob/online_decode/examples/pytorch/online-decoding/whisper-online-demo.py)
- **UEDIN team** of the [ELITR project](https://elitr.eu) for the original `line_packet.py`
- **Silero Team** for their [VAD model](https://github.com/snakers4/silero-vad) and VADIterator implementation

### Community Contributions
- **Original whisper_streaming contributors** for the foundational codebase
- **Resemblyzer team** for speaker embedding technology
- **faster-whisper developers** for optimized inference engines
- **OpenAI** for the Whisper model family

### Enhanced Implementation

This work builds upon the foundational research by Macháček et al. (2023) on real-time Whisper streaming:

[Paper PDF](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [Demo video](https://player.vimeo.com/video/840442741) | [Slides](http://ufallab.ms.mff.cuni.cz/~machacek/pre-prints/AACL23-2.11.2023-Turning-Whisper-oral.pdf)

For the original research, please cite:
```bibtex
@inproceedings{machacek-etal-2023-turning,
    title = "Turning Whisper into Real-Time Transcription System",
    author = "Mach{\'a}{\v{c}}ek, Dominik and Dabre, Raj and Bojar, Ond{\v{r}}ej",
    booktitle = "Proceedings of IJCNLP-AACL 2023: System Demonstrations",
    year = "2023",
    url = "https://aclanthology.org/2023.ijcnlp-demo.3",
    pages = "17--24"
}
```

## Contact & Support

For questions about this enhanced implementation:
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions and usage help
- **Original Research**: Contact Dominik Macháček (machacek@ufal.mff.cuni.cz) for research-related questions

