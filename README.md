# Transcription-using-Whisper

This repository provides a pipeline for automatic transcription of Hindi audio (including Hinglish) using the [Oriserve/Whisper-Hindi2Hinglish-Prime](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime) model. The pipeline supports various audio formats and outputs both plain and timestamped transcripts.

## Features

- **Audio Format Support:** Accepts mp3, m4a, wav, and more.
- **Voice Activity Detection (VAD):** Uses Silero VAD to segment speech.
- **Transcription:** Utilizes HuggingFace Transformers Whisper model for Hindi/Hinglish transcription.
- **Timestamped Segments:** Outputs both plain text and timestamped JSON files.
- **CUDA Support:** Automatically uses GPU if available.

## Directory Structure

- `transcribe_pipeline.py` — Main pipeline script for transcription.
- `config.py` — Configuration for model, device, and directories.
- `audios/` — Place your input audio files here.
- `outputs/` — Transcription results are saved here.
- `ffmpeg-7.0.2-full_build/` — FFmpeg binaries for audio processing.
- `requirements.txt` — Python dependencies.

## Installation

1. **Clone the repository**
    ```sh
   git clone https://github.com/mohsinansari0705/Transcription-using-Whisper.git
   cd Transcription-using-Whisper
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Ensure FFmpeg is available**
    - Use the provided [`ffmpeg-7.0.2-full_build`](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.0.2-full_build.7z) or install via your OS package manager.
    - Add the path to the FFmpeg `bin` directory (e.g., `ffmpeg-7.0.2-full_build/bin`) to your system's environment `PATH` variable so FFmpeg commands are recognized.

## Usage

1. Place your audio file (e.g., `sample.m4a`) in the `audios/` directory.
2. Edit `transcribe_pipeline.py` to set the correct input filename if needed.
3. Run the transcription pipeline:
   ```sh
   python transcribe_pipeline.py
   ```
4. Find results in `outputs/`:
   - `*_transcribed_plain.json`: Full transcript.
   - `*_transcribed_timestamped.json`: Segmented transcript with timestamps.

## Output Format

- **Plain JSON:** Contains metadata and the full transcript.
- **Timestamped JSON:** Contains metadata and a list of segments with start/end times and text.

## Configuration

See [`config.py`](config.py) for model selection, device, and directory paths.

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

**Author:** Mohsin Ansari