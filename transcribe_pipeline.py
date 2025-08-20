"""
    Pipeline for transcription using Oriserve/Whisper-Hindi2Hinglish-Prime
    Outputs:
     - transcript_plain.json
     - transcript_timestamped.json
"""

# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutomaticSpeechRecognitionPipeline
from config import MODEL_ID, DEVICE, OUTPUT_DIR, INPUT_DIR
import numpy as np
import ffmpeg
import torch
import json
import os


"""
    Load Model, Processor, and Pipeline once
"""
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load STT model with specified configurations
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch_dtype,        # Use appropriate precision (float16 for GPU, float32 for CPU)
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(DEVICE)

# Set the alignment_heads for word-level timestamps
alignment_heads = [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]]
model.generation_config.alignment_heads = alignment_heads

# Load the processor for audio preprocessing
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Speech recognition pipeline
pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=DEVICE,
    # generate_kwargs={
    #     "task": "transcribe",
    #     "language": "en"
    # }
)


def load_audio_any_format(audio_path, target_sr=16000):
    """
    Loads any audio format (mp3, m4a, wav, etc.) 
    and returns mono PCM float32 numpy array at target_sr.
    """
    try:
        out, _ = (
            ffmpeg
            .input(audio_path)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

        return audio, target_sr
    except Exception:
        raise RuntimeError(f"ffmpeg failed to load '{audio_path}'.")


def transcribe_audio(audio_path: str, audio_id: str):
    """
    Transcribe audio file into plain text and timestamped segments.
    Returns two dicts for saving as JSON.
    """

    # Load audio
    audio, sr = load_audio_any_format(audio_path)
    
    result = pipe(audio, return_timestamps='segment')

    # Build Plain JSON
    plain_json = {
        "audio_id": audio_id,
        "transcript": result["text"]
    }

    # Build Timestamped JSON
    segments = []
    if "chunks" in result:
        # chunks contain start, end, text
        for seg in result["chunks"]:
            segments.append({
                "start": float(seg["timestamp"][0]) if seg["timestamp"][0] is not None else None,
                "end": float(seg["timestamp"][1]) if seg["timestamp"][1] is not None else None,
                "text": seg["text"]
            })
    else:
        segments.append({
            "start": 0.0,
            "end": None,
            "text": result["text"]
        })

    timestamped_json = {
        "audio_id": audio_id,
        "segments": segments
    }

    return plain_json, timestamped_json


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # your input file
    audio_file_name = 'sample.ogg'
    audio_file = os.path.join(INPUT_DIR, audio_file_name)

    plain, timestamped = transcribe_audio(audio_file, audio_file_name)

    save_json(plain, os.path.join(OUTPUT_DIR, "transcribed_plain.json"))
    save_json(timestamped, os.path.join(OUTPUT_DIR, "transcribed_timestamped.json"))

    print(f"✅ Transcription done. Files saved in {OUTPUT_DIR}")