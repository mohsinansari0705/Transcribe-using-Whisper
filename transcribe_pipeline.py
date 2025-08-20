"""
    Pipeline for transcription using Oriserve/Whisper-Hindi2Hinglish-Prime
    Outputs:
     - transcript_plain.json
     - transcript_timestamped.json
"""
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutomaticSpeechRecognitionPipeline
from silero_vad import get_speech_timestamps, load_silero_vad
from config import MODEL_ID, DEVICE, OUTPUT_DIR, INPUT_DIR
from datetime import datetime
import numpy as np
import ffmpeg
import torch
import json
import uuid
import os


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
    

def merge_vad_segments(segments, max_pause=0.5, sr=16000):
    """Merge VAD segments with short pauses into longer chunks."""
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    for seg in segments[1:]:
        pause = (seg['start'] - current['end']) / sr
        if pause <= max_pause:
            current['end'] = seg['end']
        else:
            merged.append(current)
            current = seg.copy()

    merged.append(current)
    return merged


def transcribe_audio_with_vad(audio_path: str, audio_id: str):
    """
    Transcribe audio file into plain text and timestamped segments.
    Returns two dicts for saving as JSON.
    """

    # Load audio
    audio, sr = load_audio_any_format(audio_path)

    speech_segments = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=sr)
    merged_segments = merge_vad_segments(speech_segments, max_pause=0.5, sr=sr)

    # Load STT model once with specified configurations
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(DEVICE)

    # Load the processor for audio preprocessing
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    # Speech recognition pipeline
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=DEVICE,
        generate_kwargs={
            "task": "transcribe",
            "language": "en"
        }
    )

    segments = []
    full_text = ""
    for seg in merged_segments:
        start_sample = seg['start']
        end_sample = seg['end']
        segment_audio = audio[start_sample:end_sample]

        result = pipe(segment_audio, return_timestamps='segment')

        seg_text = result['text']
        full_text += seg_text + " "

        segments.append({
            "start": start_sample / sr,     # convert to seconds
            "end": end_sample / sr,
            "text": seg_text
        })

    # Build Plain JSON
    plain_json = {
        "job_id": str(uuid.uuid4()),
        "model": MODEL_ID,
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_file": audio_id,
        "sample_rate": sr,
        "device": DEVICE,
        "duration": len(audio) / sr,
        "transcript": full_text.strip(),
        "word_count": len(full_text.strip().split())
    }

    # Build Timestamped JSON    
    timestamped_json = {
        "job_id": str(uuid.uuid4()),
        "model": MODEL_ID,
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_file": audio_id,
        "sample_rate": sr,
        "device": DEVICE,
        "duration": len(audio) / sr,
        "segments": [
            {
                "index": idx,
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text'],
                "word_count": len(seg['text'].split())
            }
            for idx, seg in enumerate(segments)
        ]
    }

    return plain_json, timestamped_json


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # your input file
    audio_file_name = 'sample_me.mp4'
    audio_file = os.path.join(INPUT_DIR, audio_file_name)

    plain, timestamped = transcribe_audio_with_vad(audio_file, audio_file_name)

    save_json(plain, os.path.join(OUTPUT_DIR, f"{audio_file_name}_transcribed_plain.json"))
    save_json(timestamped, os.path.join(OUTPUT_DIR, f"{audio_file_name}_transcribed_timestamped.json"))

    print(f"✅ Transcription done. Files saved in {OUTPUT_DIR}")