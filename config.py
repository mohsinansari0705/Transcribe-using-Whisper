import torch
import os

MODEL_ID = "Oriserve/Whisper-Hindi2Hinglish-Prime"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
INPUT_DIR = os.path.join(ROOT_DIR, 'audios')