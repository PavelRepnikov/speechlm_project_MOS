import os
import json
import torchaudio
from torch.utils.data import Dataset
import torch


class SomosDataset(Dataset):
    def __init__(self, json_path, audio_dir, sample_rate=16000, transform=None, process_audio=None):
        """
        :param json_path: Path to JSON file (train.json or test.json)
        :param audio_dir: Root folder with audio files
        :param sample_rate: Audio sampling rate
        :param transform: Audio transformations
        :param process_audio: Function for audio preprocessing
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.process_audio = process_audio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Validate keys
        required_keys = {"text", "clean path", "mos"}
        if not required_keys.issubset(sample):
            raise ValueError(f"Sample {idx} missing required keys: {required_keys - sample.keys()}")

        text = sample["text"]
        audio_path = os.path.join(self.audio_dir, sample["clean path"])
        mos = float(sample["mos"])

        # Ensure text is a string
        if not isinstance(text, str):
            raise ValueError("Text must be a string.")

        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except FileNotFoundError:
            raise RuntimeError(f"Audio file not found: {audio_path}")

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)

        # Apply optional audio transformations
        if self.transform:
            waveform = self.transform(waveform)

        # Apply custom audio processing function
        if self.process_audio:
            waveform = self.process_audio(waveform)

        # Normalize audio
        waveform = waveform / waveform.abs().max()

        return {
            "text": text,
            "audio": waveform,  # Audio shape: (samples,)
            "mos": mos
        }
