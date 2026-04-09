import os
import random
from typing import Callable, Dict, Optional

import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset

class AudioAugmentor:
    """Safe audio augmentations for audio-text retrieval."""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.augmentations = [
            self.add_noise,
            self.random_gain,
            self.time_shift,
            self.polarity_inversion,
            self.speed_perturb,
            self.spec_augment_on_waveform,
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Apply 1-3 random augmentations
        n_augs = random.randint(1, 3)
        chosen = random.sample(self.augmentations, min(n_augs, len(self.augmentations)))
        for aug in chosen:
            waveform = aug(waveform)
        return waveform

    def add_noise(self, waveform, snr_db_range=(15, 40)):
        snr_db = random.uniform(*snr_db_range)
        noise = torch.randn_like(waveform)
        signal_power = waveform.norm(p=2)
        noise_power = noise.norm(p=2)
        if noise_power == 0:
            return waveform
        scale = signal_power / (10 ** (snr_db / 20) * noise_power)
        return waveform + scale * noise

    def random_gain(self, waveform, min_db=-6, max_db=6):
        gain_db = random.uniform(min_db, max_db)
        return waveform * (10 ** (gain_db / 20))

    def time_shift(self, waveform, max_shift=0.1):
        shift = int(waveform.shape[-1] * random.uniform(-max_shift, max_shift))
        return torch.roll(waveform, shifts=shift, dims=-1)

    def polarity_inversion(self, waveform):
        return -waveform if random.random() > 0.5 else waveform

    def speed_perturb(self, waveform, factor_range=(0.95, 1.05)):
        factor = random.uniform(*factor_range)
        resampled = torchaudio.functional.resample(
            waveform, orig_freq=self.sr, new_freq=int(self.sr * factor)
        )
        return resampled

    def spec_augment_on_waveform(self, waveform):
        """Apply small random zero-out segments (simplified SpecAugment on waveform)."""
        clip_fraction = random.uniform(0.0, 0.05)
        clip_len = int(waveform.shape[-1] * clip_fraction)
        if clip_len > 0:
            start = random.randint(0, waveform.shape[-1] - clip_len)
            waveform = waveform.clone()
            waveform[..., start : start + clip_len] = 0
        return waveform


class TextAugmentor:
    """Safe text augmentations for audio-text retrieval (meaning-preserving)."""

    def __init__(self):
        self.templates = [
            "{caption}",
            "The sound of {caption_lower}",
            "Audio recording of {caption_lower}",
            "A clip where {caption_lower}",
            "You can hear {caption_lower}",
            "An audio where {caption_lower}",
        ]

    def __call__(self, caption: str) -> str:
        # Apply one random text augmentation
        aug = random.choice(
            [
                self.template_wrap,
                self.light_synonym_replace,
                self.identity,
            ]
        )
        return aug(caption)

    def identity(self, caption):
        return caption

    def template_wrap(self, caption):
        template = random.choice(self.templates)
        return template.format(caption=caption, caption_lower=caption.lower())

    def light_synonym_replace(self, caption, n=1):
        """Replace 1 word with a simple synonym (no NLTK dependency)."""
        # Lightweight synonyms for common audio description words
        synonyms = {
            "loud": ["noisy", "blaring"],
            "quiet": ["soft", "gentle"],
            "fast": ["quick", "rapid"],
            "slow": ["unhurried", "gradual"],
            "big": ["large", "huge"],
            "small": ["tiny", "little"],
            "many": ["several", "numerous"],
            "walks": ["strolls", "steps"],
            "runs": ["jogs", "sprints"],
            "talks": ["speaks", "chats"],
            "sings": ["vocalizes", "chants"],
            "plays": ["performs"],
            "hits": ["strikes", "taps"],
            "falls": ["drops", "descends"],
            "moves": ["shifts", "travels"],
            "flows": ["streams", "runs"],
            "blows": ["gusts", "whooshes"],
        }
        words = caption.split()
        replaced = False
        indices = list(range(len(words)))
        random.shuffle(indices)
        for idx in indices:
            word_lower = words[idx].lower().strip(".,!?")
            if word_lower in synonyms and not replaced:
                replacement = random.choice(synonyms[word_lower])
                # Preserve original casing roughly
                if words[idx][0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
                replaced = True
                break
        return " ".join(words)


class ClothoDataset(Dataset):
    """
    Clotho dataset for Language-Based Audio Retrieval.

    Compatible with HuggingFace Trainer — returns a dict from __getitem__.

    Args:
        audio_dir:       Path to the directory containing .wav files
        captions_csv:    Path to the captions CSV file
        audio_processor: A callable that processes raw waveform into model inputs
                         (e.g., ClapProcessor, Wav2Vec2Processor, or a mel-spec transform)
        text_tokenizer:  A callable tokenizer (e.g., AutoTokenizer)
        sr:              Target sample rate
        max_audio_len:   Max audio length in seconds (clips/pads to this)
        train:           If True, apply augmentations
        text_max_length: Max token length for text
    """

    def __init__(
        self,
        audio_dir: str,
        captions_csv: str,
        audio_processor: Optional[Callable] = None,
        text_tokenizer: Optional[Callable] = None,
        sr: int = 44100,
        max_audio_len: float = 30.0,
        train: bool = True,
        text_max_length: int = 77,
    ):
        self.audio_dir = audio_dir
        self.sr = sr
        self.max_audio_len = max_audio_len
        self.max_samples = int(sr * max_audio_len)
        self.train = train
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length

        # Load captions and melt into (file_name, caption) pairs
        captions_df = pd.read_csv(captions_csv)
        caption_cols = [c for c in captions_df.columns if c.startswith("caption")]

        self.data = captions_df.melt(
            id_vars=["file_name"],
            value_vars=caption_cols,
            var_name="caption_number",
            value_name="caption",
        ).reset_index(drop=True)

        # Drop any rows with missing captions
        self.data = self.data.dropna(subset=["caption"]).reset_index(drop=True)

        # Augmentors (only used in train mode)
        self.audio_augmentor = AudioAugmentor(sr=sr) if train else None
        self.text_augmentor = TextAugmentor() if train else None

    def __len__(self) -> int:
        return len(self.data)

    def _load_audio(self, file_name: str) -> torch.Tensor:
        """Load and preprocess audio to a fixed length."""
        path = os.path.join(self.audio_dir, file_name)
        waveform, orig_sr = torchaudio.load(path)

        # Resample if needed
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to max_samples
        if waveform.shape[-1] > self.max_samples:
            # Random crop during training, center crop during eval
            if self.train:
                start = random.randint(0, waveform.shape[-1] - self.max_samples)
            else:
                start = (waveform.shape[-1] - self.max_samples) // 2
            waveform = waveform[..., start : start + self.max_samples]
        elif waveform.shape[-1] < self.max_samples:
            pad_len = self.max_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform  # shape: (1, max_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        file_name = row["file_name"]
        caption = row["caption"]

        # --- Load audio ---
        waveform = self._load_audio(file_name)

        # --- Audio augmentation (train only) ---
        if self.train and self.audio_augmentor:
            waveform = self.audio_augmentor(waveform)

        # --- Text augmentation (train only) ---
        if self.train and self.text_augmentor:
            caption = self.text_augmentor(caption)

        # --- Process audio through processor (e.g., CLAP, mel-spec) ---
        if self.audio_processor is not None:
            audio_inputs = self.audio_processor(
                audios=waveform.squeeze(0).numpy(),
                sampling_rate=self.sr,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_samples,
            )
            # Flatten batch dim added by processor
            audio_inputs = {k: v.squeeze(0) for k, v in audio_inputs.items()}
        else:
            # Return raw waveform if no processor
            audio_inputs = {"input_values": waveform.squeeze(0)}

        # --- Tokenize text ---
        if self.text_tokenizer is not None:
            text_inputs = self.text_tokenizer(
                caption,
                padding="max_length",
                max_length=self.text_max_length,
                truncation=True,
                return_tensors="pt",
            )
            # Flatten batch dim
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        else:
            text_inputs = {"caption": caption}

        # --- Combine into a single dict (HF Trainer compatible) ---
        output = {}
        output.update(audio_inputs)
        output["input_ids"] = text_inputs.get("input_ids", caption)
        output["attention_mask"] = text_inputs.get(
            "attention_mask",
            torch.ones_like(text_inputs.get("input_ids", torch.tensor([]))),
        )

        return output