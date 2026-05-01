import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from src.settings import settings


class CLAPEncoder:
    model_id = "laion/clap-htsat-unfused"

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, dtype=torch.float16, device_map=settings.DEVICE
        )
        self.model = AutoModel.from_pretrained(self.model_id)

    def encode_audio(self, audio):
        inputs = self.processor(
            audio, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def encode_text(self, text):
        inputs = self.processor(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class BERTEncoder:
    model_id = ""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)

    def encode_text(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class ModernBERTEncoder:
    model_id = ""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)

    def encode_text(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
