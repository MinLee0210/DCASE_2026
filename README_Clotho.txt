---
language:
  - en
license: other
task_categories:
  - audio-classification
  - text-to-audio
  - audio-to-text
tags:
  - audio-retrieval
  - audio-captioning
  - DCASE
  - CLAP
  - contrastive-learning
pretty_name: Clotho Development Subset
size_categories:
  - 1K<n<10K
---

# Clotho Development Subset

A sampled subset (~3GB) of the [Clotho v2.1](https://zenodo.org/record/3490684) development split, packaged for quick experimentation with audio-text retrieval pipelines.

## 📋 Dataset Description

This dataset is a convenience subset of the **Clotho** audio captioning dataset, created for rapid prototyping and testing of audio-text retrieval models (e.g., CLAP fine-tuning) on limited compute.

- **Source**: Clotho v2.1 (development split)
- **Original Authors**: K. Drossos, S. Lipping, T. Virtanen
- **Original Paper**: [Clotho: An Audio Captioning Dataset](https://arxiv.org/abs/1910.09387)

## 📊 Dataset Structure

### Splits

| Split | Samples | Description |
|-------|---------|-------------|
| train | ~1,300  | Training set (80%) |
| test  | ~330    | Test set (20%) |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `file_name` | `string` | Original filename from Clotho |
| `audio` | `Audio` | Audio waveform, 44.1kHz |
| `caption_1` | `string` | Human-written caption #1 |
| `caption_2` | `string` | Human-written caption #2 |
| `caption_3` | `string` | Human-written caption #3 |
| `caption_4` | `string` | Human-written caption #4 |
| `caption_5` | `string` | Human-written caption #5 |

### Audio Details

- **Duration**: 15–30 seconds per clip
- **Sample Rate**: 44,100 Hz
- **Channels**: Mono
- **Format**: WAV (stored as Parquet/Arrow on Hub)

## 🚀 Usage

```python
from datasets import load_dataset

ds = load_dataset("your-username/clotho-dev-sample")

# Access a sample
sample = ds["train"][0]
print(sample["caption_1"])   # "A dog barks in the distance"
print(sample["audio"])       # {'array': array([...]), 'sampling_rate': 44100}
```

### With CLAP

```python
from transformers import ClapProcessor, ClapModel

processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

sample = ds["train"][0]
inputs = processor(
    audios=sample["audio"]["array"],
    sampling_rate=sample["audio"]["sampling_rate"],
    text=sample["caption_1"],
    return_tensors="pt",
    padding=True,
)
outputs = model(**inputs)
```

## ⚠️ Important Notes

- This is a **subset** (~43%) of the full Clotho development split, sampled randomly with `seed=42`
- For official benchmarking, use the full Clotho dataset from [Zenodo](https://zenodo.org/record/3490684)
- This subset is intended for **pipeline testing and prototyping only**

## 📄 Citation

If you use this dataset, please cite the original Clotho paper:

```bibtex
@inproceedings{drossos2020clotho,
  title={Clotho: An Audio Captioning Dataset},
  author={Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas},
  booktitle={ICASSP 2020 - IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={736--740},
  year={2020},
  organization={IEEE}
}
```

## 🏷️ License

This dataset follows the original Clotho license. Audio clips are sourced from [Freesound](https://freesound.org/) under Creative Commons licenses. Please refer to the [original dataset](https://zenodo.org/record/3490684) for full license details.
