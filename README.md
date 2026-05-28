# DCASE2026 Challenge

Repository for participating in the [DCASE 2026 Challenge](https://dcase.community/) — Detection and Classification of Acoustic Scenes and Events.

## 🎯 Target Tasks

Task: Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring / Language-Based Audio Retrieval

## 📁 Project Structure

```text
DCASE2026/
├── data/               # Datasets and features
├── config/             # YAML configuration files (e.g., config_enhanced.yml)
├── notebooks/          # EDA & experimentation notebooks
├── reference/          # Reference papers & literature
├── src/
│   ├── core/           # Core configuration and environment settings
│   ├── data/           # Dataset loaders and vocabulary management
│   ├── models/         # Neural network architectures
│   │   ├── qd_detr/    # QD-DETR model, transformer, matcher, and postprocessing
│   │   ├── components/ # Core components (Attention, Mamba, Encoders)
│   │   └── feature_extractor/ # Feature extraction utilities
│   ├── pipelines/      # High-level execution scripts (Train, Evaluate, Submit)
│   ├── utils/          # Math, tensor, span, and logging utilities
│   └── __main__.py     # Universal CLI entrypoint
├── .env                # Environment variables (not tracked)
├── .gitignore
└── README.md
```

## 🚀 Usage

The repository uses a unified CLI to trigger different pipelines. All commands should be run from the root of the project.

### 1. Training
Run the training pipeline from scratch or fine-tune from a checkpoint:
```bash
# Train from scratch
python -m src train --config config/config_enhanced.yml

# Fine-tune from an existing checkpoint
python -m src train --config config/config_enhanced.yml --resume path/to/checkpoint.pt
```

### 2. Evaluation
Evaluate a trained model on a specific split (`val` or `test`):
```bash
python -m src evaluate --config config/config_enhanced.yml --model_path path/to/checkpoint.pt --split val
```

### 3. Create Submission
Generate private submissions for the challenge:
```bash
python -m src create_submission --config config/config_enhanced.yml --model_path path/to/checkpoint.pt
```

## 🛠️ Key Technical Features

- **Hardware Acceleration**: Built-in support for Apple Silicon (`mps`) and NVIDIA GPUs (`cuda`).
- **Mixed Precision**: Automatic AMP casting with `bfloat16` for stable training and reduced memory footprint.
- **Advanced Matcher**: Stable Hungarian Bipartite Matcher supporting Focal Loss and cross-device safety mechanisms.
- **Optimized Attention**: Uses `FlashMultiheadAttention` with dynamic hardware fallbacks.
- **Model Compilation**: Optionally integrates `torch.compile` for faster execution.

## 📚 Reference Papers

### SELD
- Spatial and Semantic Embedding Integration for Stereo SELD
- STARSS23: Audio-Visual Dataset of Spatial Recordings
- Stereo SELD with Onscreen-Offscreen Classification
- The NERC-SLIP System for Stereo SELD (DCASE 2025)
- Self-Guided Target Sound Extraction and Classification

### Language-Based Audio Retrieval
- A Cross-Modal Attention Approach to Language-Based Audio Retrieval
- AISTAT Lab System for DCASE 2025 Task 6
- Clotho: An Audio Captioning Dataset
- Dual-Encoder Audio Retrieval with PaSST and RoBERTa

## 📝 License

This project is for research/competition purposes.
