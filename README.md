# DCASE2026 Challenge

Repository for participating in the [DCASE 2026 Challenge](https://dcase.community/) — Detection and Classification of Acoustic Scenes and Events.

## Model Architecture

LCS-DETR extends [SG-DETR](https://arxiv.org/pdf/2410.01615v1) with two key mechanisms for audio moment retrieval: (1) **saliency gating** via cross-modal cosine similarity to focus on relevant audio frames, and (2) **local temporal continuity** via depthwise-separable convolution to smooth temporal dependencies.

Given an audio-text pair, [CLAP](https://github.com/microsoft/CLAP) extracts aligned embeddings. Saliency-weighted features pass through a Transformer encoder-decoder (2 layers each, 8 attention heads) with 10 learnable queries. The model outputs span predictions (center, width) with confidence scores via span regression and binary classification heads, optimized with Focal Loss and GIoU.

## Usage

The repository uses a unified CLI to trigger different pipelines. All commands should be run from the root of the project.

### 1. Training
Run the training pipeline from scratch or fine-tune from a checkpoint:
```bash
# Train from scratch
python -m src train --config config/config.yml

# Fine-tune from an existing checkpoint
python -m src train --config config/config.yml --model_path results/best_checkpoint.pth
```

### 2. Evaluation
Evaluate a trained model on a specific split (`val` or `test`):
```bash
python -m src evaluate --config config/config.yml --model_path results/best_checkpoint.pth --split test
```

### 3. Create Submission
Generate private submissions for the challenge:
```bash
python -m src create_submission --config config/config.yml --model_path results/best_checkpoint.pth
```


## Reference

```
@inproceedings{munakata2025audiomoment,
  author = {Munakata, Hokuto and Nishimura, Taichi and Nakada, Shota and Komatsu, Tatsuya},
  title = {Language-based Audio Moment Retrieval},
  booktitle = {Proc. ICASSP},
  year = {2025},
  pages = {1-5},
  _pdf = {https://arxiv.org/pdf/2409.15672}
}
```
QD-DETR citation:
```
@inproceedings{qddetr
    author = {WonJun Moon and Sangeek Hyun and SangUk Park and Dongchan Park and Jae-Pil Heo},
    title = {Query-Dependent Video Representation for Moment Retrieval and Highlight Detection},
    booktitle = {Proc. CVPR},
    year = {2023},
}
```


## Contact 

minh.leduc.0210@gmail.com