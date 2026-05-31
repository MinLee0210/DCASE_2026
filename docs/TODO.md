# TODO

## Model & Training Pipeline Improvements

### 🏃 Training Pipeline & Optimization
- [x] **Learning Rate Scheduler:** Replace `StepLR` with `CosineAnnealingLR` (or Cosine with Linear Warmup) in `src/pipelines/evaluate.py` (`setup_model`).
- [ ] **Parameter-Specific Learning Rates:** Group parameters in `AdamW` to apply different learning rates for the early encoder/projection layers versus the transformer decoder and prediction heads.
- [ ] **Enable Focal Loss:** Update configurations (e.g., `config.yml`) to set `use_focal_loss: True`. This helps handle the foreground/background imbalance inherent in temporal grounding.
- [ ] **Gradient Accumulation:** Implement gradient accumulation in `train_epoch` (`src/pipelines/train.py`) to simulate larger effective batch sizes, which stabilizes training for DETR-based models.

### 🏗️ Model Architecture (`lcs_detr`)
- [ ] **Rotary Position Embeddings (RoPE):** Replace standard absolute sine/cosine embeddings with 1D RoPE (as suggested in `model.py` comments) to better capture relative temporal distances.
- [ ] **Query Conditioning (Conditional/Anchor DETR):** Inject text embeddings (`src_txt`) into the `self.query_embed` initialization. This helps the decoder focus on relevant regions sooner by making it explicitly "query-driven".
- [x] **Conv1d in Saliency Head:** Add lightweight 1D Convolutions (e.g., kernel size 3 or 5) before the linear projections in the `LocalSaliencyHead` to capture local temporal continuity and smooth saliency scores.

### 📊 Feature-Level Augmentations
- [ ] **Temporal Feature Masking:** Implement SpecAugment-style random masking of small time-blocks in `audio_feat` during training to prevent overfitting to specific high-magnitude frames.
- [ ] **Feature Dropout/Noise:** Apply small random noise or dropout directly to input features (`src_aud` and `src_txt`) before they pass through the input projection layers to increase robustness to feature extraction artifacts.


---

Local Continuity Saliency (LCS)
The most effective component that significantly improved my score was Local Continuity Saliency (LCS). It addresses a key weakness in standard DETR-based models: the difficulty of capturing fine-grained temporal dependencies with standard Cross-Attention alone.

How it works:
Standard DETR relies on Cross-Attention between Text and Audio tokens. In your setup, audio is sampled at 2Hz, meaning consecutive tokens are 0.5s apart. A standard transformer layer has a "receptive field" of about 4-6 tokens. This means the attention mechanism has to "jump" 4-5 positions (0.5-0.6s) in a single step to connect a sound event to its description. This is hard to learn.

LCS introduces a lightweight 1D Convolutional layer directly on the Audio Encoder outputs (before the Cross-Attention). This convolution looks at immediate neighbors (e.g., 5 frames), forcing the model to learn short-term temporal continuity (e.g., the start of a sound fading into its peak). By smoothing the audio features locally, the model can better "see" the shape of the sound event when it attends to the text.

How to implement it (Code Snippet):
I added this logic to the LocalSaliencyHead (inside `src/models/components/transformer/encoder.py`). You already have the SaliencyHead, you just need to add the convolution:

# Inside LocalSaliencyHead.__init__:

# Add this line:
self.use_saliency_conv = use_saliency_conv
if self.use_saliency_conv:
    # Lightweight 1D Convolution to capture local temporal continuity
    self.audio_conv = nn.Sequential(
        nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2, groups=model_dim),
        nn.GELU(),
        nn.Conv1d(model_dim, model_dim, kernel_size=1)
    )
# Update the forward pass to use it:

# Inside LocalSaliencyHead.forward(...):

if getattr(self, "use_saliency_conv", False):
    # Apply 1D Convolution to audio features
    aud_features = src_aud.transpose(1, 2)  # [bs, model_dim, seq_len]
    aud_features = self.audio_conv(aud_features)
    aud_features = aud_features.transpose(1, 2) + src_aud  # Residual connection
else:
    aud_features = src_aud

saliency_scores = self.saliency_scores(aud_features, src_sent)