# Language-Based Audio Moment Retrieval Using a Saliency-Guided Transformer with Enhanced Training Strategies

## Abstract

We present an enhanced system for Language-Based Audio Moment Retrieval (LBAMR) submitted to the DCASE 2026 Challenge (Task 6). Our approach builds upon the LCS-DETR (Language-Conditioned Saliency Detection Transformer) architecture, introducing several targeted improvements: a convolutional saliency refinement module for local temporal modeling, a Focal Loss objective for imbalanced set prediction, and hardware-aware training optimizations. Experiments are conducted on the Clotho-Moment and Castella datasets using CLAP-based audio and text embeddings.

---

## 1. Introduction

Language-Based Audio Moment Retrieval is the task of identifying temporal segments within an audio recording that correspond to a free-text natural language query. This is fundamentally a temporal grounding problem: given an audio stream of duration $T$ seconds and a text query $q$, the system must predict a set of temporal intervals $\{[t_s, t_e]\}$ that best match the described auditory event.

We build upon the Detection Transformer (DETR) paradigm [Carion et al., 2020], which reformulates object detection as a direct set prediction problem using a Hungarian bipartite matching loss. This framework was adapted to natural language video grounding by Moment-DETR [Lei et al., 2021] and further refined for audio by the LCS-DETR baseline [LINE Yahooo, EMNLP 2024], which introduced a saliency-guided cross-modal encoding strategy.

In this work, we introduce the following improvements over the LCS-DETR baseline:

1. A lightweight 1D convolutional module integrated into the Local Saliency Head to capture local temporal continuity.
2. Focal Loss replacing standard Cross-Entropy in the set prediction objective to address the foreground/background class imbalance.
3. A Cosine Annealing learning rate schedule for smoother convergence.

---

## 2. Background and Baseline Architecture

### 2.1 DETR-Based Moment Retrieval

The LCS-DETR baseline treats moment retrieval as a direct set prediction problem. Given a sequence of audio clip features $\mathbf{A} \in \mathbb{R}^{L_a \times d_a}$ and text query features $\mathbf{T} \in \mathbb{R}^{L_t \times d_t}$, the model predicts a set of $N$ candidate temporal spans $\{\hat{s}_i\}_{i=1}^{N}$, each represented as a center-width tuple $(c_x, w)$ normalized by the total audio duration.

A fixed set of $N$ learnable object queries $Q \in \mathbb{R}^{N \times d}$ interact with the encoded audio-text memory through a standard Transformer decoder. The final predictions are produced by two heads: a span regression head (3-layer MLP) and a binary foreground/background classification head (linear layer).

### 2.2 Saliency-Guided Cross-Attention

A key component of the LCS-DETR baseline is the saliency-guided cross-attention mechanism. Before the main Transformer encoder-decoder, a **Local Saliency Head** computes a per-clip saliency score by measuring the cosine similarity between each audio clip embedding and an aggregated sentence-level text representation.

Let $\mathbf{a}_i \in \mathbb{R}^d$ denote the embedding of the $i$-th audio clip and $\hat{\mathbf{t}} \in \mathbb{R}^d$ denote the aggregated sentence embedding (obtained via a Learned Aggregation module). The saliency score for clip $i$ is:

$$
s_i = \frac{\mathbf{a}_i \cdot \hat{\mathbf{t}}}{\|\mathbf{a}_i\| \cdot \|\hat{\mathbf{t}}\|} \cdot b + a
$$

where $a$ and $b$ are learnable scalar parameters.

These saliency scores are then used to weight the cross-attention in the **Text-to-Audio (T2A) encoder**, which fuses the textual and audio representations before they are passed to the core Transformer. Furthermore, a **Saliency Amplifier** module re-weights the audio memory features emitted by the encoder, amplifying clips deemed relevant by the saliency estimator before the final span prediction.

### 2.3 Features and Input Representations

All audio and text features are derived from **CLAP** (Contrastive Language-Audio Pretraining) [Wu et al., 2023], which produces 768-dimensional embeddings for both modalities in a shared semantic space. This is particularly well-suited for audio-text grounding tasks since the audio and text embeddings are pretrained to be semantically aligned.

Audio features are augmented with **Temporal Embedding Features (TEF)**, a 2-dimensional feature per clip encoding its normalized start and end time within the recording. This provides explicit temporal position information to the model, supplementing the sinusoidal positional encoding applied within the Transformer.

---

## 3. Proposed Improvements

### 3.1 Convolutional Local Saliency Refinement

The original Local Saliency Head directly computes per-clip saliency scores from independent clip embeddings, treating each clip in isolation. This ignores the natural temporal continuity of acoustic events—a sound event typically spans several consecutive clips and its salience varies smoothly over time.

We augment the Local Saliency Head with a lightweight 1D depthwise-separable convolutional module applied to the audio embeddings prior to the saliency score computation:

$$
\tilde{\mathbf{A}} = \text{Conv}_{1}(\text{GELU}(\text{DWConv}_{5}(\mathbf{A}))) + \mathbf{A}
$$

Specifically, we apply a depthwise 1D convolution with kernel size 5 (capturing a temporal context of 5 clips, i.e., approximately 5 seconds with 1-second clips) followed by a GELU activation and a pointwise ($1 \times 1$) convolution to mix channels. A residual connection preserves the original clip information. This allows the saliency head to produce smoother, temporally-aware salience estimates that better reflect the gradual onset and offset of acoustic events.

### 3.2 Focal Loss for Set Prediction

The bipartite matching criterion in DETR-based models assigns each ground truth span to exactly one predicted query. The remaining $N - |\text{GT}|$ predictions are classified as "no-object" (background). Since $N = 10$ queries typically match only 1–2 ground truth windows per audio, the training signal is highly class-imbalanced.

Standard Cross-Entropy treats all predictions equally, allowing the model to trivially minimize the loss by predicting background for most queries. We replace the classification term in the set prediction objective with **Focal Loss** [Lin et al., 2017]:

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

The $(1 - p_t)^{\gamma}$ modulating factor down-weights the contribution of well-classified easy negatives, forcing the training signal to concentrate on the harder, foreground moment queries. We use $\gamma = 2$ and $\alpha = 0.25$ following the standard Focal Loss parameterization.

The total training objective is a weighted combination of the span regression loss $\mathcal{L}_{\text{span}}$ (L1 and GIoU), the focal classification loss $\mathcal{L}_{\text{focal}}$, and the saliency contrastive loss $\mathcal{L}_{\text{sal}}$:

$$
\mathcal{L} = \lambda_s \mathcal{L}_{\text{span}} + \lambda_g \mathcal{L}_{\text{giou}} + \lambda_f \mathcal{L}_{\text{focal}} + \lambda_{sal} \mathcal{L}_{\text{sal}}
$$

### 3.3 Cosine Annealing Learning Rate Schedule

We replace the step-decay learning rate schedule of the baseline with a **Cosine Annealing** schedule [Loshchilov & Hutter, 2017]. The learning rate follows:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)
$$

where $\eta_{\max}$ is the initial learning rate, $\eta_{\min} = 10^{-6}$ is the minimum, and $T_{\max}$ is the total number of training epochs. This provides a smooth, gradual decay that avoids the abrupt discontinuities of step scheduling, often leading to better-calibrated final model weights.

---

## 4. Experimental Setup

### 4.1 Datasets

- **Clotho-Moment**: A language-based audio moment retrieval benchmark derived from the Clotho dataset. Each example consists of a 15–30 second audio clip paired with a natural language query and annotated temporal windows.
- **Castella**: An additional audio grounding dataset used for training and evaluation, following the same annotation format.

### 4.2 Evaluation Metrics

Following the DCASE 2026 challenge protocol, we evaluate using:
- **MR-R1@0.5** and **MR-R1@0.7**: Recall at 1 at IoU thresholds of 0.5 and 0.7, respectively.
- **MR-mAP@0.5**, **MR-mAP@0.75**, and **MR-mAP (average)**: Mean Average Precision averaged across multiple IoU thresholds.

### 4.3 Training Configuration

The model is trained with the AdamW optimizer, an initial learning rate of $10^{-4}$, and a weight decay of $10^{-4}$. Input audio features have a maximum length of 300 clips, and text queries are truncated to 32 tokens. The model uses a hidden dimension of 256, 2 encoder layers, 2 decoder layers, 8 attention heads, and $N = 10$ object queries.

---

## 5. References

- Carion et al. (2020). *End-to-End Object Detection with Transformers (DETR)*. ECCV 2020.
- Lei et al. (2021). *QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries (Moment-DETR)*. NeurIPS 2021.
- Lin et al. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.
- Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR 2017.
- Wu et al. (2023). *Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation (CLAP)*. ICASSP 2023.
- LCS-DETR baseline. *Lighthouse: A User-Friendly Library for Reproducible Video Moment Retrieval and Highlight Detection*. EMNLP 2024.
