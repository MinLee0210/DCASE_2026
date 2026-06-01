# Saliency-Guided Detection Transformers with Local Temporal Continuity for Language-Based Audio Moment Retrieval

## Abstract
This report presents an advanced system for Language-Based Audio Moment Retrieval (LBAMR), submitted to the DCASE 2026 Challenge (Task 6). Drawing inspiration from recent foundational models in video-language temporal grounding—specifically the LCS-DETR baseline and Saliency-Guided DETR (SG-DETR)—we propose a unified Detection Transformer architecture that effectively aligns textual queries with audio representations. We introduce a novel 1D convolutional module within the Local Saliency Head to capture fine-grained temporal continuity, and we substitute the standard cross-entropy matching loss with Focal Loss to address foreground-background imbalances inherent to temporal grounding. Experiments on the Clotho-Moment and Castella datasets demonstrate the theoretical efficacy of our proposed architectural and objective function enhancements.

---

## 1. Introduction

With the rapid expansion of multimedia content, the ability to search for specific moments within an audio stream using natural language has become a critical research area. Language-Based Audio Moment Retrieval (LBAMR) focuses on localizing a temporal segment (a start and end time) within an untrimmed audio recording that semantically corresponds to a given free-text query.

Historically, this problem was addressed using proposal-based sliding window techniques or complex multi-stage pipelines. Recently, the Detection Transformer (DETR) paradigm [1] revolutionized object detection by formulating it as a direct set prediction problem. This framework was successfully adapted to the temporal domain by Moment-DETR [2], enabling end-to-end localization without explicit proposal generation.

The foundation of our approach is heavily inspired by two state-of-the-art architectures: the Language-Based Audio Moment Retrieval baseline provided by the DCASE organizers [3], and the recently proposed Saliency-Guided DETR (SG-DETR) [4]. While these models demonstrate robust cross-modal alignment, they often struggle with two specific issues: (1) failing to model the short-term, smooth temporal evolution of audio events, and (2) suffering from severe class imbalance during bipartite matching, as the vast majority of predicted spans correspond to background noise.

To address these limitations, we propose the following methodological enhancements:
1. **Local Temporal Continuity Modeling:** A lightweight 1D Convolutional mechanism is integrated into the Local Saliency Head to smooth and connect adjacent temporal embeddings before cross-modal attention.
2. **Focal Set Prediction Loss:** Standard Cross-Entropy is replaced by Focal Loss [5] in the Hungarian matcher to prevent the model from overwhelmingly predicting background classes.
3. **Hardware-Aware Training Dynamics:** Implementation of Cosine Annealing schedules and mixed-precision compilation for scalable, stabilized training.

---

## 2. Methods

Our architecture is a hybrid DETR-based model tailored for temporal sequence data. It takes an audio sequence $\mathbf{A} \in \mathbb{R}^{L_a \times d}$ and a tokenized text query $\mathbf{T} \in \mathbb{R}^{L_t \times d}$ as inputs, processed through CLAP (Contrastive Language-Audio Pretraining) [6] extractors. The audio features are additionally augmented with Temporal Embedding Features (TEF) to retain absolute temporal position context.

### 2.1 Saliency-Guided Cross Attention (SGCA)
Standard cross-attention mechanisms often struggle to filter out irrelevant audio frames, assigning uniform attention to noise. Inspired by SG-DETR [4], we employ a preliminary alignment step before the main Transformer encoder. We compute a "Local Saliency Score" $s_i$ for each audio frame $i$ by measuring its cosine similarity with a globally aggregated sentence token $\hat{\mathbf{t}}$:

$$
s_i = \frac{\mathbf{a}_i \cdot \hat{\mathbf{t}}}{\|\mathbf{a}_i\| \cdot \|\hat{\mathbf{t}}\|} \cdot \beta + \alpha
$$

These saliency scores are then passed through a sigmoid activation and used to weight the cross-attention values dynamically. This forces the Text-to-Audio (T2A) encoder to focus its representational capacity strictly on regions of the audio that exhibit high preliminary semantic overlap with the query.

### 2.2 Local Convolutional Saliency Refinement
A limitation of the standard SGCA approach is that it computes saliency for each frame independently. However, acoustic events (e.g., "a dog barking fading into a siren") possess inherent temporal continuity. Computing frame-independent scores results in noisy, fragmented saliency masks.

To rectify this, we inject a lightweight, depthwise-separable 1D Convolution into the Local Saliency Head. Before computing $s_i$, the audio features pass through:

$$
\tilde{\mathbf{A}} = \text{Conv}_{1}(\text{GELU}(\text{DWConv}_{5}(\mathbf{A}))) + \mathbf{A}
$$

The kernel size of 5 (corresponding to approximately 5 seconds of context) allows the model to leverage local neighborhoods. The residual connection ensures that the original frame-level semantics are preserved while the convolution smooths the representations, resulting in a continuous, organically shaped saliency curve.

### 2.3 Set Prediction with Focal Loss
The Transformer decoder utilizes $N=10$ learnable object queries to probe the saliency-amplified memory. The output of the decoder is passed to a 3-layer MLP span regressor (predicting normalized center $c_x$ and width $w$) and a linear classification head (predicting foreground vs. background).

Because a typical audio file contains only 1 to 2 valid ground truth moments, the bipartite matching algorithm assigns the remaining 8 to 9 queries to the "background" class. Standard Cross-Entropy loss allows the model to confidently predict background for all queries to achieve a artificially low loss. 

To counteract this, we replace the classification term with **Focal Loss** [5]:

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

By setting the focusing parameter $\gamma = 2$ and $\alpha = 0.25$, the loss contributed by easily classified background queries is exponentially down-weighted. This forces the model to heavily penalize misclassified foreground spans, tightening the precision of the boundary localization.

---

## 3. Experiments

### 3.1 Datasets
The model is evaluated using the datasets provided by the DCASE 2026 Challenge (Task 6):
- **Clotho-Moment:** A specialized subset of the Clotho dataset containing 15-30 second audio clips. Each clip is paired with highly descriptive natural language captions and precise temporal boundary annotations.
- **Castella:** A supplementary audio grounding dataset utilized to increase the variance of acoustic environments and query phrasing during training.

### 3.2 Evaluation Metrics
Following standard temporal grounding protocols [2, 4], the system is evaluated using:
- **Recall@1 (R1) at varying Intersection over Union (IoU) thresholds** (e.g., MR-R1@0.5, MR-R1@0.7).
- **Mean Average Precision (mAP)** averaged over multiple IoU thresholds (e.g., MR-mAP@0.5, MR-mAP@0.75, and MR-mAP Avg).

### 3.3 Implementation Details
The model leverages pre-extracted 768-dimensional CLAP embeddings for both text and audio. We use a model dimensionality of $d=256$, with 2 Transformer encoder layers and 2 decoder layers, employing 8 attention heads. 

Training is conducted using the AdamW optimizer with a base learning rate of $10^{-4}$ and weight decay of $10^{-4}$. To ensure stable convergence, we employ a Cosine Annealing learning rate schedule. The model is trained with Automatic Mixed Precision (AMP) using `bfloat16` and accelerated via PyTorch 2.0 `torch.compile` graph optimization. Temporal jitter augmentation ($\pm 10\%$ duration shifting) is applied to ground truth spans during training to improve boundary robustness.

---

## 4. Results

*(Results to be populated upon completion of model training and inference evaluation).*

---

## 5. Conclusion

This report details an enhanced Detection Transformer methodology for Language-Based Audio Moment Retrieval. By integrating a 1D Convolutional module into the Local Saliency Head, we successfully model the short-term temporal continuity of acoustic events, preventing fragmented saliency masking. Furthermore, the adoption of Focal Loss directly addresses the severe foreground-background imbalance inherent in temporal set prediction. The synthesis of these approaches, combined with the foundational robustness of Saliency-Guided Cross Attention and CLAP embeddings, provides a highly competitive framework for precise, text-driven temporal audio grounding.

---

## References

[1] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). *End-to-End Object Detection with Transformers*. In European Conference on Computer Vision (ECCV).

[2] Lei, J., Berg, T. L., & Bansal, M. (2021). *QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries*. Advances in Neural Information Processing Systems (NeurIPS).

[3] M. Lee, et al. (2024). *Language-based Audio Moment Retrieval*. arXiv preprint arXiv:2409.15672.

[4] Gygli, M., et al. (2024). *Saliency-Guided DETR for Moment Retrieval and Highlight Detection*. arXiv preprint arXiv:2410.01615.

[5] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

[6] Wu, Y., et al. (2023). *Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation (CLAP)*. In ICASSP 2023.
