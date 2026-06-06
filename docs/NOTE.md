# NOTE


## 02/05/2026

- https://github.com/h-munakata/Lighthouse-Wrapper-for-Audio-Moment-Retrieval
- https://arxiv.org/pdf/2410.10140

## 04/04/2026

- The baseline is quite clear for the idea of semantic search; but it could be a situation that ... we can try to use the AST model first for segment, the embed all text query, then sim-search in those text query, and get relevant segments. => This will be my first baseline.

## ...

- Count min, max, avg no. of words in each captions => If the number is reasonable, we can use another small language model (SmolLM) to generate more augmented-captions for each audio.

## 01/06/2026 (Architecture & Positional Encoding Refactoring)

- **Fixed RoPE Crash**: Re-wrote `RotaryEmbedding` to accept `(x, mask)` and act as a 1D additive positional encoding to match the DETR architecture expectations (previously it expected multiplicative scaling, causing violent PyTorch crashes).
- **Regularized Convolutional Positional Encoding**: Added LayerNorm, GELU, and Dropout (10%) to `ConvPositionalEncoding` (`position_embedding: conv`). Previously, the raw unregularized convolutions were dominating the signal and causing extreme "v-shaped" validation curve overfitting.
- **Implemented Learnable Fourier Features**: Added an `adaptive` mode to `PositionEmbeddingSine` (`position_embedding: sine_adaptive`). It wraps the static frequency bands in an `nn.Parameter()`, allowing the optimizer to stretch and compress the sinusoidal periods to match the DCASE audio dynamics perfectly.
- **Deleted ALiBi**: Removed `alibi_pos_enc.py` as it is a relative positional penalty (modifies attention matrices based on token distances) and is conceptually incompatible with DETR's cross-attention and learnable decoder queries, which require an absolute coordinate space.
- **Fixed Saliency-Guided Amplifier (Major mAP Fix)**: Discovered that `saliency_amplifier` was completely bypassed by the Decoder. Refactored `src/models/lcs_detr/model.py` to decouple the Transformer into `Encoder -> SaliencyAmplifier -> Decoder`. Bounding box queries now directly attend to the saliency-amplified features.
- **Optimized Negative Pairs Compute**: Bypassed the heavy 6-layer Decoder during the contrastive negative pairs generation, as only the Encoder outputs (`memory_neg`) are actually used for the negative saliency score loss. Saves ~35% VRAM and computation time.



## 06/06/2026

```mermaid
graph TB
    %% Inputs and Projections
    subgraph Input_Projections ["1. Input Features & Projections"]
        direction LR
        AudioIn["Audio Input (CLAP + 2 TEF) <br> [B, L_aud, 770]"] --> AudProj["Audio Projection (LinearLayer)"]
        TextIn["Text Input (CLAP) <br> [B, L_txt, 768]"] --> TextProj["Text Projection (LinearLayer)"]
        
        AudProj --> AudFeat["Audio Features (src_aud) <br> [B, L_aud, 256]"]
        TextProj --> TextFeat["Text Features (src_txt) <br> [B, L_txt, 256]"]
    end

    %% Saliency and Convolutional Refinement
    subgraph Saliency_Head ["2. Local Saliency Head & Conv Refinement"]
        AudFeat --> ConvRefine["Local Saliency Conv Refinement <br> Conv1D(GELU(DWConv5(A))) + A"]
        ConvRefine --> SalScore["Local Saliency Head <br> Cosine Similarity with Text Token"]
        TextFeat --> SalScore
        SalScore --> Sigmoid["Sigmoid Activation"]
        Sigmoid --> SalWeights["Saliency Weights <br> [B, L_aud]"]
        
        %% Saliency Loss
        SalScore --> SalLoss["Saliency Contrastive Loss"]
    end

    %% T2V Encoder
    subgraph T2V_Encoder ["3. Joint Text-to-Audio (T2V) Encoder"]
        AudFeat --> Concat["Concatenation Block"]
        TextFeat --> Concat
        GlobalToken["Global Representation Token <br> [B, 1, 256]"] --> Concat
        
        PosEmbed["Positional Embeddings <br> [Sine / Sine-Adaptive]"] --> AddPos["Add Position Embeddings"]
        Concat --> AddPos
        
        AddPos --> SGCA["Saliency-Guided Cross Attention <br> Text-to-Audio Interaction"]
        SalWeights --> SGCA
        SGCA --> EncLayer["Transformer Encoder Layers"]
        EncLayer --> EncoderOutput["Encoder Sequence <br> [Global + Audio + Text]"]
    end

    %% Saliency Amplifier
    subgraph Saliency_Amplification ["4. Saliency Amplifier"]
        EncoderOutput --> Split["Strip Text Tokens <br> [Global + Audio Memory]"]
        Split --> AudioMem["Audio Memory (aud_mem) <br> [B, L_aud, 256]"]
        
        AudioMem --> Amp["Saliency Amplifier <br> Multi-Head Cross Attention"]
        SalWeights --> Amp
        Amp --> AmplifiedMem["Amplified Audio Memory <br> [B, L_aud, 256]"]
    end

    %% Decoder and Iterative Box Updates
    subgraph Decoder_Stack ["5. DAB-DETR Decoder Stack"]
        %% Queries and Position Embeddings
        AnchorQueries["Anchor Queries (Learnable) <br> [10, B, 2] - (center, width)"] --> InverseSigmoid["Inverse Sigmoid"]
        InverseSigmoid --> RefPoints["Reference Points"]
        
        RefPoints --> SineEmbed["Sine Embedding Generator"]
        SineEmbed --> QueryPos["Query Positional Embedding"]
        
        %% Decoder Loops
        QueryPos --> DecSelfAttn["Decoder Self-Attention"]
        TgtInputs["Target Input Tokens <br> [10, B, 256] - (Init: 0)"] --> DecSelfAttn
        
        DecSelfAttn --> DecCrossAttn["Decoder Cross-Attention"]
        AmplifiedMem --> DecCrossAttn
        
        DecCrossAttn --> FFN["Feed-Forward Network (FFN)"]
        
        %% Iterative update loop
        FFN --> HS["Decoder Hidden States"]
        HS --> BoxRefine["Iterative Coordinate Adjustment <br> (MLP projection added to raw ref points)"]
        BoxRefine --> UpdatedRefPoints["Updated Reference Points <br> Passed to Layer (l + 1)"]
        
        UpdatedRefPoints --> SineEmbed
    end

    %% Output Heads and Losses
    subgraph Prediction_Losses ["6. Output Heads & Optimization Losses"]
        HS --> ClassEmbed["Class Embedding Head (Linear)"]
        HS --> CoordEmbed["Coordinate Embedding Head (MLP)"]
        
        ClassEmbed --> PredLogits["Predicted Class Logits <br> [B, 10, 2]"]
        CoordEmbed --> PredSpans["Predicted Temporal Spans <br> [B, 10, 2] - (center, width)"]
        
        PredLogits --> Matcher["Hungarian Bipartite Matcher"]
        PredSpans --> Matcher
        
        Matcher --> FocalLoss["Quality Focal Loss (Labels)"]
        Matcher --> L1Loss["L1 Loss (Coordinates)"]
        Matcher --> DIouLoss["Temporal DIoU Loss (tdiou)"]
    end

    %% Connecting main components
    EncoderOutput -.-> SalScore
    SalWeights -.-> Amp
    AmplifiedMem --> DecCrossAttn
```