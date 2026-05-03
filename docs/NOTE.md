# NOTE


## 02/05/2026

- https://github.com/h-munakata/Lighthouse-Wrapper-for-Audio-Moment-Retrieval
- https://arxiv.org/pdf/2410.10140

## 04/04/2026

- The baseline is quite clear for the idea of semantic search; but it could be a situation that ... we can try to use the AST model first for segment, the embed all text query, then sim-search in those text query, and get relevant segments. => This will be my first baseline.

## ...

- Count min, max, avg no. of words in each captions => If the number is reasonable, we can use another small language model (SmolLM) to generate more augmented-captions for each audio.