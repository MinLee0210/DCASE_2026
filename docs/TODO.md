# TODO

### 10-12/04/2026

- [ ] EDA Clotho-Moment
- [ ] EDA DCASE Task4

#### Phase 1: Basic Statistics ⬜
- [ ] Load and parse JSON metadata across all splits
- [ ] Count total events per split
- [ ] Calculate events per sample ratio
- [ ] Check for missing files or corrupted data

#### Phase 2: Temporal Analysis ⬜
- [ ] Event duration distribution (min, max, mean, median)
- [ ] Event start time distribution
- [ ] Analyze temporal overlap patterns between events
- [ ] Identify outliers (very short/long events)

#### Phase 3: Audio Analysis ⬜
- [ ] Extract audio features (RMS, ZCR, MFCC)
- [ ] Analyze foreground vs background audio levels (dB)
- [ ] Generate spectrograms and mel-spectrograms
- [ ] Sample rate and bit depth analysis
- [ ] Audio quality assessment

#### Phase 4: Text Analysis ⬜
- [ ] Caption length distribution
- [ ] Vocabulary size and word frequency
- [ ] Identify unique sound event types
- [ ] Text diversity and coverage analysis
- [ ] Stopword filtering and semantic analysis

#### Phase 5: Quality Assurance ⬜
- [ ] Count missing/null values
- [ ] Validate temporal annotations (start_time + duration <= total audio length)
- [ ] Check for duplicates
- [ ] Identify anomalies and edge cases

#### Phase 6: Visualizations ⬜
- [ ] Distribution histograms (duration, start_time, caption length)
- [ ] Scatter plots (bg dB vs fg dB)
- [ ] Waveform and spectrogram plots for sample events
- [ ] Word frequency bar charts
- [ ] Temporal distribution heatmaps

- [ ] Re-run task 6 baseline.


### 03/04/2026

[x] Read baseline
    - [x] Task 3
    - [x] Task 6
[x] Run experiment dataset:
    - Cloth-Moment
  
Note: Task 3 have no experimental dataset.