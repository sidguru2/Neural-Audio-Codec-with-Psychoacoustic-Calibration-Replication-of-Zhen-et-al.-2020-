# Neural-Audio-Codec-with-Psychoacoustic-Calibration-Replication-of-Zhen-et-al.-2020-
This project replicates and extends "Psychoacoustic Calibration of Loss Functions for Efficient End-to-End Neural Audio Coding" (Zhen et al., 2020). It implements a neural audio codec using lightweight autoencoders, soft-to-hard quantization, SSE loss, mel-frequency loss, and priority weighting(psychoacoustic loss informed by masking thresholds).

Completed:
Model-A and Model-B architectures
Priority weighting, mel-frequency, and SSE loss
Custom PAM-1 implementation for global masking
Audio reconstruction with overlap-add synthesis

Work in progress: 
Model-C (training) and Model-D, noise modulation, temporal masking integration

Dataset: Custom curated multi-genre dataset (~5.5 hrs audio)
Author: Sid Gurumurthi — based on Zhen et al. (2020)
PI: Professor Thomas Moon
