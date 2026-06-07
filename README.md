# Evaluating the Quality of the N-BEATS Model for Predicting Outliers

**National Research University Higher School of Economics (HSE)**  
Faculty of Computer Science  
Bachelor’s Programme “Data Science and Business Analytics”  

**Fulfilled by:** Kondrashov Mikhail Antonovich (Group #БПАД242)  
**Assessed by:** Petr Pavlovich Lukianchenko (Research Fellow, FCS, HSE)  

---

## Abstract

This research aims to evaluate the quality of the N-BEATS (Neural Basis Expansion Analysis Time Series) model for outlier prediction. While N-BEATS has recommended itself as a strong model for univariate and multivariate time series forecasting, its outlier prediction facilities remain largely unexplored. This repository contains the implementation of the N-BEATS algorithm for multi-step time series forecasting, utilized specifically to evaluate whether the forecasting errors of a predictive neural network can be used as reliable indicators for structural breaks without explicit labels.

Our empirical evaluation demonstrates that N-BEATS forecasting residuals can successfully indicate structural breaks. We additionally characterize the temporal structure of these anomalies, showing that optimal hyperparameters and resultant inter-break gap distributions are dependent on the spectral color of background noise.

## Research Tasks and Objectives

**Tasks:**
* Review relevant literature on time series forecasting and anomaly detection methods.
* Implement the N-BEATS model for multi-step forecasting.
* Implement several anomaly detection algorithms (PELT, LSTM-AD, Anomaly-Transformer, AnomalyBERT, Sub-LoF, VLT-AnomalyDetection). *(Note: This repository primarily highlights the N-BEATS evaluation pipeline).*
* Prepare a dataset with labeled outliers.
* Compare N-BEATS forecast errors with detector outputs.
* Decide on appropriate metrics.

**Objectives:**
* Determine if N-BEATS forecasting errors can indicate outliers.
* Compare results of outlier indication with N-BEATS to specialized detectors.
* Provide reproducible analysis and code.

## Data and Methodology

**Mathematics of Synthetic Data**  
To evaluate the capability of N-BEATS to indicate structural breaks in an unsupervised forecasting setting, we generate synthetic time series representing a Binary Telegraph Process (BTP). The latent piecewise-constant signal is corrupted by various stochastic noise regimes to test the model's sensitivity to temporal correlation: White, Pink, Red, Blue, Green, and Black noise.

**Forecasting as Outlier Detection**  
Unlike explicit classifiers trained on binary break labels, this approach relies entirely on the autoregressive forecasting residual. The structural break indication score is derived directly from the absolute forecasting error:
|x_{t+1} - \hat{x}_{t+1}|

A breakpoint is flagged if this error exceeds a strict upper percentile threshold (e.g., 95th or 98th) of the validation errors. To align with standard change-point detection evaluation and account for temporal imprecision, we use a strict tolerance margin of w = 2 time steps, evaluating performance via Precision, Recall, margin-based F1-score, PR-AUC, and ROC-AUC metrics via the Ruptures library.

## Key Results and Discussion

* **Forecasting Errors and Explicit Classification:** Our evaluation reveals a distinct theoretical mechanism. Regime transitions violently disrupt the local temporal patterns learned by the network, resulting in transient error spikes. This zero-shot anomaly detection paradigm is highly viable, but its success strongly depends on the underlying noise color.
* **Impact of Noise Color:** Highly structured, persistent regimes (e.g., Red and Green noise) yielded the best detection performance (achieving a Recall of 1.000 and 0.989). N-BEATS excels at modeling the slow, predictable drift of correlated noise, making structural breaks easily distinguishable. Conversely, in weakly structured or high-frequency regimes (White noise), the baseline forecasting error is inherently high and volatile, entirely masking the actual structural break.
* **Signed Predictions:** Beyond merely identifying that a break occurred, highly correlated noises allow the model to accurately predict the direction of the structural shift (upward vs. downward step), proving that the geometry of the residual contains deep directional intelligence.
* **Distribution of Predicted Inter-Break Gaps:** N-BEATS exhibits distinct detection rhythms depending on the noise color. The inter-break gaps successfully fit parametric target distributions (Normal, Student-t, Cauchy), minimizing the Kolmogorov-Smirnov distance depending on the configuration.

## Repository Structure and Execution

* `5267nbeats_pipeline_synth.py`: Main executable script containing the `RobustBTP` synthetic data generator, the simplified `SimpleNBeats` PyTorch architecture, and the experimental pipeline.
* `requirements.txt`: Python environment dependencies.
* `.gitignore`: Configuration to exclude generated datasets, temporary environments, and local system files.

### Execution Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   cd YOUR_REPOSITORY_NAME
