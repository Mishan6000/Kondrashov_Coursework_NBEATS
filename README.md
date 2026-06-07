# Evaluating the Quality of the N-BEATS Model for Predicting Outliers

**National Research University Higher School of Economics (HSE)**  
Faculty of Computer Science  
**Author:** Kondrashov Mikhail Antonovich 

---

## Overview

This repository contains the experimental pipeline for evaluating N-BEATS (Neural Basis Expansion Analysis Time Series) in the context of unsupervised anomaly detection. The project investigates whether absolute forecasting errors from an N-BEATS network can serve as reliable indicators for structural breaks in time series data without explicit labels.

## Methodology

* **Data:** Synthetic Binary Telegraph Process (BTP) corrupted by varying stochastic noise regimes (White, Pink, Red, Blue, Green, Black) to test sensitivity to temporal correlation.
* **Approach:** N-BEATS is trained purely on an unsupervised forecasting task. Structural breaks are flagged when the absolute forecasting residual exceeds a strict upper percentile threshold (e.g., 95th).
* **Evaluation:** Margin-based classification metrics (Precision, Recall, F1-score) are computed via the Ruptures library using a tolerance margin of w = 2 time steps.

## Key Findings

* The zero-shot anomaly detection paradigm is viable, but success strongly depends on the underlying noise color.
* N-BEATS yields high detection performance in structured, correlated regimes (e.g., Red and Green noise) where breaks violently disrupt predictable drift.
* The model struggles in pure entropy settings (White noise) where baseline forecasting error is inherently volatile.
* In correlated environments, the residual geometry contains deep directional intelligence, allowing the model to accurately classify the direction of the structural shift.

## Execution Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
